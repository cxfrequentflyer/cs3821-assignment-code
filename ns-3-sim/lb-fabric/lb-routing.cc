#include "lb-routing.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace ns3
{
namespace lb
{

NS_LOG_COMPONENT_DEFINE("LbRouting");
NS_OBJECT_ENSURE_REGISTERED(LbRouting);

namespace
{

uint64_t
Mix(uint64_t x)
{
    // splitmix64 finalizer: a good 64-bit mixer, so hash % n is uniform
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

const std::vector<std::pair<std::string, Scheme>> kSchemes = {
    {"ecmp", Scheme::Ecmp},
    {"wcmp", Scheme::Wcmp},
    {"dchoice", Scheme::DChoice},
    {"flowlet", Scheme::Flowlet},
    {"conga", Scheme::Conga},
    {"presto", Scheme::Presto},
    {"random", Scheme::Random},
    {"rr", Scheme::RoundRobin},
    {"drill", Scheme::Drill},
    {"drill-comp", Scheme::DrillComp},
};

} // namespace

Scheme
ParseScheme(const std::string& name)
{
    for (const auto& [n, s] : kSchemes)
    {
        if (n == name)
        {
            return s;
        }
    }
    throw std::invalid_argument("unknown scheme '" + name + "'");
}

std::string
SchemeName(Scheme s)
{
    for (const auto& [n, sc] : kSchemes)
    {
        if (sc == s)
        {
            return n;
        }
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Fabric
// ---------------------------------------------------------------------------

void
Fabric::Init(uint32_t nLeaves, uint32_t nSpines)
{
    leaves = nLeaves;
    spines = nSpines;
    up.assign(leaves, std::vector<Ptr<PointToPointNetDevice>>(spines));
    down.assign(spines, std::vector<Ptr<PointToPointNetDevice>>(leaves));
    upRate.assign(leaves, std::vector<double>(spines, 0.0));
    downRate.assign(spines, std::vector<double>(leaves, 0.0));
    spineAddr.assign(leaves, std::vector<Ipv4Address>(spines));
    retired.assign(leaves, std::vector<bool>(spines, false));
}

void
Fabric::SetRetired(uint32_t l, uint32_t s, bool value)
{
    retired[l][s] = value;
    Build();
}

void
Fabric::Build()
{
    usable.assign(leaves, std::vector<std::vector<uint32_t>>(leaves));
    for (uint32_t a = 0; a < leaves; a++)
    {
        for (uint32_t b = 0; b < leaves; b++)
        {
            for (uint32_t s = 0; s < spines && a != b; s++)
            {
                if (InService(a, s) && down[s][b])
                {
                    usable[a][b].push_back(s);
                }
            }
        }
    }
    // Link labels: the (src, dst) leaf pairs that can use the link. Two links
    // are symmetric when their labels match; two paths when their links do,
    // hop by hop, with the same capacity factor (DRILL §3.4.1 and §3.4.3).
    auto upLabel = [&](uint32_t a, uint32_t s) {
        std::vector<uint32_t> lab;
        for (uint32_t b = 0; b < leaves; b++)
        {
            if (a != b && InService(a, s) && down[s][b])
            {
                lab.push_back(a * leaves + b);
            }
        }
        return lab;
    };
    auto downLabel = [&](uint32_t s, uint32_t b) {
        std::vector<uint32_t> lab;
        for (uint32_t a = 0; a < leaves; a++)
        {
            if (a != b && InService(a, s) && down[s][b])
            {
                lab.push_back(a * leaves + b);
            }
        }
        return lab;
    };
    comps.assign(leaves, std::vector<std::vector<Component>>(leaves));
    for (uint32_t a = 0; a < leaves; a++)
    {
        for (uint32_t b = 0; b < leaves; b++)
        {
            using Score = std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, int64_t>;
            std::map<Score, size_t> index;
            for (uint32_t s : usable[a][b])
            {
                Score key{upLabel(a, s),
                          downLabel(s, b),
                          static_cast<int64_t>(1e6 * upRate[a][s] / downRate[s][b])};
                auto it = index.find(key);
                if (it == index.end())
                {
                    index[key] = comps[a][b].size();
                    comps[a][b].push_back(Component{{}, 0.0});
                    it = index.find(key);
                }
                auto& c = comps[a][b][it->second];
                c.spines.push_back(s);
                c.weight += std::min(upRate[a][s], downRate[s][b]);
            }
        }
    }
}

uint32_t
Fabric::LeafOf(Ipv4Address host)
{
    return (host.Get() >> 16) & 0xff;
}

double
Fabric::UpBacklog(uint32_t l, uint32_t s) const
{
    return up[l][s]->GetQueue()->GetNBytes() * 8.0 / upRate[l][s];
}

double
Fabric::DownBacklog(uint32_t s, uint32_t l) const
{
    return down[s][l]->GetQueue()->GetNBytes() * 8.0 / downRate[s][l];
}

// ---------------------------------------------------------------------------
// LbRouting
// ---------------------------------------------------------------------------

TypeId
LbRouting::GetTypeId()
{
    static TypeId tid = TypeId("ns3::lb::LbRouting")
                            .SetParent<Ipv4RoutingProtocol>()
                            .SetGroupName("Internet")
                            .AddConstructor<LbRouting>();
    return tid;
}

LbRouting::LbRouting()
    : m_rng(CreateObject<UniformRandomVariable>())
{
}

void
LbRouting::Setup(uint32_t leaf, const Fabric* fabric, const Config& config)
{
    m_leaf = leaf;
    m_fabric = fabric;
    m_cfg = config;
    m_engines.assign(std::max(1u, config.engines), Engine{});
}

Ptr<Ipv4Route>
LbRouting::RouteOutput(Ptr<Packet>, const Ipv4Header&, Ptr<NetDevice>, Socket::SocketErrno& sockerr)
{
    sockerr = Socket::ERROR_NOROUTETOHOST; // a switch originates nothing
    return nullptr;
}

double
LbRouting::Backlog(uint32_t engine, uint32_t spine)
{
    if (m_cfg.staleness.IsZero())
    {
        return m_fabric->UpBacklog(m_leaf, spine);
    }
    Time now = Simulator::Now();
    if (m_snapshotAt.IsNegative() || now - m_snapshotAt >= m_cfg.staleness)
    {
        m_snapshot.assign(m_fabric->spines, 0.0);
        for (uint32_t s = 0; s < m_fabric->spines; s++)
        {
            if (m_fabric->up[m_leaf][s])
            {
                m_snapshot[s] = m_fabric->up[m_leaf][s]->GetQueue()->GetNBytes();
            }
        }
        for (auto& e : m_engines)
        {
            e.own.assign(m_fabric->spines, 0.0);
        }
        m_snapshotAt = now;
    }
    // the engine's estimate: the snapshot plus its own packets, less what the
    // link can have sent since the snapshot (a queue cannot go below empty)
    double rate = m_fabric->upRate[m_leaf][spine];
    double bytes = m_snapshot[spine] + m_engines[engine].own[spine] -
                   rate / 8 * (now - m_snapshotAt).GetSeconds();
    return std::max(0.0, bytes) * 8.0 / rate;
}

uint32_t
LbRouting::DrillPick(const std::vector<uint32_t>& allowed,
                     uint32_t engine,
                     uint64_t memKey,
                     uint32_t d,
                     uint32_t memory)
{
    const uint32_t n = allowed.size();
    // min(d, n) distinct samples in random order (a partial Fisher-Yates
    // shuffle). Shuffling even when d >= n matters: ties (empty queues are
    // common) go to the first candidate, and a fixed order would send them
    // all to the lowest-numbered spine.
    std::vector<uint32_t> cands;
    std::vector<uint32_t> pool = allowed;
    for (uint32_t k = 0; k < std::min(d, n); k++)
    {
        uint32_t j = k + m_rng->GetInteger(0, n - 1 - k);
        std::swap(pool[k], pool[j]);
        cands.push_back(pool[k]);
    }
    std::vector<uint32_t>* mem = nullptr;
    if (memory > 0)
    {
        mem = &m_engines[engine].memory[memKey];
        for (uint32_t s : *mem)
        {
            if (std::find(allowed.begin(), allowed.end(), s) != allowed.end())
            {
                cands.push_back(s);
            }
        }
    }
    std::vector<double> seen(cands.size());
    size_t best = 0;
    for (size_t i = 0; i < cands.size(); i++)
    {
        seen[i] = Backlog(engine, cands[i]);
        if (seen[i] < seen[best])
        {
            best = i; // strictly less: ties keep the earlier (random) sample
        }
    }
    uint32_t pick = cands[best];
    if (mem)
    {
        // remember the chosen uplink, then the other distinct candidates, least loaded first
        std::vector<std::pair<double, uint32_t>> rest;
        for (size_t i = 0; i < cands.size(); i++)
        {
            bool dup = cands[i] == pick;
            for (const auto& r : rest)
            {
                dup = dup || r.second == cands[i];
            }
            if (!dup)
            {
                rest.emplace_back(seen[i], cands[i]);
            }
        }
        std::stable_sort(rest.begin(), rest.end(), [](const auto& x, const auto& y) {
            return x.first < y.first;
        });
        mem->assign(1, pick);
        for (size_t i = 0; i < rest.size() && mem->size() < memory; i++)
        {
            mem->push_back(rest[i].second);
        }
    }
    return pick;
}

uint32_t
LbRouting::Choose(uint32_t dstLeaf, uint64_t flowHash, uint32_t payload, uint32_t engine)
{
    const Fabric& f = *m_fabric;
    const std::vector<uint32_t>& allowed = f.usable[m_leaf][dstLeaf];
    const uint32_t n = allowed.size();
    Time now = Simulator::Now();
    auto weighted = [&](const std::vector<uint32_t>& spines, uint64_t h) {
        double total = 0;
        for (uint32_t s : spines)
        {
            total += std::min(f.upRate[m_leaf][s], f.downRate[s][dstLeaf]);
        }
        double u = (h >> 11) * (1.0 / 9007199254740992.0) * total; // 53 bits -> [0, total)
        for (uint32_t s : spines)
        {
            u -= std::min(f.upRate[m_leaf][s], f.downRate[s][dstLeaf]);
            if (u < 0)
            {
                return s;
            }
        }
        return spines.back();
    };

    switch (m_cfg.scheme)
    {
    case Scheme::Ecmp:
        return allowed[flowHash % n];
    case Scheme::Wcmp:
        return weighted(allowed, flowHash);
    case Scheme::Random:
        return allowed[m_rng->GetInteger(0, n - 1)];
    case Scheme::RoundRobin:
        return allowed[m_engines[engine].rr++ % n];
    case Scheme::Drill:
        return DrillPick(allowed, engine, dstLeaf, m_cfg.d, m_cfg.memory);
    default:
        break;
    }

    FlowState& st = m_flows[flowHash];
    // a pinned spine may have been retired since (Fabric::SetRetired): choose again
    auto inService = [&](uint32_t s) {
        return std::find(allowed.begin(), allowed.end(), s) != allowed.end();
    };
    switch (m_cfg.scheme)
    {
    case Scheme::DChoice:
        if (!st.init || !inService(st.spine))
        {
            st.spine = DrillPick(allowed, engine, dstLeaf, m_cfg.d, 0);
        }
        break;
    case Scheme::Flowlet:
    case Scheme::Conga:
        if (!st.init || now - st.last > m_cfg.flowletGap || !inService(st.spine))
        {
            if (m_cfg.scheme == Scheme::Flowlet)
            {
                st.spine = allowed[m_rng->GetInteger(0, n - 1)];
            }
            else
            {
                // least congested path; ties broken from a random starting point
                uint32_t off = m_rng->GetInteger(0, n - 1);
                double best = 0;
                for (uint32_t k = 0; k < n; k++)
                {
                    uint32_t s = allowed[(off + k) % n];
                    double c = std::max(f.UpBacklog(m_leaf, s), f.DownBacklog(s, dstLeaf));
                    if (k == 0 || c < best)
                    {
                        best = c;
                        st.spine = s;
                    }
                }
            }
        }
        st.last = now;
        break;
    case Scheme::Presto:
        if (!st.init)
        {
            st.start = Mix(flowHash) % n;
        }
        if (payload > 0)
        {
            if (st.cellBytes + payload > m_cfg.cellBytes && st.cellBytes > 0)
            {
                st.cellIndex++;
                st.cellBytes = 0;
            }
            st.cellBytes += payload;
        }
        st.spine = allowed[(st.start + st.cellIndex) % n];
        break;
    case Scheme::DrillComp: {
        const auto& comps = f.comps[m_leaf][dstLeaf];
        if (!st.init || st.comp >= static_cast<int32_t>(comps.size()))
        {
            double total = 0;
            for (const auto& c : comps)
            {
                total += c.weight;
            }
            double u = (Mix(flowHash) >> 11) * (1.0 / 9007199254740992.0) * total;
            st.comp = comps.size() - 1;
            for (size_t i = 0; i < comps.size(); i++)
            {
                u -= comps[i].weight;
                if (u < 0)
                {
                    st.comp = i;
                    break;
                }
            }
        }
        st.init = true;
        return DrillPick(comps[st.comp].spines,
                         engine,
                         uint64_t(dstLeaf) * 1024 + st.comp,
                         m_cfg.d,
                         m_cfg.memory);
    }
    default:
        NS_FATAL_ERROR("unhandled scheme");
    }
    st.init = true;
    return st.spine;
}

Ptr<Ipv4Route>
LbRouting::RouteVia(uint32_t spine, const Ipv4Header& header)
{
    Ptr<Ipv4Route> r = m_routes[spine];
    r->SetDestination(header.GetDestination());
    r->SetSource(header.GetSource());
    return r;
}

bool
LbRouting::RouteInput(Ptr<const Packet> p,
                      const Ipv4Header& header,
                      Ptr<const NetDevice> idev,
                      const UnicastForwardCallback& ucb,
                      const MulticastForwardCallback&,
                      const LocalDeliverCallback&,
                      const ErrorCallback&)
{
    Ipv4Address dst = header.GetDestination();
    if (dst.IsMulticast() || dst.IsBroadcast() || (dst.Get() >> 24) != 10)
    {
        return false;
    }
    uint32_t dstLeaf = Fabric::LeafOf(dst);
    if (dstLeaf == m_leaf || dstLeaf >= m_fabric->leaves)
    {
        return false; // a local host: static routing delivers it
    }
    if (m_routes.empty())
    {
        m_routes.resize(m_fabric->spines);
        m_sent.assign(m_fabric->spines, 0);
        for (uint32_t s = 0; s < m_fabric->spines; s++)
        {
            if (m_fabric->up[m_leaf][s])
            {
                m_routes[s] = Create<Ipv4Route>();
                m_routes[s]->SetGateway(m_fabric->spineAddr[m_leaf][s]);
                m_routes[s]->SetOutputDevice(m_fabric->up[m_leaf][s]);
            }
        }
    }
    if (m_fabric->usable[m_leaf][dstLeaf].empty())
    {
        return false;
    }

    uint16_t sport = 0;
    uint16_t dport = 0;
    uint32_t payload = 0;
    if (header.GetProtocol() == 6)
    {
        TcpHeader th;
        p->PeekHeader(th);
        sport = th.GetSourcePort();
        dport = th.GetDestinationPort();
        payload = p->GetSize() - th.GetSerializedSize();
    }
    else if (header.GetProtocol() == 17)
    {
        UdpHeader uh;
        p->PeekHeader(uh);
        sport = uh.GetSourcePort();
        dport = uh.GetDestinationPort();
        payload = p->GetSize() - uh.GetSerializedSize();
    }
    uint64_t key = (uint64_t(header.GetSource().Get()) << 32) ^ dst.Get();
    key = Mix(key ^ (uint64_t(sport) << 16 | dport) ^ (uint64_t(header.GetProtocol()) << 40) ^
              (uint64_t(m_cfg.hashSeed) << 48));
    uint32_t iif = m_ipv4->GetInterfaceForDevice(idev);
    uint32_t engine = iif % m_engines.size();

    uint32_t spine = Choose(dstLeaf, key, payload, engine);
    m_sent[spine]++;
    if (!m_cfg.staleness.IsZero() && !m_engines[engine].own.empty())
    {
        m_engines[engine].own[spine] += p->GetSize() + header.GetSerializedSize();
    }
    ucb(RouteVia(spine, header), p, header);
    return true;
}

void
LbRouting::NotifyInterfaceUp(uint32_t)
{
}

void
LbRouting::NotifyInterfaceDown(uint32_t)
{
}

void
LbRouting::NotifyAddAddress(uint32_t, Ipv4InterfaceAddress)
{
}

void
LbRouting::NotifyRemoveAddress(uint32_t, Ipv4InterfaceAddress)
{
}

void
LbRouting::SetIpv4(Ptr<Ipv4> ipv4)
{
    m_ipv4 = ipv4;
}

void
LbRouting::PrintRoutingTable(Ptr<OutputStreamWrapper> stream, Time::Unit) const
{
    *stream->GetStream() << "LbRouting on leaf " << m_leaf << ", scheme "
                         << SchemeName(m_cfg.scheme) << "\n";
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

LbRoutingHelper::LbRoutingHelper(const Fabric* fabric,
                                 const Config& config,
                                 const std::unordered_map<uint32_t, uint32_t>* leafOf)
    : m_fabric(fabric),
      m_cfg(config),
      m_leafOf(leafOf)
{
}

LbRoutingHelper*
LbRoutingHelper::Copy() const
{
    return new LbRoutingHelper(*this);
}

Ptr<Ipv4RoutingProtocol>
LbRoutingHelper::Create(Ptr<Node> node) const
{
    Ptr<LbRouting> r = CreateObject<LbRouting>();
    r->Setup(m_leafOf->at(node->GetId()), m_fabric, m_cfg);
    return r;
}

} // namespace lb
} // namespace ns3
