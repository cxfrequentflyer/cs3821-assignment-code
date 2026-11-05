/*
 * lb-fabric: packet-level load balancing in a leaf-spine fabric.
 *
 * The ns-3 counterpart of ecmp-sim's experiments: the same questions (how
 * unevenly do paths load up, what do d choices / memory / stale parallel
 * decisions / finer granularity / global information buy, what happens with
 * asymmetry), now with real TCP, queues, drops and reordering, measured as
 * flow completion times (pFabric's metric), duplicate ACKs (DRILL Fig. 11),
 * queue-length spread (DRILL Fig. 2-3) and link imbalance (CONGA's chi).
 *
 *   ./ns3 run "lb-fabric --scheme=drill --d=2 --memory=1 --load=0.6 --flows=2000"
 *   ./ns3 run "lb-fabric --PrintHelp"
 *
 * Topology: `leaves` leaf switches, `spines` spine switches, `hosts` hosts per
 * leaf. Host i of leaf l is 10.l.i.1; the leaf-spine link (l, s) is
 * 11.l.s.0/24. Links can be failed (--fail=0-0), slowed (--slow=1-1:0.25) or
 * striped (--striping: leaf l doubled to spines l and l+1, DRILL Fig. 13), and
 * taken out of or brought into service mid-run (--retire=0-1@5000,
 * --add=0-3@5000, in microseconds: dynamic-sim's retireBin / addBin).
 *
 * Traffic: `flows` TCP flows with Poisson arrivals. Sizes come from --sizes:
 *   empirical:<file>  a CDF file in 1460-byte packets (ecmp-sim/workloads/<name>.txt)
 *   fixed:<bytes> | exp:<mean> | pareto:<alpha>:<mean> | lognormal:<sigma>:<mean>
 *   | bimodal:<p>:<ratio>:<mean>
 * Pattern (--pattern): uniform (random source host, destination host under
 * another leaf), stride:<k> (destination leaf = source leaf + k), or
 * pair:<a>:<b> (leaf a's hosts to leaf b's: a single ECMP group, as in ecmp-sim
 * level 1). The arrival rate makes the offered load `load` x the uplink
 * capacity of the sending leaves.
 *
 * Output: one row appended to --out (CSV, header written if the file is new),
 * and optionally one row per flow to --flowsOut.
 */
#include "lb-routing.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("LbFabric");

namespace
{

const uint16_t kBasePort = 10000; // flow i listens on kBasePort + i (ephemeral ports start at 49152)
const uint32_t kMaxFlows = 39000;
const double kPacketBytes = 1460;  // unit of the CDF files

// ---------------------------------------------------------------------------
// Flow sizes
// ---------------------------------------------------------------------------

struct SizeDist
{
    std::string kind;
    std::vector<double> p;    // parameters
    std::vector<double> x, F; // empirical CDF, in bytes
    double mean = 0;          // bytes

    static SizeDist Parse(const std::string& spec)
    {
        SizeDist d;
        std::vector<std::string> parts;
        std::stringstream ss(spec);
        std::string item;
        while (std::getline(ss, item, ':'))
        {
            parts.push_back(item);
        }
        NS_ABORT_MSG_IF(parts.empty(), "empty --sizes");
        d.kind = parts[0];
        if (d.kind == "empirical")
        {
            NS_ABORT_MSG_IF(parts.size() != 2, "use empirical:<file>");
            d.LoadCdf(parts[1]);
            return d;
        }
        for (size_t i = 1; i < parts.size(); i++)
        {
            d.p.push_back(std::stod(parts[i]));
        }
        const std::map<std::string, size_t> nparams = {
            {"fixed", 1}, {"exp", 1}, {"pareto", 2}, {"lognormal", 2}, {"bimodal", 3}};
        NS_ABORT_MSG_IF(!nparams.count(d.kind), "unknown size distribution " << d.kind);
        NS_ABORT_MSG_IF(d.p.size() != nparams.at(d.kind), "wrong number of parameters: " << spec);
        d.mean = d.p.back();
        NS_ABORT_MSG_IF(d.kind == "pareto" && d.p[0] <= 1, "pareto needs alpha > 1");
        return d;
    }

    void LoadCdf(const std::string& path)
    {
        std::ifstream in(path);
        NS_ABORT_MSG_IF(!in, "cannot open " << path);
        std::string line;
        while (std::getline(in, line))
        {
            line = line.substr(0, line.find('#'));
            std::replace(line.begin(), line.end(), ',', ' ');
            std::stringstream ss(line);
            std::vector<std::string> cols;
            std::string c;
            while (ss >> c)
            {
                cols.push_back(c);
            }
            if (cols.size() < 2)
            {
                continue;
            }
            try
            {
                double size = std::stod(cols.front());
                double cdf = std::stod(cols.back());
                x.push_back(size * kPacketBytes);
                F.push_back(cdf);
            }
            catch (const std::exception&)
            {
                // header line
            }
        }
        NS_ABORT_MSG_IF(x.size() < 2, "no CDF points in " << path);
        if (std::abs(F.back() - 100) < 1e-6)
        {
            for (auto& v : F)
            {
                v /= 100;
            }
        }
        mean = F[0] * x[0]; // piecewise linear, as ecmpsim.flows.Empirical
        for (size_t i = 1; i < x.size(); i++)
        {
            mean += (F[i] - F[i - 1]) * (x[i] + x[i - 1]) / 2;
        }
    }

    double Sample(Ptr<UniformRandomVariable> u) const
    {
        double v = u->GetValue();
        if (kind == "fixed")
        {
            return p[0];
        }
        if (kind == "exp")
        {
            return -p[0] * std::log(1 - v);
        }
        if (kind == "pareto")
        {
            double a = p[0];
            return p[1] * (a - 1) / a * std::pow(1 - v, -1 / a);
        }
        if (kind == "lognormal")
        {
            double s = p[0];
            double z = std::sqrt(-2 * std::log(1 - v)) * std::cos(2 * M_PI * u->GetValue());
            return p[1] * std::exp(s * z - s * s / 2);
        }
        if (kind == "bimodal")
        {
            double small = p[2] / (1 - p[0] + p[0] * p[1]);
            return v < p[0] ? p[1] * small : small;
        }
        // empirical: invert the piecewise-linear CDF
        if (v <= F[0])
        {
            return x[0];
        }
        size_t i = std::upper_bound(F.begin(), F.end(), v) - F.begin();
        if (i >= F.size())
        {
            return x.back();
        }
        double t = (F[i] == F[i - 1]) ? 0 : (v - F[i - 1]) / (F[i] - F[i - 1]);
        return x[i - 1] + t * (x[i] - x[i - 1]);
    }
};

// ---------------------------------------------------------------------------
// Per-flow and global measurements
// ---------------------------------------------------------------------------

struct Flow
{
    uint32_t src, dst, srcLeaf, dstLeaf;
    uint64_t size;
    Time start;
    Time end = Seconds(-1);
    uint64_t rx = 0;
    uint32_t dupAcks = 0;
    bool haveAck = false;
    uint32_t lastAck = 0;
    uint32_t ooo = 0;        // data segments arriving below the highest byte seen
    uint32_t reordered = 0;  // ...of which the sender had sent only once: true reordering
    uint32_t maxRx = 0;
    std::unordered_map<uint32_t, uint16_t> sends; // times each segment was sent
    uint32_t retx = 0;       // data segments sent below the highest byte sent
    uint32_t maxTx = 0;
};

std::vector<Flow> g_flows;
uint32_t g_completed = 0;
lb::Fabric g_fabric;
std::vector<uint64_t> g_drops(4, 0); // host NIC, leaf->host, leaf->spine, spine->leaf
std::vector<std::vector<uint64_t>> g_upBytes, g_upWin, g_downBytes;
double g_winImbSum = 0, g_idleSum = 0;
uint64_t g_winImbN = 0, g_idleN = 0;
Time g_eventAt = Seconds(-1); // first timed link event, if any
double g_upStdSum = 0, g_downStdSum = 0, g_upQSum = 0, g_lastHopQSum = 0;
uint64_t g_upStdN = 0, g_downStdN = 0, g_upQN = 0, g_lastHopQN = 0;
uint32_t g_upQMax = 0;
std::vector<Ptr<PointToPointNetDevice>> g_lastHop; // leaf -> host devices
bool g_running = true;
int g_debugFlow = -1; // print every TCP segment of this flow
Time g_measureEnd; // queue sampling covers [first arrival, last arrival]

int
FlowOfPort(uint16_t port)
{
    if (port < kBasePort || port >= kBasePort + g_flows.size())
    {
        return -1;
    }
    return port - kBasePort;
}

void
OnSinkRx(uint32_t id, Ptr<const Packet> p, const Address&)
{
    Flow& f = g_flows[id];
    f.rx += p->GetSize();
    if (f.rx >= f.size && f.end.IsNegative())
    {
        f.end = Simulator::Now();
        if (++g_completed == g_flows.size())
        {
            g_running = false;
            Simulator::Stop();
        }
    }
}

void
DebugSegment(const char* what, const TcpHeader& th, uint32_t payload)
{
    int id = std::max(FlowOfPort(th.GetDestinationPort()), FlowOfPort(th.GetSourcePort()));
    if (id >= 0 && id == g_debugFlow)
    {
        std::cout << Simulator::Now().GetMicroSeconds() << "us " << what << " "
                  << TcpHeader::FlagsToString(th.GetFlags()) << " seq=" << th.GetSequenceNumber()
                  << " ack=" << th.GetAckNumber() << " len=" << payload
                  << " win=" << th.GetWindowSize() << "\n";
    }
}

void
OnSendOutgoing(const Ipv4Header& h, Ptr<const Packet> p, uint32_t)
{
    if (h.GetProtocol() != 6)
    {
        return;
    }
    TcpHeader th;
    p->PeekHeader(th);
    uint32_t payload = p->GetSize() - th.GetSerializedSize();
    DebugSegment("send", th, payload);
    if (payload > 0)
    {
        int id = FlowOfPort(th.GetDestinationPort()); // sender -> sink data
        if (id >= 0)
        {
            Flow& f = g_flows[id];
            uint32_t seq = th.GetSequenceNumber().GetValue();
            if (seq < f.maxTx)
            {
                f.retx++;
            }
            f.sends[seq]++;
            f.maxTx = std::max(f.maxTx, seq + payload);
        }
    }
    else if (th.GetFlags() == TcpHeader::ACK)
    {
        int id = FlowOfPort(th.GetSourcePort()); // sink -> sender ACKs
        if (id >= 0)
        {
            Flow& f = g_flows[id];
            uint32_t ack = th.GetAckNumber().GetValue();
            if (f.haveAck && ack == f.lastAck)
            {
                f.dupAcks++;
            }
            f.haveAck = true;
            f.lastAck = ack;
        }
    }
}

void
OnLocalDeliver(const Ipv4Header& h, Ptr<const Packet> p, uint32_t)
{
    if (h.GetProtocol() != 6)
    {
        return;
    }
    TcpHeader th;
    p->PeekHeader(th);
    uint32_t payload = p->GetSize() - th.GetSerializedSize();
    DebugSegment("recv", th, payload);
    int id = FlowOfPort(th.GetDestinationPort());
    if (payload == 0 || id < 0)
    {
        return;
    }
    Flow& f = g_flows[id];
    uint32_t seq = th.GetSequenceNumber().GetValue();
    if (seq < f.maxRx)
    {
        f.ooo++;
        auto it = f.sends.find(seq);
        if (it != f.sends.end() && it->second == 1)
        {
            f.reordered++; // late, yet never retransmitted: overtaken on another path
        }
    }
    f.maxRx = std::max(f.maxRx, seq + payload);
}

void
OnDrop(uint32_t category, Ptr<const Packet>)
{
    g_drops[category]++;
}

void
OnUpTx(uint32_t l, uint32_t s, Ptr<const Packet> p)
{
    g_upBytes[l][s] += p->GetSize();
    g_upWin[l][s] += p->GetSize();
}

void
OnDownTx(uint32_t s, uint32_t l, Ptr<const Packet> p)
{
    g_downBytes[s][l] += p->GetSize();
}

/** (max - min) / mean of utilization over a leaf's uplinks; -1 if idle. */
double
Imbalance(const std::vector<uint64_t>& bytes, const std::vector<double>& rate)
{
    std::vector<double> u;
    for (size_t s = 0; s < bytes.size(); s++)
    {
        if (rate[s] > 0)
        {
            u.push_back(bytes[s] * 8.0 / rate[s]);
        }
    }
    if (u.empty())
    {
        return -1;
    }
    double mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
    if (mean <= 0)
    {
        return -1;
    }
    return (*std::max_element(u.begin(), u.end()) - *std::min_element(u.begin(), u.end())) / mean;
}

double
StdDev(const std::vector<double>& v)
{
    double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double s = 0;
    for (double x : v)
    {
        s += (x - m) * (x - m);
    }
    return std::sqrt(s / v.size());
}

void
SampleQueues(Time interval)
{
    if (!g_running || Simulator::Now() > g_measureEnd)
    {
        return;
    }
    const lb::Fabric& f = g_fabric;
    for (uint32_t l = 0; l < f.leaves; l++)
    {
        std::vector<double> q;
        for (uint32_t s = 0; s < f.spines; s++)
        {
            if (f.up[l][s])
            {
                uint32_t n = f.up[l][s]->GetQueue()->GetNPackets();
                q.push_back(n);
                g_upQSum += n;
                g_upQN++;
                g_upQMax = std::max(g_upQMax, n);
            }
        }
        if (q.size() > 1)
        {
            g_upStdSum += StdDev(q);
            g_upStdN++;
        }
        q.clear();
        for (uint32_t s = 0; s < f.spines; s++)
        {
            if (f.down[s][l])
            {
                q.push_back(f.down[s][l]->GetQueue()->GetNPackets());
            }
        }
        if (q.size() > 1)
        {
            g_downStdSum += StdDev(q);
            g_downStdN++;
        }
    }
    for (const auto& d : g_lastHop)
    {
        g_lastHopQSum += d->GetQueue()->GetNPackets();
        g_lastHopQN++;
    }
    Simulator::Schedule(interval, &SampleQueues, interval);
}

void
WindowImbalance(Time window)
{
    if (!g_running || Simulator::Now() > g_measureEnd)
    {
        return;
    }
    for (uint32_t l = 0; l < g_fabric.leaves; l++)
    {
        std::vector<double> rate(g_fabric.spines, 0.0); // links in service only
        uint32_t active = 0, idle = 0;
        for (uint32_t s = 0; s < g_fabric.spines; s++)
        {
            if (g_fabric.InService(l, s))
            {
                rate[s] = g_fabric.upRate[l][s];
                active++;
                idle += g_upWin[l][s] == 0;
            }
        }
        double imb = Imbalance(g_upWin[l], rate);
        if (imb >= 0)
        {
            g_winImbSum += imb;
            g_winImbN++;
            g_idleSum += double(idle) / active; // ecmp-sim's idle_frac, per window
            g_idleN++;
        }
        std::fill(g_upWin[l].begin(), g_upWin[l].end(), 0);
    }
    Simulator::Schedule(window, &WindowImbalance, window);
}

double
Quantile(std::vector<double> v, double q)
{
    if (v.empty())
    {
        return NAN;
    }
    std::sort(v.begin(), v.end());
    double pos = q * (v.size() - 1);
    size_t i = static_cast<size_t>(pos);
    double t = pos - i;
    return i + 1 < v.size() ? v[i] * (1 - t) + v[i + 1] * t : v[i];
}

double
Mean(const std::vector<double>& v)
{
    return v.empty() ? NAN : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

std::vector<std::string>
Split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
    {
        if (!item.empty())
        {
            out.push_back(item);
        }
    }
    return out;
}

} // namespace

int
main(int argc, char* argv[])
{
    // ----- parameters --------------------------------------------------------
    uint32_t leaves = 4, spines = 4, hosts = 8;
    std::string hostRate = "10Gbps", coreRate = "20Gbps", linkDelay = "1us";
    uint32_t switchQueue = 100, hostQueue = 1000; // packets
    std::string scheme = "ecmp";
    uint32_t d = 2, memory = 1, engines = 1;
    double stalenessUs = 0, flowletUs = 100;
    uint32_t cellKB = 64;
    std::string sizes = "exp:100000";
    std::string pattern = "uniform";
    double load = 0.5;
    uint32_t nFlows = 1000;
    uint32_t seed = 1;
    std::string fail, slow, retire, add;
    bool striping = false;
    double maxSeconds = 5;
    double maxFlowBytes = 0;
    double sampleUs = 10, windowUs = 1000;
    std::string tcp = "TcpNewReno";
    double minRtoMs = 5;
    double connTimeoutMs = 10;
    uint32_t delAck = 1;
    std::string out = "lb-fabric.csv", flowsOut, tag;

    CommandLine cmd(__FILE__);
    cmd.AddValue("leaves", "leaf switches", leaves);
    cmd.AddValue("spines", "spine switches", spines);
    cmd.AddValue("hosts", "hosts per leaf", hosts);
    cmd.AddValue("hostRate", "host link rate", hostRate);
    cmd.AddValue("coreRate", "leaf-spine link rate", coreRate);
    cmd.AddValue("delay", "per-link propagation delay", linkDelay);
    cmd.AddValue("switchQueue", "switch port buffer, packets", switchQueue);
    cmd.AddValue("hostQueue", "host NIC buffer, packets", hostQueue);
    cmd.AddValue("scheme",
                 "ecmp | wcmp | dchoice | flowlet | conga | presto | random | rr | drill | drill-comp",
                 scheme);
    cmd.AddValue("d", "samples for drill / dchoice", d);
    cmd.AddValue("memory", "remembered uplinks for drill", memory);
    cmd.AddValue("engines", "forwarding engines per leaf", engines);
    cmd.AddValue("staleness", "age of an engine's queue snapshot, microseconds (0 = live)",
                 stalenessUs);
    cmd.AddValue("flowletGap", "flowlet idle gap, microseconds", flowletUs);
    cmd.AddValue("cellKB", "presto flowcell size, KB", cellKB);
    cmd.AddValue("sizes", "flow sizes, see the file header", sizes);
    cmd.AddValue("pattern", "uniform | stride:<k> | pair:<a>:<b>", pattern);
    cmd.AddValue("load", "offered load, fraction of the sending leaves' uplink capacity", load);
    cmd.AddValue("flows", "number of flows", nFlows);
    cmd.AddValue("seed", "run number for the random streams", seed);
    cmd.AddValue("fail", "failed leaf-spine links, e.g. 0-0,2-1", fail);
    cmd.AddValue("slow", "slowed links, e.g. 1-1:0.25", slow);
    cmd.AddValue("retire", "take links out of service mid-run, e.g. 0-1@5000 (us)", retire);
    cmd.AddValue("add", "links that come into service mid-run, e.g. 0-3@5000 (us)", add);
    cmd.AddValue("striping", "leaf l doubled to spines l and l+1 (DRILL Fig. 13)", striping);
    cmd.AddValue("maxSeconds", "simulated time limit", maxSeconds);
    cmd.AddValue("maxFlowBytes", "cap on a flow's size (0 = none)", maxFlowBytes);
    cmd.AddValue("sampleUs", "queue sampling interval, microseconds", sampleUs);
    cmd.AddValue("windowUs", "window for the uplink imbalance, microseconds", windowUs);
    cmd.AddValue("tcp", "TCP variant (ns3::<name>)", tcp);
    cmd.AddValue("minRto", "TCP minimum RTO, ms", minRtoMs);
    cmd.AddValue("delAck",
                 "segments per ACK. Above 1, a lone segment after a timeout waits for the "
                 "200 ms delayed-ACK timer, longer than the RTO, and the flow crawls",
                 delAck);
    cmd.AddValue("connTimeout", "SYN retransmission timeout, ms (ns-3's default is 3 s)",
                 connTimeoutMs);
    cmd.AddValue("out", "summary CSV (appended)", out);
    cmd.AddValue("flowsOut", "per-flow CSV (optional)", flowsOut);
    cmd.AddValue("tag", "free-form label stored with the results", tag);
    cmd.AddValue("debugFlow", "print every TCP segment of this flow id", g_debugFlow);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(nFlows == 0 || nFlows > kMaxFlows, "flows must be in 1.." << kMaxFlows);
    NS_ABORT_MSG_IF(leaves < 2 || leaves > 250 || spines < 1 || spines > 250 || hosts < 1 ||
                        hosts > 250,
                    "bad topology size");
    RngSeedManager::SetSeed(3821);
    RngSeedManager::SetRun(seed);

    lb::Config cfg;
    cfg.scheme = lb::ParseScheme(scheme);
    cfg.d = d;
    cfg.memory = memory;
    cfg.engines = engines;
    cfg.staleness = MicroSeconds(stalenessUs);
    cfg.cellBytes = cellKB * 1024;
    cfg.flowletGap = MicroSeconds(flowletUs);
    cfg.hashSeed = seed;

    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       TypeIdValue(TypeId::LookupByName("ns3::" + tcp)));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
    Config::SetDefault("ns3::TcpSocket::InitialCwnd", UintegerValue(10));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(1 << 20));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(1 << 20));
    Config::SetDefault("ns3::TcpSocketBase::MinRto", TimeValue(MilliSeconds(minRtoMs)));
    Config::SetDefault("ns3::TcpSocketBase::ClockGranularity", TimeValue(MicroSeconds(10)));
    Config::SetDefault("ns3::TcpSocket::ConnTimeout", TimeValue(MilliSeconds(connTimeoutMs)));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(delAck));

    // ----- link capacities ----------------------------------------------------
    std::vector<std::vector<double>> factor(leaves, std::vector<double>(spines, 1.0));
    if (striping)
    {
        for (uint32_t l = 0; l < leaves; l++)
        {
            factor[l][l % spines] = factor[l][(l + 1) % spines] = 2.0;
        }
    }
    for (const auto& item : Split(fail, ','))
    {
        auto ls = Split(item, '-');
        NS_ABORT_MSG_IF(ls.size() != 2, "bad --fail entry " << item);
        factor.at(std::stoul(ls[0])).at(std::stoul(ls[1])) = 0;
    }
    for (const auto& item : Split(slow, ','))
    {
        auto parts = Split(item, ':');
        auto ls = Split(parts.at(0), '-');
        NS_ABORT_MSG_IF(ls.size() != 2 || parts.size() != 2, "bad --slow entry " << item);
        factor.at(std::stoul(ls[0])).at(std::stoul(ls[1])) = std::stod(parts[1]);
    }

    // ----- nodes and stacks ---------------------------------------------------
    NodeContainer leafNodes, spineNodes, hostNodes;
    leafNodes.Create(leaves);
    spineNodes.Create(spines);
    hostNodes.Create(leaves * hosts);
    std::unordered_map<uint32_t, uint32_t> leafOf;
    for (uint32_t l = 0; l < leaves; l++)
    {
        leafOf[leafNodes.Get(l)->GetId()] = l;
    }
    g_fabric.Init(leaves, spines);

    InternetStackHelper plain;
    plain.Install(spineNodes);
    plain.Install(hostNodes);
    Ipv4StaticRoutingHelper staticHelper;
    lb::LbRoutingHelper lbHelper(&g_fabric, cfg, &leafOf);
    Ipv4ListRoutingHelper list;
    list.Add(staticHelper, 0);
    list.Add(lbHelper, 10);
    InternetStackHelper leafStack;
    leafStack.SetRoutingHelper(list);
    leafStack.Install(leafNodes);

    // ----- links --------------------------------------------------------------
    PointToPointHelper hostLink;
    hostLink.SetDeviceAttribute("DataRate", StringValue(hostRate));
    hostLink.SetChannelAttribute("Delay", StringValue(linkDelay));
    hostLink.SetQueue("ns3::DropTailQueue<Packet>",
                      "MaxSize",
                      QueueSizeValue(QueueSize(std::to_string(hostQueue) + "p")));
    hostLink.DisableFlowControl();
    Ipv4AddressHelper addr;
    std::vector<Ipv4Address> leafAddrForHost(leaves * hosts);
    for (uint32_t l = 0; l < leaves; l++)
    {
        for (uint32_t i = 0; i < hosts; i++)
        {
            Ptr<Node> host = hostNodes.Get(l * hosts + i);
            NetDeviceContainer devs = hostLink.Install(host, leafNodes.Get(l));
            auto leafDev = DynamicCast<PointToPointNetDevice>(devs.Get(1));
            leafDev->GetQueue()->SetMaxSize(QueueSize(std::to_string(switchQueue) + "p"));
            leafDev->GetQueue()->TraceConnectWithoutContext("Drop", MakeBoundCallback(&OnDrop, 1));
            DynamicCast<PointToPointNetDevice>(devs.Get(0))
                ->GetQueue()
                ->TraceConnectWithoutContext("Drop", MakeBoundCallback(&OnDrop, 0));
            g_lastHop.push_back(leafDev);
            std::ostringstream net;
            net << "10." << l << "." << i << ".0";
            addr.SetBase(net.str().c_str(), "255.255.255.0");
            Ipv4InterfaceContainer ifs = addr.Assign(devs);
            leafAddrForHost[l * hosts + i] = ifs.GetAddress(1);
        }
    }
    DataRate core(coreRate);
    for (uint32_t l = 0; l < leaves; l++)
    {
        for (uint32_t s = 0; s < spines; s++)
        {
            if (factor[l][s] <= 0)
            {
                continue;
            }
            double bps = core.GetBitRate() * factor[l][s];
            PointToPointHelper coreLink;
            coreLink.SetDeviceAttribute("DataRate", DataRateValue(DataRate(uint64_t(bps))));
            coreLink.SetChannelAttribute("Delay", StringValue(linkDelay));
            coreLink.SetQueue("ns3::DropTailQueue<Packet>",
                              "MaxSize",
                              QueueSizeValue(QueueSize(std::to_string(switchQueue) + "p")));
            coreLink.DisableFlowControl();
            NetDeviceContainer devs = coreLink.Install(leafNodes.Get(l), spineNodes.Get(s));
            std::ostringstream net;
            net << "11." << l << "." << s << ".0";
            addr.SetBase(net.str().c_str(), "255.255.255.0");
            Ipv4InterfaceContainer ifs = addr.Assign(devs);
            auto up = DynamicCast<PointToPointNetDevice>(devs.Get(0));
            auto down = DynamicCast<PointToPointNetDevice>(devs.Get(1));
            g_fabric.up[l][s] = up;
            g_fabric.down[s][l] = down;
            g_fabric.upRate[l][s] = bps;
            g_fabric.downRate[s][l] = bps;
            g_fabric.spineAddr[l][s] = ifs.GetAddress(1);
            up->GetQueue()->TraceConnectWithoutContext("Drop", MakeBoundCallback(&OnDrop, 2));
            down->GetQueue()->TraceConnectWithoutContext("Drop", MakeBoundCallback(&OnDrop, 3));

            // spine s reaches leaf l's hosts over this link
            Ptr<Ipv4> sip = spineNodes.Get(s)->GetObject<Ipv4>();
            std::ostringstream hostsNet;
            hostsNet << "10." << l << ".0.0";
            staticHelper.GetStaticRouting(sip)->AddNetworkRouteTo(
                Ipv4Address(hostsNet.str().c_str()),
                Ipv4Mask("255.255.0.0"),
                ifs.GetAddress(0),
                sip->GetInterfaceForDevice(down));
        }
    }
    // The drop-tail device queues are the switch buffers. With flow control
    // disabled the devices have no NetDeviceQueueInterface, so Ipv4AddressHelper
    // installs no queue disc (FqCoDel) on them.
    // timed link events: dynamic-sim's retireBin / addBin
    for (auto [spec, toRetire] : {std::pair{retire, true}, std::pair{add, false}})
    {
        for (const auto& item : Split(spec, ','))
        {
            auto at = Split(item, '@');
            auto ls = Split(at.at(0), '-');
            NS_ABORT_MSG_IF(at.size() != 2 || ls.size() != 2, "use l-s@us, got " << item);
            uint32_t l = std::stoul(ls[0]), s = std::stoul(ls[1]);
            NS_ABORT_MSG_IF(l >= leaves || s >= spines || !g_fabric.up[l][s],
                            "no link " << at[0] << " to schedule");
            Time when = MicroSeconds(std::stod(at[1]));
            if (!toRetire)
            {
                g_fabric.retired[l][s] = true; // out of service until `when`
            }
            Simulator::Schedule(when, &lb::Fabric::SetRetired, &g_fabric, l, s, toRetire);
            if (g_eventAt.IsNegative() || when < g_eventAt)
            {
                g_eventAt = when;
            }
        }
    }
    g_fabric.Build();
    for (uint32_t h = 0; h < leaves * hosts; h++)
    {
        Ptr<Ipv4> ip = hostNodes.Get(h)->GetObject<Ipv4>();
        staticHelper.GetStaticRouting(ip)->SetDefaultRoute(leafAddrForHost[h], 1);
        ip->TraceConnectWithoutContext("SendOutgoing", MakeCallback(&OnSendOutgoing));
        ip->TraceConnectWithoutContext("LocalDeliver", MakeCallback(&OnLocalDeliver));
    }
    g_upBytes.assign(leaves, std::vector<uint64_t>(spines, 0));
    g_upWin.assign(leaves, std::vector<uint64_t>(spines, 0));
    g_downBytes.assign(spines, std::vector<uint64_t>(leaves, 0));
    for (uint32_t l = 0; l < leaves; l++)
    {
        for (uint32_t s = 0; s < spines; s++)
        {
            if (g_fabric.up[l][s])
            {
                g_fabric.up[l][s]->TraceConnectWithoutContext("PhyTxEnd",
                                                              MakeBoundCallback(&OnUpTx, l, s));
                g_fabric.down[s][l]->TraceConnectWithoutContext("PhyTxEnd",
                                                                MakeBoundCallback(&OnDownTx, s, l));
            }
        }
    }

    // ----- traffic -------------------------------------------------------------
    SizeDist dist = SizeDist::Parse(sizes);
    std::vector<uint32_t> senders;
    uint32_t strideK = 0, pairA = 0, pairB = 1;
    auto pat = Split(pattern, ':');
    if (pat[0] == "uniform")
    {
        for (uint32_t l = 0; l < leaves; l++)
        {
            senders.push_back(l);
        }
    }
    else if (pat[0] == "stride" && pat.size() == 2)
    {
        strideK = std::stoul(pat[1]) % leaves;
        NS_ABORT_MSG_IF(strideK == 0, "stride must not be a multiple of leaves");
        for (uint32_t l = 0; l < leaves; l++)
        {
            senders.push_back(l);
        }
    }
    else if (pat[0] == "pair" && pat.size() == 3)
    {
        pairA = std::stoul(pat[1]);
        pairB = std::stoul(pat[2]);
        NS_ABORT_MSG_IF(pairA == pairB || pairA >= leaves || pairB >= leaves, "bad pair");
        senders.push_back(pairA);
    }
    else
    {
        NS_FATAL_ERROR("unknown pattern " << pattern);
    }
    double capacity = 0;
    for (uint32_t l : senders)
    {
        for (uint32_t s = 0; s < spines; s++)
        {
            capacity += g_fabric.upRate[l][s];
        }
    }
    double lambda = load * capacity / (8 * dist.mean); // flows per second
    Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
    Ptr<ExponentialRandomVariable> gap = CreateObject<ExponentialRandomVariable>();
    gap->SetAttribute("Mean", DoubleValue(1 / lambda));

    double t = 0.001; // let routing settle
    g_flows.reserve(nFlows);
    for (uint32_t i = 0; i < nFlows; i++)
    {
        t += gap->GetValue();
        Flow f;
        f.srcLeaf = senders[rng->GetInteger(0, senders.size() - 1)];
        if (pat[0] == "uniform")
        {
            f.dstLeaf = (f.srcLeaf + rng->GetInteger(1, leaves - 1)) % leaves;
        }
        else if (pat[0] == "stride")
        {
            f.dstLeaf = (f.srcLeaf + strideK) % leaves;
        }
        else
        {
            f.dstLeaf = pairB;
        }
        f.src = f.srcLeaf * hosts + rng->GetInteger(0, hosts - 1);
        f.dst = f.dstLeaf * hosts + rng->GetInteger(0, hosts - 1);
        double size = dist.Sample(rng);
        if (maxFlowBytes > 0)
        {
            size = std::min(size, maxFlowBytes);
        }
        f.size = std::max<uint64_t>(1, uint64_t(std::llround(size)));
        f.start = Seconds(t);
        NS_ABORT_MSG_IF(g_fabric.usable[f.srcLeaf][f.dstLeaf].empty(),
                        "leaf " << f.srcLeaf << " cannot reach leaf " << f.dstLeaf);
        g_flows.push_back(f);

        uint16_t port = kBasePort + i;
        Ptr<Node> dstNode = hostNodes.Get(f.dst);
        Ipv4Address dstAddr = dstNode->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();
        PacketSinkHelper sinkHelper("ns3::TcpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sink = sinkHelper.Install(dstNode);
        sink.Start(Seconds(0));
        sink.Get(0)->TraceConnectWithoutContext("Rx", MakeBoundCallback(&OnSinkRx, i));
        BulkSendHelper src("ns3::TcpSocketFactory", InetSocketAddress(dstAddr, port));
        src.SetAttribute("MaxBytes", UintegerValue(f.size));
        src.SetAttribute("SendSize", UintegerValue(65536));
        ApplicationContainer app = src.Install(hostNodes.Get(f.src));
        app.Start(f.start);
    }

    g_measureEnd = g_flows.back().start;
    Time first = g_flows.front().start;
    Simulator::Schedule(first + MicroSeconds(sampleUs), &SampleQueues, MicroSeconds(sampleUs));
    Simulator::Schedule(first + MicroSeconds(windowUs), &WindowImbalance, MicroSeconds(windowUs));
    // windows shorter than the arrival period would never close: shrink to fit
    NS_ABORT_MSG_IF(g_measureEnd - first < MicroSeconds(windowUs),
                    "arrivals span less than one --windowUs window; use more flows or a smaller window");
    Simulator::Stop(Seconds(maxSeconds));
    auto wall0 = std::chrono::steady_clock::now();
    Simulator::Run();
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    Time simEnd = Simulator::Now();

    // ----- results ----------------------------------------------------------------
    double hostBps = DataRate(hostRate).GetBitRate();
    double prop = Time(linkDelay).GetSeconds();
    double mtuCore = 1500 * 8.0 / core.GetBitRate();
    double mtuHost = 1500 * 8.0 / hostBps;
    // Best case on an empty fabric: a handshake RTT, then the bytes (with headers)
    // at the host rate, plus one packet's store-and-forward and propagation.
    double baseOneWay = 4 * prop + 2 * mtuHost + 2 * mtuCore;
    std::vector<double> fct, nfct, fctSmall, fctMid, fctLarge, nfctSmall, nfctLarge;
    std::vector<double> fctBefore, fctAfter; // flows starting before / after the first link event
    uint32_t anyDup = 0, manyDup = 0, anyOoo = 0, anyReorder = 0, unfinished = 0;
    uint64_t reorderedSegs = 0;
    uint64_t retx = 0;
    for (const auto& f : g_flows)
    {
        if (f.end.IsNegative())
        {
            unfinished++;
            continue;
        }
        double x = (f.end - f.start).GetSeconds();
        double ideal = 2 * baseOneWay + baseOneWay + f.size * (1500.0 / 1448) * 8 / hostBps;
        fct.push_back(x);
        nfct.push_back(x / ideal);
        if (!g_eventAt.IsNegative())
        {
            (f.start < g_eventAt ? fctBefore : fctAfter).push_back(x);
        }
        if (f.size < 100e3)
        {
            fctSmall.push_back(x);
            nfctSmall.push_back(x / ideal);
        }
        else if (f.size > 10e6)
        {
            fctLarge.push_back(x);
            nfctLarge.push_back(x / ideal);
        }
        else
        {
            fctMid.push_back(x);
        }
        anyDup += f.dupAcks > 0;
        manyDup += f.dupAcks > 3;
        anyOoo += f.ooo > 0;
        anyReorder += f.reordered > 0;
        reorderedSegs += f.reordered;
        retx += f.retx;
    }
    double imbRun = 0, imbDown = 0;
    uint32_t nImb = 0, nImbDown = 0;
    for (uint32_t l = 0; l < leaves; l++)
    {
        double v = Imbalance(g_upBytes[l], g_fabric.upRate[l]);
        if (v >= 0)
        {
            imbRun += v;
            nImb++;
        }
        std::vector<uint64_t> bytes(spines);
        std::vector<double> rate(spines);
        for (uint32_t s = 0; s < spines; s++)
        {
            bytes[s] = g_downBytes[s][l];
            rate[s] = g_fabric.downRate[s][l];
        }
        v = Imbalance(bytes, rate);
        if (v >= 0)
        {
            imbDown += v;
            nImbDown++;
        }
    }
    uint32_t done = fct.size();
    auto pct = [&](uint32_t k) { return done ? double(k) / done : NAN; };

    std::vector<std::pair<std::string, std::string>> row;
    auto put = [&](const std::string& k, auto v) {
        std::ostringstream o;
        o.precision(8);
        o << v;
        row.emplace_back(k, o.str());
    };
    put("tag", tag);
    put("scheme", scheme);
    put("d", d);
    put("memory", memory);
    put("engines", engines);
    put("staleness_us", stalenessUs);
    put("flowlet_us", flowletUs);
    put("cell_kb", cellKB);
    put("leaves", leaves);
    put("spines", spines);
    put("hosts", hosts);
    put("host_rate", hostRate);
    put("core_rate", coreRate);
    put("switch_queue", switchQueue);
    put("fail", fail);
    put("slow", slow);
    put("striping", striping);
    put("retire", retire);
    put("add", add);
    put("pattern", pattern);
    put("sizes", sizes);
    put("load", load);
    put("flows", nFlows);
    put("seed", seed);
    put("max_flow_bytes", maxFlowBytes);
    put("window_us", windowUs);
    put("sample_us", sampleUs);
    put("tcp", tcp);
    put("min_rto_ms", minRtoMs);
    put("del_ack", delAck);
    put("mean_size_bytes", dist.mean);
    put("completed", done);
    put("unfinished", unfinished);
    put("sim_seconds", simEnd.GetSeconds());
    put("fct_mean_ms", 1e3 * Mean(fct));
    put("fct_p99_ms", 1e3 * Quantile(fct, 0.99));
    put("nfct_mean", Mean(nfct));
    put("nfct_p99", Quantile(nfct, 0.99));
    put("small_fct_mean_ms", 1e3 * Mean(fctSmall));
    put("small_fct_p99_ms", 1e3 * Quantile(fctSmall, 0.99));
    put("small_nfct_mean", Mean(nfctSmall));
    put("small_nfct_p99", Quantile(nfctSmall, 0.99));
    put("mid_fct_mean_ms", 1e3 * Mean(fctMid));
    put("large_fct_mean_ms", 1e3 * Mean(fctLarge));
    put("large_nfct_mean", Mean(nfctLarge));
    put("fct_before_event_ms", 1e3 * Mean(fctBefore));
    put("fct_after_event_ms", 1e3 * Mean(fctAfter));
    put("flows_with_dupack", pct(anyDup));
    put("flows_over_3_dupacks", pct(manyDup));
    put("flows_with_ooo", pct(anyOoo));
    put("flows_reordered", pct(anyReorder));
    put("reordered_segments", reorderedSegs);
    put("retx_segments", retx);
    put("drops_host_nic", g_drops[0]);
    put("drops_last_hop", g_drops[1]);
    put("drops_uplink", g_drops[2]);
    put("drops_downlink", g_drops[3]);
    put("uplink_imbalance_run", nImb ? imbRun / nImb : NAN);
    put("uplink_imbalance_window", g_winImbN ? g_winImbSum / g_winImbN : NAN);
    put("uplink_idle_frac_window", g_idleN ? g_idleSum / g_idleN : NAN);
    put("downlink_imbalance_run", nImbDown ? imbDown / nImbDown : NAN);
    put("uplink_queue_std", g_upStdN ? g_upStdSum / g_upStdN : NAN);
    put("downlink_queue_std", g_downStdN ? g_downStdSum / g_downStdN : NAN);
    put("uplink_queue_mean", g_upQN ? g_upQSum / g_upQN : NAN);
    put("uplink_queue_max", g_upQMax);
    put("last_hop_queue_mean", g_lastHopQN ? g_lastHopQSum / g_lastHopQN : NAN);
    put("wall_seconds", wall);

    bool fresh = true;
    {
        std::ifstream probe(out);
        fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    }
    std::ofstream os(out, std::ios::app);
    auto quote = [](const std::string& v) {
        return v.find(',') == std::string::npos ? v : "\"" + v + "\"";
    };
    if (fresh)
    {
        for (size_t i = 0; i < row.size(); i++)
        {
            os << (i ? "," : "") << row[i].first;
        }
        os << "\n";
    }
    for (size_t i = 0; i < row.size(); i++)
    {
        os << (i ? "," : "") << quote(row[i].second);
    }
    os << "\n";

    if (!flowsOut.empty())
    {
        std::ofstream fo(flowsOut);
        fo << "id,src,dst,src_leaf,dst_leaf,size,start_s,fct_s,dupacks,ooo,reordered,retx\n";
        for (size_t i = 0; i < g_flows.size(); i++)
        {
            const auto& f = g_flows[i];
            fo << i << "," << f.src << "," << f.dst << "," << f.srcLeaf << "," << f.dstLeaf << ","
               << f.size << "," << f.start.GetSeconds() << ","
               << (f.end.IsNegative() ? NAN : (f.end - f.start).GetSeconds()) << "," << f.dupAcks
               << "," << f.ooo << "," << f.reordered << "," << f.retx << "\n";
        }
    }

    std::cout << scheme << " load=" << load << " flows=" << done << "/" << nFlows
              << " fct_mean=" << 1e3 * Mean(fct) << "ms nfct_mean=" << Mean(nfct)
              << " dupack_flows=" << pct(anyDup) << " reordered_flows=" << pct(anyReorder)
              << " uplink_imb=" << (nImb ? imbRun / nImb : NAN)
              << " sim=" << simEnd.GetSeconds() << "s wall=" << wall << "s\n";
    Simulator::Destroy();
    return 0;
}
