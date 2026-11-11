/*
 * Load balancing at the leaf switches of a leaf-spine fabric.
 *
 * LbRouting is an Ipv4RoutingProtocol installed (in an Ipv4ListRouting, above
 * static routing) on every leaf switch. Packets for hosts under the same leaf
 * are left to static routing; packets for a remote leaf are sent up to a
 * spine chosen by the configured scheme. Spines and hosts use static routes:
 * in a two-tier fabric the only path choice is the spine.
 *
 * Schemes (granularity / information):
 *   ecmp       flow     / none      hash of the 5-tuple
 *   wcmp       flow     / static    hash, weighted by path capacity
 *   dchoice    flow     / local     d sampled uplinks at the flow's first packet, then pinned
 *                                   (ecmp-sim's DChoice; DRILL's "per-flow DRILL" strawman)
 *   flowlet    flowlet  / none      new random spine after an idle gap (LetFlow-style)
 *   conga      flowlet  / global    least-congested path at each new flowlet: max of the
 *                                   uplink and the spine->destination downlink queues.
 *                                   An oracle: it reads remote queues directly instead of
 *                                   CONGA's piggybacked feedback, so it is an upper bound.
 *   presto     flowcell / none      each flow round-robins fixed-size cells (64KB default)
 *                                   from a hashed start
 *   random     packet   / none      uniform random uplink per packet ("per-packet Random")
 *   rr         packet   / none      round robin per forwarding engine
 *   drill      packet   / local     DRILL(d, m): d random uplinks plus m remembered ones,
 *                                   shortest queue wins; per forwarding engine
 *   drill-comp packet   / local     DRILL's asymmetry handling: hash the flow to a
 *                                   symmetric component (weighted by capacity), DRILL inside it
 *
 * Forwarding engines (DRILL §3.2.1): a packet is handled by engine
 * (input interface mod F), and each engine has its own memory. With
 * staleness > 0 the switch's queue-length snapshot is refreshed only every
 * `staleness`; between refreshes an engine estimates each queue as the
 * snapshot plus what IT has sent since, less what the link can have drained
 * meanwhile, and never sees what the other engines sent. One engine is therefore
 * (nearly) up to date, and F engines decide in parallel on partly shared,
 * stale information: the mechanism behind DRILL's synchronization effect
 * (and the shared-snapshot rounds of ecmp-sim's `engines=F`). Queue "length" is backlog in time (bytes / link rate), so links of
 * different speeds compare fairly (as ecmp-sim's utilization and dynamic-sim's
 * backlog do).
 */
#ifndef LB_ROUTING_H
#define LB_ROUTING_H

#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ns3
{
namespace lb
{

enum class Scheme
{
    Ecmp,
    Wcmp,
    DChoice,
    Flowlet,
    Conga,
    Presto,
    Random,
    RoundRobin,
    Drill,
    DrillComp
};

Scheme ParseScheme(const std::string& name);
std::string SchemeName(Scheme s);

/** What every leaf knows about the fabric: built once, before traffic starts. */
struct Fabric
{
    struct Component
    {
        std::vector<uint32_t> spines;
        double weight; // total capacity of its paths
    };

    uint32_t leaves = 0;
    uint32_t spines = 0;
    // up[l][s]: leaf l's device towards spine s (null if the link is missing or failed)
    std::vector<std::vector<Ptr<PointToPointNetDevice>>> up;
    // down[s][l]: spine s's device towards leaf l
    std::vector<std::vector<Ptr<PointToPointNetDevice>>> down;
    std::vector<std::vector<double>> upRate;   // bit/s, 0 if missing
    std::vector<std::vector<double>> downRate; // bit/s, 0 if missing
    std::vector<std::vector<Ipv4Address>> spineAddr; // spine s's address on link (l, s)
    // retired[l][s]: the link exists but routing no longer uses it (or not yet):
    // dynamic-sim's retireBin / addBin. Packets already queued still drain.
    std::vector<std::vector<bool>> retired;

    // usable[a][b]: spines on a path a -> s -> b; comps[a][b]: those split into
    // symmetric components (DRILL §3.4)
    std::vector<std::vector<std::vector<uint32_t>>> usable;
    std::vector<std::vector<std::vector<Component>>> comps;

    void Init(uint32_t nLeaves, uint32_t nSpines);
    void Build(); // fills usable and comps from the links in service
    void SetRetired(uint32_t l, uint32_t s, bool value); // and rebuild
    bool InService(uint32_t l, uint32_t s) const
    {
        return up[l][s] && !retired[l][s];
    }
    static uint32_t LeafOf(Ipv4Address host); // hosts are 10.<leaf>.<host>.1
    double UpBacklog(uint32_t l, uint32_t s) const;   // seconds to drain
    double DownBacklog(uint32_t s, uint32_t l) const; // seconds to drain
};

struct Config
{
    Scheme scheme = Scheme::Ecmp;
    uint32_t d = 2;            // samples for drill / dchoice
    uint32_t memory = 1;       // remembered uplinks for drill
    uint32_t engines = 1;      // forwarding engines per leaf
    Time staleness = Seconds(0); // how old an engine's queue snapshot may get
    uint32_t cellBytes = 65536;  // presto flowcell
    Time flowletGap = MicroSeconds(100);
    uint32_t hashSeed = 0;
};

class LbRouting : public Ipv4RoutingProtocol
{
  public:
    static TypeId GetTypeId();
    LbRouting();

    void Setup(uint32_t leaf, const Fabric* fabric, const Config& config);

    Ptr<Ipv4Route> RouteOutput(Ptr<Packet> p,
                               const Ipv4Header& header,
                               Ptr<NetDevice> oif,
                               Socket::SocketErrno& sockerr) override;
    bool RouteInput(Ptr<const Packet> p,
                    const Ipv4Header& header,
                    Ptr<const NetDevice> idev,
                    const UnicastForwardCallback& ucb,
                    const MulticastForwardCallback& mcb,
                    const LocalDeliverCallback& lcb,
                    const ErrorCallback& ecb) override;
    void NotifyInterfaceUp(uint32_t interface) override;
    void NotifyInterfaceDown(uint32_t interface) override;
    void NotifyAddAddress(uint32_t interface, Ipv4InterfaceAddress address) override;
    void NotifyRemoveAddress(uint32_t interface, Ipv4InterfaceAddress address) override;
    void SetIpv4(Ptr<Ipv4> ipv4) override;
    void PrintRoutingTable(Ptr<OutputStreamWrapper> stream,
                           Time::Unit unit = Time::S) const override;

    /** Packets this leaf sent up to each spine (for checks). */
    const std::vector<uint64_t>& SentPerSpine() const
    {
        return m_sent;
    }

  private:
    struct FlowState
    {
        bool init = false;
        uint32_t spine = 0;
        Time last;
        uint64_t cellBytes = 0;
        uint32_t cellIndex = 0;
        uint32_t start = 0;
        int32_t comp = -1;
    };

    struct Engine
    {
        std::vector<double> own; // bytes this engine sent per spine since the snapshot
        uint64_t rr = 0;
        std::unordered_map<uint64_t, std::vector<uint32_t>> memory; // per (dst leaf, component)
    };

    uint32_t Choose(uint32_t dstLeaf, uint64_t flowHash, uint32_t payload, uint32_t engine);
    uint32_t DrillPick(const std::vector<uint32_t>& allowed,
                       uint32_t engine,
                       uint64_t memKey,
                       uint32_t d,
                       uint32_t memory);
    double Backlog(uint32_t engine, uint32_t spine);
    Ptr<Ipv4Route> RouteVia(uint32_t spine, const Ipv4Header& header);

    Ptr<Ipv4> m_ipv4;
    uint32_t m_leaf = 0;
    const Fabric* m_fabric = nullptr;
    Config m_cfg;
    std::vector<Engine> m_engines;
    std::vector<double> m_snapshot; // shared queue snapshot (bytes per spine), when stale
    Time m_snapshotAt = Seconds(-1);
    std::unordered_map<uint64_t, FlowState> m_flows;
    std::vector<Ptr<Ipv4Route>> m_routes;
    std::vector<uint64_t> m_sent;
    Ptr<UniformRandomVariable> m_rng;
};

/** Creates LbRouting on the leaves: Create() looks the node up in `leafOf`. */
class LbRoutingHelper : public Ipv4RoutingHelper
{
  public:
    LbRoutingHelper(const Fabric* fabric,
                    const Config& config,
                    const std::unordered_map<uint32_t, uint32_t>* leafOf);
    LbRoutingHelper* Copy() const override;
    Ptr<Ipv4RoutingProtocol> Create(Ptr<Node> node) const override;

  private:
    const Fabric* m_fabric;
    Config m_cfg;
    const std::unordered_map<uint32_t, uint32_t>* m_leafOf;
};

} // namespace lb
} // namespace ns3

#endif
