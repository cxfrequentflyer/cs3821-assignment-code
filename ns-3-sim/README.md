# ns-3-sim: packet-level load balancing in a leaf-spine fabric

`lb-fabric` is one configurable ns-3 program that brings the project's
experiments down to packets. It covers the static balls-into-bins
experiments (`ecmp-sim` exp01-exp08) and the queueing ones (`dynamic-sim`'s
lab and transition matrix), with real TCP, switch queues, drops and
reordering. The Python scripts in `experiments/` sweep it and summarize the
results, one script per experiment.

## Build

```bash
cd /path/to/ns-3.49
rm -rf scratch && ln -s /path/to/cs3821-assignment-code/ns-3-sim scratch
./ns3 configure -d optimized --enable-modules=core,network,internet,point-to-point,applications,traffic-control,flow-monitor
./ns3 build lb-fabric
./ns3 run "lb-fabric --PrintHelp"
./ns3 run "lb-fabric --scheme=drill --d=2 --memory=1 --load=0.6 --flows=1000"
```

`-d optimized` matters: sweeps run the binary hundreds of times. The
module list is optional; it just shortens the first build (about 10 minutes
on 4 cores). Each run prints a one-line summary and appends a row of metrics
to `--out` (default `lb-fabric.csv`); `--flowsOut=flows.csv` adds one row per
flow.

## Running the experiments

```bash
export NS3_DIR=/path/to/ns-3.49
python3 ns-3-sim/experiments/ns06_granularity.py --quick --seeds 1   # smoke test, ~30 s
python3 ns-3-sim/experiments/ns06_granularity.py                     # full: more flows, 2 seeds
```

Flags shared by every script: `--seeds N` (runs per configuration, default 2),
`--jobs N` (parallel runs, default: all cores), `--quick` (fewer flows),
`--rerun` (ignore the cache), `--ns3 DIR` (instead of `$NS3_DIR`). Results are
cached in `experiments/results/<name>.csv` (gitignored, like `ecmp-sim/results`):
rerunning a script only simulates configurations it has not seen.

| Script | Mirrors | Question | Paper hook |
|---|---|---|---|
| `ns01_single_group.py` | ecmp-sim exp01 + exp02 | one ECMP group (leaf 0 -> leaf 1 over n uplinks): imbalance and FCT vs n and flow-size distribution, for ECMP, per-flow 2 choices and DRILL | balls into bins |
| `ns03_conga_bound.py` | exp03 | windowed uplink imbalance under ECMP vs CONGA's sqrt(8 n ln n (1+CV^2)/m) | CONGA Thm 2 |
| `ns04_herding.py` | exp04 | DRILL(d, m) x forwarding engines x stale queue information | DRILL §3.2.3, Fig. 2-3 |
| `ns05_speeds.py` | exp05 | one 4x-faster spine: memoryless d choices vs memory, WCMP, CONGA | DRILL Thm 1-2 |
| `ns06_granularity.py` | exp06 | flow -> flowlet -> flowcell -> packet: FCT gained vs reordering paid (writes a figure) | Presto, CONGA, DRILL Fig. 11 |
| `ns07_workloads.py` | exp07 | web search vs data mining across loads, normalized FCT for small and large flows | pFabric §5.4 |
| `ns08_asymmetry.py` | exp08 | failed link, quarter-speed link, imbalanced striping: ECMP, WCMP, CONGA, DRILL, DRILL with symmetric components | CONGA Fig. 2-3, DRILL Fig. 4, 13 |
| `ns09_dynamic.py` | dynamic-sim lab, transition-matrix | heterogeneous uplinks, links retired / added mid-run, idle uplinks per window | dynamic-sim |

There is no ns02: exp02's flow-size sweep is part of ns01.

## What lb-fabric models

**Topology.** `--leaves`, `--spines`, `--hosts` per leaf, `--hostRate`,
`--coreRate`, `--delay`, `--switchQueue` (drop-tail, packets). Links can be
failed (`--fail=0-0`), run at a different speed (`--slow=1-1:0.25`), striped
(`--striping`: leaf l doubled to spines l and l+1), and taken out of or
brought into service mid-run (`--retire=0-1@5000`, `--add=0-3@5000`, in
microseconds). Routing knows the topology at all times, as if link state
converged instantly, and a retired link's queue still drains.

**Load balancing** happens at the leaves (`lb-routing.{h,cc}`), the only hop
with a choice in a two-tier fabric:

| `--scheme` | granularity | information | notes |
|---|---|---|---|
| `ecmp` | flow | none | 5-tuple hash |
| `wcmp` | flow | static | hash weighted by path capacity |
| `dchoice` | flow | local | d sampled uplinks at the first packet, then pinned (ecmp-sim's DChoice; DRILL's "per-flow DRILL") |
| `flowlet` | flowlet | none | new random uplink after `--flowletGap` us idle (LetFlow-style) |
| `conga` | flowlet | global | least congested path, max(uplink, spine downlink) backlog, at each flowlet |
| `presto` | `--cellKB` cell | none | each flow round-robins its cells from a hashed start |
| `random` | packet | none | per-packet Random |
| `rr` | packet | none | round robin per engine |
| `drill` | packet | local | DRILL(`--d`, `--memory`) on `--engines` forwarding engines |
| `drill-comp` | packet | local | hash each flow to a symmetric component (weighted by capacity), DRILL inside it |

**Traffic.** `--flows` TCP flows (NewReno by default) with Poisson arrivals
at `--load` times the uplink capacity of the sending leaves. Sizes:
`empirical:<file>` (the CDFs in `ecmp-sim/workloads/`), `fixed:`, `exp:`,
`pareto:<alpha>:`, `lognormal:<sigma>:`, `bimodal:<p>:<ratio>:` (each ending
in the mean, in bytes). Patterns: `uniform`, `stride:<k>`, `pair:<a>:<b>`
(a single ECMP group).

**Metrics** (one CSV row per run):

- **FCT:** mean and p99, overall and for small (< 100KB) and large (> 10MB) flows.
- **Normalized FCT:** divided by a best case on an empty fabric, pFabric's metric.
- **Before / after a link event:** FCT split at the first `--retire` / `--add` time.
- **Reordering:**
  - `flows_reordered`: a segment arrived late although it was sent once.
  - `flows_with_ooo`: any late segment, retransmissions included.
  - `flows_over_3_dupacks`: TCP's fast-retransmit threshold, DRILL Fig. 11.
  - Retransmitted segments.
- **Drops:** per hop.
- **Queues:** the spread (std) of a leaf's uplink queue lengths, sampled every
  `--sampleUs` (DRILL's metric), plus mean and max.
- **Imbalance:** (max - min) / mean utilization of a leaf's uplinks, over the
  run and per `--windowUs` window (CONGA's chi, ecmp-sim's `imbalance`).
- **Idle uplinks:** the share per window (ecmp-sim's `idle_frac`).

Queue and window metrics cover the arrival period only, so the drain at the
end doesn't dilute them.

## Modeling choices to state

- **Queue length** is backlog in time (bytes / link rate), so links of
  different speeds compare fairly. The packet being transmitted is not
  counted, so queues are often empty and tie; candidates are always shuffled,
  so ties are random.
- **Forwarding engines and staleness:** a packet is handled by engine (input
  port mod F). With `--staleness=T` the switch's queue snapshot refreshes
  every T us. In between, an engine estimates each queue as the snapshot,
  plus what *it* has sent since, minus what the link can have drained; it
  never sees the other engines' packets. One engine is therefore nearly up to
  date, while F engines decide in parallel on partly shared, stale
  information. That is DRILL's synchronization mechanism, and the dynamic
  version of ecmp-sim's shared-snapshot rounds.
- **CONGA here is an oracle:** it reads the spine's downlink queue directly,
  instead of CONGA's piggybacked, quantized feedback. Treat it as an upper
  bound on what CONGA's information can buy.
- **TCP:**
  - Minimum RTO is 5 ms and the SYN timeout 10 ms (ns-3's default of 3 s
    dominates every tail otherwise), so the small-flow p99 often sits at about
    10 ms: one lost SYN.
  - The receiver ACKs every segment (`--delAck=1`). With delayed ACKs and an
    RTO below the 200 ms delayed-ACK timer, a flow that times out crawls one
    segment per RTO; that bug showed up during testing and is why the default
    is 1.
  - No ECN or DCTCP: pFabric's and DRILL's evaluations use other transports,
    so compare trends, not absolute numbers.
- **Normalized FCT** divides by a handshake RTT plus one-way delay plus the
  bytes at the host rate. It ignores slow start, so mid-size flows sit well
  above 1 for every scheme.
- **Oversubscription:** the defaults (4 x 4, 8 hosts per leaf, 10G hosts,
  20G core) are 1:1. ns06-ns08 use a 10G core (2:1) so that the uplinks,
  where the choice is made, are the bottleneck; at 1:1 the last hop masks the
  differences.
- **Not modeled:** host-side flowcell splitting and reorder buffers (Presto's
  shim), incast applications, multi-tier Clos.

## First observations (quick runs: few flows, one seed)

These come from `--quick --seeds 1` smoke runs, so they're for orientation,
not for quoting. Rerun the full settings before using any number.

- **ns06 (granularity):** the trade-off is clear.
  - ECMP reorders almost nothing (~2% of flows) but drops heavily at the
    uplinks and has the slowest flows.
  - Flowlets and CONGA stay below 5% reordered and are faster.
  - 64KB flowcells remove uplink drops at ~45% of flows reordered.
  - Per-packet schemes are fastest with 85-95% reordered.
  - DRILL reorders only slightly less than per-packet random here, unlike the
    paper's 8x. Possible reasons: queues are measured differently, the core
    runs at the host rate (10G) where DRILL's ran at 4x (40G), and there is no shim.
- **ns08 (asymmetry):** with a quarter-speed link, plain per-packet DRILL's
  mean FCT roughly quadruples, while DRILL with symmetric components and
  CONGA stay close to the symmetric case. This is DRILL §3.4's point.
- **ns07 (workloads):** data mining has lower normalized FCTs than web search
  at the same load: pFabric's "easier" result.
- **ns05 (unequal speeds):** at 80% load, one memory slot cuts DRILL(1,0)'s
  mean FCT about in half.
- **ns04 (herding):** staleness hurts monotonically; the engine-count effect
  is small next to seed-to-seed noise at quick settings, so run it with
  several seeds.
