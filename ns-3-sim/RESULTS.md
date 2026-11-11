# lb-fabric results (full sweeps)

All numbers are means over 2 seeds at each script's default settings
(`python3 ns-3-sim/experiments/nsXX_*.py`, ns-3.49, optimized build). Rerun
a script to get the per-run CSV in `experiments/results/`; the tables below
are excerpts of what the scripts print. Two seeds are enough to see the
effects below, but not to put tight error bars on them; use `--seeds 5` or
more before quoting a difference of a few percent.

Shorthand used throughout: **FCT** is flow completion time; **NFCT** is FCT
divided by the best case on an empty fabric (pFabric's metric); **load** is
the offered traffic as a share of the sending leaves' uplink capacity.

## ns01: one ECMP group (ecmp-sim exp01 + exp02)

Leaf 0's n uplinks carry all traffic to leaf 1; load 0.7; flow sizes with a
100KB mean. Windowed uplink imbalance ((busiest - least loaded) / mean per
1 ms window) at n = 16:

| sizes | ECMP | per-flow 2 choices | DRILL(2,1) |
|---|---|---|---|
| fixed | 1.03 | 0.81 | 0.11 |
| exponential | 1.29 | 1.13 | 0.12 |
| Pareto(1.5) | 1.41 | 1.24 | 0.13 |
| lognormal(1.5) | 1.54 | 1.41 | 0.12 |
| bimodal (5% x100) | 2.29 | 2.12 | 0.14 |

- **Same ordering as the static model:** imbalance grows with the group size
  n and with the spread of flow sizes, under ECMP and under per-flow choices.
- **Per-flow 2 choices** cut ECMP's imbalance only by 8-22% here, much less
  than in the static model. A flow is placed when it starts, and the queue it
  saw at that moment says little about the bytes it will send later.
- **Per-packet DRILL** is about 10x more even than either, and never dropped a
  packet at the uplinks. ECMP lost 1.3k-4.3k packets there per run.

## ns03: CONGA's bound with packets (exp03)

ECMP over 8 uplinks, load 0.6. Here m = flows arriving per window.

| window | m | fixed | exponential | lognormal(1.5) | Pareto(1.5) | bound (exp) |
|---|---|---|---|---|---|---|
| 100 us | 6 | 1.71 | 1.76 | 1.84 | 1.82 | 6.66 |
| 300 us | 18 | 1.39 | 1.55 | 1.67 | 1.60 | 3.85 |
| 1 ms | 60 | 0.94 | 1.16 | 1.38 | 1.29 | 2.11 |
| 3 ms | 180 | 0.58 | 0.77 | 1.06 | 0.97 | 1.22 |

- **The bound holds in every case**, and is loosest for short windows.
- **The decay with m is slower than 1/sqrt(m).** Over 30x more flows per
  window, exponential sizes only drop 2.3x, not 5.5x. Bytes in a window come
  from flows that started earlier and are still sending, so windows are
  correlated; the theorem counts each flow's bytes once, when it arrives.

## ns04: herding (exp04)

8 uplinks per leaf, 16 hosts (so up to 16 engines), load 0.8. The table
shows the standard deviation of a leaf's uplink queue lengths, in packets
(DRILL's metric).

| staleness | engines | DRILL(2,0) | DRILL(2,1) | DRILL(8,1) | DRILL(1,4) |
|---|---|---|---|---|---|
| live | any | 0.63 | 0.56 | 0.44 | 0.49 |
| 20 us | 1 | 1.27 | 1.26 | 1.19 | 1.26 |
| 20 us | 4 | 2.03 | 2.05 | 2.08 | 2.11 |
| 20 us | 16 | 2.07 | 2.10 | 2.16 | 2.20 |
| 100 us | 16 | 2.30 | 2.39 | 2.35 | 2.40 |

- **Live information:** more choices help, and the engine count doesn't matter.
- **Stale information:** more engines spread the queues more (1.27 to 2.07 at
  20 us), and extra choices or memory stop helping. With 16 engines, DRILL(2,0)
  is the most even and DRILL(1,4) the least. That is the direction of DRILL's
  synchronization effect and of ecmp-sim exp04.
- **The effect is small at the flow level:** mean FCT stays within about
  ±10% (1.07-1.25 ms) across the whole sweep.

## ns05: memory and unequal speeds (exp05)

Every leaf's link to spine 0 is 4x faster. Load 0.8:

| scheme | uplink queue (pkts) | mean FCT (ms) | uplink drops |
|---|---|---|---|
| ECMP | 39.2 | 4.15 | 20.5k |
| WCMP | 20.5 | 2.00 | 6.2k |
| CONGA | 14.5 | 1.51 | 1.7k |
| DRILL(1,0) | 46.2 | 3.90 | 9.8k |
| DRILL(1,1) | 2.8 | 1.03 | 5 |
| DRILL(2,0) | 15.7 | 1.33 | 565 |
| DRILL(2,1) | 2.3 | 0.98 | 9 |

- **DRILL's Theorems 1-2 at packet level:** without memory, the fast spine is
  under-used and the slow uplinks queue and drop. One memory slot fixes it,
  even with a single sample.

## ns06: granularity, gain vs cost (exp06)

Web search, load 0.6, 2:1 oversubscribed core. `results/ns06_granularity.png`
plots mean FCT against the share of flows reordered.

| granularity | mean FCT (ms) | small-flow p99 (ms) | flows reordered | > 3 dup ACKs | uplink drops |
|---|---|---|---|---|---|
| ECMP (flow) | 6.63 | 10.9 | 0.9% | 42% | 18.3k |
| flowlet 100 us | 5.22 | 10.4 | 2.4% | 41% | 17.0k |
| CONGA 100 us | 4.22 | 5.3 | 2.9% | 36% | 15.3k |
| Presto 64KB | 5.98 | 0.6 | 52% | 50% | 1.1k |
| Presto 16KB | 4.67 | 0.3 | 74% | 69% | 0 |
| random (packet) | 3.92 | 0.8 | 98% | 79% | 0 |
| DRILL(2,1) (packet) | 3.69 | 4.1 | 95% | 70% | 257 |

- **Finer granularity:** faster flows and fewer drops, paid for in reordering.
  Per-packet schemes are about 1.8x faster than ECMP but reorder nearly every
  flow.
- **CONGA** gets most of the gain at almost no reordering.
- **DRILL vs random:** DRILL reorders only slightly fewer flows than
  per-packet random here. DRILL's paper reports 8x fewer, with 40G core links
  against our 10G and with a different queue model.
- **Dup ACKs come from loss as well as reordering**, which is why ECMP also
  shows 42%. "Flows reordered" counts only segments overtaken on another path.

## ns07: web search vs data mining (exp07)

Mean NFCT; 2:1 core.

| workload | load | ECMP | CONGA | Presto | DRILL |
|---|---|---|---|---|---|
| web search | 0.3 | 11.5 | 4.0 | 2.1 | 3.9 |
| web search | 0.7 | 14.9 | 5.6 | 4.4 | 4.1 |
| data mining | 0.3 | 10.6 | 12.2 | 2.0 | 7.6 |
| data mining | 0.7 | 9.1 | 4.0 | 1.6 | 9.1 |

- **pFabric's "data mining is easier" holds for ECMP and Presto**, and for
  CONGA at moderate to high load.
- **It reverses for per-packet DRILL:** small-flow p99 NFCT is about 180 on
  data mining vs about 50-80 on web search. Spraying the packets of very large
  flows reorders them heavily, and the resulting fast retransmits load the
  queues that the small flows share.

## ns08: asymmetry (exp08)

Mean FCT in ms; 2:1 core, load 0.5.

| scenario | ECMP | WCMP | CONGA | Presto | DRILL | DRILL + components |
|---|---|---|---|---|---|---|
| symmetric | 1.88 | 1.81 | 0.70 | 0.84 | 0.50 | 0.50 |
| 1 failed link | 2.00 | 2.03 | 0.74 | 0.98 | 0.53 | 0.64 |
| quarter-speed link | 6.94 | 1.91 | 0.76 | 7.75 | 5.00 | 0.60 |
| imbalanced striping | 2.51 | 2.02 | 0.97 | 2.27 | 0.96 | 1.15 |

- **A slow link breaks load-oblivious and purely local schemes:** ECMP, Presto
  and plain DRILL all slow down 4-10x. DRILL's symmetric components and CONGA's
  path-wide view barely notice it. That is DRILL §3.4's point and CONGA
  Fig. 2-3's.
- **With imbalanced striping**, components cost DRILL a little (1.15 vs
  0.96 ms): they pin each flow to a subset of its paths.

## ns09: bins over time (dynamic-sim)

One ECMP group of 8 uplinks, log-normal sizes (sigma 1), 2 seeds.

| scenario | load | ECMP | 2 choices | DRILL(2,0) | DRILL(2,1) |
|---|---|---|---|---|---|
| steady, mean FCT (ms) | 0.9 | 3.05 | 2.63 | 2.40 | 2.35 |
| uneven speeds, mean FCT (ms) | 0.9 | 11.05 | 2.94 | 2.50 | 1.98 |
| uneven speeds, uplink queue (pkts) | 0.9 | 29.7 | 24.4 | 14.8 | 0.70 |
| retire 2 links, FCT before -> after (ms) | 0.6 | 1.00 -> 1.57 | 0.72 -> 0.87 | 0.55 -> 0.58 | 0.56 -> 0.67 |

- **Uneven speeds with the same total capacity** hurt ECMP 3.6x. Memory
  matters again: DRILL(2,1) keeps the queues at 0.7 packets where DRILL(2,0)
  builds 14.8.
- **Retiring two links** at 60% load costs ECMP 57% on FCT, and per-packet
  DRILL with no memory almost nothing. At 90% load the remaining 6 links are
  overloaded for every scheme.
- **Idle uplinks per 50 us window** (the transition-matrix question): about 6%
  under ECMP and 2 choices at load 0.6, and 0 under DRILL, which keeps every
  link busy.
