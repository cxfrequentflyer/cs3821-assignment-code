# ecmpsim: balls into bins for ECMP groups, leaf-spine fabrics and fat-trees

Level 1 of the COMP3821 project simulator: flows (balls) are placed onto the
`n` equal-cost paths (bins) of a single ECMP group, and we measure how unevenly
the paths end up loaded. It is small on purpose, checked against exact answers,
and built so that every number in the report can be regenerated from a seed.

On top of that sit the datacenter load-balancing extensions drawn from CONGA
(SIGCOMM 2014), DRILL (SIGCOMM 2017) and pFabric (SIGCOMM 2013): weighted
imbalance theory, parallel stale decisions and memory, unequal path speeds,
chunking, real workloads, and a two-hop leaf-spine level. See
[Extensions](#extensions-datacenter-load-balancing) below.

Three further levels drop the model's biggest simplifications: a three-tier
**fat-tree** (several ECMP decisions per flow), **real hashes** (CRC and XOR
over structured 5-tuples, and hash polarization across switches), and **time**
(flows arrive and leave, paths are draining queues). See
[Fat-tree, real hashes and time](#fat-tree-real-hashes-and-time).

## Quick start

```bash
pip3 install -r requirements.txt        # numpy, pandas, matplotlib, scipy, pytest
python3 validate.py                     # the proposal's Example 1: exact vs simulated
python3 -m pytest                       # ~225 tests, ~15 seconds
python3 experiments/exp01_unit_accuracy.py   # core Q1 template -> results/*.csv, *.png
python3 experiments/exp02_flow_sizes.py      # core Q2/Q3 first look (printed tables)
python3 experiments/exp03_conga_bound.py     # extensions: exp03 ... exp08, see below
python3 experiments/exp09_fattree.py         # fat-tree, hashes, time: exp09 ... exp11
```

## Layout

```
ecmpsim/
  flows.py      flow-size distributions (weights), all scaled to mean 1, with their CV^2
  policies.py   where each flow goes: ECMP, DChoice (DRILL(d, m), engines), Greedy,
                LPT, RoundRobin, Spray
  chunking.py   split flows into chunks (flowcells / packets)
  metrics.py    path loads -> max load, gap, imbalance, max/lower-bound, idle paths, throughput
  runner.py     run(), sweep(), summarize(), paired_diff(), bootstrap_ci()
  exact.py      exact answers: max-load distribution (one choice), Markov chain (d choices)
  theory.py     asymptotic and heuristic predictions to test against, incl. CONGA's
                bound, the heavy-tail decay exponent and DRILL's counting bound
  leafspine.py  level 2: two-hop leaf-spine fabric, its policies and the LP optimum
  fattree.py    level 3: k-ary fat-tree, hop-by-hop policies, hash ECMP, the LP optimum
  hashing.py    CRC-16/CCITT, CRC-16/ARC, CRC-32 and XOR fold over 5-tuples; HashECMP
  dynamic.py    time: Poisson arrivals into draining FIFO paths, run_dynamic()
workloads/      web-search and data-mining flow-size CDFs (pFabric / CONGA)
tests/          reference implementations, exact-vs-brute-force, statistical checks
experiments/    one script per experiment; each writes a CSV, then plots from it
results/        CSVs and figures (regenerable, so you don't have to commit big ones)
validate.py     reproduces the proposal's worked example
```

## The model in one table

| Balls-into-bins | ECMP | In the code |
|---|---|---|
| ball | a flow | one column of the `(T, m)` weight array `w` |
| bin | an equal-cost path (next hop) | an integer in `0..n-1` |
| ball weight | flow rate demand (snapshot) or bytes (time window) | `flows.*`, mean 1 |
| allocation rule | the switch's path choice | `policies.*` |
| maximum load | the busiest link | `max_load` |

Because weights have mean 1, `m / n` (the `t` column) is always the expected
average load per path, so different flow-size distributions share an axis.

## Using it

```python
from ecmpsim import run, summarize, paired_diff, Pareto, ECMP, DChoice

df = run(n=16, m=64, flows=Pareto(1.5),
         policies=[ECMP(), DChoice(2, by="bytes", replace=False)],
         trials=2000, seed=3821)          # one row per (policy, trial)
summarize(df, metrics=["max_over_lb", "idle_frac"])
paired_diff(df, "max_over_lb", baseline="ecmp")
```

Metrics (all per trial, see `metrics.py`): `max_load`, `gap` (max - total/n,
what the theorems talk about), `max_over_mean`, `imbalance` ((max - min) /
(total/n), CONGA's chi), `max_over_lb` (max divided by max(total/n, largest
flow), the load no placement can beat), `lb_is_elephant`, `idle_frac`,
`max_count`, and `throughput` when you pass `capacity=`. With `speeds=` the
ratios use utilization (load / speed); with `chunk=` there is also `chunks`.

## Reproducibility: how the randomness works

Every random stream is keyed by what it is for, not by the order code runs in:

- flow sizes: `(seed, "flows", flows.label, m, trial)`
- policy draws: `(seed, "policy", policy.seed_name or name, n, m, trial)`, plus
  `("chunk", c)` for chunked runs. `seed_name` lets variants that differ only
  in `engines` share draws, so the engine sweep is paired too.

Consequences: every policy sees identical flow sets, so comparisons are
paired and much less noisy (`paired_diff`). Adding a policy does not change
any other policy's numbers. `batch_size` only affects speed and memory. The
tests check all three. Policies split into `draw` (random numbers for one
trial) and `assign` (deterministic), which is what makes this possible.

## How the simulator is validated

Work up the ladder before trusting any experiment, and re-run it after any
change to the core:

1. **Brute force**: enumerate every random outcome for tiny `m, n`.
2. **Exact**: the exact max-load distribution for one choice (Poissonization,
   checked against brute force and big-integer counting), and an exact Markov
   chain over load profiles for d choices (checked against brute force).
3. **Differential tests**: each fast policy agrees *exactly* with a slow,
   obviously-correct reference in `tests/reference.py`, given the same draws.
4. **Statistical tests**: simulated means land within 4 standard errors of the
   exact values; the n = 1000 two-choice case lands on tanh(1).
5. **validate.py**: every number in the proposal's Example 1 reproduces
   (68.4%, 80.8%, 2.13, 1.71, 76.2%).

## Adding a policy

1. Write a class with a stable `name`, `draw(rng, m, n)` returning one trial's
   random numbers (an array, a tuple of arrays, or `None`) and
   `assign(w, draws, n, **context)` returning `(T, m)` paths. Context is passed
   by keyword: `speeds` to everyone, `flow` and `rank` only to policies with
   `uses_flow = True`; accept `**_` for the rest.
2. Have someone *else* write the slow version in `tests/reference.py` from
   the description alone, and add it to `test_fast_matches_reference`.
3. If an exact answer exists for a small case, add it to `test_exact.py`.

## Statistics notes

- With Pareto alpha < 2, raw loads have infinite variance, so "mean ± 1.96 SE"
  is not a valid confidence interval for them. Report `max_over_lb` (bounded
  between 1 and n), medians and high quantiles, with `bootstrap_ci`.
- Compare policies with `paired_diff` rather than overlapping CIs.
- In exp01, 1000 trials pin E[max load] down to between ±0.03 flows (two
  choices) and ±0.26 flows (one choice at 64 flows per path). Scale trials to
  the effect you need to resolve, not to a round number.

## Modeling choices to state in the report

- `ECMP()`'s hash is ideal: independent and uniform per flow. Real switches use
  deterministic hashes (CRC, XOR) of header fields: `HashECMP`, see below.
- A weight is a rate demand in a snapshot; a path delivers min(capacity, load).
- `DChoice(replace=True)` samples with replacement, as in the classical
  analysis and the proposal's n = 4 numbers; `replace=False` (what a switch
  would do) is noticeably better at small n (84.7% vs 80.8% of paths in use).
- Ties go to a uniformly random tied path. For symmetric metrics the
  tie-breaking rule cannot change the results.
- `run()` is static: all flows present at once, none leaving. `run_dynamic()`
  is the version with a clock, see below.

## Extensions: datacenter load balancing

Each experiment simulates once into `results/`, prints its tables, and (where a
figure helps) plots from the CSV. Pass `--rerun` to simulate again.

| Script | Idea | Paper hook | Headline so far (default settings) |
|---|---|---|---|
| `exp03_conga_bound.py` | test CONGA's imbalance theorem | CONGA §6.2, Thm 2 | The bound holds and is ~25% loose for finite variance. With Pareto(alpha < 2) it says nothing, and imbalance decays like m^-(1-1/alpha) (fitted -0.34 vs -1/3 for alpha = 1.5) instead of m^-1/2 |
| `exp04_herding.py` | F engines deciding from one stale snapshot | DRILL §3.2.3, Fig. 2-3; their open problem | One engine: more choices are better. 16 engines: DRILL(2,1) / (1,2) is best and more d or memory hurts. 64 engines: greedy is 4.5x the lower bound, far worse than ECMP's 1.7 |
| `exp05_speeds.py` | memory and unequal path speeds | DRILL §3.2.4, Thm 1-2 | Memoryless d choices sit above the counting bound (d = 2: 1.81 vs bound 1.69); one memory slot brings every d to ~1.1-1.2 |
| `exp06_chunking.py` | split flows into chunks of size c | Presto via DRILL, CONGA flowlets | Below c ~ 0.5 the flow-size distribution stops mattering; ECMP on chunks improves like sqrt(c), spray and d = 2 roughly like c |
| `exp07_workloads.py` | web search vs data mining | pFabric §5.4, CONGA §6.2 | Data mining has 12x the CV^2 and a bigger raw imbalance, yet at low flow counts every policy is at its lower bound: one elephant dominates and there is nothing left to balance. A static version of pFabric's "easier" |
| `exp08_leafspine.py` | two hops: asymmetry and PoA | CONGA Fig. 2-3, Thm 1; DRILL Fig. 4, 13 | One-shot CONGA greedy ~1.14x optimum, its Nash equilibrium ~1.01x; ESF is optimal when symmetric but 2.9x with a slow link; local schemes move the bottleneck downstream |

Modeling choices to state for the extensions:

- **Engines** (`engines=F`): flow j goes to engine j mod F; flows are decided
  in rounds of F from the snapshot at the start of the round, each engine
  with its own memory. Loads never drain, so a remembered path never goes
  stale over time the way DRILL's queues do. `run_dynamic` (below) is the
  version where they drain.
- **Ties under stale information matter**: greedy with "first" ties sends a
  whole round to one path. Both tie rules are implemented; report both.
- **DRILL memory** remembers the chosen path, then the other distinct
  candidates from least to most loaded, as seen at decision time.
- **Speeds** are relative path capacities; load-aware rules compare load /
  speed. ECMP ignores them; `LSWCMP` is the weighted version in the leaf-spine.
- **Chunks** are fixed-size pieces in units of the mean flow; their order is
  interleaved across flows by default. Flowlets (gap-defined) and the cost of
  reordering are not modeled; that is the ns-3 track's question.
- **Workload CDFs** in `workloads/` are the commonly circulated transcriptions;
  `tests/test_workloads.py` checks them against pFabric's stated summaries,
  but compare with the original figures before quoting exact values.
- **Leaf-spine**: fluid links (utilization can exceed 1), no host links, no
  TCP. DRILL's Fig. 4 argument (TCP coupling the rates of a split flow) is
  therefore not reproduced; the static load distribution is. The benchmark is
  the splittable multicommodity LP (scipy), which equals ESF in a symmetric
  fabric (tested). The Nash policy plays CONGA's game with whole flows, while
  CONGA's PoA = 2 theorem is about the splittable game, so look at `over_lb`
  too. CONGA's Fig. 17 lower-bound instance is not reproduced.
- **Reference implementations** for the new options (`tests/reference.py`)
  were written by the same author as the fast code. Have someone else
  re-derive them from the docstrings, as the section above asks.

## Fat-tree, real hashes and time

Three levels that used to be listed here as "not modeled yet".

| Script | Idea | Paper hook | Headline so far (default settings) |
|---|---|---|---|
| `exp09_fattree.py` | k = 8 fat-tree: two ECMP decisions per flow, failures, a slow link | DRILL §3.4, Fig. 4; CONGA Fig. 2-3 | Symmetric: ECMP 1.59x the optimum, each switch's least-loaded uplink 1.47x, and the bottleneck moves to the core and aggregation downlinks nobody chose. A slow core link: ECMP 2.9x, local 2.7x and even equal split 2.7x, DRILL's components 1.5x, global path information 1.1x |
| `exp10_hashes.py` | CRC16 / CRC32 / XOR over structured 5-tuples; hash polarization | Al-Fares et al. (fat-tree) addressing; switch CRC16 hashing | On realistic tuples the CRCs are indistinguishable from the ideal hash. A port counter is spread *better* than random (XOR: a perfect round robin). XOR with source port = destination port leaves 75% of 64 paths idle. Same hash on both tiers: 4 of 16 cores used, 4.0x the optimum; XOR seeds: all cores but still 2.7x; another polynomial or other bits: 1.4x, the ideal value |
| `exp11_time.py` | Poisson arrivals, draining FIFO paths | DRILL §3.2.3-3.2.4, Thm 1-2; M/G/1 | ECMP matches Pollaczek-Khinchine to within the CI. At load 0.9, n = 16: mean wait 9.3 (ECMP), 1.3 (d = 2), 0.9 (DRILL(2,1)), 0.37 (greedy). With one 10x path, ECMP is unstable above load 0.53 and d = 2 above 0.59, piling up exactly the counting bound's share (0.308 and 0.208 at load 0.8); one unit of memory is stable at every load tested |

### Fat-tree (`fattree.py`)

```python
from ecmpsim import Exponential, Hash
from ecmpsim.fattree import (FatTree, FTECMP, FTHashECMP, FTLocal, FTGlobal,
                             FTComponents, FTESF, run_fattree)

net = FatTree.kary(8).fail_core_link(pod=0, agg=0, core=0)
df = run_fattree(net, m=2048, flows=Exponential(), trials=100, seed=3821,
                 policies=[FTECMP(), FTHashECMP("crc16"), FTLocal(), FTGlobal(),
                           FTComponents(), FTESF()])
df.groupby("policy")[["over_opt", "agg_up_max", "cores_used"]].mean()
```

A k-ary fat-tree has k pods of k/2 edge and k/2 aggregation switches, and
(k/2)^2 cores; core (a, c) hangs off aggregation switch a of every pod. A flow
between pods is routed by two choices, the aggregation switch a (made by the
edge switch) and the core c (made by that aggregation switch), and loads four
links; inside a pod there is only a. `FatTree.symmetric(pods, edges, aggs,
cores)` builds other shapes, `set_edge_link` / `set_core_link` / `fail_*`
change capacities. Columns: `max_util`, the busiest link of each tier
(`edge_up_max`, `agg_up_max`, `core_down_max`, `agg_down_max`), `cores_used`,
and `opt` / `over_opt` / `over_lb` against the splittable LP as in level 2.

Checked by: a dict-of-links reference for the loads, hand-worked optima,
ESF = LP in symmetric trees, `over_opt >= 1` for every policy on asymmetric
ones, the per-hop probabilities of ideal ECMP next to a failure, and DRILL's
components on a tree with one failed core link.

### Real hashes (`hashing.py`)

```python
from ecmpsim import run, Unit, ECMP, HashECMP
run(64, 1024, Unit(), [ECMP(), HashECMP("crc16", "datacenter"),
                       HashECMP("xor16", "symmetric")], trials=500, seed=3821)
```

`Hash(kind, shift=0)` is `crc16` (CCITT), `crc16-arc`, `crc32` or `xor16`,
vectorized over flows and checked against the published check values, a
bit-at-a-time reference and `zlib.crc32`. `HashECMP(hash, traffic)` is a
normal policy for `run()` and `run_dynamic()`: its "random draws" are the
flows' 5-tuples (`five_tuples`: `random`, `datacenter`, `sequential`,
`symmetric`, `incast`), keyed by the traffic kind so every hash sees the same
headers. In the fat-tree the 5-tuples belong to the traffic (addresses
`10.pod.edge.host`, `ports=` chooses the port pattern) and
`FTHashECMP(edge, agg, seeds)` hashes them at both tiers.

Polarization in one line: aggregation switch a only receives flows with
hash mod k/2 = a, so if it computes the same hash it sends them all to its
core number a. A CRC is linear, so a per-switch seed (initial value, or seed
bytes before or after the header) only XORs a constant into the result, which
permutes the buckets instead of mixing them: the test suite checks that
identity and that each edge switch still reaches only k/2 of its (k/2)^2
paths. A different polynomial or different bits at the second tier do fix it.

### Time (`dynamic.py`)

```python
from ecmpsim import run_dynamic, summarize_dynamic, Exponential, ECMP, DChoice
df = run_dynamic(n=16, load=0.9, flows=Exponential(), arrivals=60_000, trials=20, seed=3821,
                 policies=[ECMP(), DChoice(2, replace=False),
                           DChoice(1, replace=False, memory=1)])
summarize_dynamic(df)        # wait_mean, wait_p99, queue_max, idle_frac, growth (+ CIs)
```

Flows arrive as a Poisson process (or `process="constant"`) at offered load
`load` = arrival rate x mean size / total speed. A flow adds its bytes to the
queue of its path; path i sends `speeds[i]` bytes per unit time, first come
first served; a flow has left when its last byte is sent. The existing
policies are reused unchanged (ECMP, DChoice with memory / engines / ties,
Greedy, RoundRobin, HashECMP): the load-aware ones now compare what is still
queued, as clearing time. Metrics per run: `wait_mean`, `wait_p99`,
`fct_mean`, `slowdown_mean`, `queue_mean`, `queue_max`, `idle_frac`,
`backlog_mean`, and `growth` (share of the offered traffic piling up: 0 when
stable).

Seeding: flow sizes and policy draws come from the same streams as the static
`run()`, and arrival times from `(seed, "gaps", process, arrivals, trial)`
scaled by the rate, so runs are paired across policies, loads and speeds.

Checked by: a one-trial reference for the placement over time (memory,
engines, speeds), the Lindley recursion for the waits, "no time passing =
the static model", M/M/1 and M/D/1 means and idle fractions within 4 standard
errors, and DRILL's Theorem 1 pile-up rate (`theory.dchoice_pileup`).

### Modeling choices to state for these levels

- **Fat-tree**: fluid links, no host links, no TCP, static (all flows at once),
  as in the leaf-spine level. `FTLocal` and `FTComponents` decide hop by hop
  on uplink utilization only; `FTGlobal` is a yardstick, not a protocol
  (CONGA's feedback is two-tier only). `FTESF` is the *hop-by-hop* equal
  split, which is not the optimum next to a failure (tested). Components are
  this code's reading of DRILL §3.4 (same set of edge-switch pairs per link,
  same capacity ratios); check it against the paper before quoting.
- **Next-hop lists under failures**: a switch only lists next hops that can
  still reach the destination, and takes hash mod (list length). Real routing
  converges to that; the transient is not modeled.
- **Hashes**: the 13-byte 5-tuple in network order is the whole hash input (no
  VLAN, no ingress port, no flow label). Switch chips differ in field order,
  folding and bucket selection, and their exact functions are rarely public,
  so read the results as "a CRC" and "an XOR fold", not as a specific ASIC.
  Seeds are modeled as an XOR, exact for CRCs by linearity; a vendor mixing
  the seed in non-linearly would behave more like "other polynomial".
- **Traffic kinds** are stylized on purpose (one counter, exact port symmetry);
  they show what *can* happen, not how often it does in a real trace.
- **Time**: one FIFO queue per path with no buffer limit, so nothing is dropped
  and an overloaded path grows for ever; a flow's bytes all arrive at once (no
  TCP, no rates, no packets); policies compare bytes still queued, not queue
  length in packets as DRILL does (`by="count"` is rejected). Systems start
  empty and the first quarter of the arrivals is warm-up by default: at loads
  above ~0.95 lengthen both. Metrics are sampled at arrivals, which is the
  time average for Poisson arrivals only.
- **Mean waits under heavy tails**: with infinite-variance sizes (Pareto alpha
  <= 2) the M/G/1 mean wait is infinite at any load, so a simulated mean is
  meaningless; use quantiles, as the statistics notes say for the static case.
- **Reference implementations** for these levels were again written by the
  same author as the fast code.

## Not modeled yet (natural next levels)

- **A fat-tree over time**: `dynamic.py` is one ECMP group; queues at every
  tier (and DRILL's per-packet decisions at each hop) need the two combined.
- **Packets and TCP**: reordering, flowlet gaps, rate adaptation, buffer
  limits and drops. That is the ns-3 track's question.
- **Queue length as the signal**: tracking the number of flows or packets
  still on a path, instead of bytes.
- **Hash details of real ASICs**: field selection, symmetric hashing,
  resilient hashing and bucket tables (the remap when a next hop fails).
