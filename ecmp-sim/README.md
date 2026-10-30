# ecmpsim: balls into bins for ECMP groups and leaf-spine fabrics

Level 1 of the COMP3821 project simulator: flows (balls) are placed onto the
`n` equal-cost paths (bins) of a single ECMP group, and we measure how unevenly
the paths end up loaded. It is small on purpose, checked against exact answers,
and built so that every number in the report can be regenerated from a seed.

On top of that sit the datacenter load-balancing extensions drawn from CONGA
(SIGCOMM 2014), DRILL (SIGCOMM 2017) and pFabric (SIGCOMM 2013): weighted
imbalance theory, parallel stale decisions and memory, unequal path speeds,
chunking, real workloads, and a two-hop leaf-spine level. See
[Extensions](#extensions-datacenter-load-balancing) below.

## Quick start

```bash
pip3 install -r requirements.txt        # numpy, pandas, matplotlib, scipy, pytest
python3 validate.py                     # the proposal's Example 1: exact vs simulated
python3 -m pytest                       # ~155 tests, ~15 seconds
python3 experiments/exp01_unit_accuracy.py   # core Q1 template -> results/*.csv, *.png
python3 experiments/exp02_flow_sizes.py      # core Q2/Q3 first look (printed tables)
python3 experiments/exp03_conga_bound.py     # extensions: exp03 ... exp08, see below
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

- The hash is ideal: independent and uniform per flow. Real switches use
  deterministic hashes (CRC, XOR) of header fields.
- A weight is a rate demand in a snapshot; a path delivers min(capacity, load).
- `DChoice(replace=True)` samples with replacement, as in the classical
  analysis and the proposal's n = 4 numbers; `replace=False` (what a switch
  would do) is noticeably better at small n (84.7% vs 80.8% of paths in use).
- Ties go to a uniformly random tied path. For symmetric metrics the
  tie-breaking rule cannot change the results.
- Everything is static: all flows present at once, none leaving.

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
  stale over time the way DRILL's queues do. That part needs `dynamic-sim`.
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

## Not modeled yet (natural next levels)

- **Fat-tree / 3 tiers**: several ECMP decisions per flow. The leaf-spine
  level covers two hops; DRILL §3.4 shows how symmetric components extend to
  multi-stage Clos.
- **Real hashes**: CRC16 over structured 5-tuples, and correlated choices
  across switches (hash polarization).
- **Time**: flows arriving and leaving, so loads are a queueing process
  (`../dynamic-sim` does this for one group; DRILL's draining queues and
  stability over time belong there).
