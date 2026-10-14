# ecmpsim: balls into bins for one ECMP group

Level 1 of the COMP3821 project simulator: flows (balls) are placed onto the
`n` equal-cost paths (bins) of a single ECMP group, and we measure how unevenly
the paths end up loaded. It is small on purpose, checked against exact answers,
and built so that every number in the report can be regenerated from a seed.

## Quick start

```bash
pip3 install -r requirements.txt        # numpy, pandas, matplotlib, pytest
python3 validate.py                     # the proposal's Example 1: exact vs simulated
python3 -m pytest                       # ~75 tests, a few seconds
python3 experiments/exp01_unit_accuracy.py   # core Q1 template -> results/*.csv, *.png
python3 experiments/exp02_flow_sizes.py      # core Q2/Q3 first look (printed tables)
```

## Layout

```
ecmpsim/
  flows.py      flow-size distributions (weights), all scaled to mean 1
  policies.py   where each flow goes: ECMP, DChoice, Greedy, LPT
  metrics.py    path loads -> max load, gap, max/lower-bound, idle paths, throughput
  runner.py     run(), sweep(), summarize(), paired_diff(), bootstrap_ci()
  exact.py      exact answers: max-load distribution (one choice), Markov chain (d choices)
  theory.py     asymptotic and heuristic predictions to test against
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
what the theorems talk about), `max_over_mean`, `max_over_lb` (max divided by
max(total/n, largest flow), the load no placement can beat), `lb_is_elephant`,
`idle_frac`, `max_count`, and `throughput` when you pass `capacity=`.

## Reproducibility: how the randomness works

Every random stream is keyed by what it is for, not by the order code runs in:

- flow sizes: `(seed, "flows", flows.label, m, trial)`
- policy draws: `(seed, "policy", policy.name, n, m, trial)`

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
   random numbers (or `None`) and `assign(w, draws, n)` returning `(T, m)` paths.
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

## Not modeled yet (natural next levels)

- **Fat-tree (level 2)**: several ECMP decisions per flow, downlinks with no
  choice at all, and flows between different host pairs. Optimal routing there
  is a *multicommodity* flow problem, so a single max-flow is only a valid
  benchmark when all flows share one source and one destination.
- **Real hashes**: CRC16 over structured 5-tuples, and correlated choices
  across switches (hash polarization).
- **Time**: flows arriving and leaving, so loads are a queueing process.
