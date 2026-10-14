"""The fast policies must agree exactly with the slow reference versions."""
import numpy as np
import pytest

import reference as ref
from ecmpsim import (ECMP, LPT, DChoice, Greedy, Pareto, Unit, opt_lower_bound,
                     path_loads, rng_for, sample_distinct)

N, M, T = 7, 60, 6

DCHOICE_VARIANTS = [DChoice(d, by, replace)
                    for d in (1, 2, 3) for by in ("bytes", "count") for replace in (True, False)]


def make_batch(policy, flows, seed=11):
    w = np.stack([flows.sample(rng_for(seed, "flows", flows.label, M, t), M) for t in range(T)])
    draws = [policy.draw(rng_for(seed, "policy", policy.name, N, M, t), M, N) for t in range(T)]
    return w, (None if draws[0] is None else np.stack(draws))


def reference_paths(policy, w_row, draws_row):
    if isinstance(policy, ECMP):
        return ref.ecmp(w_row, draws_row, N)
    if isinstance(policy, DChoice):
        return ref.dchoice(w_row, draws_row, N, policy.by)
    if isinstance(policy, Greedy):
        return ref.greedy(w_row, N)
    if isinstance(policy, LPT):
        return ref.lpt(w_row, N)
    raise TypeError(policy)


@pytest.mark.parametrize("flows", [Unit(), Pareto(1.5)], ids=lambda f: f.label)
@pytest.mark.parametrize("policy", [ECMP(), *DCHOICE_VARIANTS, Greedy(), LPT()],
                         ids=lambda p: p.name)
def test_fast_matches_reference(policy, flows):
    # Unit flows create lots of ties, so this also pins down tie-breaking.
    w, draws = make_batch(policy, flows)
    fast = policy.assign(w, draws, N)
    assert fast.shape == (T, M)
    for t in range(T):
        expected = reference_paths(policy, list(w[t]), None if draws is None else draws[t].tolist())
        assert fast[t].tolist() == expected


def test_sample_distinct_is_uniform_over_ordered_samples():
    n, d, size = 5, 3, 300_000
    picks = sample_distinct(np.random.default_rng(0), size, n, d)
    assert all(len(set(row)) == d for row in picks[:2000].tolist())
    codes = (picks[:, 0] * n + picks[:, 1]) * n + picks[:, 2]
    counts = np.bincount(codes, minlength=n ** 3)
    valid = counts[counts > 0]
    assert len(valid) == n * (n - 1) * (n - 2)       # every ordered triple shows up...
    expected = size / len(valid)
    assert np.abs(valid - expected).max() < 5 * np.sqrt(expected)   # ...about equally often


def test_graham_list_scheduling_bound():
    # Any greedy list schedule satisfies max <= total/n + (1 - 1/n) * largest flow.
    flows = Pareto(1.5)
    for policy in (Greedy(), LPT()):
        w, draws = make_batch(policy, flows, seed=5)
        load = path_loads(policy.assign(w, draws, N), w, N)
        bound = w.sum(axis=1) / N + (1 - 1 / N) * w.max(axis=1)
        assert np.all(load.max(axis=1) <= bound + 1e-9)


@pytest.mark.parametrize("policy", [ECMP(), DChoice(2), Greedy(), LPT()], ids=lambda p: p.name)
def test_nothing_beats_the_lower_bound(policy):
    w, draws = make_batch(policy, Pareto(1.5), seed=9)
    load = path_loads(policy.assign(w, draws, N), w, N)
    assert np.all(load.max(axis=1) >= opt_lower_bound(w, N) - 1e-9)
