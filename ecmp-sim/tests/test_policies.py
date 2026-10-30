"""The fast policies must agree exactly with the slow reference versions."""
import numpy as np
import pytest

import reference as ref
from ecmpsim.runner import _stack
from ecmpsim import (ECMP, LPT, DChoice, Greedy, Pareto, RoundRobin, Spray, Unit, chunk_flows,
                     opt_lower_bound, path_counts, path_loads, rng_for, sample_distinct)

N, M, T = 7, 60, 6

DCHOICE_VARIANTS = [DChoice(d, by, replace)
                    for d in (1, 2, 3) for by in ("bytes", "count") for replace in (True, False)]


def make_batch(policy, flows, seed=11):
    w = np.stack([flows.sample(rng_for(seed, "flows", flows.label, M, t), M) for t in range(T)])
    draws = [policy.draw(rng_for(seed, "policy", policy.name, N, M, t), M, N) for t in range(T)]
    return w, _stack(draws)


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


# --- DRILL options: memory, engines, random ties, unequal speeds -----------------------

EXTRA_VARIANTS = [
    DChoice(2, replace=False, memory=1),
    DChoice(1, memory=1),
    DChoice(2, memory=3, ties="random"),
    DChoice(3, "count", replace=False, engines=4),
    DChoice(2, replace=False, memory=2, engines=3),
    DChoice(2, replace=False, memory=1, engines=5, ties="random"),
]
SPEEDS = [3.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0]


@pytest.mark.parametrize("speeds", [None, SPEEDS], ids=["equal", "speeds"])
@pytest.mark.parametrize("flows", [Unit(), Pareto(1.5)], ids=lambda f: f.label)
@pytest.mark.parametrize("policy", EXTRA_VARIANTS, ids=lambda p: p.name)
def test_drill_options_match_reference(policy, flows, speeds):
    w, draws = make_batch(policy, flows)
    picks, keys = draws if policy.ties == "random" else (draws, None)
    fast = policy.assign(w, draws, N, speeds=speeds)
    for t in range(T):
        expected = ref.drill(list(w[t]), picks[t].tolist(), N, policy.by, policy.memory,
                             policy.engines, None if keys is None else keys[t].tolist(), speeds)
        assert fast[t].tolist() == expected


@pytest.mark.parametrize("speeds", [None, SPEEDS], ids=["equal", "speeds"])
@pytest.mark.parametrize("policy", [Greedy(engines=1), Greedy(engines=4), Greedy(ties="random"),
                                    Greedy(engines=3, ties="random")], ids=lambda p: p.name)
def test_greedy_options_match_reference(policy, speeds):
    w, draws = make_batch(policy, Pareto(1.5))
    fast = policy.assign(w, draws, N, speeds=speeds)
    for t in range(T):
        keys = None if draws is None else draws[t].tolist()
        assert fast[t].tolist() == ref.greedy_engines(list(w[t]), N, policy.engines, keys, speeds)


@pytest.mark.parametrize("engines", [1, 3, 8])
def test_round_robin_matches_reference(engines):
    policy = RoundRobin(engines)
    w, draws = make_batch(policy, Unit())
    fast = policy.assign(w, draws, N)
    for t in range(T):
        assert fast[t].tolist() == ref.round_robin(M, N, draws[t].tolist())


def test_greedy_herds_with_stale_snapshots():
    # F engines, unit flows, ties to the lowest index: each round of F flows reads
    # the same snapshot, so all F land on one path. Random ties spread them out.
    # (Random ties only help while everything is tied: here, the first round.)
    F = N
    w = np.ones((T, F))
    herd = path_counts(Greedy(engines=F).assign(w, None, N), N)
    assert (herd.max(axis=1) == F).all()
    keys = np.random.default_rng(0).random((T, F, N))
    spread = path_counts(Greedy(engines=F, ties="random").assign(w, keys, N), N)
    assert spread.max(axis=1).mean() < F / 2
    # with fully up-to-date loads (one engine) unit flows spread perfectly
    assert (path_counts(Greedy().assign(w, None, N), N).max(axis=1) == 1).all()


def test_one_engine_is_the_classical_model():
    for a, b in [(DChoice(2, engines=1), DChoice(2)), (Greedy(engines=1), Greedy())]:
        assert a.name == b.name
        w, draws = make_batch(a, Pareto(1.5))
        assert (a.assign(w, draws, N) == b.assign(w, draws, N)).all()


def test_engines_share_draws_across_engine_counts():
    assert DChoice(2, memory=1, engines=8).seed_name == DChoice(2, memory=1).seed_name
    assert Greedy(engines=4).seed_name == Greedy().seed_name


def test_spray_without_chunks_is_ecmp():
    rng = np.random.default_rng(0)
    draws = np.stack([Spray().draw(rng, M, N) for _ in range(T)])
    w = np.ones((T, M))
    assert (Spray().assign(w, draws, N) == ECMP().assign(w, draws, N)).all()


def test_spray_round_robins_each_flow():
    w = np.array([[2.5, 0.4, 1.0]])
    sizes, flow, rank = chunk_flows(w[0], 1.0)
    starts = np.array([[5, 1, 6]])
    paths = Spray().assign(sizes[None], starts, N, flow=flow[None], rank=rank[None])[0]
    got = {(int(f), int(r)): int(p) for f, r, p in zip(flow, rank, paths)}
    assert got == {(0, 0): 5, (0, 1): 6, (0, 2): 0, (1, 0): 1, (2, 0): 6}


@pytest.mark.parametrize("c", [1.0, 0.3, 0.05])
def test_chunk_flows_matches_reference(c):
    w = Pareto(1.5).sample(np.random.default_rng(3), 40)
    sizes, flow, rank = chunk_flows(w, c, order="flow")
    expected = ref.chunk_flows(list(w), c)
    assert [(int(f), int(r)) for f, r in zip(flow, rank)] == [(f, r) for _, f, r in expected]
    assert np.allclose(sizes, [s for s, _, _ in expected])
    assert np.all(sizes > 0) and np.all(sizes <= c * (1 + 1e-9))
    assert np.allclose(np.bincount(flow, weights=sizes), w)        # nothing lost
    inter = chunk_flows(w, c)                                       # same chunks, other order
    assert sorted(zip(inter[1], inter[2])) == sorted(zip(flow, rank))
    assert list(inter[2]) == sorted(inter[2])                       # rank-major


def test_fast_policies_with_speeds_beat_the_utilization_bound():
    for policy in (ECMP(), DChoice(2), DChoice(2, memory=1), Greedy()):
        w, draws = make_batch(policy, Pareto(1.5), seed=9)
        load = path_loads(policy.assign(w, draws, N, speeds=SPEEDS), w, N)
        lb = opt_lower_bound(w, N, SPEEDS)
        assert np.all((load / np.array(SPEEDS)).max(axis=1) >= lb - 1e-9)
