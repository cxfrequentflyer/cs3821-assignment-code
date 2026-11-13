"""Time: the draining-queue model against a slow reference, the static model
it must reduce to, and exact queueing theory."""
import numpy as np
import pytest

import reference as ref
from ecmpsim import (ECMP, LPT, DChoice, Exponential, Greedy, HashECMP, Pareto, RoundRobin, Unit,
                     queue_metrics, rng_for, run, run_dynamic, summarize_dynamic, theory)
from ecmpsim.dynamic import unit_gaps
from ecmpsim.runner import _stack

N, M, T = 5, 80, 4
SPEEDS = np.array([3.0, 1.0, 1.0, 0.5, 2.0])


def batch(policy, flows=Exponential(), rate=4.0, seed=3):
    w = np.stack([flows.sample(rng_for(seed, "flows", flows.label, M, t), M) for t in range(T)])
    gaps = np.stack([unit_gaps(rng_for(seed, "gaps", "poisson", M, t), M) for t in range(T)]) / rate
    draws = _stack([policy.draw(rng_for(seed, "policy", policy.seed_name, N, M, t), M, N)
                    for t in range(T)])
    return w, gaps, draws


@pytest.mark.parametrize("speeds", [None, SPEEDS], ids=["equal", "speeds"])
@pytest.mark.parametrize("policy", [
    DChoice(1), DChoice(2), DChoice(3, replace=False), DChoice(2, memory=1),
    DChoice(1, replace=False, memory=2), DChoice(2, engines=3), DChoice(2, memory=1, engines=4),
], ids=lambda p: p.name)
def test_fast_matches_reference_over_time(policy, speeds):
    w, gaps, draws = batch(policy)
    fast = policy.assign(w, draws, N, speeds=speeds, gaps=gaps)
    for t in range(T):
        expected = ref.queue_drill(list(w[t]), draws[t].tolist(), list(gaps[t]), N,
                                   memory=policy.memory, engines=policy.engines, speeds=speeds)
        assert fast[t].tolist() == expected


def test_greedy_over_time_matches_reference():
    w, gaps, _ = batch(DChoice(1))
    fast = Greedy().assign(w, None, N, speeds=SPEEDS, gaps=gaps)
    everything = [list(range(N))] * M
    for t in range(T):
        assert fast[t].tolist() == ref.queue_drill(list(w[t]), everything, list(gaps[t]), N,
                                                   speeds=SPEEDS)


@pytest.mark.parametrize("speeds", [None, SPEEDS], ids=["equal", "speeds"])
def test_waits_match_the_lindley_recursion(speeds):
    policy = DChoice(2)
    w, gaps, draws = batch(policy)
    paths = policy.assign(w, draws, N, speeds=speeds, gaps=gaps)
    met = queue_metrics(paths, w, gaps, N, speeds)
    s = np.ones(N) if speeds is None else speeds
    for t in range(T):
        waits = np.array(ref.queue_waits(paths[t].tolist(), list(w[t]), list(gaps[t]), N, speeds))
        assert met["wait_mean"][t] == pytest.approx(waits.mean())
        assert met["fct_mean"][t] == pytest.approx((waits + w[t] / s[paths[t]]).mean())
    # with a warm-up only the later flows count
    late = queue_metrics(paths, w, gaps, N, speeds, warmup=30)
    waits = np.array(ref.queue_waits(paths[0].tolist(), list(w[0]), list(gaps[0]), N, speeds))
    assert late["wait_mean"][0] == pytest.approx(waits[30:].mean())


def test_no_time_passing_is_the_static_model():
    for policy in (DChoice(2), DChoice(2, memory=1, engines=3), Greedy()):
        w, gaps, draws = batch(policy) if hasattr(policy, "d") else (*batch(DChoice(1))[:2], None)
        still = policy.assign(w, draws, N, gaps=np.zeros_like(gaps))
        assert (still == policy.assign(w, draws, N)).all()


def test_oblivious_policies_use_the_same_paths_as_the_static_run():
    # same flow sizes, same policy stream: so an ECMP run over time is the
    # static experiment with a clock attached
    pols = [ECMP(), RoundRobin(), HashECMP("crc16")]
    static = run(4, 400, Unit(), pols, trials=6, seed=8)
    for pol in pols:
        w = np.ones((6, 400))
        draws = _stack([pol.draw(rng_for(8, "policy", getattr(pol, "seed_name", pol.name), 4, 400, t),
                                 400, 4) for t in range(6)])
        counts = np.stack([np.bincount(p, minlength=4) for p in pol.assign(w, draws, 4)])
        assert (counts.max(axis=1) == static[static.policy == pol.name].max_load.values).all()
    df = run_dynamic(4, 0.5, Unit(), pols, arrivals=400, trials=6, seed=8)
    assert len(df) == 18 and set(df.policy) == {p.name for p in pols}


@pytest.mark.parametrize("flows, rho", [(Exponential(), 0.5), (Exponential(), 0.8), (Unit(), 0.7)],
                         ids=["M/M/1 0.5", "M/M/1 0.8", "M/D/1 0.7"])
def test_ecmp_is_n_independent_mg1_queues(flows, rho):
    df = run_dynamic(4, rho, flows, [ECMP()], arrivals=40_000, warmup=4_000, trials=40, seed=1)
    wait, idle = df.wait_mean, df.idle_frac
    exact = theory.mg1_wait(rho, flows.cv2)
    assert abs(wait.mean() - exact) < 4 * wait.std(ddof=1) / np.sqrt(len(wait)) + 0.01 * exact
    assert abs(idle.mean() - theory.mg1_idle(rho)) < 4 * idle.std(ddof=1) / np.sqrt(len(idle)) + 0.003
    assert abs(df.growth.mean()) < 0.01


def test_mg1_formulas():
    assert theory.mg1_wait(0.5, 1.0) == pytest.approx(1.0)          # M/M/1: rho / (1 - rho)
    assert theory.mg1_wait(0.5, 0.0) == pytest.approx(0.5)          # M/D/1: half of that
    assert theory.mg1_wait(0.5, 1.0, speed=2.0) == pytest.approx(0.5)
    assert theory.mg1_wait(1.0, 1.0) == np.inf
    assert theory.mg1_wait(0.5, Pareto(1.5).cv2) == np.inf


def test_more_information_means_less_waiting():
    pols = [ECMP(), DChoice(2, replace=False), Greedy()]
    df = run_dynamic(8, 0.9, Exponential(), pols, arrivals=8_000, trials=10, seed=2)
    wait = df.groupby("policy").wait_mean.mean()
    assert wait["greedy"] < wait["d2-bytes-distinct"] < 0.5 * wait["ecmp"]
    s = summarize_dynamic(df)
    assert list(s.policy) == [p.name for p in pols] and "wait_mean_ci95" in s


def test_drill_theorems_over_time():
    # one fast path (10x) and nine slow ones at 80% load: ECMP and DRILL(2, 0)
    # cannot find the fast path often enough and their queues grow for ever, at
    # the rate the counting argument predicts; one unit of memory is stable
    speeds = np.array([10.0] + [1.0] * 9)
    pols = [ECMP(), DChoice(2, replace=False), DChoice(1, replace=False, memory=1)]
    df = run_dynamic(10, 0.8, Exponential(), pols, arrivals=30_000, trials=10, seed=3,
                     speeds=speeds)
    growth = df.groupby("policy").growth.mean()
    assert growth["ecmp"] == pytest.approx(theory.dchoice_pileup(speeds, 1, 0.8), abs=0.02)
    assert growth["d2-bytes-distinct"] == pytest.approx(theory.dchoice_pileup(speeds, 2, 0.8),
                                                        abs=0.02)
    assert abs(growth["d1-bytes-distinct-mem1"]) < 0.005
    assert theory.dchoice_pileup(speeds, 1, 0.8) == pytest.approx(0.9 - 9 / (0.8 * 19))
    assert theory.dchoice_pileup(speeds, 2, 0.5) == 0.0
    assert theory.dchoice_stable_load(speeds, 2) == pytest.approx(9 / (0.8 * 19))
    assert theory.dchoice_stable_load(np.ones(8), 2) == 1.0


def test_results_do_not_depend_on_batching_or_on_other_policies():
    kw = dict(n=6, load=0.8, flows=Exponential(), arrivals=600, trials=7, seed=4)
    a = run_dynamic(policies=[ECMP(), DChoice(2)], **kw)
    b = run_dynamic(policies=[DChoice(2)], batch_size=2, **kw)
    assert np.allclose(a[a.policy == "d2-bytes"].wait_mean.values, b.wait_mean.values)


def test_constant_arrivals_and_bad_arguments():
    df = run_dynamic(4, 0.9, Unit(), [RoundRobin()], arrivals=400, trials=2, process="constant")
    assert (df.wait_mean == 0).all()                    # D/D/1 below capacity never queues
    for bad in ([LPT()], [DChoice(2, by="count")], [ECMP(), ECMP()]):
        with pytest.raises(ValueError):
            run_dynamic(4, 0.5, Unit(), bad, arrivals=100, trials=2)
    with pytest.raises(ValueError):
        run_dynamic(4, 0.5, Unit(), [ECMP()], arrivals=100, trials=2, process="bursty")
