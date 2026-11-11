"""The simulator's averages land on the exact answers, and the plumbing is reproducible."""
import math

import numpy as np
import pandas as pd
import pytest

from ecmpsim import (ECMP, Bimodal, DChoice, Empirical, Exponential, Greedy, LogNormal, Pareto,
                     Spray, Unit, exact, run, theory)


def within(sample, target, sigmas=4.0):
    sample = np.asarray(sample, dtype=float)
    se = sample.std(ddof=1) / math.sqrt(len(sample))
    return abs(sample.mean() - float(target)) < sigmas * se + 1e-12


def test_ecmp_mean_max_load_matches_exact():
    df = run(8, 32, Unit(), [ECMP()], trials=20_000, seed=1)
    assert within(df.max_load, exact.expected_max_load(32, 8))
    assert within(df.idle_frac, exact.expected_idle_frac(32, 8))


@pytest.mark.parametrize("replace", [True, False])
def test_two_choice_small_matches_exact(replace):
    st = exact.profile_stats(exact.dchoice_exact(4, 4, 2, replace=replace, exact=True))
    df = run(4, 4, Unit(), [DChoice(2, replace=replace)], trials=20_000, seed=2)
    assert within(df.max_load, st["max_load"])
    assert within(df.idle_frac, st["idle_frac"])


def test_two_choice_large_n_matches_fluid_limit():
    df = run(1000, 1000, Unit(), [DChoice(2)], trials=40, seed=3)
    assert abs((1 - df.idle_frac).mean() - math.tanh(1)) < 0.005


@pytest.mark.parametrize("flows", [Unit(), Exponential(), LogNormal(1.0), Bimodal(0.05, 100)],
                         ids=lambda f: f.label)
def test_flow_sizes_have_mean_one(flows):
    x = flows.sample(np.random.default_rng(0), 1_000_000)
    assert within(x, 1.0, sigmas=5)


def test_pareto_tail():
    f = Pareto(1.5)
    x = f.sample(np.random.default_rng(0), 1_000_000)
    assert x.min() >= f.x_min
    for q in (0.5, 1.0, 3.0, 10.0):
        want = min(1.0, (f.x_min / q) ** f.alpha)
        assert abs((x > q).mean() - want) < 5 * math.sqrt(want * (1 - want) / len(x)) + 1e-9


def test_empirical_cdf_and_mean():
    f = Empirical(sizes=(1, 10, 100), cdf=(0.5, 0.9, 1.0))
    assert f.raw_mean == pytest.approx(0.5 * 1 + 0.4 * 5.5 + 0.1 * 55)
    x = f.sample(np.random.default_rng(0), 1_000_000) * f.raw_mean
    assert (x <= 1).mean() == pytest.approx(0.5, abs=0.003)
    assert (x <= 10).mean() == pytest.approx(0.9, abs=0.003)
    assert within(x / f.raw_mean, 1.0, sigmas=5)


def test_empirical_from_csv(tmp_path):
    path = tmp_path / "websearch_like.txt"
    path.write_text("# size  unused  cdf(percent)\nsize x cdf\n6 1 0\n13 1 50\n\n19 1 100\n")
    f = Empirical.from_csv(path)
    assert f.sizes == (6.0, 13.0, 19.0)
    assert f.cdf == (0.0, 0.5, 1.0)
    assert f.label == "websearch_like"


def test_batch_size_does_not_change_results():
    policies = [ECMP(), DChoice(2), DChoice(2, by="count", replace=False)]
    a = run(6, 30, Pareto(1.5), policies, trials=23, seed=4, batch_size=3)
    b = run(6, 30, Pareto(1.5), policies, trials=23, seed=4, batch_size=50)
    pd.testing.assert_frame_equal(a, b)


def test_adding_a_policy_changes_nothing_else():
    alone = run(6, 30, Exponential(), [ECMP()], trials=50, seed=5)
    mixed = run(6, 30, Exponential(), [DChoice(3), ECMP()], trials=50, seed=5)
    pd.testing.assert_frame_equal(alone, mixed[mixed.policy == "ecmp"].reset_index(drop=True))


def test_example_1_throughput_is_fraction_of_busy_paths():
    df = run(4, 4, Unit(), [ECMP()], trials=500, seed=6, capacity=1.0)
    assert np.allclose(df.throughput, 1 - df.idle_frac)


# --- Flow-size CV^2 and CONGA's imbalance theorem ------------------------------------

@pytest.mark.parametrize("flows", [Unit(), Exponential(), LogNormal(1.0), Bimodal(0.05, 100),
                                   Empirical((1, 2, 10), (0.2, 0.5, 1.0))], ids=lambda f: f.label)
def test_cv2_matches_samples(flows):
    x = flows.sample(np.random.default_rng(1), 2_000_000)
    assert x.var() / x.mean() ** 2 == pytest.approx(flows.cv2, rel=0.02, abs=1e-12)


def test_pareto_cv2():
    assert math.isinf(Pareto(1.5).cv2) and math.isinf(Pareto(2.0).cv2)
    assert Pareto(3.0).cv2 == pytest.approx(1 / 3)
    assert math.isinf(theory.conga_imbalance_bound(4096, 16, Pareto(1.5).cv2))


def test_imbalance_metric_is_max_minus_min_over_mean():
    from ecmpsim import path_loads, trial_metrics
    rng = np.random.default_rng(7)
    w, paths = rng.exponential(size=(20, 40)), rng.integers(0, 5, size=(20, 40))
    load = path_loads(paths, w, 5)
    want = (load.max(axis=1) - load.min(axis=1)) / (w.sum(axis=1) / 5)
    assert np.allclose(trial_metrics(paths, w, 5)["imbalance"], want)


@pytest.mark.parametrize("flows", [Unit(), Exponential(), LogNormal(1.5)], ids=lambda f: f.label)
def test_conga_bound_holds_for_finite_variance(flows):
    n, m = 16, 4096
    df = run(n, m, flows, [ECMP()], trials=300, seed=8)
    assert df.imbalance.mean() < theory.conga_imbalance_bound(m, n, flows.cv2)


@pytest.mark.parametrize("flows,alpha", [(Exponential(), None), (Pareto(1.5), 1.5)],
                         ids=["exp", "pareto1.5"])
def test_imbalance_decays_at_the_predicted_rate(flows, alpha):
    ms = [256, 1024, 4096, 16384]
    med = [run(16, m, flows, [ECMP()], trials=600, seed=9).imbalance.median() for m in ms]
    assert theory.fitted_exponent(ms, med) == pytest.approx(
        theory.imbalance_decay_exponent(alpha), abs=0.06)


# --- Unequal speeds: DRILL's Theorem 1 and 2 in a snapshot ----------------------------

SPEEDS = np.array([10.0] + [1.0] * 9)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_memoryless_dchoice_cannot_beat_the_counting_bound(d):
    df = run(10, 400, Exponential(), [DChoice(d, replace=False)], trials=300, seed=10, speeds=SPEEDS)
    bound = theory.dchoice_util_lower_bound(SPEEDS, d)
    assert bound > 1.4
    assert df.max_over_mean.mean() > 0.98 * bound


def test_one_slot_of_memory_escapes_the_bound():
    df = run(10, 400, Exponential(), [DChoice(1, memory=1), DChoice(2, replace=False)],
             trials=300, seed=10, speeds=SPEEDS)
    mean = df.groupby("policy").max_over_mean.mean()
    assert mean["d1-bytes-mem1"] < 1.3 < theory.dchoice_util_lower_bound(SPEEDS, 2) < mean["d2-bytes-distinct"]


def test_counting_bound_simple_cases():
    assert theory.dchoice_util_lower_bound(np.ones(8), 2) == 1.0          # equal speeds: no bound
    assert theory.dchoice_util_lower_bound(SPEEDS, 10) == 1.0             # d = n sees everything
    assert theory.dchoice_util_lower_bound(SPEEDS, 1) == pytest.approx(0.9 * 19 / 9)
    assert theory.dchoice_util_lower_bound(SPEEDS, 2, replace=True) == pytest.approx(0.81 * 19 / 9)


def test_equal_speeds_change_nothing():
    pols = [ECMP(), DChoice(2), Greedy()]
    a = run(6, 30, Pareto(1.5), pols, trials=30, seed=11)
    b = run(6, 30, Pareto(1.5), pols, trials=30, seed=11, speeds=np.ones(6))
    pd.testing.assert_frame_equal(a, b)


# --- Chunking --------------------------------------------------------------------------

def test_chunked_runs_do_not_depend_on_batch_size():
    pols = [ECMP(), Spray(), DChoice(2, by="count", replace=False), Greedy(engines=3)]
    a = run(6, 30, Pareto(1.5), pols, trials=17, seed=12, chunk=0.3, batch_size=4)
    b = run(6, 30, Pareto(1.5), pols, trials=17, seed=12, chunk=0.3, batch_size=50)
    pd.testing.assert_frame_equal(a, b)


def test_chunking_keeps_the_flows_and_masks_padding():
    plain = run(8, 40, Pareto(1.5), [ECMP()], trials=50, seed=13)
    chunked = run(8, 40, Pareto(1.5), [ECMP()], trials=50, seed=13, chunk=0.5)
    # same flows, so the same total, hence the same mean load per path
    assert np.allclose(plain.max_load - plain.gap, chunked.max_load - chunked.gap)
    assert (chunked.chunks >= 40).all()
    assert (chunked.idle_frac >= 0).all() and (chunked.max_count <= chunked.chunks).all()


def test_smaller_chunks_balance_better():
    out = [run(16, 64, Pareto(1.5), [ECMP(), Spray()], trials=300, seed=14, chunk=c)
           .groupby("policy").imbalance.median() for c in (2.0, 0.5, 0.1)]
    for pol in ("ecmp", "spray"):
        assert out[0][pol] > out[1][pol] > out[2][pol]
    assert out[2]["spray"] < out[2]["ecmp"]          # round robin beats random at fine grain
