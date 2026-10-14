"""The simulator's averages land on the exact answers, and the plumbing is reproducible."""
import math

import numpy as np
import pandas as pd
import pytest

from ecmpsim import (ECMP, Bimodal, DChoice, Empirical, Exponential, LogNormal, Pareto, Unit,
                     exact, run, theory)


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
