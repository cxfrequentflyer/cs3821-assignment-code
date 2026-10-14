"""The exact formulas agree with brute force, with each other, and with the proposal."""
import math
from fractions import Fraction

import numpy as np
import pytest

from ecmpsim import exact, theory


def brute_force_max_cdf(m, n):
    pmf = exact.profile_stats(exact.brute_force(m, n, d=1))["max_pmf"]
    return [float(sum(p for k, p in pmf.items() if k <= kk)) for kk in range(m + 1)]


@pytest.mark.parametrize("m,n", [(4, 4), (5, 3), (6, 2), (3, 5), (7, 4)])
def test_poissonized_cdf_matches_brute_force(m, n):
    expected = brute_force_max_cdf(m, n)
    got = [exact.max_load_cdf(m, n, k) for k in range(m + 1)]
    assert np.allclose(got, expected, atol=1e-12)


def counting_cdf(m, n, k):
    """P(max <= k) by counting assignments with big integers: an independent method."""
    ways = [1] + [0] * m                 # ways[s]: put s labeled balls in the bins so far
    for _ in range(n):
        ways = [sum(math.comb(s, j) * ways[s - j] for j in range(min(k, s) + 1))
                for s in range(m + 1)]
    return Fraction(ways[m], n ** m)


@pytest.mark.parametrize("m,n", [(30, 6), (40, 10), (25, 25), (60, 4)])
def test_poissonized_cdf_matches_counting(m, n):
    for k in range(-(-m // n), m + 1):
        want = float(counting_cdf(m, n, k))
        assert exact.max_load_cdf(m, n, k) == pytest.approx(want, rel=1e-10, abs=1e-14)
        if want > 1 - 1e-15:
            break


def test_expected_max_small_cases():
    assert exact.expected_max_load(4, 4) == pytest.approx(17 / 8)
    assert exact.expected_max_load(2, 2) == pytest.approx(1.5)
    assert exact.expected_max_load(1, 9) == pytest.approx(1.0)
    assert exact.expected_max_load(7, 1) == pytest.approx(7.0)


@pytest.mark.parametrize("m,n,d,replace", [(4, 4, 1, True), (4, 4, 2, True), (4, 4, 2, False),
                                           (5, 3, 2, True), (4, 3, 3, False)])
def test_markov_chain_matches_brute_force(m, n, d, replace):
    assert exact.dchoice_exact(m, n, d, replace, exact=True) == exact.brute_force(m, n, d, replace)


def test_proposal_example_1():
    """Every number in the proposal's worked example (n = m = 4)."""
    assert 1 - exact.expected_idle_frac(4, 4) == pytest.approx(175 / 256)    # 68.4%
    assert exact.expected_max_load(4, 4) == pytest.approx(17 / 8)            # 2.13
    two = exact.profile_stats(exact.dchoice_exact(4, 4, 2, replace=True, exact=True))
    assert 1 - two["idle_frac"] == Fraction(13231, 16384)                    # 80.8%
    assert two["max_load"] == Fraction(439, 256)                             # 1.71
    assert 1 - theory.fluid_idle_frac(1.0, 2) == pytest.approx(math.tanh(1), abs=1e-9)  # 76.2%


@pytest.mark.parametrize("t", [0.7, 3.0])
def test_fluid_limit_one_choice_is_poisson(t):
    s = theory.fluid_tails(t, 1)
    for i in range(1, 6):
        poisson_tail = 1 - sum(math.exp(-t) * t ** j / math.factorial(j) for j in range(i))
        assert s[i] == pytest.approx(poisson_tail, abs=1e-8)


@pytest.mark.parametrize("t", [0.5, 1.0, 3.0])
def test_fluid_limit_two_choices_is_tanh(t):
    assert theory.fluid_tails(t, 2)[1] == pytest.approx(math.tanh(t), abs=1e-8)


def test_fluid_heuristic_matches_its_formula_for_one_choice():
    # How *accurate* the heuristic is, is an experimental question (exp01).
    # Here we only check the code computes what the docstring says.
    m, n = 96, 8
    t = m / n

    def poisson_tail(i):
        return 1 - sum(math.exp(-t) * t ** j / math.factorial(j) for j in range(i))

    want = sum(1.0 if i <= math.ceil(t) else -math.expm1(-n * poisson_tail(i))
               for i in range(1, 80))
    assert theory.fluid_expected_max(m, n, 1) == pytest.approx(want, abs=1e-7)


def test_exact_expected_max_is_fast_and_sane_for_big_groups():
    e = exact.expected_max_load(4096, 64)
    assert 64 < e < 64 + 4 * math.sqrt(64 * 2 * math.log(64))


def test_fluid_heuristic_respects_pigeonhole():
    for m, n in [(4, 4), (256, 4), (10, 3)]:
        assert theory.fluid_expected_max(m, n, 2) >= -(-m // n)


def test_expected_max_of_normals():
    assert theory.expected_max_std_normal(1) == pytest.approx(0.0, abs=1e-7)
    assert theory.expected_max_std_normal(2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-7)
    assert theory.expected_max_std_normal(3) == pytest.approx(1.5 / math.sqrt(math.pi), abs=1e-7)


def test_inverse_gamma():
    assert theory.inverse_gamma(24) == pytest.approx(5.0, abs=1e-9)
    assert theory.gonnet(2) == pytest.approx(1.5, abs=1e-9)
