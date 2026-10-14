"""Asymptotic and heuristic predictions to hold the simulations up against.

Logs are natural. Most of these are leading-order statements: the o(1) and
O(1) terms they hide can be large at n = 4..64, and measuring how large is the
point of the first core experiment. Double-check each formula against the
original paper before quoting it in the report.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz     # NumPy 2 renamed trapz


# --------------------------------------------------------------------------
# One choice
# --------------------------------------------------------------------------

def raab_steger(m: int, n: int, alpha: float = 1.0) -> float:
    """Raab & Steger (1998), regime n log n << m <= n polylog(n).

    The max load exceeds m/n + alpha * sqrt(2 (m/n) ln n) with probability o(1)
    when alpha > 1 and with probability 1 - o(1) when alpha < 1. So alpha is a
    threshold multiplier (not a confidence level) and alpha = 1 gives the
    leading-order prediction. It describes the typical max load, which for
    these distributions sits close to the mean.
    """
    return m / n + alpha * math.sqrt(2 * (m / n) * math.log(n))


def m_equals_n_leading(n: int) -> float:
    """The textbook leading term for m = n: ln n / ln ln n (needs n >= 3)."""
    return math.log(n) / math.log(math.log(n))


def inverse_gamma(y: float) -> float:
    """The x >= 2 with Gamma(x) = y (y >= 1), by bisection on lgamma."""
    target, lo, hi = math.log(y), 2.0, 4.0
    while math.lgamma(hi) < target:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if math.lgamma(mid) < target else (lo, mid)
    return (lo + hi) / 2


def gonnet(n: int) -> float:
    """Gonnet (1981): for m = n, E[max load] = Gamma^{-1}(n) - 3/2 + o(1)."""
    return inverse_gamma(n) - 1.5


@lru_cache(maxsize=None)
def expected_max_std_normal(n: int) -> float:
    """E[max of n iid standard normals], by numerical integration."""
    x = np.linspace(-12.0, 12.0, 96001)
    erf = np.vectorize(math.erf)
    F = (0.5 * (1 + erf(x / math.sqrt(2)))) ** n           # CDF of the maximum
    pos, neg = x >= 0, x <= 0
    return float(_trapezoid(1 - F[pos], x[pos]) - _trapezoid(F[neg], x[neg]))


def gaussian_max(m: int, n: int) -> float:
    """Normal approximation to E[max load], one choice.

    Each path's load is Binomial(m, 1/n), roughly N(m/n, (m/n)(1 - 1/n));
    pretend the n loads are independent and take the expected maximum. A
    finite-n heuristic rather than a theorem. Comparing it with raab_steger
    shows how much of that formula's error comes from sqrt(2 ln n) being a
    slow approximation to the expected maximum of n normals.
    """
    return m / n + math.sqrt((m / n) * (1 - 1 / n)) * expected_max_std_normal(n)


# --------------------------------------------------------------------------
# d choices
# --------------------------------------------------------------------------

def heavily_loaded_dchoice(m: int, n: int, d: int) -> float:
    """Berenbrink, Czumaj, Steger & Vöcking (STOC 2000; SIAM J. Comput. 2006):
    with d >= 2 choices the max load is m/n + ln ln n / ln d + O(1) w.h.p., for
    every m. Note the gap does not grow with m. The O(1) is unspecified, so
    this gives the shape of the prediction, not its value.
    """
    return m / n + math.log(math.log(n)) / math.log(d)


@lru_cache(maxsize=None)
def _fluid_tails_cached(t: float, d: int, imax: int, h: float) -> tuple:
    s = np.zeros(imax + 1)
    s[0] = 1.0

    def f(s):
        sd = s ** d
        ds = np.zeros_like(s)
        ds[1:] = sd[:-1] - sd[1:]
        return ds

    steps = max(1, math.ceil(t / h))
    h = t / steps
    for _ in range(steps):
        k1 = f(s)
        k2 = f(s + h / 2 * k1)
        k3 = f(s + h / 2 * k2)
        k4 = f(s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return tuple(s)


def fluid_tails(t: float, d: int, imax: int | None = None, h: float = 0.005) -> np.ndarray:
    """Fluid limit for d-choice (Mitzenmacher 1996/2001; d = 1 is plain hashing).

    s[i] is the fraction of bins with load >= i after t*n balls, as n -> inf:

        ds_i/dt = s_{i-1}^d - s_i^d,   s_0 = 1,   s_i(0) = 0 for i >= 1,

    integrated with classical RK4. Checks: s_1 = 1 - e^{-t} for d = 1, and
    s_1 = tanh(t) for d = 2 (the proposal's Example 1).
    """
    if imax is None:
        imax = int(t + 12 * math.sqrt(t + 1) + 20)
    return np.array(_fluid_tails_cached(float(t), int(d), int(imax), float(h)))


def fluid_idle_frac(t: float, d: int) -> float:
    """Predicted fraction of idle paths at t = m/n flows per path, n -> infinity."""
    return float(1 - fluid_tails(t, d)[1])


def fluid_expected_max(m: int, n: int, d: int) -> float:
    """Turn the fluid limit into a finite-n prediction of E[max load].

    Treat the number of bins with load >= i as Poisson with mean n * s_i, so
    P(max >= i) ~ 1 - exp(-n s_i), except that P(max >= i) = 1 exactly for
    i <= ceil(m/n) (pigeonhole). Then E[max] = sum_{i >= 1} P(max >= i). This
    is a heuristic (it ignores finite-n corrections and correlations between
    bins), which is exactly what makes it worth testing against simulation.
    """
    s = fluid_tails(m / n, d)
    i = np.arange(len(s))
    p_at_least = np.where(i <= -(-m // n), 1.0, -np.expm1(-n * s))
    return float(p_at_least[1:].sum())
