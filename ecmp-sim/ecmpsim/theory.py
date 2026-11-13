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


# --------------------------------------------------------------------------
# Weighted flows: CONGA's imbalance theorem and what replaces it
# --------------------------------------------------------------------------

def conga_imbalance_bound(m: float, n: int, cv2: float) -> float:
    """CONGA (Alizadeh et al., SIGCOMM 2014), Theorem 2, read in this model.

    Flows of random size S arrive as a Poisson process of rate lambda, each to
    a uniformly random one of n links. If E[exp(theta S)] is finite near 0,
    then for t large enough

        E[chi(t)] <= 1 / sqrt(lambda_e t) + o(1 / sqrt(t)),
        lambda_e = lambda / (8 n ln n (1 + (sigma_S / E[S])^2)),

    where chi(t) = (busiest - least loaded link) / (expected load per link)
    is metrics' `imbalance`. Reading lambda t as the number of flows m gives
    sqrt(8 n ln n (1 + CV^2) / m).

    Caveats to state when quoting it: it is asymptotic in t (the o-term is
    unspecified, so "the bound holds at m = 4096" is an observation, not a
    guarantee); the theorem has Poisson(m) flows rather than exactly m; and
    the light-tail condition fails for lognormal and Pareto. The formula
    still needs only CV^2 < inf, which lognormal has and Pareto(alpha <= 2)
    does not; then it returns inf.
    """
    if math.isinf(cv2):
        return math.inf
    return math.sqrt(8 * n * math.log(n) * (1 + cv2) / m)


def imbalance_decay_exponent(alpha: float | None = None) -> float:
    """How fast the imbalance shrinks with the number of flows: E[chi] ~ m^exponent.

    With finite variance the per-link loads obey the central limit theorem,
    so fluctuations grow like sqrt(m) against a mean of order m: exponent
    -1/2, the 1/sqrt(t) in CONGA's theorem. With a Pareto tail of index
    1 < alpha < 2 the variance is infinite and the generalized CLT applies
    instead: sums fluctuate on the scale m^(1/alpha), so the exponent is
    -(1 - 1/alpha). For alpha = 1.5 that is -1/3, i.e. 16x more flows cut
    the imbalance by 16^(1/3) = 2.5x, not 4x. alpha = None means finite
    variance. This is a scaling heuristic (no constant), to fit slopes against.
    """
    if alpha is None or alpha >= 2:
        return -0.5
    if alpha <= 1:
        raise ValueError("alpha must be > 1 (finite mean)")
    return -(1 - 1 / alpha)


def fitted_exponent(m, imbalance) -> float:
    """Least-squares slope of log(imbalance) against log(m)."""
    return float(np.polyfit(np.log(np.asarray(m, float)), np.log(np.asarray(imbalance, float)), 1)[0])


# --------------------------------------------------------------------------
# Unequal path speeds: DRILL's Theorem 1, in a static snapshot
# --------------------------------------------------------------------------

def dchoice_util_lower_bound(speeds, d: int, replace: bool = False) -> float:
    """Lower bound on E[busiest utilization] / ideal for ANY rule that picks
    among d random samples with no memory (DRILL(d, 0), Ghorbani et al.,
    SIGCOMM 2017, Theorem 1), when paths have unequal speeds.

    The argument is the theorem's counting step. Take A = the a fastest
    paths. A flow can only land in A if one of its samples does, which
    happens with probability 1 - q_a, where q_a = C(n - a, d) / C(n, d)
    (distinct samples) or (1 - a/n)^d (with replacement). So in expectation
    at least a fraction q_a of the bytes lands outside A, on paths with total
    speed S - S_A, and the busiest of those runs at utilization at least
    q_a * total / (S - S_A). Dividing by the ideal total / S and taking the
    best a gives the bound. It ignores fluctuations of the total, so it is a
    statement about means. It is 1 or less when d samples are enough; memory
    (DRILL(d, m >= 1)) escapes it, which is DRILL's Theorem 2.
    """
    s = np.sort(np.asarray(speeds, dtype=float))[::-1]
    n, S = len(s), s.sum()
    best = 1.0
    for a in range(1, n):
        if replace:
            q = (1 - a / n) ** d
        else:
            q = math.comb(n - a, d) / math.comb(n, d) if n - a >= d else 0.0
        best = max(best, q * S / (S - s[:a].sum()))
    return best


# --------------------------------------------------------------------------
# Time: queues that drain (dynamic.py)
# --------------------------------------------------------------------------

def mg1_wait(rho: float, cv2: float, speed: float = 1.0) -> float:
    """Mean time a flow waits in an M/G/1 queue (Pollaczek-Khinchine):

        E[W] = rho (1 + CV^2) E[service] / (2 (1 - rho)),   E[service] = 1 / speed.

    With ECMP and Poisson arrivals every path is exactly this queue (a
    Poisson stream split at random is Poisson), at rho = the offered load
    when speeds are equal. Infinite for rho >= 1 or infinite variance: with
    Pareto(alpha <= 2) sizes the mean wait is infinite at ANY load.
    """
    if rho >= 1 or math.isinf(cv2):
        return math.inf
    return rho * (1 + cv2) / (2 * (1 - rho) * speed)


def mg1_idle(rho: float) -> float:
    """Fraction of time a single-server queue at load rho < 1 is empty."""
    return max(1 - rho, 0.0)


def dchoice_pileup(speeds, d: int, load: float, replace: bool = False) -> float:
    """Lower bound on the share of the offered traffic that piles up for ever
    under ANY memoryless rule choosing among d sampled paths, at offered load
    `load` (DRILL Theorem 1, run over time; metric `growth` in dynamic.py).

    The counting step of dchoice_util_lower_bound: a fraction q_a of the
    flows samples none of the a fastest paths, and the remaining paths can
    only carry (S - S_A) / (load * S) of the offered traffic. 0 means the
    argument finds no overload. d = 1 is ECMP, for which the bound is exact
    with a = the set of paths that are not overloaded.
    """
    s = np.sort(np.asarray(speeds, dtype=float))[::-1]
    n, S = len(s), s.sum()
    worst = 0.0
    for a in range(0, n):
        if replace:
            q = (1 - a / n) ** d
        else:
            q = math.comb(n - a, d) / math.comb(n, d) if n - a >= d else 0.0
        worst = max(worst, q - (S - s[:a].sum()) / (load * S))
    return worst


def dchoice_stable_load(speeds, d: int, replace: bool = False) -> float:
    """The largest offered load at which a memoryless d-choice rule can still
    be stable according to the counting bound: 1 / dchoice_util_lower_bound."""
    return 1.0 / dchoice_util_lower_bound(speeds, d, replace)
