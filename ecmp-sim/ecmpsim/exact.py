"""Exact answers: the ground truth the simulator is checked against.

    max_load_cdf / max_load_pmf / expected_max_load
        one choice (ECMP), unit flows, any m and n
    expected_idle_frac
        one choice: (1 - 1/n)^m
    dchoice_exact + profile_stats
        d choices, unit flows, small m and n, by tracking every load profile
    brute_force
        enumerate every random outcome; tiny cases only, used by the tests
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations, product

import numpy as np


# --------------------------------------------------------------------------
# One choice: exact max-load distribution by Poissonization
# --------------------------------------------------------------------------

def _fft_conv(a: np.ndarray, b: np.ndarray, size: int) -> np.ndarray:
    """Linear convolution of two non-negative sequences, truncated to `size` terms."""
    full = len(a) + len(b) - 1
    nfft = 1 << (full - 1).bit_length() # this takes the next power of 2 >= full, faster than full
    out = np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:min(full, size)]
    return np.clip(out, 0.0, None)          # FFT rounding can dip a hair below 0


def _conv_power(q: np.ndarray, times: int, size: int) -> np.ndarray:
    """Distribution of the sum of `times` iid draws from pmf q, truncated to `size` terms."""
    result, base = np.array([1.0]), q[:size].copy()
    while times:
        if times & 1:
            result = _fft_conv(result, base, size)
        times >>= 1
        if times:
            base = _fft_conv(base, base, size)
    return result


def max_load_cdf(m: int, n: int, k: int) -> float:
    """P(max load <= k) when m unit balls go into n bins uniformly at random.

    Exact up to floating-point rounding. Independent Poisson(m/n) loads,
    conditioned on summing to m, are exactly multinomial, so

        P(max <= k) = P(X <= k)^n * P(Y_1 + ... + Y_n = m) / P(Poisson(m) = m)

    with X ~ Poisson(m/n) and each Y_i distributed as X given X <= k. The
    middle factor is an n-fold convolution of a short, all-positive vector.
    """
    if m == 0 or k >= m:
        return 1.0 if k >= 0 else 0.0
    if k * n < m:
        return 0.0
    lam = m / n
    j = np.arange(k + 1)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, k + 1)))])
    p = np.exp(j * math.log(lam) - lam - log_fact)        # Poisson pmf on 0..k
    p_le_k = p.sum()
    total = _conv_power(p / p_le_k, n, m + 1)
    if len(total) <= m or total[m] <= 0:
        return 0.0
    log_poisson_m_at_m = m * math.log(m) - m - math.lgamma(m + 1)
    return min(1.0, math.exp(n * math.log(p_le_k) + math.log(total[m]) - log_poisson_m_at_m))


def max_load_pmf(m: int, n: int) -> np.ndarray:
    """Array p with p[k] = P(max load = k) for k = 0..m (one choice, unit flows)."""
    cdf = np.ones(m + 1)
    for k in range(m + 1):
        cdf[k] = max_load_cdf(m, n, k)
        # Stop once the remaining tail is negligible. Rounding keeps the computed
        # CDF about 1e-13 short of 1, so waiting for exactly 1.0 would never end.
        if cdf[k] > 1 - 1e-11 or (k and cdf[k] > 1 - 1e-8 and cdf[k] - cdf[k - 1] < 1e-15):
            cdf[k:] = 1.0
            break
    cdf = np.maximum.accumulate(cdf)
    return np.diff(cdf, prepend=0.0)


def expected_max_load(m: int, n: int) -> float:
    """E[max load] for m unit flows hashed uniformly onto n paths."""
    pmf = max_load_pmf(m, n)
    return float(np.dot(np.arange(len(pmf)), pmf))


def expected_idle_frac(m: int, n: int) -> float:
    """Expected fraction of paths with no flow, one choice: (1 - 1/n)^m."""
    return (1 - 1 / n) ** m


# --------------------------------------------------------------------------
# d choices: exact Markov chain on sorted load profiles
# --------------------------------------------------------------------------

def _landing_probs(profile, n, d, replace, one):
    """P(the next ball lands on a bin whose load is v), for each load v present.

    The ball lands on load v exactly when the smallest sampled load is v:
    P(min >= v) - P(min > v).
    """
    out = []
    for v in sorted(set(profile)):
        ge = sum(1 for x in profile if x >= v)
        gt = sum(1 for x in profile if x > v)
        if replace:
            pr = (one * ge / n) ** d - (one * gt / n) ** d
        else:
            pr = one * (math.comb(ge, d) - math.comb(gt, d)) / math.comb(n, d)
        if pr:
            out.append((v, pr))
    return out


def dchoice_exact(m: int, n: int, d: int, replace: bool = True, exact: bool = False) -> dict:
    """Exact distribution of the sorted load profile after m unit balls with d choices.

    Returns {profile (descending tuple): probability}. Bins are interchangeable
    under uniform sampling, so the sorted profile is a Markov chain and we push
    probability through it ball by ball. With exact=True all arithmetic uses
    Fractions. The number of profiles grows like the number of partitions of
    m, so keep m small (m <= 40 with n <= 8 is instant). d = 1 is ECMP.
    """
    if not replace and d > n:
        raise ValueError("cannot sample more distinct bins than exist")
    one = Fraction(1) if exact else 1.0
    dist = {tuple([0] * n): one}
    for _ in range(m):
        nxt = defaultdict(int)
        for profile, p in dist.items():
            for v, pr in _landing_probs(profile, n, d, replace, one):
                new = list(profile)
                new[new.index(v)] += 1          # first bin with load v; stays sorted
                nxt[tuple(new)] += p * pr
        dist = dict(nxt)
    return dist


def profile_stats(dist: dict) -> dict:
    """Expected max load, expected idle fraction and P(max = k) from a profile distribution."""
    n = len(next(iter(dist)))
    e_max = sum(p * prof[0] for prof, p in dist.items())
    e_idle = sum(p * sum(1 for x in prof if x == 0) / n for prof, p in dist.items())
    max_pmf = defaultdict(int)
    for prof, p in dist.items():
        max_pmf[prof[0]] += p
    return {"max_load": e_max, "idle_frac": e_idle, "max_pmf": dict(sorted(max_pmf.items()))}


def brute_force(m: int, n: int, d: int = 1, replace: bool = True) -> dict:
    """Enumerate every equally likely sequence of samples (tiny m and n only).

    Each ball draws d samples and goes to the first sampled bin with the
    smallest load, exactly like DChoice. Returns {sorted profile: Fraction}.
    """
    per_ball = list(product(range(n), repeat=d)) if replace else list(permutations(range(n), d))
    counts = Counter()
    for seq in product(per_ball, repeat=m):
        load = [0] * n
        for cand in seq:
            load[min(cand, key=lambda b: load[b])] += 1     # min() keeps the first minimum
        counts[tuple(sorted(load, reverse=True))] += 1
    total = len(per_ball) ** m
    return {prof: Fraction(c, total) for prof, c in counts.items()}
