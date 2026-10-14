"""Placement policies: which path each flow goes on.

Every policy has two halves, kept apart on purpose:

    draw(rng, m, n)       the random numbers it needs for ONE trial (or None)
    assign(w, draws, n)   paths for a whole batch of trials, deterministically

Keeping the randomness out of `assign` buys three things:
  * per-trial seeds, so results never depend on how trials are batched;
  * differential testing: a slow, obviously-correct reference implementation
    can be fed the same draws and must agree exactly (tests/reference.py);
  * common random numbers: the same draws can be replayed against different
    flow sets.

Shapes: `w` is (T, m), the flow weights of T trials, in arrival order.
`draws` is the per-trial draws stacked along a new first axis.
`assign` returns a (T, m) integer array of paths in 0..n-1.

To add a policy, give it a stable `name` plus these two methods, then add a
reference version and a comparison test.
"""
from __future__ import annotations

import numpy as np


def sample_distinct(rng: np.random.Generator, m: int, n: int, d: int) -> np.ndarray:
    """For each of m flows, an ordered sample of d distinct paths, uniformly at random.

    The k-th pick is drawn from the n - k paths not chosen yet, then shifted
    past the earlier picks in increasing order. Costs O(m d^2) and never
    builds an (m, n) array.
    """
    if not 1 <= d <= n:
        raise ValueError(f"cannot sample {d} distinct paths out of {n}")
    picks = np.empty((m, d), dtype=np.int64)
    for k in range(d):
        r = rng.integers(0, n - k, size=m)
        if k:
            for prev in np.sort(picks[:, :k], axis=1).T:       # earlier picks, ascending
                r += r >= prev
        picks[:, k] = r
    return picks


class ECMP:
    """ECMP with an ideal hash: every flow picks a path uniformly at random,
    independently of everything else and blind to load. This is the classical
    one-choice balls-into-bins process."""

    name = "ecmp"

    def draw(self, rng: np.random.Generator, m: int, n: int) -> np.ndarray:
        return rng.integers(0, n, size=m)

    def assign(self, w: np.ndarray, draws: np.ndarray, n: int) -> np.ndarray:
        return np.asarray(draws, dtype=np.int64)

    def __repr__(self):
        return "ECMP()"


class DChoice:
    """Power of d choices: sample d paths, send the flow to the least-loaded one.

    by="bytes"   compare the total weight already on each sampled path. Needs
                 link-utilization feedback (the kind of signal CONGA or DRILL
                 use), but not the size of the new flow.
    by="count"   compare the number of flows already on each path. Cheaper,
                 and blind to flow sizes.
    replace      True samples with replacement: the textbook model, and what
                 the proposal's n = 4 example assumes. False samples d distinct
                 paths, which is what a real switch would do.

    Ties go to the earliest sample. The samples are exchangeable, so that is a
    uniformly random choice among the tied paths. d = 1 is ECMP.
    """

    def __init__(self, d: int = 2, by: str = "bytes", replace: bool = True):
        if d < 1:
            raise ValueError("d must be at least 1")
        if by not in ("bytes", "count"):
            raise ValueError("by must be 'bytes' or 'count'")
        self.d, self.by, self.replace = d, by, replace
        self.name = f"d{d}-{by}" + ("" if replace else "-distinct")

    def draw(self, rng: np.random.Generator, m: int, n: int) -> np.ndarray:
        if self.replace:
            return rng.integers(0, n, size=(m, self.d))
        return sample_distinct(rng, m, n, self.d)

    def assign(self, w: np.ndarray, draws: np.ndarray, n: int) -> np.ndarray:
        T, m = w.shape
        rows = np.arange(T)
        load = np.zeros((T, n))
        inc = w if self.by == "bytes" else np.ones_like(w)
        paths = np.empty((T, m), dtype=np.int64)
        for j in range(m):                        # flows arrive one at a time...
            cand = draws[:, j, :]                 # ...but all T trials move together
            seen = np.take_along_axis(load, cand, axis=1)
            pick = cand[rows, np.argmin(seen, axis=1)]   # argmin = first minimum
            paths[:, j] = pick
            load[rows, pick] += inc[:, j]
        return paths

    def __repr__(self):
        return f"DChoice(d={self.d}, by={self.by!r}, replace={self.replace})"


class Greedy:
    """Online least-loaded: each flow joins the path carrying the least weight
    right now (d = n with full information). A strong online yardstick; no
    switch-local scheme sees this much. Ties go to the lowest-numbered path,
    which cannot change any symmetric metric."""

    name = "greedy"

    def draw(self, rng, m, n):
        return None

    def assign(self, w: np.ndarray, draws, n: int) -> np.ndarray:
        T, m = w.shape
        rows = np.arange(T)
        load = np.zeros((T, n))
        paths = np.empty((T, m), dtype=np.int64)
        for j in range(m):
            pick = np.argmin(load, axis=1)
            paths[:, j] = pick
            load[rows, pick] += w[:, j]
        return paths

    def __repr__(self):
        return "Greedy()"


class LPT:
    """Offline longest-processing-time first: sort flows largest first, then
    place greedily. It needs every flow's size up front, so no switch could
    run it; it is the near-optimal yardstick. Graham (1969): its max load is at
    most (4/3 - 1/(3n)) times the optimal max load."""

    name = "lpt"

    def draw(self, rng, m, n):
        return None

    def assign(self, w: np.ndarray, draws, n: int) -> np.ndarray:
        order = np.argsort(-w, axis=1, kind="stable")
        placed = Greedy().assign(np.take_along_axis(w, order, axis=1), None, n)
        paths = np.empty_like(placed)
        np.put_along_axis(paths, order, placed, axis=1)   # back to arrival order
        return paths

    def __repr__(self):
        return "LPT()"
