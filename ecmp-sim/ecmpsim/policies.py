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
`draws` is the per-trial draws stacked along a new first axis (a tuple of
draws is stacked component by component). `assign` returns a (T, m) integer
array of paths in 0..n-1.

Extra context, passed by keyword only to the policies that use it:
    speeds      (n,) relative path capacities. Load-aware policies compare
                utilization, load / speed, instead of raw load.
    flow, rank  (T, m) for chunked runs (runner.run(chunk=...)): the flow each
                chunk belongs to and its position inside that flow. Only
                policies with `uses_flow = True` get them (Spray).

Parallel decisions (DRILL's forwarding engines). DChoice, Greedy and
RoundRobin take `engines=F`. Flow j is handled by engine j mod F, and the
flows are decided in rounds of F: everyone in a round reads the same snapshot
of the loads, taken at the start of the round, so they cannot see each
other's choices. That is the stale, simultaneous information that causes
DRILL's "synchronization effect" (herding). engines=1 is the classical,
fully up-to-date model.

To add a policy, give it a stable `name` plus these two methods, then add a
reference version and a comparison test. A policy may also define
`seed_name`, used instead of `name` to key its random stream, when two
variants should share draws (e.g. the same DChoice with different `engines`).
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

    def assign(self, w: np.ndarray, draws: np.ndarray, n: int, **_) -> np.ndarray:
        return np.asarray(draws, dtype=np.int64)

    def __repr__(self):
        return "ECMP()"


def _place(w, cand, n, *, inc, speeds=None, engines=1, memory=0, tiekeys=None):
    """The sequential placement loop shared by DChoice and Greedy.

    cand(j) gives the (T, k) candidate paths of flow j. Each flow goes to the
    candidate with the lowest utilization in its engine's view; ties go to the
    earliest candidate, or to the lowest tie key when `tiekeys` is given.
    With `memory`, each engine also considers the paths it remembered last
    time and then remembers the `memory` best distinct candidates it just
    saw, the chosen path first (DRILL(d, m)). Memory slots start empty.
    """
    T, m = w.shape
    rows = np.arange(T)
    inv = np.ones(n) if speeds is None else 1.0 / np.asarray(speeds, dtype=float)
    util = np.zeros((T, n))
    view = util
    mem = np.full((T, engines, memory), -1, dtype=np.int64)
    paths = np.empty((T, m), dtype=np.int64)
    for j in range(m):                        # flows arrive one at a time...
        f = j % engines
        if engines > 1 and f == 0:
            view = util.copy()                # a new round: everyone reads this snapshot
        c = cand(j)                           # ...but all T trials move together
        if memory:
            c = np.concatenate([c, mem[:, f, :]], axis=1)
        seen = np.where(c >= 0, np.take_along_axis(view, np.maximum(c, 0), axis=1), np.inf)
        if tiekeys is None:
            idx = np.argmin(seen, axis=1)     # argmin = first minimum
        else:
            tied = seen == seen.min(axis=1, keepdims=True)
            idx = np.argmin(np.where(tied, tiekeys[:, j, :c.shape[1]], np.inf), axis=1)
        pick = c[rows, idx]
        paths[:, j] = pick
        util[rows, pick] += inc[:, j] * inv[pick]
        if memory:
            rank = np.where(c == pick[:, None], -np.inf, seen)
            for k in range(1, c.shape[1]):    # each path counts once
                rank[(c[:, k:k + 1] == c[:, :k]).any(axis=1), k] = np.inf
            order = np.argsort(rank, axis=1, kind="stable")[:, :memory]
            keep = np.take_along_axis(c, order, axis=1)
            keep[np.take_along_axis(rank, order, axis=1) == np.inf] = -1   # empty or repeat
            mem[:, f, :] = keep
    return paths


def _suffix(engines: int, ties: str) -> str:
    return (f"-e{engines}" if engines > 1 else "") + ("-rtie" if ties == "random" else "")


def _check(engines: int, ties: str):
    if engines < 1:
        raise ValueError("engines must be at least 1")
    if ties not in ("first", "random"):
        raise ValueError("ties must be 'first' or 'random'")


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
    memory       DRILL(d, m): also consider the `memory` best paths this
                 engine saw at its previous decision (named `memory` because
                 `m` already means the number of flows here). DRILL's
                 Theorem 1: with memory = 0 and d < n, no rule can keep up
                 with unequal path speeds; Theorem 2: DRILL(1, 1) can.
    engines      F forwarding engines deciding in parallel from a shared
                 stale snapshot (see the module docstring).
    ties         "first": the earliest candidate. The samples are
                 exchangeable, so without memory that is a uniformly random
                 choice among tied samples. "random": an explicit random key.

    d = 1 with no memory is ECMP.
    """

    def __init__(self, d: int = 2, by: str = "bytes", replace: bool = True, memory: int = 0,
                 engines: int = 1, ties: str = "first"):
        if d < 1:
            raise ValueError("d must be at least 1")
        if by not in ("bytes", "count"):
            raise ValueError("by must be 'bytes' or 'count'")
        if memory < 0:
            raise ValueError("memory must be >= 0")
        _check(engines, ties)
        self.d, self.by, self.replace = d, by, replace
        self.memory, self.engines, self.ties = memory, engines, ties
        self.seed_name = (f"d{d}-{by}" + ("" if replace else "-distinct")
                          + (f"-mem{memory}" if memory else "")
                          + ("-rtie" if ties == "random" else ""))
        self.name = self.seed_name.replace("-rtie", "") + _suffix(engines, ties)

    def draw(self, rng: np.random.Generator, m: int, n: int):
        picks = (rng.integers(0, n, size=(m, self.d)) if self.replace
                 else sample_distinct(rng, m, n, self.d))
        if self.ties == "random":
            return picks, rng.random((m, self.d + self.memory))
        return picks

    def assign(self, w: np.ndarray, draws, n: int, speeds=None, **_) -> np.ndarray:
        picks, keys = draws if self.ties == "random" else (draws, None)
        inc = w if self.by == "bytes" else (w > 0).astype(float)
        return _place(w, lambda j: picks[:, j, :], n, inc=inc, speeds=speeds,
                      engines=self.engines, memory=self.memory, tiekeys=keys)

    def __repr__(self):
        return (f"DChoice(d={self.d}, by={self.by!r}, replace={self.replace}, "
                f"memory={self.memory}, engines={self.engines}, ties={self.ties!r})")


class Greedy:
    """Online least-loaded: each flow joins the path carrying the least weight
    (utilization, with speeds) right now: d = n with full information. A strong
    online yardstick; no switch-local scheme sees this much.

    Ties go to the lowest-numbered path, which cannot change any symmetric
    metric when engines = 1. With engines > 1 it matters a lot: every flow in
    a round sees the same snapshot, so with ties="first" they all pile onto
    the same path, while ties="random" spreads them over the tied paths.
    Report both."""

    def __init__(self, engines: int = 1, ties: str = "first"):
        _check(engines, ties)
        self.engines, self.ties = engines, ties
        self.seed_name = "greedy" + ("-rtie" if ties == "random" else "")
        self.name = "greedy" + _suffix(engines, ties)

    def draw(self, rng, m, n):
        return rng.random((m, n)) if self.ties == "random" else None

    def assign(self, w: np.ndarray, draws, n: int, speeds=None, **_) -> np.ndarray:
        T = w.shape[0]
        everything = np.broadcast_to(np.arange(n), (T, n))
        return _place(w, lambda j: everything, n, inc=w, speeds=speeds,
                      engines=self.engines, tiekeys=draws)

    def __repr__(self):
        return f"Greedy(engines={self.engines}, ties={self.ties!r})"


class RoundRobin:
    """Per-packet round robin (DRILL's RR baseline): each engine keeps a pointer,
    starting at a random path, and sends successive flows to successive paths.
    Load-oblivious. With one engine and unit flows it is perfectly even; with
    many engines the independent pointers collide, a herding effect that
    needs no load information at all."""

    def __init__(self, engines: int = 1):
        _check(engines, "first")
        self.engines = engines
        self.seed_name = "rr"
        self.name = "rr" + _suffix(engines, "first")

    def draw(self, rng, m, n):
        return rng.integers(0, n, size=self.engines)

    def assign(self, w: np.ndarray, draws, n: int, **_) -> np.ndarray:
        j = np.arange(w.shape[1])
        return (draws[:, j % self.engines] + j // self.engines) % n

    def __repr__(self):
        return f"RoundRobin(engines={self.engines})"


class Spray:
    """Presto-style spraying: every flow picks a random starting path and sends
    its successive chunks round-robin from there. Load-oblivious, like ECMP,
    but at chunk granularity. Without chunking (each flow is one chunk) it is
    exactly ECMP. Use it with runner.run(chunk=c)."""

    name = "spray"
    uses_flow = True

    def draw(self, rng, m, n):
        return rng.integers(0, n, size=m)      # m = number of flows, not chunks

    def assign(self, w: np.ndarray, draws, n: int, flow=None, rank=None, **_) -> np.ndarray:
        if flow is None:
            return np.asarray(draws, dtype=np.int64)
        start = np.take_along_axis(draws, np.maximum(flow, 0), axis=1)
        return (start + rank) % n

    def __repr__(self):
        return "Spray()"


class LPT:
    """Offline longest-processing-time first: sort flows largest first, then
    place greedily. It needs every flow's size up front, so no switch could
    run it; it is the near-optimal yardstick. Graham (1969): its max load is at
    most (4/3 - 1/(3n)) times the optimal max load (equal speeds; with speeds
    it places on the least-utilized path, which is a heuristic)."""

    name = "lpt"

    def draw(self, rng, m, n):
        return None

    def assign(self, w: np.ndarray, draws, n: int, speeds=None, **_) -> np.ndarray:
        order = np.argsort(-w, axis=1, kind="stable")
        placed = Greedy().assign(np.take_along_axis(w, order, axis=1), None, n, speeds=speeds)
        paths = np.empty_like(placed)
        np.put_along_axis(paths, order, placed, axis=1)   # back to arrival order
        return paths

    def __repr__(self):
        return "LPT()"
