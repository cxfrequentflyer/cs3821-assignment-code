"""Level 2: a two-tier leaf-spine fabric, where one ECMP group is no longer enough.

A flow goes from a source leaf up to a spine and down to a destination leaf,
so choosing a spine loads TWO links: the uplink (src, spine) and the downlink
(spine, dst). That is what a single ECMP group cannot show: the right split at
a leaf depends on downstream congestion, i.e. on everybody else's traffic
(CONGA Fig. 2-3, DRILL Fig. 4). Host links (the last hop) have no choice and
are not modeled.

    net = LeafSpine.symmetric(leaves=9, spines=4).fail(0, 0)
    df = run_leafspine(net, m=200, flows=Exponential(), trials=500, seed=1,
                       policies=[LSECMP(), LSLocal(), LSConga(), LSComponents(), ESF()])

Policies (all see the same flow sets; seeding works as in runner.py):
    LSECMP        uniform random spine among the usable ones (load-oblivious)
    LSWCMP        random spine with probability proportional to the path's
                  static capacity (WCMP; Presto's failover)
    LSLocal       least-utilized *uplink*: a leaf's own view (DRILL-style
                  local information, decided per flow)
    LSConga       least-congested *path*: min over spines of max(uplink,
                  downlink) utilization (CONGA's global metric, greedy)
    LSComponents  DRILL's asymmetry handling: split the paths into symmetric
                  components, pick a component with probability proportional
                  to its capacity, then the least-utilized uplink inside it
    ESF           Equal Split Fluid: every flow split equally over its usable
                  spines (splittable; optimal in a symmetric fabric)
    LSNash        CONGA's game: start from another policy, then let flows move
                  to a spine with a smaller own bottleneck until none can

Benchmark: `optimum_util` solves the splittable multicommodity LP for the
smallest possible busiest-link utilization (needs scipy). No placement of
unsplittable flows can beat it, so `over_opt` >= 1 for every policy. In a
symmetric fabric it equals ESF, which the tests check.

Modeling choices to state: flows are rate demands in a snapshot, links are
fluid (utilization can exceed 1), and there is no TCP. In particular DRILL's
Fig. 4 argument (TCP coupling the rates of a split flow) is not reproduced;
what is reproduced is the static load distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .runner import rng_for


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LeafSpine:
    """Link capacities: up[l, s] from leaf l to spine s, down[s, l] back down.
    A capacity of 0 means the link is missing or failed."""

    up: np.ndarray
    down: np.ndarray
    label: str = "leafspine"
    _paths: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        up = np.asarray(self.up, dtype=float)
        down = np.asarray(self.down, dtype=float)
        if up.ndim != 2 or down.shape != up.T.shape:
            raise ValueError("up must be (leaves, spines) and down (spines, leaves)")
        if np.any(up < 0) or np.any(down < 0):
            raise ValueError("capacities must be >= 0")
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "down", down)
        L, S = up.shape
        # usable[a, b, s]: the path a -> s -> b exists
        usable = (up[:, None, :] > 0) & (down.T[None, :, :] > 0)
        usable[np.arange(L), np.arange(L), :] = False
        object.__setattr__(self, "_paths", usable)

    @classmethod
    def symmetric(cls, leaves: int, spines: int, capacity: float = 1.0) -> "LeafSpine":
        return cls(np.full((leaves, spines), capacity), np.full((spines, leaves), capacity),
                   f"sym({leaves}x{spines})")

    @property
    def leaves(self) -> int:
        return self.up.shape[0]

    @property
    def spines(self) -> int:
        return self.up.shape[1]

    @property
    def usable(self) -> np.ndarray:
        """(L, L, S) boolean: can traffic from leaf a to leaf b go through spine s?"""
        return self._paths

    def set_link(self, leaf: int, spine: int, capacity: float, label: str | None = None):
        """A copy with the leaf-spine link (both directions) at a new capacity."""
        up, down = self.up.copy(), self.down.copy()
        up[leaf, spine] = down[spine, leaf] = capacity
        return LeafSpine(up, down, label or f"{self.label}|L{leaf}S{spine}={capacity:g}")

    def fail(self, leaf: int, spine: int) -> "LeafSpine":
        return self.set_link(leaf, spine, 0.0, f"{self.label}|fail L{leaf}-S{spine}")

    def path_capacity(self) -> np.ndarray:
        """(L, L, S): the capacity of path a -> s -> b (its slowest link), 0 if unusable."""
        cap = np.minimum(self.up[:, None, :], self.down.T[None, :, :])
        return np.where(self.usable, cap, 0.0)

    def components(self) -> list:
        """DRILL §3.4: for every leaf pair, its usable spines split into symmetric
        components, with weights proportional to their capacity.

        Two links are symmetric when the same set of (src, dst) pairs can use
        them; two paths are symmetric when their links are, hop by hop, with
        the same capacity factor (input rate over output rate at the spine).
        Returns comps[a][b] = list of (spines array, weight).
        """
        L, S = self.leaves, self.spines
        u = self.usable
        up_label = {(a, s): frozenset((a, b) for b in range(L) if u[a, b, s])
                    for a in range(L) for s in range(S)}
        down_label = {(s, b): frozenset((a, b) for a in range(L) if u[a, b, s])
                      for s in range(S) for b in range(L)}
        pcap = self.path_capacity()
        comps = [[[] for _ in range(L)] for _ in range(L)]
        for a in range(L):
            for b in range(L):
                groups: dict = {}
                for s in np.flatnonzero(u[a, b]):
                    score = (up_label[a, s], down_label[s, b],
                             round(self.up[a, s] / self.down[s, b], 12))
                    groups.setdefault(score, []).append(s)
                comps[a][b] = [(np.array(g), float(pcap[a, b, g].sum())) for g in groups.values()]
        return comps


# --------------------------------------------------------------------------
# Traffic
# --------------------------------------------------------------------------

def sample_pairs(rng: np.random.Generator, m: int, leaves: int, pattern: str = "uniform",
                 stride: int = 1):
    """Source and destination leaves for m flows.

    uniform   source uniform, destination uniform among the other leaves
              (pFabric's and CONGA's setup)
    stride    source uniform, destination = source + stride (mod leaves), the
              synthetic Stride pattern in DRILL Table 1
    """
    src = rng.integers(0, leaves, size=m)
    if pattern == "uniform":
        dst = (src + rng.integers(1, leaves, size=m)) % leaves
    elif pattern == "stride":
        if stride % leaves == 0:
            raise ValueError("stride must not be a multiple of the number of leaves")
        dst = (src + stride) % leaves
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    return src, dst


# --------------------------------------------------------------------------
# Placement: every policy returns the spine of each flow, (T, m), or loads
# --------------------------------------------------------------------------

def link_loads(net: LeafSpine, src, dst, w, spine):
    """(T, L, S) uplink loads and (T, S, L) downlink loads of a placement."""
    T, m = w.shape
    L, S = net.leaves, net.spines
    t = np.repeat(np.arange(T), m)
    up = np.zeros((T, L, S))
    down = np.zeros((T, S, L))
    np.add.at(up, (t, src.ravel(), spine.ravel()), w.ravel())
    np.add.at(down, (t, spine.ravel(), dst.ravel()), w.ravel())
    return up, down


def _util(load, cap):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cap > 0, load / np.where(cap > 0, cap, 1.0), np.where(load > 0, np.inf, 0.0))


def max_util(net: LeafSpine, up_load, down_load) -> np.ndarray:
    """(T,) utilization of the busiest link."""
    return np.maximum(_util(up_load, net.up).max(axis=(1, 2)),
                      _util(down_load, net.down).max(axis=(1, 2)))


def _pick_weighted(weights, u):
    """Row-wise: index chosen with probability proportional to weights, using uniforms u."""
    cum = np.cumsum(weights, axis=1)
    return np.minimum((cum < (u * cum[:, -1])[:, None]).sum(axis=1), weights.shape[1] - 1)


class _Sequential:
    """Flows are placed one at a time (all T trials together), each looking at
    the loads the earlier flows created. Subclasses define `score`, the value
    to minimize over usable spines; ties go to the lowest random key."""

    def draw(self, rng, m, net):
        return rng.random((m, 1 + net.spines))

    def place(self, net, src, dst, w, draws):
        T, m = w.shape
        rows = np.arange(T)
        S = net.spines
        up_u = np.zeros((T, net.leaves, S))       # utilizations so far
        down_u = np.zeros((T, S, net.leaves))
        spine = np.empty((T, m), dtype=np.int64)
        up_cap = np.where(net.up > 0, net.up, np.inf)
        down_cap = np.where(net.down > 0, net.down, np.inf)
        for j in range(m):
            a, b = src[:, j], dst[:, j]
            ok = net.usable[a, b]                                   # (T, S)
            score = self.score(net, a, b, ok, up_u[rows, a, :], down_u[rows, :, b], draws[:, j, 0])
            score = np.where(ok, score, np.inf)
            tied = score == score.min(axis=1, keepdims=True)
            s = np.argmin(np.where(tied, draws[:, j, 1:], np.inf), axis=1)
            spine[:, j] = s
            up_u[rows, a, s] += w[:, j] / up_cap[a, s]
            down_u[rows, s, b] += w[:, j] / down_cap[s, b]
        return spine


class LSECMP(_Sequential):
    name = "ecmp"

    def score(self, net, a, b, ok, up, down, u):
        return np.zeros_like(up)                  # all tied: a uniform usable spine


class LSWCMP(_Sequential):
    name = "wcmp"

    def score(self, net, a, b, ok, up, down, u):
        pick = _pick_weighted(np.where(ok, net.path_capacity()[a, b], 0.0), u)
        out = np.ones_like(up)
        out[np.arange(len(a)), pick] = 0.0
        return out


class LSLocal(_Sequential):
    name = "local"

    def score(self, net, a, b, ok, up, down, u):
        return up


class LSConga(_Sequential):
    name = "conga"

    def score(self, net, a, b, ok, up, down, u):
        return np.maximum(up, down)


class LSComponents(_Sequential):
    name = "drill-comp"

    def place(self, net, src, dst, w, draws):
        self._comps = net.components()
        return super().place(net, src, dst, w, draws)

    def score(self, net, a, b, ok, up, down, u):
        T, S = up.shape
        allowed = np.zeros((T, S), dtype=bool)
        for i in range(T):                        # small: one leaf pair per trial
            comps = self._comps[a[i]][b[i]]
            weights = np.array([c[1] for c in comps])
            k = int(_pick_weighted(weights[None, :], np.array([u[i]]))[0])
            allowed[i, comps[k][0]] = True
        return np.where(allowed, up, np.inf)


class ESF:
    """Equal Split Fluid (DRILL §3.1): every flow is split equally over all its
    usable spines. Splittable, so it returns loads rather than spines."""

    name = "esf"
    splittable = True

    def draw(self, rng, m, net):
        return None

    def loads(self, net, src, dst, w):
        u = net.usable[src, dst]                                  # (T, m, S)
        share = w[..., None] * u / u.sum(axis=-1, keepdims=True)
        T, m = w.shape
        up = np.zeros((T, net.leaves, net.spines))
        down = np.zeros((T, net.spines, net.leaves))
        t = np.repeat(np.arange(T), m)
        np.add.at(up, (t, src.ravel()), share.reshape(T * m, -1))
        np.add.at(down.transpose(0, 2, 1), (t, dst.ravel()), share.reshape(T * m, -1))
        return up, down


class LSNash:
    """CONGA's bottleneck routing game, played by whole flows.

    Start from `start`'s placement, then repeatedly let any flow move to a
    spine where its own bottleneck (the busiest link on its path, counting
    itself) would be strictly smaller. Each move lowers the sorted vector of
    link utilizations, so this stops, at a pure Nash equilibrium of the
    unsplittable game. CONGA's Theorem 1 (PoA = 2) is about the splittable
    game; with many small flows the two get close.
    """

    def __init__(self, start=None, max_rounds: int = 200, tol: float = 1e-12):
        self.start = start or LSConga()
        self.max_rounds, self.tol = max_rounds, tol
        self.name = f"nash({self.start.name})"

    def draw(self, rng, m, net):
        return self.start.draw(rng, m, net)

    def place(self, net, src, dst, w, draws):
        spine = self.start.place(net, src, dst, w, draws)
        T, m = w.shape
        rows = np.arange(T)
        up_cap = np.where(net.up > 0, net.up, np.inf)
        down_cap = np.where(net.down > 0, net.down, np.inf)
        up, down = link_loads(net, src, dst, w, spine)
        up_u, down_u = up / up_cap, down / down_cap
        self.rounds = 0
        for self.rounds in range(1, self.max_rounds + 1):
            moved = False
            for j in range(m):
                a, b, s = src[:, j], dst[:, j], spine[:, j]
                wu, wd = w[:, j][:, None] / up_cap[a], w[:, j][:, None] / down_cap[:, b].T
                cur_up, cur_down = up_u[rows, a, :], down_u[rows, :, b]
                now = np.maximum(cur_up[rows, s], cur_down[rows, s])
                # the flow's bottleneck on each other spine, if it moved there
                alt = np.maximum(cur_up + wu, cur_down + wd)
                alt[rows, s] = now
                alt = np.where(net.usable[a, b], alt, np.inf)
                best = np.argmin(alt, axis=1)
                go = alt[rows, best] < now - self.tol
                if go.any():
                    moved = True
                    r, old, new = rows[go], s[go], best[go]
                    up_u[r, a[go], old] -= wu[go, old]
                    down_u[r, old, b[go]] -= wd[go, old]
                    up_u[r, a[go], new] += wu[go, new]
                    down_u[r, new, b[go]] += wd[go, new]
                    spine[go, j] = new
            if not moved:
                break
        return spine


# --------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------

def optimum_util(net: LeafSpine, src, dst, w) -> float:
    """Smallest possible busiest-link utilization for ONE trial if flows could
    be split arbitrarily: a multicommodity flow LP (all flows between the same
    leaf pair are one commodity).

        minimize z  s.t.  sum_s x[a,b,s] = D[a,b]   (only over usable spines)
                          sum_b x[a,b,s] <= z up[a,s],  sum_a x[a,b,s] <= z down[s,b]
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    L, S = net.leaves, net.spines
    D = np.zeros((L, L))
    np.add.at(D, (src, dst), w)
    pairs = [(a, b) for a in range(L) for b in range(L) if D[a, b] > 0]
    var = [(a, b, s) for (a, b) in pairs for s in np.flatnonzero(net.usable[a, b])]
    if any(not net.usable[a, b].any() for a, b in pairs):
        return np.inf
    nv = len(var) + 1                                   # the last variable is z
    eq_r, eq_c, ub_r, ub_c, ub_v = [], [], [], [], []
    pair_row = {p: i for i, p in enumerate(pairs)}
    up_row = {(a, s): i for i, (a, s) in enumerate((a, s) for a in range(L) for s in range(S))}
    down_row = {(s, b): len(up_row) + i
                for i, (s, b) in enumerate((s, b) for s in range(S) for b in range(L))}
    for k, (a, b, s) in enumerate(var):
        eq_r.append(pair_row[a, b]); eq_c.append(k)
        ub_r += [up_row[a, s], down_row[s, b]]; ub_c += [k, k]; ub_v += [1.0, 1.0]
    caps = np.concatenate([net.up.ravel(), net.down.ravel()])
    for r in range(len(caps)):
        ub_r.append(r); ub_c.append(nv - 1); ub_v.append(-caps[r])
    A_eq = coo_matrix((np.ones(len(eq_r)), (eq_r, eq_c)), shape=(len(pairs), nv))
    A_ub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(len(caps), nv))
    c = np.zeros(nv); c[-1] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(caps)), A_eq=A_eq,
                  b_eq=np.array([D[a, b] for a, b in pairs]), bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.x[-1])


def elephant_bound(net: LeafSpine, src, dst, w) -> np.ndarray:
    """(T,) a flow that cannot be split runs at least at w / (its best path's
    capacity) somewhere, whatever the policy."""
    best = net.path_capacity().max(axis=2)[src, dst]
    with np.errstate(divide="ignore"):
        return (w / best).max(axis=1)


# --------------------------------------------------------------------------
# Running experiments
# --------------------------------------------------------------------------

def run_leafspine(net: LeafSpine, m: int, flows, policies, trials: int = 200, seed: int = 0,
                  pattern: str = "uniform", stride: int = 1, optimum: bool = True,
                  label: str | None = None) -> pd.DataFrame:
    """One row per (policy, trial).

    max_util   utilization of the busiest link
    up_max, down_max   the busiest uplink and downlink utilizations (where
               the bottleneck is: CONGA's point is that it can be downstream)
    opt        the splittable LP optimum (if optimum=True), and
    over_opt   max_util / opt, at least 1
    over_lb    max_util / max(opt, elephant bound); just over_opt for ESF,
               which splits flows and so is not bound by the elephant term
    """
    label = label or net.label
    L = net.leaves
    keys = (seed, "ls-flows", flows.label, pattern, stride, L, m)
    w = np.stack([flows.sample(rng_for(*keys, t, "w"), m) for t in range(trials)])
    pairs = [sample_pairs(rng_for(*keys, t, "pairs"), m, L, pattern, stride) for t in range(trials)]
    src = np.stack([p[0] for p in pairs])
    dst = np.stack([p[1] for p in pairs])
    if not net.usable[src, dst].any(axis=-1).all():
        raise ValueError("some flow has no usable path in this topology")
    opt = (np.array([optimum_util(net, src[t], dst[t], w[t]) for t in range(trials)])
           if optimum else np.full(trials, np.nan))
    lb = np.maximum(opt, elephant_bound(net, src, dst, w)) if optimum else opt
    frames = []
    for pol in policies:
        if hasattr(pol, "loads"):
            up, down = pol.loads(net, src, dst, w)
        else:
            draws = np.stack([pol.draw(rng_for(seed, "ls-policy", pol.name, L, m, t), m, net)
                              for t in range(trials)])
            up, down = link_loads(net, src, dst, w, pol.place(net, src, dst, w, draws))
        mu = max_util(net, up, down)
        frames.append(pd.DataFrame({
            "policy": pol.name, "trial": np.arange(trials), "max_util": mu,
            "up_max": _util(up, net.up).max(axis=(1, 2)),
            "down_max": _util(down, net.down).max(axis=(1, 2)),
            "opt": opt, "over_opt": mu / opt,
            # the elephant bound only binds placements that cannot split flows
            "over_lb": mu / (opt if getattr(pol, "splittable", False) else lb)}))
    df = pd.concat(frames, ignore_index=True)
    for i, (col, val) in enumerate([("topology", label), ("flows", flows.label),
                                    ("pattern", pattern), ("m", m), ("seed", seed)]):
        df.insert(i, col, val)
    return df
