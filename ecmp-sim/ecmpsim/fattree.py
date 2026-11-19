"""Level 3: a three-tier fat-tree, where one flow takes several ECMP decisions.

A k-ary fat-tree (Al-Fares et al., SIGCOMM 2008) has k pods, each with k/2
edge and k/2 aggregation switches, and (k/2)^2 core switches. Core (a, c) is
the c-th core attached to aggregation switch a of EVERY pod. A flow between
pods climbs edge -> agg -> core and then has no choice on the way down:

    edge (ps, es) --a--> agg (ps, a) --c--> core (a, c) ---> agg (pd, a) ---> edge (pd, ed)

so its path is the pair (a, c): two ECMP decisions, taken by two different
switches, loading four links. A flow inside a pod only chooses a (two links).
Flows between hosts on the same edge switch never enter the fabric and are
not generated; host links are not modeled (as in leafspine.py).

    net = FatTree.kary(8).fail_core_link(pod=0, agg=0, core=0)
    df = run_fattree(net, m=2000, flows=Exponential(), trials=100, seed=1,
                     policies=[FTECMP(), FTHashECMP("crc16"), FTLocal(), FTGlobal(),
                               FTComponents(), FTESF()])

What three tiers add over the leaf-spine level:
  * hop-by-hop choices: a switch picks among ITS next hops, so with failures
    "uniform at every hop" is not "uniform over paths";
  * the upstream switch cannot see the downstream load (FTLocal), and here
    even a clever edge switch only controls half of the decision;
  * hash polarization: if the edge and the aggregation switch compute the
    same hash, the flows an aggregation switch receives all agree on it, and
    it sends every one of them to the same core (FTHashECMP).

Policies (all see the same flow sets and the same 5-tuples):
    FTECMP        ideal ECMP: an independent uniform choice at each hop
    FTHashECMP    real ECMP: hash(5-tuple) mod (usable next hops) at each
                  hop, with configurable hashes, per-switch seeds and bits
    FTLocal       DRILL-style: each switch sends the flow to its least-
                  utilized usable uplink (local information only)
    FTGlobal      least-congested path: min over (a, c) of the busiest of the
                  four links (a CONGA-like global yardstick; CONGA itself is
                  two-tier only)
    FTComponents  DRILL §3.4 for a multi-stage Clos: paths are grouped into
                  symmetric components, one is picked with probability
                  proportional to its capacity, then least-utilized uplinks
                  inside it
    FTESF         Equal Split Fluid, hop by hop (splittable)

Benchmark: `optimum_util`, the splittable multicommodity LP, as in the
leaf-spine level. In a symmetric fat-tree it equals FTESF (tested).

All link arrays are indexed pod first:
    edge_up[p, e, a]    edge e -> agg a          agg_down[p, a, e]   agg a -> edge e
    agg_up[p, a, c]     agg a -> core (a, c)     core_down[p, a, c]  core (a, c) -> agg a
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import hashing
from .runner import rng_for

TIERS = ("edge_up", "agg_up", "core_down", "agg_down")


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FatTree:
    """Link capacities of a three-tier fat-tree; 0 means missing or failed.

    Shapes (P pods, E edge and A aggregation switches per pod, C cores per
    aggregation switch): edge_up (P, E, A), agg_down (P, A, E), agg_up and
    core_down (P, A, C). `hosts` per edge switch only matters for addresses
    and for how traffic is sampled.
    """

    edge_up: np.ndarray
    agg_down: np.ndarray
    agg_up: np.ndarray
    core_down: np.ndarray
    hosts: int = 2
    label: str = "fattree"
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        for name in TIERS:
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.ndim != 3 or np.any(arr < 0):
                raise ValueError(f"{name} must be a 3-D array of capacities >= 0")
            object.__setattr__(self, name, arr)
        P, E, A = self.edge_up.shape
        if (self.agg_down.shape != (P, A, E) or self.agg_up.shape[:2] != (P, A)
                or self.core_down.shape != self.agg_up.shape):
            raise ValueError("inconsistent link array shapes")
        if self.hosts < 1:
            raise ValueError("hosts must be at least 1")

    @classmethod
    def symmetric(cls, pods: int, edges: int, aggs: int, cores: int, hosts: int | None = None,
                  capacity: float = 1.0, label: str | None = None) -> "FatTree":
        """`cores` is per aggregation switch: aggs * cores core switches in all."""
        if pods < 2 or min(edges, aggs, cores) < 1:
            raise ValueError("need at least 2 pods and one switch of every kind")
        return cls(np.full((pods, edges, aggs), capacity), np.full((pods, aggs, edges), capacity),
                   np.full((pods, aggs, cores), capacity), np.full((pods, aggs, cores), capacity),
                   hosts or edges, label or f"ft({pods}p,{edges}e,{aggs}a,{cores}c)")

    @classmethod
    def kary(cls, k: int, capacity: float = 1.0) -> "FatTree":
        """The standard k-ary fat-tree: k pods, k/2 of everything else, k^3/4 hosts."""
        if k < 2 or k % 2:
            raise ValueError("k must be even and at least 2")
        h = k // 2
        return cls.symmetric(k, h, h, h, h, capacity, f"fattree(k={k})")

    pods = property(lambda self: self.edge_up.shape[0])
    edges = property(lambda self: self.edge_up.shape[1])
    aggs = property(lambda self: self.edge_up.shape[2])
    cores = property(lambda self: self.agg_up.shape[2])

    def _replace(self, label, **arrays) -> "FatTree":
        cur = {name: getattr(self, name) for name in TIERS}
        cur.update(arrays)
        return FatTree(**cur, hosts=self.hosts, label=label)

    def set_edge_link(self, pod: int, edge: int, agg: int, capacity: float, label=None):
        """A copy with the edge-agg link (both directions) at a new capacity."""
        up, down = self.edge_up.copy(), self.agg_down.copy()
        up[pod, edge, agg] = down[pod, agg, edge] = capacity
        return self._replace(label or f"{self.label}|P{pod}:E{edge}-A{agg}={capacity:g}",
                             edge_up=up, agg_down=down)

    def set_core_link(self, pod: int, agg: int, core: int, capacity: float, label=None):
        """A copy with the link between agg (pod, agg) and core (agg, core), both
        directions, at a new capacity."""
        up, down = self.agg_up.copy(), self.core_down.copy()
        up[pod, agg, core] = down[pod, agg, core] = capacity
        return self._replace(label or f"{self.label}|P{pod}:A{agg}-C{agg}.{core}={capacity:g}",
                             agg_up=up, core_down=down)

    def fail_edge_link(self, pod: int, edge: int, agg: int) -> "FatTree":
        return self.set_edge_link(pod, edge, agg, 0.0, f"{self.label}|fail P{pod}:E{edge}-A{agg}")

    def fail_core_link(self, pod: int, agg: int, core: int) -> "FatTree":
        return self.set_core_link(pod, agg, core, 0.0,
                                  f"{self.label}|fail P{pod}:A{agg}-C{agg}.{core}")

    # ---- paths: everything below is indexed [..., a, c] ----

    def _hops(self, ps, es, pd, ed):
        """Capacities of the four links of every path, each (..., A, C), and
        the inter-pod mask (..., 1, 1). Intra-pod flows have no core hops."""
        ps, es, pd, ed = (np.asarray(x) for x in (ps, es, pd, ed))
        shape = (*ps.shape, self.aggs, self.cores)
        inter = (ps != pd)[..., None, None]
        eu = np.broadcast_to(self.edge_up[ps, es][..., None], shape)
        ad = np.broadcast_to(self.agg_down[pd, :, ed][..., None], shape)
        return eu, self.agg_up[ps], self.core_down[pd], ad, inter

    def usable(self, ps, es, pd, ed) -> np.ndarray:
        """(..., A, C) boolean: can a flow from edge (ps, es) to edge (pd, ed)
        take path (a, c)? An intra-pod path is written (a, 0)."""
        eu, au, cd, ad, inter = self._hops(ps, es, pd, ed)
        first = np.arange(self.cores) == 0
        return (eu > 0) & (ad > 0) & np.where(inter, (au > 0) & (cd > 0), first)

    def path_capacity(self, ps, es, pd, ed) -> np.ndarray:
        """(..., A, C): capacity of each path (its slowest link), 0 if unusable."""
        eu, au, cd, ad, inter = self._hops(ps, es, pd, ed)
        cap = np.minimum(np.minimum(eu, ad), np.where(inter, np.minimum(au, cd), np.inf))
        return np.where(self.usable(ps, es, pd, ed), cap, 0.0)

    def components(self, ps: int, es: int, pd: int, ed: int) -> list:
        """DRILL §3.4, for one pair of edge switches: its usable paths split into
        symmetric components, with weights proportional to their capacity.

        Two links are symmetric when the same set of (source edge, destination
        edge) pairs can use them; two paths are symmetric when their links
        are, hop by hop, with the same capacity ratio between consecutive
        hops. Returns a list of ((A, C) boolean mask, weight).
        """
        key = (int(ps), int(es), int(pd), int(ed))
        if "labels" not in self._cache:
            self._cache["labels"] = self._link_labels()
            self._cache["components"] = {}
        done = self._cache["components"]
        if key not in done:
            labels = self._cache["labels"]
            ps, es, pd, ed = key
            ok = self.usable(ps, es, pd, ed)
            cap = self.path_capacity(ps, es, pd, ed)
            groups: dict = {}
            for a, c in zip(*np.nonzero(ok)):
                hops = [("edge_up", ps, es, a), ("agg_down", pd, a, ed)]
                if ps != pd:
                    hops[1:1] = [("agg_up", ps, a, c), ("core_down", pd, a, c)]
                caps = [getattr(self, t)[i, j, k] for t, i, j, k in hops]
                sig = (tuple(labels[h] for h in hops),
                       tuple(round(x / y, 12) for x, y in zip(caps, caps[1:])))
                groups.setdefault(sig, []).append((a, c))
            out = []
            for members in groups.values():
                mask = np.zeros(ok.shape, dtype=bool)
                mask[tuple(np.array(members).T)] = True
                out.append((mask, float(cap[mask].sum())))
            done[key] = out
        return done[key]

    def _link_labels(self) -> dict:
        """link -> an id for the set of edge pairs that can use it."""
        users: dict = {}
        P, E = self.pods, self.edges
        pair = 0
        for ps in range(P):
            for es in range(E):
                for pd in range(P):
                    for ed in range(E):
                        if (ps, es) == (pd, ed):
                            continue
                        for a, c in zip(*np.nonzero(self.usable(ps, es, pd, ed))):
                            links = [("edge_up", ps, es, a), ("agg_down", pd, a, ed)]
                            if ps != pd:
                                links += [("agg_up", ps, a, c), ("core_down", pd, a, c)]
                            for link in links:
                                users.setdefault(link, set()).add(pair)
                        pair += 1
        ids: dict = {}
        return {link: ids.setdefault(frozenset(p), len(ids)) for link, p in users.items()}


# --------------------------------------------------------------------------
# Traffic
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Traffic:
    """T trials of m flows: sizes w, source (ps, es, hs) and destination
    (pd, ed, hd) as pod / edge switch / host, all (T, m), and the flows'
    5-tuples, headers (T, m, 13)."""

    w: np.ndarray
    ps: np.ndarray
    es: np.ndarray
    hs: np.ndarray
    pd: np.ndarray
    ed: np.ndarray
    hd: np.ndarray
    headers: np.ndarray

    @property
    def inter(self) -> np.ndarray:
        return self.ps != self.pd


PATTERNS = ("uniform", "interpod", "stride")


def sample_hosts(rng: np.random.Generator, m: int, net: FatTree, pattern: str = "uniform",
                 stride: int = 1):
    """Endpoints of m flows: (ps, es, hs, pd, ed, hd).

    uniform    source host uniform; destination uniform among the hosts on
               every OTHER edge switch (so a mix of intra- and inter-pod)
    interpod   destination uniform among the hosts of the other pods
    stride     destination = the same edge and host index, `stride` pods on
    """
    P, E, H = net.pods, net.edges, net.hosts
    ps, es, hs = rng.integers(0, P, m), rng.integers(0, E, m), rng.integers(0, H, m)
    if pattern == "uniform":
        other = (ps * E + es + rng.integers(1, P * E, m)) % (P * E)
        pd, ed, hd = other // E, other % E, rng.integers(0, H, m)
    elif pattern == "interpod":
        pd = (ps + rng.integers(1, P, m)) % P
        ed, hd = rng.integers(0, E, m), rng.integers(0, H, m)
    elif pattern == "stride":
        if stride % P == 0:
            raise ValueError("stride must not be a multiple of the number of pods")
        pd, ed, hd = (ps + stride) % P, es, hs
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    return ps, es, hs, pd, ed, hd


def make_headers(rng, ps, es, hs, pd, ed, hd, ports: str = "ephemeral") -> np.ndarray:
    """5-tuples with Al-Fares addressing: host h on edge e of pod p is 10.p.e.(h + 2)."""
    src_port, dst_port = hashing.sample_ports(rng, len(ps), ports)
    return hashing.pack(hashing.ip(10, ps, es, hs + 2), hashing.ip(10, pd, ed, hd + 2),
                        src_port, dst_port)


# --------------------------------------------------------------------------
# Loads
# --------------------------------------------------------------------------

def link_loads(net: FatTree, tr: Traffic, a: np.ndarray, c: np.ndarray) -> dict:
    """Loads of a placement (a, c), each (T, m): tier name -> (T, P, ., .)."""
    T, m = tr.w.shape
    t = np.broadcast_to(np.arange(T)[:, None], (T, m))
    out = {name: np.zeros((T, *getattr(net, name).shape)) for name in TIERS}
    np.add.at(out["edge_up"], (t, tr.ps, tr.es, a), tr.w)
    np.add.at(out["agg_down"], (t, tr.pd, a, tr.ed), tr.w)
    i = tr.inter
    np.add.at(out["agg_up"], (t[i], tr.ps[i], a[i], c[i]), tr.w[i])
    np.add.at(out["core_down"], (t[i], tr.pd[i], a[i], c[i]), tr.w[i])
    return out


def _util(load, cap):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cap > 0, load / np.where(cap > 0, cap, 1.0), np.where(load > 0, np.inf, 0.0))


def tier_utils(net: FatTree, loads: dict) -> dict:
    """tier name -> (T,) utilization of its busiest link."""
    return {name: _util(loads[name], getattr(net, name)).max(axis=(1, 2, 3)) for name in TIERS}


def max_util(net: FatTree, loads: dict) -> np.ndarray:
    """(T,) utilization of the busiest link anywhere."""
    return np.max(list(tier_utils(net, loads).values()), axis=0)


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------

def _argmin_ties(score, ok, keys):
    """Row-wise index of the smallest score among `ok`; ties to the lowest key."""
    score = np.where(ok, score, np.inf)
    tied = score == score.min(axis=1, keepdims=True)
    return np.argmin(np.where(tied, keys, np.inf), axis=1)


def _kth_true(mask, k):
    """Index of the k-th (0-based) True along the last axis."""
    return (np.cumsum(mask, axis=-1) > k[..., None]).argmax(axis=-1)


def _pick_weighted(weights, u):
    """Index chosen with probability proportional to weights, from a uniform u."""
    cum = np.cumsum(weights)
    return min(int((cum < u * cum[-1]).sum()), len(weights) - 1)


class _Sequential:
    """Flows are placed one at a time (all T trials together), each looking at
    the utilizations the earlier flows created.

    Subclasses define either `hop_scores` (hop by hop: the edge switch
    minimizes one score over its usable aggregation switches, then that
    switch minimizes another over its usable cores) or `path_scores` (one
    score over whole paths). Ties go to the lowest random key. `allowed`
    can restrict the paths further.
    """

    path_scores = None

    def draw(self, rng, m, net):
        A, C = net.aggs, net.cores
        return rng.random((m, 1 + A + A * C))    # one uniform, agg keys, core keys per agg

    def allowed(self, net, tr, j, ok, u):
        return ok

    def place(self, net, tr, draws):
        T, m = tr.w.shape
        A, C = net.aggs, net.cores
        rows = np.arange(T)
        cap = {name: np.where(getattr(net, name) > 0, getattr(net, name), np.inf) for name in TIERS}
        U = {name: np.zeros((T, *getattr(net, name).shape)) for name in TIERS}   # utilizations
        a_out = np.empty((T, m), dtype=np.int64)
        c_out = np.empty((T, m), dtype=np.int64)
        for j in range(m):
            ps, es, pd, ed, x = tr.ps[:, j], tr.es[:, j], tr.pd[:, j], tr.ed[:, j], tr.w[:, j]
            inter = ps != pd
            ok = self.allowed(net, tr, j, net.usable(ps, es, pd, ed), draws[:, j, 0])
            eu, ad = U["edge_up"][rows, ps, es], U["agg_down"][rows, pd, :, ed]        # (T, A)
            au = np.where(inter[:, None, None], U["agg_up"][rows, ps], 0.0)            # (T, A, C)
            cd = np.where(inter[:, None, None], U["core_down"][rows, pd], 0.0)
            ka, kc = draws[:, j, 1:1 + A], draws[:, j, 1 + A:].reshape(T, A, C)
            if self.path_scores is not None:
                score = self.path_scores(eu, au, cd, ad)
                best = _argmin_ties(score.reshape(T, -1), ok.reshape(T, -1), kc.reshape(T, -1))
                a, c = best // C, best % C
            else:
                agg_score, core_score = self.hop_scores(eu, au)
                a = _argmin_ties(agg_score, ok.any(axis=2), ka)
                c = _argmin_ties(core_score[rows, a], ok[rows, a], kc[rows, a])
            a_out[:, j], c_out[:, j] = a, c
            U["edge_up"][rows, ps, es, a] += x / cap["edge_up"][ps, es, a]
            U["agg_down"][rows, pd, a, ed] += x / cap["agg_down"][pd, a, ed]
            r = rows[inter]
            U["agg_up"][r, ps[r], a[r], c[r]] += x[r] / cap["agg_up"][ps[r], a[r], c[r]]
            U["core_down"][r, pd[r], a[r], c[r]] += x[r] / cap["core_down"][pd[r], a[r], c[r]]
        return a_out, c_out


class FTECMP(_Sequential):
    """Ideal ECMP: every switch picks uniformly among its usable next hops,
    independently of every other switch and of the load."""

    name = "ecmp"

    def hop_scores(self, eu, au):
        return np.zeros_like(eu), np.zeros_like(au)       # all tied: uniform


class FTLocal(_Sequential):
    """Each switch sends the flow to its own least-utilized usable uplink
    (DRILL's information: local queues only, nothing about downstream)."""

    name = "local"

    def hop_scores(self, eu, au):
        return eu, au


class FTGlobal(_Sequential):
    """The path whose busiest link is least utilized, over all (a, c): needs
    every link on every path, which no fat-tree switch has. A yardstick."""

    name = "global"

    def path_scores(self, eu, au, cd, ad):
        return np.maximum(np.maximum(eu, ad)[:, :, None], np.maximum(au, cd))


class FTComponents(FTLocal):
    """DRILL's asymmetry handling: pick one symmetric component of the flow's
    paths with probability proportional to its capacity, then choose least-
    utilized uplinks hop by hop among that component's paths only."""

    name = "drill-comp"

    def allowed(self, net, tr, j, ok, u):
        out = np.zeros_like(ok)
        for i in range(len(u)):                   # small: one edge pair per trial
            comps = net.components(tr.ps[i, j], tr.es[i, j], tr.pd[i, j], tr.ed[i, j])
            out[i] = comps[_pick_weighted(np.array([wt for _, wt in comps]), u[i])][0]
        return out


class FTHashECMP:
    """ECMP as switches do it: each hop takes hash(5-tuple) mod (its number of
    usable next hops) and uses that entry of its next-hop list.

    edge, agg   the hash of the edge and of the aggregation switches: a Hash
                or a kind ("crc16", "crc16-arc", "crc32", "xor16"); `agg`
                defaults to the same as `edge`
    seeds       "none": every switch computes exactly the same function.
                "xor": every switch XORs its own random seed into the hash,
                which is what seeding a CRC amounts to (see hashing.py)

    The same hash at both tiers polarizes: aggregation switch a only receives
    flows with hash = a (mod A), and in a k-ary fat-tree A = C, so it sends
    ALL of them to its core number a and leaves its other uplinks idle. Remedies to compare: a different polynomial per tier
    (agg="crc16-arc"), different bits per tier (agg=Hash("crc16", shift=8)),
    or a fan-out that is not a power of two together with seeds.
    """

    def __init__(self, edge="crc16", agg=None, seeds: str = "none"):
        if seeds not in ("none", "xor"):
            raise ValueError("seeds must be 'none' or 'xor'")
        self.edge = hashing.as_hash(edge)
        self.agg = hashing.as_hash(agg if agg is not None else edge)
        self.seeds = seeds
        self.name = ("hash(" + self.edge.label
                     + (f",{self.agg.label}" if self.agg != self.edge else "")
                     + (",seeded" if seeds == "xor" else "") + ")")

    def draw(self, rng, m, net):
        """Per-switch seeds: one per edge switch, then one per aggregation switch."""
        return rng.integers(0, 2 ** 32, size=net.pods * (net.edges + net.aggs))

    def place(self, net, tr, draws):
        P, E, A = net.pods, net.edges, net.aggs
        ok = net.usable(tr.ps, tr.es, tr.pd, tr.ed)                       # (T, m, A, C)
        t = np.arange(len(tr.w))[:, None]
        ok_a = ok.any(axis=3)
        seed = draws[t, tr.ps * E + tr.es] if self.seeds == "xor" else 0
        a = _kth_true(ok_a, self.edge(tr.headers, seed) % ok_a.sum(axis=2))
        ok_c = np.take_along_axis(ok, a[..., None, None], axis=2)[:, :, 0, :]
        seed = draws[t, P * E + tr.ps * A + a] if self.seeds == "xor" else 0
        c = _kth_true(ok_c, self.agg(tr.headers, seed) % ok_c.sum(axis=2))
        return a, c


class FTESF:
    """Equal Split Fluid, hop by hop: the edge switch splits a flow equally over
    its usable aggregation switches, and each of those equally over its usable
    cores. Splittable, so it returns loads rather than paths."""

    name = "esf"
    splittable = True

    def draw(self, rng, m, net):
        return None

    def loads(self, net, tr):
        T, m = tr.w.shape
        ok = net.usable(tr.ps, tr.es, tr.pd, tr.ed)                       # (T, m, A, C)
        ok_a = ok.any(axis=3)
        per_agg = tr.w[..., None] * ok_a / ok_a.sum(axis=2, keepdims=True)
        per_path = per_agg[..., None] * ok / np.maximum(ok.sum(axis=3, keepdims=True), 1)
        per_path = per_path * tr.inter[..., None, None]
        t = np.broadcast_to(np.arange(T)[:, None], (T, m))
        out = {name: np.zeros((T, *getattr(net, name).shape)) for name in TIERS}
        np.add.at(out["edge_up"], (t, tr.ps, tr.es), per_agg)
        np.add.at(out["agg_down"].transpose(0, 1, 3, 2), (t, tr.pd, tr.ed), per_agg)
        np.add.at(out["agg_up"], (t, tr.ps), per_path)
        np.add.at(out["core_down"], (t, tr.pd), per_path)
        return out


# --------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------

def optimum_util(net: FatTree, ps, es, pd, ed, w) -> float:
    """Smallest possible busiest-link utilization for ONE trial if flows could
    be split arbitrarily: a multicommodity flow LP over paths (all flows
    between the same pair of edge switches are one commodity).

        minimize z  s.t.  sum over usable paths x[pair, a, c] = D[pair]
                          for every link: (flow on it) <= z * capacity
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    P, E, A, C = net.pods, net.edges, net.aggs, net.cores
    src, dst = np.asarray(ps) * E + es, np.asarray(pd) * E + ed
    D = np.zeros((P * E, P * E))
    np.add.at(D, (src, dst), w)
    s, d = np.nonzero(D)
    ok = net.usable(s // E, s % E, d // E, d % E)                         # (pairs, A, C)
    if not ok.any(axis=(1, 2)).all():
        return np.inf
    k, a, c = np.nonzero(ok)                                              # one variable each
    sp, se, dp, de = s[k] // E, s[k] % E, d[k] // E, d[k] % E
    inter = sp != dp
    nv = len(k) + 1                                                       # the last one is z
    offs = np.concatenate([[0], np.cumsum([getattr(net, name).size for name in TIERS])])
    var = np.arange(len(k))
    rows = np.concatenate([offs[0] + (sp * E + se) * A + a,                # in TIERS order
                           offs[1] + ((sp * A + a) * C + c)[inter],
                           offs[2] + ((dp * A + a) * C + c)[inter],
                           offs[3] + (dp * A + a) * E + de])
    cols = np.concatenate([var, var[inter], var[inter], var])
    caps = np.concatenate([getattr(net, name).ravel() for name in TIERS])
    A_ub = coo_matrix((np.concatenate([np.ones(len(rows)), -caps]),
                       (np.concatenate([rows, np.arange(len(caps))]),
                        np.concatenate([cols, np.full(len(caps), nv - 1)]))),
                      shape=(len(caps), nv))
    A_eq = coo_matrix((np.ones(len(k)), (k, var)), shape=(len(s), nv))
    obj = np.zeros(nv)
    obj[-1] = 1.0
    res = linprog(obj, A_ub=A_ub, b_ub=np.zeros(len(caps)), A_eq=A_eq, b_eq=D[s, d],
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.x[-1])


def elephant_bound(net: FatTree, tr: Traffic) -> np.ndarray:
    """(T,) a flow that cannot be split runs at least at w / (its best path's
    capacity) somewhere, whatever the policy."""
    best = net.path_capacity(tr.ps, tr.es, tr.pd, tr.ed).max(axis=(2, 3))
    with np.errstate(divide="ignore"):
        return (tr.w / best).max(axis=1)


# --------------------------------------------------------------------------
# Running experiments
# --------------------------------------------------------------------------

def make_traffic(net: FatTree, m: int, flows, trials: int, seed: int = 0,
                 pattern: str = "uniform", stride: int = 1, ports: str = "ephemeral") -> Traffic:
    """The flow sets of run_fattree: sizes, endpoints and 5-tuples, keyed by
    what they are for, so every policy (and every topology with the same
    dimensions) sees the same flows."""
    dims = (net.pods, net.edges, net.hosts)
    keys = (seed, "ft-flows", flows.label, pattern, stride, *dims, m)
    w = np.stack([flows.sample(rng_for(*keys, t, "w"), m) for t in range(trials)])
    ends = [sample_hosts(rng_for(*keys, t, "hosts"), m, net, pattern, stride) for t in range(trials)]
    headers = np.stack([make_headers(rng_for(*keys, t, "ports", ports), *e, ports=ports)
                        for t, e in enumerate(ends)])
    return Traffic(w, *(np.stack([e[i] for e in ends]) for i in range(6)), headers)


def run_fattree(net: FatTree, m: int, flows, policies, trials: int = 100, seed: int = 0,
                pattern: str = "uniform", stride: int = 1, ports: str = "ephemeral",
                optimum: bool = True, label: str | None = None) -> pd.DataFrame:
    """One row per (policy, trial).

    max_util     utilization of the busiest link
    edge_up_max, agg_up_max, core_down_max, agg_down_max
                 the busiest link of each tier: where the bottleneck sits
    cores_used   fraction of the core switches that carry any traffic (hash
                 polarization shows up here first)
    opt          the splittable LP optimum (if optimum=True), and
    over_opt     max_util / opt, at least 1
    over_lb      max_util / max(opt, elephant bound); just over_opt for the
                 splittable FTESF
    """
    if len(set(p.name for p in policies)) != len(policies):
        raise ValueError("policy names must be distinct")
    label = label or net.label
    tr = make_traffic(net, m, flows, trials, seed, pattern, stride, ports)
    if not net.usable(tr.ps, tr.es, tr.pd, tr.ed).any(axis=(2, 3)).all():
        raise ValueError("some flow has no usable path in this topology")
    opt = (np.array([optimum_util(net, tr.ps[t], tr.es[t], tr.pd[t], tr.ed[t], tr.w[t])
                     for t in range(trials)]) if optimum else np.full(trials, np.nan))
    lb = np.maximum(opt, elephant_bound(net, tr)) if optimum else opt
    dims = (net.pods, net.edges, net.aggs, net.cores)
    frames = []
    for pol in policies:
        if hasattr(pol, "loads"):
            loads = pol.loads(net, tr)
        else:
            draws = np.stack([pol.draw(rng_for(seed, "ft-policy", pol.name, *dims, m, t), m, net)
                              for t in range(trials)])
            loads = link_loads(net, tr, *pol.place(net, tr, draws))
        tiers = tier_utils(net, loads)
        mu = np.max(list(tiers.values()), axis=0)
        frames.append(pd.DataFrame({
            "policy": pol.name, "trial": np.arange(trials), "max_util": mu,
            **{f"{name}_max": u for name, u in tiers.items()},
            "cores_used": (loads["agg_up"].sum(axis=1) > 0).mean(axis=(1, 2)),
            "opt": opt, "over_opt": mu / opt,
            "over_lb": mu / (opt if getattr(pol, "splittable", False) else lb)}))
    df = pd.concat(frames, ignore_index=True)
    for i, (col, val) in enumerate([("topology", label), ("flows", flows.label),
                                    ("pattern", pattern), ("m", m), ("seed", seed)]):
        df.insert(i, col, val)
    return df
