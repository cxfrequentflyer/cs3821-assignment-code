"""Level 3: the fat-tree against hand-worked cases, a slow reference, the LP
optimum, and the hash-polarization arithmetic."""
import numpy as np
import pytest

import reference as ref
from ecmpsim import Exponential, Hash, Pareto, Unit
from ecmpsim.fattree import (FTECMP, FTESF, FatTree, FTComponents, FTGlobal, FTHashECMP, FTLocal,
                             Traffic, link_loads, make_traffic, max_util, optimum_util,
                             run_fattree, tier_utils)

POLICIES = [FTECMP(), FTHashECMP("crc16"), FTHashECMP("crc16", "crc16-arc", seeds="xor"),
            FTLocal(), FTGlobal(), FTComponents(), FTESF()]


def traffic(flows):
    """One trial from a list of (ps, es, pd, ed, w)."""
    cols = [np.array([[f[i] for f in flows]]) for i in range(5)]
    zeros = np.zeros_like(cols[0])
    return Traffic(cols[4].astype(float), cols[0], cols[1], zeros, cols[2], cols[3], zeros,
                   np.zeros((1, len(flows), 13), dtype=np.uint8))


def test_kary_dimensions_and_path_counts():
    net = FatTree.kary(6)
    assert (net.pods, net.edges, net.aggs, net.cores, net.hosts) == (6, 3, 3, 3, 3)
    assert net.usable(0, 0, 1, 2).sum() == 9            # (k/2)^2 paths between pods
    intra = net.usable(2, 0, 2, 1)
    assert intra.sum() == 3 and intra[:, 0].all()       # k/2 inside a pod, written (a, 0)
    assert net.path_capacity(0, 0, 1, 2).tolist() == [[1.0] * 3] * 3
    with pytest.raises(ValueError):
        FatTree.kary(5)


def test_failures_remove_the_right_paths():
    net = FatTree.kary(4).fail_core_link(0, 0, 0)
    assert net.usable(0, 1, 2, 0).tolist() == [[False, True], [True, True]]   # out of pod 0
    assert net.usable(2, 0, 0, 1).tolist() == [[False, True], [True, True]]   # and into it
    assert net.usable(1, 0, 2, 0).all()
    assert net.usable(0, 0, 0, 1)[:, 0].all()                                 # intra-pod unaffected
    net = FatTree.kary(4).fail_edge_link(1, 0, 1)
    assert net.usable(1, 0, 3, 0).tolist() == [[True, True], [False, False]]
    assert net.usable(3, 0, 1, 0).tolist() == [[True, True], [False, False]]
    assert net.usable(1, 0, 1, 1).tolist() == [[True, False], [False, False]]


def test_link_loads_match_the_reference():
    net = FatTree.symmetric(3, 2, 3, 2)
    tr = make_traffic(net, 80, Exponential(), trials=4, seed=1)
    rng = np.random.default_rng(0)
    a, c = rng.integers(0, 3, (4, 80)), rng.integers(0, 2, (4, 80))
    loads = link_loads(net, tr, a, c)
    for t in range(4):
        flows = list(zip(tr.ps[t], tr.es[t], tr.pd[t], tr.ed[t], tr.w[t]))
        expected = ref.fattree_loads(flows, list(zip(a[t], c[t])))
        for (tier, i, j, k), value in expected.items():
            assert loads[tier][t, i, j, k] == pytest.approx(value)
        assert sum(x[t].sum() for x in loads.values()) == pytest.approx(sum(expected.values()))
    # every flow crosses one edge uplink and one agg downlink; only inter-pod ones a core
    assert np.allclose(loads["edge_up"].sum(axis=(1, 2, 3)), tr.w.sum(axis=1))
    assert np.allclose(loads["agg_up"].sum(axis=(1, 2, 3)), (tr.w * tr.inter).sum(axis=1))
    assert np.allclose(loads["core_down"].sum(axis=(1, 2, 3)), (tr.w * tr.inter).sum(axis=1))


def test_hand_worked_optimum():
    net = FatTree.kary(4)
    one = [(0, 0, 1, 0, 1.0)]
    # one unit between two edge switches: two edge uplinks, so 1/2 each
    assert optimum_util(net, [0], [0], [1], [0], [1.0]) == pytest.approx(0.5)
    assert optimum_util(net.fail_core_link(0, 0, 0), [0], [0], [1], [0], [1.0]) == pytest.approx(0.5)
    assert optimum_util(net.fail_edge_link(0, 0, 0), [0], [0], [1], [0], [1.0]) == pytest.approx(1.0)
    assert max_util(net, FTESF().loads(net, traffic(one)))[0] == pytest.approx(0.5)


def test_equal_split_is_fooled_by_a_remote_failure():
    # DRILL Fig. 4, one tier further up. Edge (0,0) lost its link to agg 0, so its
    # unit of traffic to edge (1,0) all arrives through agg 1 of pod 1. Edge (2,0)
    # also sends one unit there: the optimum sends all of it via agg 0 (every link
    # at 1), but an equal split puts half on the agg 1 downlink that is already full.
    net = FatTree.kary(4).fail_edge_link(0, 0, 0)
    flows = [(0, 0, 1, 0, 0.01)] * 100 + [(2, 0, 1, 0, 0.01)] * 100
    tr = traffic(flows)
    assert optimum_util(net, tr.ps[0], tr.es[0], tr.pd[0], tr.ed[0], tr.w[0]) == pytest.approx(1.0)
    tiers = tier_utils(net, FTESF().loads(net, tr))
    assert tiers["agg_down"][0] == pytest.approx(1.5)
    assert tiers["edge_up"][0] == pytest.approx(1.0)


def test_equal_split_is_optimal_in_a_symmetric_fat_tree():
    for net in (FatTree.kary(4), FatTree.symmetric(3, 2, 3, 2)):
        df = run_fattree(net, 150, Pareto(1.5), [FTESF()], trials=15, seed=1)
        assert np.allclose(df.over_opt, 1.0)


@pytest.mark.parametrize("net", [
    FatTree.kary(4),
    FatTree.kary(4).fail_core_link(0, 0, 0).set_edge_link(2, 1, 0, 0.5),
    FatTree.symmetric(3, 2, 3, 2).fail_edge_link(1, 0, 2).set_core_link(0, 1, 1, 2.0),
], ids=["symmetric", "asymmetric", "odd-shape"])
def test_nothing_beats_the_optimum(net):
    df = run_fattree(net, 120, Exponential(), POLICIES, trials=12, seed=2)
    assert (df.over_opt >= 1 - 1e-7).all()
    assert (df.over_lb >= 1 - 1e-7).all()
    tiers = df[["edge_up_max", "agg_up_max", "core_down_max", "agg_down_max"]]
    assert np.isfinite(df.max_util).all()               # nothing was sent down a failed link
    assert (df.max_util == tiers.max(axis=1)).all()


def test_symmetric_components():
    net = FatTree.kary(4)
    for pair in [(0, 0, 1, 1), (2, 1, 2, 0)]:
        (mask, weight), = net.components(*pair)
        assert (mask == net.usable(*pair)).all() and weight == mask.sum()
    net = net.fail_core_link(0, 0, 0)
    # between two healthy pods, the path through core (0,0) is special: that core
    # cannot reach pod 0, so its links serve a different set of pairs
    got = sorted((m.astype(int).tolist(), wt) for m, wt in net.components(1, 0, 2, 0))
    assert got == [([[0, 1], [1, 1]], 3.0), ([[1, 0], [0, 0]], 1.0)]
    (mask, weight), = net.components(0, 0, 2, 0)        # from pod 0 that path does not exist
    assert mask.astype(int).tolist() == [[0, 1], [1, 1]] and weight == 3.0
    (mask, weight), = net.components(1, 0, 1, 1)        # inside a pod nothing changed
    assert mask.astype(int).tolist() == [[1, 0], [1, 0]] and weight == 2.0


def test_ideal_ecmp_is_uniform_per_hop_not_per_path():
    # with core link (0,0,0) down, a flow from pod 0 picks agg 0 or 1 with
    # probability 1/2, and agg 0 has one core left: paths get 1/2, 1/4, 1/4
    net = FatTree.kary(4).fail_core_link(0, 0, 0)
    m = 30_000
    tr = traffic([(0, 0, 1, 0, 1.0)] * m)
    pol = FTECMP()
    a, c = pol.place(net, tr, pol.draw(np.random.default_rng(3), m, net)[None])
    share = np.bincount(a[0] * 2 + c[0], minlength=4) / m
    assert share == pytest.approx([0, 0.5, 0.25, 0.25], abs=0.012)


def place_hash(pol, net, m=4000, seed=4, pattern="interpod"):
    tr = make_traffic(net, m, Unit(), trials=2, seed=seed, pattern=pattern)
    draws = np.stack([pol.draw(np.random.default_rng(t), m, net) for t in range(2)])
    return tr, pol.place(net, tr, draws)


def test_the_same_hash_at_both_tiers_polarizes():
    net = FatTree.kary(8)
    tr, (a, c) = place_hash(FTHashECMP("crc16"), net)
    assert (a == c).all()                               # every agg sends everything to one core
    assert len(np.unique(a)) == 4                       # while the edge tier is spread out
    df = run_fattree(net, 2000, Unit(), [FTHashECMP("crc16"), FTECMP()], trials=5, seed=4,
                     pattern="interpod", optimum=False)
    used = df.groupby("policy").cores_used.mean()
    assert used["hash(crc16)"] == pytest.approx(1 / 4) and used["ecmp"] == 1.0


def test_xor_seeds_do_not_cure_polarization_but_other_bits_do():
    net = FatTree.kary(8)
    tr, (a, c) = place_hash(FTHashECMP("crc16", seeds="xor"), net)
    for t in range(2):
        # a seed permutes the buckets, it does not mix them: the flows one edge
        # switch hands to one agg switch still all leave on ONE uplink, so an
        # edge switch reaches 4 of its 16 paths (which 4 differs per switch)
        hop = (tr.ps[t] * net.edges + tr.es[t]) * net.aggs + a[t]
        assert all(len(np.unique(c[t][hop == h])) == 1 for h in np.unique(hop))
        assert len(np.unique(a[t] * 4 + c[t])) == 16    # all cores are in use, though
    for cure in (FTHashECMP("crc16", "crc16-arc"), FTHashECMP("crc16", Hash("crc16", shift=8))):
        tr, (a, c) = place_hash(cure, net)
        switch = tr.ps[0] * net.aggs + a[0]
        assert all(len(np.unique(c[0][switch == s])) == 4 for s in np.unique(switch))
        counts = np.bincount(a[0] * 4 + c[0], minlength=16)
        assert counts.min() > 0.8 * 4000 / 16           # all 16 paths, about equally


def test_hash_ecmp_only_uses_usable_paths():
    net = FatTree.kary(4).fail_core_link(0, 0, 0).fail_edge_link(1, 1, 0)
    for pol in (FTHashECMP("crc16"), FTHashECMP("crc32", "xor16", seeds="xor")):
        tr, (a, c) = place_hash(pol, net, m=3000, pattern="uniform")
        ok = net.usable(tr.ps, tr.es, tr.pd, tr.ed)
        t, j = np.indices(a.shape)
        assert ok[t, j, a, c].all()
        assert (c[~tr.inter] == 0).all()


def test_names_and_policies_are_paired():
    assert FTHashECMP("crc16").name == "hash(crc16)"
    assert FTHashECMP("crc16", Hash("crc16", 8), "xor").name == "hash(crc16,crc16>>8,seeded)"
    net = FatTree.kary(4)
    both = run_fattree(net, 100, Exponential(), [FTECMP(), FTLocal()], trials=8, seed=5,
                       optimum=False)
    alone = run_fattree(net, 100, Exponential(), [FTLocal()], trials=8, seed=5, optimum=False)
    assert (both[both.policy == "local"].max_util.values == alone.max_util.values).all()
    with pytest.raises(ValueError):
        run_fattree(net, 10, Unit(), [FTECMP(), FTECMP()], trials=2)


def test_global_information_helps_under_asymmetry():
    net = FatTree.kary(4).set_core_link(1, 0, 0, 0.25)
    df = run_fattree(net, 400, Exponential(), [FTECMP(), FTLocal(), FTGlobal()], trials=30, seed=6)
    mean = df.groupby("policy").over_opt.mean()
    assert mean["global"] < mean["local"] < mean["ecmp"]
