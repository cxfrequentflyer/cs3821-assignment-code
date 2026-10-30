"""Level 2: the leaf-spine model against hand-worked cases and the LP optimum."""
import numpy as np
import pytest

from ecmpsim import Exponential, Pareto, Unit
from ecmpsim.leafspine import (ESF, LSECMP, LSWCMP, LeafSpine, LSComponents, LSConga, LSLocal,
                               LSNash, link_loads, max_util, optimum_util, run_leafspine)

POLICIES = [LSECMP(), LSWCMP(), LSLocal(), LSConga(), LSComponents(), ESF(), LSNash()]


def drill_fig4():
    """DRILL Fig. 4a: 4 leaves, 3 spines, every link 40G (here 1), L0-S0 failed.
    L0 and L3 each send 40G (here 1) to L1, as many small flows."""
    net = LeafSpine.symmetric(4, 3).fail(0, 0)
    src = np.array([0] * 100 + [3] * 100)
    return net, src, np.ones(200, dtype=int), np.full(200, 0.01)


def test_fig4_optimum_and_equal_split():
    net, src, dst, w = drill_fig4()
    # L3 should send 2/3 of its traffic via S0, so every downlink into L1 is at 2/3
    assert optimum_util(net, src, dst, w) == pytest.approx(2 / 3)
    # an equal split puts 1/2 + 1/3 on S1->L1 and S2->L1
    up, down = ESF().loads(net, src[None], dst[None], w[None])
    assert max_util(net, up, down)[0] == pytest.approx(5 / 6)


def test_fig4_symmetric_components():
    net, *_ = drill_fig4()
    comps = net.components()[3][1]                      # L3 -> L1
    got = sorted((sorted(s.tolist()), wt) for s, wt in comps)
    assert got == [([0], 1.0), ([1, 2], 2.0)]           # {P0} weight 1, {P1, P2} weight 2
    assert [sorted(s.tolist()) for s, _ in net.components()[0][1]] == [[1, 2]]


def test_symmetric_fabric_is_one_component_per_pair():
    net = LeafSpine.symmetric(5, 4)
    for a in range(5):
        for b in range(5):
            if a != b:
                assert [s.tolist() for s, _ in net.components()[a][b]] == [[0, 1, 2, 3]]


def test_equal_split_is_optimal_in_a_symmetric_fabric():
    df = run_leafspine(LeafSpine.symmetric(6, 3), 80, Pareto(1.5), [ESF()], trials=30, seed=1)
    assert np.allclose(df.over_opt, 1.0)


@pytest.mark.parametrize("net", [LeafSpine.symmetric(5, 3),
                                 LeafSpine.symmetric(5, 3).fail(0, 0).set_link(2, 1, 0.5)],
                         ids=["symmetric", "asymmetric"])
def test_nothing_beats_the_optimum(net):
    df = run_leafspine(net, 60, Exponential(), POLICIES, trials=25, seed=2)
    assert (df.over_opt >= 1 - 1e-7).all()
    assert (df.over_lb >= 1 - 1e-7).all()
    assert (df.max_util == np.maximum(df.up_max, df.down_max)).all()


def test_failed_links_carry_nothing():
    net = LeafSpine.symmetric(4, 3).fail(1, 2)
    df = run_leafspine(net, 50, Unit(), [LSECMP(), LSLocal()], trials=10, seed=3, optimum=False)
    assert np.isfinite(df.max_util).all()


def test_link_loads_conserve_traffic():
    net = LeafSpine.symmetric(4, 3)
    rng = np.random.default_rng(4)
    src = rng.integers(0, 4, (5, 30))
    dst = (src + rng.integers(1, 4, (5, 30))) % 4
    w, spine = rng.exponential(size=(5, 30)), rng.integers(0, 3, (5, 30))
    up, down = link_loads(net, src, dst, w, spine)
    assert np.allclose(up.sum(axis=(1, 2)), w.sum(axis=1))
    assert np.allclose(down.sum(axis=(1, 2)), w.sum(axis=1))


def test_wcmp_follows_path_capacity():
    net = LeafSpine.symmetric(2, 3).set_link(1, 0, 2.0).set_link(0, 0, 2.0)  # path via S0 is 2x
    src, dst = np.zeros((1, 30_000), int), np.ones((1, 30_000), int)
    w = np.ones((1, 30_000))
    pol = LSWCMP()
    spine = pol.place(net, src, dst, w, pol.draw(np.random.default_rng(5), 30_000, net)[None])
    share = np.bincount(spine[0], minlength=3) / 30_000
    assert share == pytest.approx([0.5, 0.25, 0.25], abs=0.01)


def test_nash_is_an_equilibrium():
    net = LeafSpine.symmetric(5, 3).fail(0, 0)
    rng = np.random.default_rng(6)
    m = 40
    src = rng.integers(0, 5, (3, m))
    dst = (src + rng.integers(1, 5, (3, m))) % 5
    w = rng.exponential(size=(3, m))
    pol = LSNash(LSECMP())
    draws = np.stack([pol.draw(np.random.default_rng(t), m, net) for t in range(3)])
    spine = pol.place(net, src, dst, w, draws)
    up, down = link_loads(net, src, dst, w, spine)
    uu, du = up / np.where(net.up > 0, net.up, np.inf), down / np.where(net.down > 0, net.down, np.inf)
    for t in range(3):
        for j in range(m):
            a, b, s, x = src[t, j], dst[t, j], spine[t, j], w[t, j]
            now = max(uu[t, a, s], du[t, s, b])
            for s2 in np.flatnonzero(net.usable[a, b]):
                if s2 != s:
                    alt = max(uu[t, a, s2] + x / net.up[a, s2], du[t, s2, b] + x / net.down[s2, b])
                    assert alt >= now - 1e-9


def test_conga_beats_local_under_asymmetry():
    # CONGA Fig. 2-3: with a half-capacity downlink, a leaf that only looks at its
    # own uplinks sends too much into the slow link; a path metric does not.
    net = LeafSpine.symmetric(6, 4).set_link(1, 0, 0.25)
    df = run_leafspine(net, 300, Exponential(), [LSLocal(), LSConga()], trials=40, seed=7)
    mean = df.groupby("policy").over_opt.mean()
    assert mean["conga"] < mean["local"]
