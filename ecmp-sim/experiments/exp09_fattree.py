"""Level 3: a fat-tree, where every flow takes two ECMP decisions.

A k-ary fat-tree (ecmpsim.fattree): the edge switch picks an aggregation
switch, which picks a core; the way down is then fixed. Scenarios (k = 8 by
default: 8 pods, 16 cores, 16 paths between pods):

    symmetric         nothing wrong
    core link down    agg (0,0) lost one of its 4 core uplinks
    edge link down    edge (0,0) lost one of its 4 uplinks (DRILL Fig. 4, one
                      tier up: other pods should now avoid agg 0 for that edge)
    slow core link    one agg-core link at 1/4 capacity (CONGA Fig. 2-3)
    inter-pod only    symmetric, every flow leaves its pod (stresses the cores)

Policies: ideal ECMP hop by hop, each switch's least-utilized uplink (local,
DRILL's information), the least-congested whole path (global, a yardstick no
switch can compute), DRILL's symmetric components, and hop-by-hop equal split
(ESF, splittable). Everything is relative to the splittable LP optimum.

    python experiments/exp09_fattree.py [--rerun] [--k 8] [--trials 100]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from ecmpsim import Empirical, Exponential
from ecmpsim.fattree import (FTECMP, FTESF, FatTree, FTComponents, FTGlobal, FTLocal,
                             run_fattree)

SEED, PER_HOST = 3821, 16
FLOWS = [Exponential(), Empirical.from_csv(ROOT / "workloads" / "websearch.txt")]
POLICIES = [FTECMP(), FTLocal(), FTGlobal(), FTComponents(), FTESF()]
TIERS = ["edge_up_max", "agg_up_max", "core_down_max", "agg_down_max"]


def scenarios(k):
    net = FatTree.kary(k)
    return {
        "symmetric": (net, "uniform"),
        "core link down": (net.fail_core_link(0, 0, 0), "uniform"),
        "edge link down": (net.fail_edge_link(0, 0, 0), "uniform"),
        "slow core link": (net.set_core_link(1, 1, 1, 0.25), "uniform"),
        "inter-pod only": (net, "interpod"),
    }


def simulate(k, trials, csv) -> pd.DataFrame:
    m = PER_HOST * k ** 3 // 4
    parts = []
    for name, (net, pattern) in scenarios(k).items():
        for f in FLOWS:
            parts.append(run_fattree(net, m, f, POLICIES, trials=trials, seed=SEED,
                                     pattern=pattern, label=name))
            print(f"  {name} / {f.label} done", flush=True)
    df = pd.concat(parts, ignore_index=True)
    csv.parent.mkdir(exist_ok=True)
    df.to_csv(csv, index=False)
    return df


def report(df, k):
    order = [p.name for p in POLICIES]
    for f, g in df.groupby("flows", sort=False):
        print(f"\n{f} flow sizes, k = {k}, m = {g.m.iloc[0]}: mean busiest link / splittable "
              "optimum (1 = optimal)")
        print(g.pivot_table(index="topology", columns="policy", values="over_opt", sort=False)
              [order].round(3).to_string())
    e = df[df.flows == FLOWS[0].label]
    print("\nWhere the bottleneck sits: mean busiest link of each tier / optimum, exponential sizes")
    t = e[TIERS].div(e.opt, axis=0).assign(topology=e.topology, policy=e.policy)
    print(t.groupby(["topology", "policy"], sort=False)[TIERS].mean().round(2)
          .rename(columns=lambda c: c[:-4]).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()
    csv = ROOT / "results" / f"exp09_fattree_k{args.k}.csv"
    df = simulate(args.k, args.trials, csv) if args.rerun or not csv.exists() else pd.read_csv(csv)
    report(df, args.k)


if __name__ == "__main__":
    main()
