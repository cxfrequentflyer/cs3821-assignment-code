"""Idea 5: two hops, asymmetry, and CONGA's Price of Anarchy (level 2).

A leaf-spine fabric (ecmpsim.leafspine): each flow picks a spine, loading an
uplink and a downlink. Scenarios:

    symmetric       9 leaves x 4 spines, pFabric's 144-host fabric scaled down
    1 failed link   the same with leaf 0 - spine 0 down (DRILL Fig. 4, CONGA Fig. 16)
    slow link       leaf 1 - spine 1 at 1/4 capacity (CONGA Fig. 2-3)
    striping        8 x 8 with leaf i double-linked to spines i and i+1
                    (DRILL Fig. 13's imbalanced striping)
    stride          symmetric fabric, every leaf sends to the next one (DRILL Table 1)

Policies: ECMP, WCMP, local (uplink only), CONGA-like (path max), DRILL's
symmetric components, ESF (splittable equal split), and the Nash equilibrium
of CONGA's bottleneck game reached from CONGA's and from ECMP's placements.
Everything is relative to the splittable LP optimum (over_opt >= 1).
CONGA's Theorem 1: a (splittable) Nash flow is within 2x of the optimum.
Our flows are unsplittable, so over_opt can exceed 2 on an unlucky elephant;
over_lb also accounts for that.

    python experiments/exp08_leafspine.py [--rerun] [--trials 200]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ecmpsim import Empirical, Exponential
from ecmpsim.leafspine import (ESF, LSECMP, LSWCMP, LeafSpine, LSComponents, LSConga, LSLocal,
                               LSNash, run_leafspine)

M, SEED = 300, 3821
FLOWS = [Exponential(), Empirical.from_csv(ROOT / "workloads" / "websearch.txt")]
POLICIES = [LSECMP(), LSWCMP(), LSLocal(), LSConga(), LSComponents(), ESF(),
            LSNash(LSConga()), LSNash(LSECMP())]
CSV = ROOT / "results" / "exp08_leafspine.csv"


def striping(leaves=8):
    up = np.ones((leaves, leaves))
    for i in range(leaves):
        up[i, i] = up[i, (i + 1) % leaves] = 2.0
    return LeafSpine(up, up.T.copy(), "striping 8x8")


SCENARIOS = {
    "symmetric": (LeafSpine.symmetric(9, 4), "uniform"),
    "1 failed link": (LeafSpine.symmetric(9, 4).fail(0, 0), "uniform"),
    "slow link": (LeafSpine.symmetric(9, 4).set_link(1, 1, 0.25), "uniform"),
    "striping": (striping(), "uniform"),
    "stride": (LeafSpine.symmetric(9, 4), "stride"),
}


def simulate(trials) -> pd.DataFrame:
    parts = []
    for name, (net, pattern) in SCENARIOS.items():
        for f in FLOWS:
            parts.append(run_leafspine(net, M, f, POLICIES, trials=trials, seed=SEED,
                                       pattern=pattern, label=name))
            print(f"  {name} / {f.label} done", flush=True)
    df = pd.concat(parts, ignore_index=True)
    CSV.parent.mkdir(exist_ok=True)
    df.to_csv(CSV, index=False)
    return df


def report(df):
    order = [p.name for p in POLICIES]
    for f, g in df.groupby("flows", sort=False):
        print(f"\n{f} flow sizes, m = {M}: mean busiest link / splittable optimum (1 = optimal)")
        print(g.pivot_table(index="topology", columns="policy", values="over_opt", sort=False)
              [order].round(3).to_string())
    print("\nWhere the bottleneck sits (mean downlink max / uplink max), exponential sizes:")
    e = df[df.flows == FLOWS[0].label]
    print(e.assign(r=e.down_max / e.up_max)
          .pivot_table(index="topology", columns="policy", values="r", sort=False)[order]
          .round(2).to_string())
    nash = df[df.policy.str.startswith("nash")]
    print("\nPrice of anarchy, observed: worst Nash / optimum over all trials "
          "(CONGA Theorem 1 bounds the splittable game by 2)")
    print(nash.groupby(["topology", "flows", "policy"], sort=False)[["over_opt", "over_lb"]]
          .max().round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    report(simulate(args.trials) if args.rerun or not CSV.exists() else pd.read_csv(CSV))


if __name__ == "__main__":
    main()
