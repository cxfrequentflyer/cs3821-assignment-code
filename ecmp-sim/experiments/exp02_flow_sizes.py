"""Core questions 2 and 3 (first look): what happens to ECMP's imbalance when
flows have realistic sizes, and how much of it do d choices win back?

    python experiments/exp02_flow_sizes.py                 # n = 16 paths, 4 flows per path
    python experiments/exp02_flow_sizes.py --n 64 --t 16

The headline metric is max_over_lb: the busiest path divided by
max(total/n, largest flow), the load no placement could beat. It is 1 for a
perfect placement, it stays finite under infinite-variance flow sizes (unlike
the raw max load), and it separates "ECMP was unlucky" from "one flow was
simply bigger than a fair share". The d-choice policies here sample distinct
paths, as a switch would.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from ecmpsim import (ECMP, LPT, Bimodal, DChoice, Exponential, Greedy, LogNormal, Pareto, Unit,
                     paired_diff, run)

FLOWS = [Unit(), Exponential(), LogNormal(1.5), Bimodal(0.05, 100), Pareto(1.5)]
POLICIES = [ECMP(), DChoice(2, "count", replace=False), DChoice(2, "bytes", replace=False),
            DChoice(3, "bytes", replace=False), Greedy(), LPT()]
SHORT = {"ecmp": "ecmp", "d2-count-distinct": "d2 count", "d2-bytes-distinct": "d2 bytes",
         "d3-bytes-distinct": "d3 bytes", "greedy": "greedy", "lpt": "lpt (offline)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="paths in the ECMP group")
    ap.add_argument("--t", type=int, default=4, help="flows per path (m = n * t)")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3821)
    args = ap.parse_args()
    n, m = args.n, args.n * args.t

    df = pd.concat([run(n, m, f, POLICIES, trials=args.trials, seed=args.seed) for f in FLOWS],
                   ignore_index=True)
    out = ROOT / "results" / f"exp02_flow_sizes_n{n}_t{args.t}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    order = [f.label for f in FLOWS]
    cols = [SHORT[p.name] for p in POLICIES]
    df["policy"] = df.policy.map(SHORT)
    g = df.groupby(["flows", "policy"], sort=False).max_over_lb
    print(f"\nn = {n} paths, m = {m} flows, {args.trials} trials per row")
    for title, stat in [("median", g.median()), ("99th percentile", g.quantile(0.99))]:
        table = stat.unstack()[cols].reindex(order)
        print(f"\nBusiest path / lower bound, {title} (1.00 = no placement could do better)")
        print(table.round(2).to_string())

    elephant = df[df.policy == "ecmp"].groupby("flows").lb_is_elephant.mean().reindex(order)
    print("\nShare of flow sets where one flow alone exceeds a fair share (total/n):")
    print((100 * elephant).round(1).astype(str).add("%").to_string())

    diff = paired_diff(df.assign(policy=df.policy), "max_over_lb", baseline="ecmp", by=["flows"])
    diff = diff.pivot(index="flows", columns="policy", values="mean")[cols[1:]].reindex(order)
    print("\nPaired change vs ECMP in busiest/lower-bound (mean over identical flow sets):")
    print(diff.round(3).to_string())
    print(f"\nper-trial results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
