"""Idea 4: the pFabric / CONGA workloads in the balls-into-bins model.

Web search (DCTCP) and data mining (VL2) flow-size CDFs, from workloads/.
pFabric (§5.4) finds data mining *easier*, although it is the more skewed
one: "it is less likely that multiple large flows are dynamically active at
the same time". CONGA's Theorem 2 points the other way: data mining has a far
larger CV^2, so a bigger imbalance bound, and CONGA §6.2 says that ECMP does
worse on it. The two are measuring different things: pFabric the completion
time of the many small flows (dynamic), CONGA the bytes on each link. This
experiment measures the static side: per-link imbalance under ECMP, d
choices, greedy and the offline LPT, and how much 64KB chunks (Presto's
flowcells) recover.

    python experiments/exp07_workloads.py [--rerun] [--n 16]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from ecmpsim import ECMP, LPT, DChoice, Empirical, Greedy, Spray, run, theory

WORKLOADS = [Empirical.from_csv(ROOT / "workloads" / f"{w}.txt") for w in ("websearch", "datamining")]
PKT = 1460                                   # bytes per packet in the CDF files
FLOWCELL = 64 * 1024                         # Presto's chunk, bytes
LOADS = [4, 16, 64]                          # flows per path
TRIALS, SEED = 500, 3821
POLICIES = [ECMP(), DChoice(2, replace=False), Greedy(), LPT()]


def simulate(n, csv) -> pd.DataFrame:
    rows = []
    for f in WORKLOADS:
        c = FLOWCELL / (f.raw_mean * PKT)    # a flowcell in units of the mean flow
        for t in LOADS:
            m = n * t
            parts = [run(n, m, f, POLICIES, trials=TRIALS, seed=SEED).assign(granularity="flow")]
            if t <= 16:                      # chunked runs get large: tens of thousands of chunks
                parts.append(run(n, m, f, [ECMP(), Spray()], trials=TRIALS, seed=SEED, chunk=c)
                             .assign(granularity="64KB"))
            for df in parts:
                for pol, g in df.groupby("policy", sort=False):
                    rows.append({"flows": f.label, "cv2": f.cv2, "n": n, "m": m, "t": t,
                                 "granularity": g.granularity.iloc[0], "policy": pol,
                                 "imbalance_median": g.imbalance.median(),
                                 "imbalance_mean": g.imbalance.mean(),
                                 "max_over_lb_median": g.max_over_lb.median(),
                                 "max_over_mean_median": g.max_over_mean.median(),
                                 "elephant_share": g.lb_is_elephant.mean(),
                                 "conga_bound": theory.conga_imbalance_bound(m, n, f.cv2)})
            print(f"  {f.label} t={t} done", flush=True)
    out = pd.DataFrame(rows)
    csv.parent.mkdir(exist_ok=True)
    out.to_csv(csv, index=False)
    return out


def report(df):
    print("\nCV^2 of flow sizes:", {f.label: round(f.cv2, 1) for f in WORKLOADS})
    print("Mean flow size:", {f.label: f"{f.raw_mean * PKT / 1e6:.2f} MB" for f in WORKLOADS})
    print(f"\nWhole flows: mean imbalance vs CONGA's bound, {TRIALS} trials")
    flow = df[(df.granularity == "flow") & (df.policy == "ecmp")]
    print(flow[["flows", "t", "imbalance_mean", "conga_bound", "elephant_share"]]
          .round(3).to_string(index=False))
    print("\nMedian busiest path / lower bound (1 = no placement could do better)")
    t = df.assign(col=df.policy + " / " + df.granularity).pivot_table(
        index=["flows", "t"], columns="col", values="max_over_lb_median")
    print(t.round(3).to_string())
    print("\nMedian busiest path / mean load (comparable across granularities)")
    t = df.assign(col=df.policy + " / " + df.granularity).pivot_table(
        index=["flows", "t"], columns="col", values="max_over_mean_median")
    print(t.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    csv = ROOT / "results" / f"exp07_workloads_n{args.n}.csv"
    report(simulate(args.n, csv) if args.rerun or not csv.exists() else pd.read_csv(csv))


if __name__ == "__main__":
    main()
