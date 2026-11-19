"""Idea 2b: memory buys stability when paths have unequal speeds.

DRILL (SIGCOMM 2017, §3.2.4). Theorem 1: with no memory, DRILL(d, 0) cannot
keep up with every admissible traffic pattern for any d < n. Theorem 2:
DRILL(1, 1) can (100% throughput). The proof of Theorem 1 is a counting
argument that works just as well in a static snapshot: a fast path can only
receive flows that sample it, a fraction of about d/n, so the slow paths are
overloaded however clever the choice among the samples is
(theory.dchoice_util_lower_bound). Memory escapes this by remembering the
fast path.

Paths here have relative speeds; load-aware policies compare utilization
(load / speed). The metric is the busiest utilization over the ideal
total / sum(speeds): 1 is a perfect split.

    python experiments/exp05_speeds.py [--rerun]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ecmpsim import ECMP, DChoice, Exponential, Greedy, run, theory

PROFILES = {
    "1 fast (10x) + 9 slow": np.array([10.0] + [1.0] * 9),
    "4 fast (4x) + 12 slow": np.array([4.0] * 4 + [1.0] * 12),
    "16 equal": np.ones(16),
}
DS = [1, 2, 3, 4, 6]
TRIALS, SEED, PER_PATH = 1000, 3821, 40
CSV = ROOT / "results" / "exp05_speeds.csv"


def simulate() -> pd.DataFrame:
    rows = []
    for label, speeds in PROFILES.items():
        n = len(speeds)
        pols = ([ECMP(), Greedy()] + [DChoice(d, replace=False) for d in DS]
                + [DChoice(d, replace=False, memory=1) for d in DS])
        df = run(n, PER_PATH * n, Exponential(), pols, trials=TRIALS, seed=SEED, speeds=speeds)
        for p in pols:
            g = df[df.policy == p.name].max_over_mean
            d = getattr(p, "d", None)
            mem = getattr(p, "memory", None)
            bound = theory.dchoice_util_lower_bound(speeds, d) if d and not mem else np.nan
            if isinstance(p, ECMP):
                bound = theory.dchoice_util_lower_bound(speeds, 1)
            rows.append({"speeds": label, "n": n, "policy": p.name, "d": d, "memory": mem,
                         "util_over_ideal": g.mean(), "p99": g.quantile(0.99),
                         "counting_bound": bound})
        print(f"  {label} done", flush=True)
    out = pd.DataFrame(rows)
    CSV.parent.mkdir(exist_ok=True)
    out.to_csv(CSV, index=False)
    return out


def report(df):
    print(f"\nBusiest utilization / ideal (1 = perfect), {TRIALS} trials, {PER_PATH} flows per path")
    for label, g in df.groupby("speeds", sort=False):
        print(f"\n{label}")
        print(g[["policy", "util_over_ideal", "p99", "counting_bound"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    report(simulate() if args.rerun or not CSV.exists() else pd.read_csv(CSV))


if __name__ == "__main__":
    main()
