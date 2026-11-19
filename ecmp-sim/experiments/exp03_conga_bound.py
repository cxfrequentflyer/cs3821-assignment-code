"""Idea 1: test CONGA's imbalance theorem (core question 2).

CONGA's Theorem 2 bounds the imbalance chi = (busiest - least loaded path) /
(expected load per path) under ECMP by sqrt(8 n ln n (1 + CV^2) / m), assuming
light-tailed flow sizes. We check where it holds, how loose it is, and what
replaces it when the variance is infinite: the generalized-CLT decay
m^-(1 - 1/alpha) instead of m^-1/2 (theory.imbalance_decay_exponent).

    python experiments/exp03_conga_bound.py            # simulate (if needed) + report + plot
    python experiments/exp03_conga_bound.py --rerun

Medians, not means, for the decay fits: under Pareto(alpha < 2) the mean
imbalance is dominated by rare giant flows and its sample mean is unstable.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd

from ecmpsim import ECMP, Exponential, LogNormal, Pareto, Unit, run, theory

N = 16
MS = [256, 1024, 4096, 16384]
FLOWS = [Unit(), Exponential(), LogNormal(1.5), Pareto(2.5), Pareto(1.8), Pareto(1.5)]
TRIALS, SEED = 1000, 3821
CSV = ROOT / "results" / "exp03_conga_bound.csv"
PNG = ROOT / "results" / "exp03_conga_bound.png"


def simulate() -> pd.DataFrame:
    rows = []
    for f in FLOWS:
        for m in MS:
            x = run(N, m, f, [ECMP()], trials=TRIALS, seed=SEED).imbalance
            rows.append({"flows": f.label, "alpha": getattr(f, "alpha", np.nan), "cv2": f.cv2,
                         "n": N, "m": m, "mean": x.mean(), "median": x.median(),
                         "p99": x.quantile(0.99),
                         "bound": theory.conga_imbalance_bound(m, N, f.cv2)})
        print(f"  {f.label} done", flush=True)
    df = pd.DataFrame(rows)
    CSV.parent.mkdir(exist_ok=True)
    df.to_csv(CSV, index=False)
    return df


def report(df: pd.DataFrame) -> None:
    df = df.copy()
    df["mean/bound"] = (df["mean"] / df.bound).where(np.isfinite(df.bound))
    print(f"\nECMP imbalance (max - min) / (m/n), n = {N}, {TRIALS} trials per row")
    print(df[["flows", "m", "mean", "median", "p99", "bound", "mean/bound"]]
          .round(3).to_string(index=False))
    print("\nDecay with m: fitted log-log slope of the median vs the prediction")
    for label, g in df.groupby("flows", sort=False):
        alpha = g.alpha.iloc[0]
        pred = theory.imbalance_decay_exponent(None if np.isnan(alpha) else alpha)
        print(f"  {label:18s} fitted {theory.fitted_exponent(g.m, g['median']):+.3f}"
              f"   predicted {pred:+.3f}")


def plot(df: pd.DataFrame) -> None:
    import plotstyle as ps
    plt = ps.setup()
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=ps.SURFACE)
    for i, (label, g) in enumerate(df.groupby("flows", sort=False)):
        if np.isfinite(g.bound).all():          # the bound is on the mean
            ps.line(ax, g.m, g["mean"], i, label)
            ps.line(ax, g.m, g.bound, i, None, dashed=True, marker=False)
        else:                                    # infinite variance: the mean is unstable
            ps.line(ax, g.m, g["median"], i, f"{label}, median")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(MS, [str(m) for m in MS])
    ps.style(ax, f"ECMP imbalance vs flows, n = {N}: mean (solid), CONGA bound (dashed)",
             "flows m", "(max - min) / mean load")
    ax.set_xlim(right=MS[-1] * 4)
    fig.tight_layout()
    fig.savefig(PNG, dpi=160, facecolor=ps.SURFACE)
    print(f"\nfigure: {PNG.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    df = simulate() if args.rerun or not CSV.exists() else pd.read_csv(CSV)
    report(df)
    plot(df)


if __name__ == "__main__":
    main()
