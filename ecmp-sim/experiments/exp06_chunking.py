"""Idea 3: place traffic in smaller pieces (connects core questions 2 and 3).

ECMP places whole flows, CONGA flowlets, Presto 64KB flowcells, DRILL single
packets. Splitting each flow into chunks of size c (in mean flow sizes)
replaces one heavy ball with many light ones and lowers the CV of the ball
sizes, which is what CONGA's Theorem 2 says drives imbalance. The dashed
line is that bound evaluated for the chunks (their count and CV^2).

Policies: ECMP on chunks (per-packet Random when c is tiny), Spray (Presto:
each flow round-robins its chunks from a random start) and DRILL-style
DChoice(2) on chunks. Smaller chunks balance better; what they cost in
reordering is not modeled here, and is the question for the ns-3 track.

    python experiments/exp06_chunking.py [--rerun] [--n 16] [--t 4]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd

from ecmpsim import (ECMP, DChoice, Exponential, LogNormal, Pareto, Spray, chunk_flows, rng_for,
                     run, theory)

CHUNKS = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]
FLOWS = [Exponential(), LogNormal(1.5), Pareto(1.5)]
POLICIES = [ECMP(), Spray(), DChoice(2, replace=False)]
TRIALS, SEED = 500, 3821
SHORT = {"ecmp": "ecmp", "spray": "spray", "d2-bytes-distinct": "d = 2"}


def chunk_bound(f, n, m, c):
    """Median over trials of CONGA's bound with the chunks as the balls: their
    number and CV^2. A heuristic: chunks of one flow are not independent
    draws (most are exactly c), which the theorem assumes."""
    vals = []
    for t in range(TRIALS):
        sizes, _, _ = chunk_flows(f.sample(rng_for(SEED, "flows", f.label, m, t), m), c)
        cv2 = sizes.var() / sizes.mean() ** 2
        vals.append(theory.conga_imbalance_bound(len(sizes), n, cv2))
    return float(np.median(vals))


def simulate(n, m, csv) -> pd.DataFrame:
    rows = []
    for f in FLOWS:
        whole = run(n, m, f, POLICIES, trials=TRIALS, seed=SEED)
        for pol, g in whole.groupby("policy", sort=False):
            rows.append({"flows": f.label, "chunk": np.inf, "policy": pol,
                         "imbalance": g.imbalance.median(), "max_over_mean": g.max_over_mean.median(),
                         "chunks": m, "bound": theory.conga_imbalance_bound(m, n, f.cv2)})
        for c in CHUNKS:
            df = run(n, m, f, POLICIES, trials=TRIALS, seed=SEED, chunk=c)
            bound = chunk_bound(f, n, m, c)
            for pol, g in df.groupby("policy", sort=False):
                rows.append({"flows": f.label, "chunk": c, "policy": pol,
                             "imbalance": g.imbalance.median(),
                             "max_over_mean": g.max_over_mean.median(),
                             "chunks": g.chunks.mean(), "bound": bound})
        print(f"  {f.label} done", flush=True)
    out = pd.DataFrame(rows)
    csv.parent.mkdir(exist_ok=True)
    out.to_csv(csv, index=False)
    return out


def report(df, n, m):
    print(f"\nMedian imbalance (max - min)/mean, n = {n}, m = {m} flows, {TRIALS} trials "
          "(chunk = inf: whole flows)")
    for label, g in df.groupby("flows", sort=False):
        t = g.pivot(index="chunk", columns="policy", values="imbalance")[[p.name for p in POLICIES]]
        t["chunk bound"] = g.groupby("chunk").bound.first()
        t["chunks/trial"] = g.groupby("chunk").chunks.first().round(0)
        print(f"\n{label}")
        print(t.sort_index(ascending=False).round(3).to_string())


def plot(df, n, m, png):
    import plotstyle as ps
    plt = ps.setup()
    fig, axes = plt.subplots(1, len(FLOWS), figsize=(13, 4.4), facecolor=ps.SURFACE, sharey=True)
    for ax, (label, g) in zip(axes, df.groupby("flows", sort=False)):
        g = g[np.isfinite(g.chunk)]
        for i, p in enumerate(POLICIES):            # largest chunk first: labels land on the right
            s = g[g.policy == p.name].sort_values("chunk", ascending=False)
            ps.line(ax, s.chunk, s.imbalance, i, SHORT[p.name])
        b = g[g.policy == POLICIES[0].name].sort_values("chunk", ascending=False)
        ps.line(ax, b.chunk, b.bound, 3, "bound", dashed=True, marker=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(CHUNKS[0] * 1.5, CHUNKS[-1] / 5)          # reversed: smaller to the right
        ps.style(ax, label, "chunk size c (mean flow = 1), smaller to the right", "")
    axes[0].set_ylabel("median (max - min) / mean load")
    fig.suptitle(f"Splitting flows into chunks, n = {n}, m = {m}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=160, facecolor=ps.SURFACE)
    print(f"\nfigure: {png.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--t", type=int, default=4, help="flows per path")
    args = ap.parse_args()
    n, m = args.n, args.n * args.t
    csv = ROOT / "results" / f"exp06_chunking_n{n}_t{args.t}.csv"
    df = simulate(n, m, csv) if args.rerun or not csv.exists() else pd.read_csv(csv)
    report(df, n, m)
    plot(df, n, m, csv.with_suffix(".png"))


if __name__ == "__main__":
    main()
