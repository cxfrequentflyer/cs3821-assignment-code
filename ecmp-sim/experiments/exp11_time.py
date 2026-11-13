"""Time: flows arrive and leave, and the load is a queueing process.

ecmpsim.dynamic: flows arrive as a Poisson process at offered load rho, each
path is a FIFO queue draining at its speed, and policies compare what is
still queued. Four parts:

  A  check: with ECMP every path is an M/G/1 queue, so the mean wait must
     match Pollaczek-Khinchine, rho (1 + CV^2) / (2 (1 - rho))
  B  waiting time against load for ECMP, power of two choices, DRILL(1, 1),
     DRILL(2, 1) and greedy (n = 16), exponential and web-search sizes
  C  herding over time: the same rules on 16 forwarding engines that share a
     stale snapshot (DRILL §3.2.3; exp04 is the static version)
  D  stability with unequal path speeds: one path 10x faster than the other
     nine. DRILL Theorem 1: memoryless d < n choices cannot keep up below
     rho = 1; Theorem 2: one unit of memory can. `growth` is the share of the
     offered traffic that piles up for ever, against the counting bound
     (theory.dchoice_pileup); exp05 is the static version

    python experiments/exp11_time.py [--rerun] [--arrivals 60000] [--trials 20]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ecmpsim import (ECMP, DChoice, Empirical, Exponential, Greedy, LogNormal, RoundRobin, Unit,
                     run_dynamic, summarize_dynamic, theory)

SEED, N = 3821, 16
WEB = Empirical.from_csv(ROOT / "workloads" / "websearch.txt")
LOADS = [0.5, 0.7, 0.8, 0.9, 0.95]
POLICIES = [ECMP(), DChoice(2, replace=False), DChoice(1, replace=False, memory=1),
            DChoice(2, replace=False, memory=1), Greedy()]
SHORT = {"ecmp": "ECMP", "d2-bytes-distinct": "d = 2", "d1-bytes-distinct-mem1": "DRILL(1,1)",
         "d2-bytes-distinct-mem1": "DRILL(2,1)", "greedy": "greedy"}
ENGINES = 16
HERD = [DChoice(2, replace=False, engines=ENGINES), DChoice(2, replace=False, memory=1, engines=ENGINES),
        DChoice(4, replace=False, engines=ENGINES), DChoice(1, replace=False, memory=4, engines=ENGINES),
        Greedy(engines=ENGINES), Greedy(engines=ENGINES, ties="random"), RoundRobin(engines=ENGINES)]
SPEEDS = np.array([10.0] + [1.0] * 9)
SPEED_LOADS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SPEED_POLICIES = [ECMP(), DChoice(2, replace=False), DChoice(4, replace=False),
                  DChoice(1, replace=False, memory=1), DChoice(2, replace=False, memory=1)]
OUT = ROOT / "results"


def simulate(arrivals, trials) -> dict:
    kw = dict(arrivals=arrivals, trials=trials, seed=SEED)
    a = pd.concat([run_dynamic(4, rho, f, [ECMP()], **kw).assign(cv2=f.cv2)
                   for f in (Unit(), Exponential(), LogNormal(1.0)) for rho in (0.5, 0.8, 0.9)])
    print("  A done", flush=True)
    b = pd.concat([run_dynamic(N, rho, f, POLICIES, **kw) for f in (Exponential(), WEB)
                   for rho in LOADS])
    print("  B done", flush=True)
    c = pd.concat([run_dynamic(N, rho, Exponential(), [ECMP()] + HERD, **kw) for rho in (0.7, 0.9)])
    print("  C done", flush=True)
    d = pd.concat([run_dynamic(len(SPEEDS), rho, Exponential(), SPEED_POLICIES, speeds=SPEEDS, **kw)
                   for rho in SPEED_LOADS])
    print("  D done", flush=True)
    parts = {"a": a, "b": b, "c": c, "d": d}
    OUT.mkdir(exist_ok=True)
    for name, df in parts.items():
        df.to_csv(OUT / f"exp11_time_{name}.csv", index=False)
    return parts


def report(parts):
    a = parts["a"]
    s = summarize_dynamic(a, metrics=["wait_mean", "idle_frac"], by=["flows", "cv2", "load"])
    s["theory"] = [theory.mg1_wait(rho, cv2) for rho, cv2 in zip(s.load, s.cv2)]
    print(f"\nA. ECMP, n = 4: mean wait against M/G/1 (Pollaczek-Khinchine), "
          f"{a.arrivals.iloc[0]} arrivals x {a.trial.nunique()} runs")
    print(s[["flows", "load", "wait_mean", "wait_mean_ci95", "theory", "idle_frac"]]
          .round(3).to_string(index=False))

    b = parts["b"]
    order = [p.name for p in POLICIES]
    for f, g in b.groupby("flows", sort=False):
        print(f"\nB. n = {N}, {f} sizes: mean wait (in mean flow transmission times)")
        print(g.pivot_table(index="load", columns="policy", values="wait_mean", sort=False)[order]
              .rename(columns=SHORT).round(3).to_string())
        print("   99th percentile wait")
        print(g.pivot_table(index="load", columns="policy", values="wait_p99", sort=False)[order]
              .rename(columns=SHORT).round(2).to_string())

    c = parts["c"]
    print(f"\nC. n = {N}, {ENGINES} engines on one stale snapshot, exponential sizes: mean wait "
          "(ECMP and section B's one-engine numbers for scale)")
    print(c.pivot_table(index="policy", columns="load", values="wait_mean", sort=False)
          .round(3).to_string())

    d = parts["d"]
    print("\nD. speeds = one 10x path + nine 1x paths: share of the offered traffic piling up "
          "(growth); 0 = stable")
    t = d.pivot_table(index="load", columns="policy", values="growth", sort=False)
    t = t[[p.name for p in SPEED_POLICIES]].round(3)
    for dd in (1, 2, 4):
        t[f"bound d={dd}"] = [round(theory.dchoice_pileup(SPEEDS, dd, rho), 3) for rho in t.index]
    print(t.to_string())
    print("   memoryless rules can only be stable up to load "
          + ", ".join(f"{theory.dchoice_stable_load(SPEEDS, dd):.3f} (d = {dd})" for dd in (1, 2, 4)))
    print("   mean wait where stable")
    w = d.pivot_table(index="load", columns="policy", values="wait_mean", sort=False)
    g = d.pivot_table(index="load", columns="policy", values="growth", sort=False)
    print(w.where(g.abs() < 0.005)[[p.name for p in SPEED_POLICIES]].round(3).to_string(na_rep="-"))


def plot(parts, png):
    import plotstyle as ps
    plt = ps.setup()
    b = parts["b"]
    flows = list(b.flows.unique())
    fig, axes = plt.subplots(1, len(flows), figsize=(12, 4.4), facecolor=ps.SURFACE)
    for ax, f in zip(axes, flows):
        g = b[b.flows == f].groupby(["policy", "load"], sort=False).wait_mean.mean().reset_index()
        ends = []
        for i, p in enumerate(POLICIES):
            s = g[g.policy == p.name]
            ps.line(ax, s.load, s.wait_mean, i, "")
            ends.append((np.log10(s.wait_mean.iloc[-1]), SHORT[p.name]))
        y = -np.inf                                 # direct labels, nudged apart where lines meet
        for end, name in sorted(ends):
            y = max(end, y + 0.2)
            ax.annotate(name, (LOADS[-1], 10 ** y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=ps.INK_2)
        if f == "exp":
            x = np.linspace(LOADS[0], LOADS[-1], 100)
            ax.plot(x, [theory.mg1_wait(r, 1.0) for r in x], color=ps.MUTED, lw=1, ls=(0, (2, 2)),
                    zorder=0)
            ax.annotate("M/M/1", (x[40], theory.mg1_wait(x[40], 1.0)), xytext=(-6, 6),
                        textcoords="offset points", ha="right", fontsize=8.5, color=ps.MUTED)
        ax.set_yscale("log")
        ax.set_xlim(LOADS[0] - 0.02, LOADS[-1] + 0.09)
        ps.style(ax, f"{f} flow sizes", "offered load", "")
    axes[0].set_ylabel("mean wait (mean flow transmission times)")
    fig.suptitle(f"Queues over time: waiting time against load, n = {N} paths", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=160, facecolor=ps.SURFACE)
    print(f"\nfigure: {png.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--arrivals", type=int, default=60_000)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    files = {k: OUT / f"exp11_time_{k}.csv" for k in "abcd"}
    if args.rerun or not all(f.exists() for f in files.values()):
        parts = simulate(args.arrivals, args.trials)
    else:
        parts = {k: pd.read_csv(f) for k, f in files.items()}
    report(parts)
    plot(parts, OUT / "exp11_time.png")


if __name__ == "__main__":
    main()
