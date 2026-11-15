"""Poster figures for the ECMP box, drawn from the experiment CSVs in results/.

Run exp03, exp09 and exp11 first (they write the CSVs), then:

    python experiments/poster_figures.py [--transparent]

Writes PNG (300 dpi) and PDF versions into results/poster/:

    poster_snapshot   one ECMP trial: load on each of 16 paths against the average
    poster_decay      imbalance vs number of flows: unit flows against Pareto(1.5)
    poster_fattree    busiest link / optimum in a k = 8 fat-tree, ECMP vs global information
    poster_wait       waiting time vs load: ECMP on the queueing formula, two choices below it
    poster_schematic  illustration (not data): a switch hashing flows onto 4 equal-cost paths

Fonts are sized for a half-width column of an A2 poster.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd

import plotstyle as ps
from ecmpsim import theory

RES = ROOT / "results"
OUT = RES / "poster"
BLUE, ORANGE = ps.SERIES[0], ps.SERIES[1]
SEED = 3821
BG, SUFFIX = ps.SURFACE, ""      # --transparent: no background, files named *_transparent

plt = ps.setup()
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})


def finish(fig, ax, title, xlabel, ylabel, name):
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color=ps.GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=ps.INK, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}{SUFFIX}.{ext}", dpi=300, facecolor=BG)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def snapshot():
    n, m = 16, 64
    loads = np.bincount(np.random.default_rng(SEED).integers(0, n, m), minlength=n)
    loads = np.sort(loads)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=BG)
    ax.bar(np.arange(1, n + 1), loads, width=0.72, color=BLUE, edgecolor=BG, lw=2)
    ax.axhline(m / n, color=ps.INK_2, lw=1.5, ls=(0, (4, 3)))
    ax.annotate(f"perfectly even: {m // n} per path", (n + 0.5, m / n), xytext=(0, 6),
                textcoords="offset points", ha="right", va="bottom", color=ps.INK_2, fontsize=13)
    top = int((loads == loads[0]).sum())
    ax.annotate(f"busiest path{'s' if top > 1 else ''}: {loads[0]} flows each" if top > 1
                else f"busiest path: {loads[0]} flows", (0.65, loads[0]), xytext=(0, 8),
                textcoords="offset points", va="bottom", color=ps.INK, fontsize=13)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlim(0.3, n + 0.7)
    ax.set_ylim(0, loads[0] + 1.6)
    finish(fig, ax, f"One ECMP hash of {m} flows onto {n} paths",
           "path (sorted, busiest first)", "flows on the path", "poster_snapshot")
    print(f"    loads: {loads.tolist()}")


def decay():
    df = pd.read_csv(RES / "exp03_conga_bound.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=BG)
    series = [("unit", BLUE, "equal-size flows", -0.5, r"theory $m^{-1/2}$"),
              ("pareto(a=1.5)", ORANGE, r"heavy-tailed flows (Pareto $\alpha$ = 1.5)", -1 / 3,
               r"theory $m^{-1/3}$")]
    for label, color, name, slope, tname in series:
        g = df[df.flows == label].sort_values("m")
        x, y = g.m.to_numpy(), g["median"].to_numpy()
        fit = theory.fitted_exponent(g.m, g["median"])
        ax.plot(x, y[0] * (x / x[0]) ** slope, color=ps.MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
        ax.plot(x, y, color=color, lw=2.5, marker="o", ms=10, mec=BG, mew=2, zorder=2)
        text = f"{name}\nfitted slope \u2212{abs(fit):.2f}  ({tname}, dashed)"
        if color == ORANGE:
            ax.annotate(text, (x[1], y[1]), xytext=(14, 16), textcoords="offset points",
                        color=ps.INK, fontsize=12.5, va="bottom")
        else:
            ax.annotate(text, (x[0], 0.105), xytext=(-8, 0), textcoords="offset points",
                        color=ps.INK, fontsize=12.5, va="bottom")
        print(f"    {label}: fitted {fit:+.3f}, last median {y[-1]:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ms = sorted(df.m.unique())
    ax.set_xticks(ms, [f"{m:,}" for m in ms])
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_ylim(0.08, 4)
    finish(fig, ax, "More flows even out the load, but elephants slow that down",
           "number of flows m (16 paths)", "imbalance: (max − min) / average", "poster_decay")


def fattree():
    df = pd.read_csv(RES / "exp09_fattree_k8.csv")
    e = df[df.flows == "exp"].groupby(["topology", "policy"]).over_opt.mean()
    scen = [("symmetric", "healthy fabric"), ("slow core link", "one slow core link")]
    pols = [("ecmp", "ECMP", BLUE), ("global", "least-congested path", ORANGE)]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=BG)
    w = 0.34
    for j, (p, pname, color) in enumerate(pols):
        vals = [e[(s, p)] for s, _ in scen]
        xs = np.arange(len(scen)) + (j - 0.5) * (w + 0.03)
        ax.bar(xs, vals, width=w, color=color, label=pname)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}×", (x, v), xytext=(0, 5), textcoords="offset points",
                        ha="center", color=ps.INK, fontsize=15, fontweight="bold")
        print(f"    {p}: " + ", ".join(f"{s} {v:.3f}" for (s, _), v in zip(scen, vals)))
    ax.axhline(1, color=ps.INK_2, lw=1.5, ls=(0, (4, 3)))
    ax.annotate("optimum", (1.83, 1), xytext=(0, 5), textcoords="offset points", ha="right",
                color=ps.INK_2, fontsize=13)
    ax.set_xticks(range(len(scen)), [s for _, s in scen], fontsize=15)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 1.85)
    ax.set_ylim(0, 3.4)
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    finish(fig, ax, "Fat-tree (k = 8): ECMP's busiest link against the optimum",
           "", "busiest link / optimum", "poster_fattree")


def wait():
    b = pd.read_csv(RES / "exp11_time_b.csv")
    g = b[b.flows == "exp"].groupby(["policy", "load"]).wait_mean.mean()
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=BG)
    x = np.linspace(0.5, 0.95, 200)
    ax.plot(x, [theory.mg1_wait(r, 1.0) for r in x], color=ps.MUTED, lw=1.5, ls=(0, (4, 3)),
            zorder=1)
    for p, name, color, off in [("ecmp", "ECMP (dashed: queueing theory)", BLUE, (-12, 8)),
                                ("d2-bytes-distinct", "two choices", ORANGE, (-12, -18))]:
        s = g[p]
        ax.plot(s.index, s.values, color=color, lw=2.5, marker="o", ms=10, mec=BG, mew=2,
                zorder=2)
        ax.annotate(name, (s.index[2], s.values[2]), xytext=off, textcoords="offset points",
                    ha="right" if color == BLUE else "left", color=ps.INK, fontsize=13,
                    va="bottom" if color == BLUE else "top")
        print(f"    {p}: " + ", ".join(f"{l}: {v:.2f}" for l, v in s.items()))
    e9, d9 = g[("ecmp", 0.9)], g[("d2-bytes-distinct", 0.9)]
    ax.annotate("", (0.9, d9 * 1.25), (0.9, e9 / 1.25),
                arrowprops=dict(arrowstyle="<->", color=ps.INK_2, lw=1.2))
    ax.annotate(f"{e9 / d9:.0f}× shorter wait", (0.9, d9 * 1.75), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center", color=ps.INK_2, fontsize=13)
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
    ax.minorticks_off()
    ax.set_xticks([0.5, 0.7, 0.8, 0.9, 0.95], ["50%", "70%", "80%", "90%", "95%"])
    finish(fig, ax, "Waiting time as the 16 paths fill up",
           "offered load", "mean wait (flow transmission times)", "poster_wait")


def schematic():
    """An illustration, not a simulation: six flows, one of them an elephant, on four paths."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=BG)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    MOUSE, ELEPHANT, H = 0.7, 1.7, 0.46

    def box(x, y, w, h, fc, ec="none", lw=0, r=0.12, z=2):
        ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, zorder=z, fc=fc, ec=ec,
                                    lw=lw, boxstyle=f"round,pad={r}"))

    def flow(x, yc, big=False):
        w = ELEPHANT if big else MOUSE
        box(x, yc - H / 2, w, H, ORANGE if big else BLUE, z=3)
        return x + w + 0.1

    # the flows waiting on the left, one of them much larger than the rest
    for i, big in enumerate([False, True, False, False, False, False]):
        flow(0.25, 4.95 - i * 0.68, big)
    ax.text(0.25, 5.6, "flows = balls", fontsize=14, fontweight="bold", color=ps.INK)
    ax.text(0.25, 0.75, "one elephant,\nfive mice", fontsize=12, color=ps.INK_2, va="top")

    # the switch
    box(3.1, 2.25, 2.3, 2.1, BG, ec=ps.INK_2, lw=1.8, r=0.2)
    ax.text(4.25, 3.3, "switch\nhash(5-tuple)\nmod 4", ha="center", va="center", fontsize=13,
            color=ps.INK, linespacing=1.35)
    ax.annotate("", (3.05, 3.3), (2.15, 3.3),
                arrowprops=dict(arrowstyle="-|>", color=ps.INK_2, lw=1.8))

    # four equal-cost paths, loaded as one unlucky hash would load them
    lanes = [(5.0, [True, False, False], "busiest link"), (3.7, [False, False], ""),
             (2.4, [False], ""), (1.1, [], "idle")]
    for k, (yc, flows, note) in enumerate(lanes):
        ax.annotate("", (6.45, yc), (5.45, 3.3 + (yc - 3.05) * 0.25),
                    arrowprops=dict(arrowstyle="-|>", color=ps.INK_2, lw=1.5))
        box(6.5, yc - 0.4, 3.75, 0.8, ps.GRID, r=0.15, z=1)
        x = 6.65
        for big in flows:
            x = flow(x, yc, big)
        ax.text(10.4, yc, f"path {k + 1}", fontsize=13, color=ps.INK_2, va="center")
        if note:
            ax.text(11.55, yc, note, fontsize=12, color=ps.INK, va="center", fontweight="bold",
                    ha="left")
    ax.text(6.5, 5.6, "equal-cost paths = bins", fontsize=14, fontweight="bold", color=ps.INK)
    ax.set_title("ECMP: every flow is hashed onto one path, whatever its load", loc="left",
                 color=ps.INK, fontweight="bold", pad=6)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"poster_schematic{SUFFIX}.{ext}", dpi=300, facecolor=BG)
    plt.close(fig)
    print("  poster_schematic.png / .pdf")


if __name__ == "__main__":
    if "--transparent" in sys.argv:
        BG, SUFFIX = "none", "_transparent"
    print(f"writing to {OUT.relative_to(ROOT)}/")
    schematic()
    snapshot()
    decay()
    fattree()
    wait()
