"""Core question 1 (template): how good are the classical max-load predictions
at realistic ECMP group sizes? Unit flows, one and two choices.

    python experiments/exp01_unit_accuracy.py            # simulate (if needed) + plot
    python experiments/exp01_unit_accuracy.py --rerun    # force a fresh simulation

Pattern worth copying for every experiment: simulate once into a CSV in
results/, then plot from the CSV, so restyling a figure never re-runs anything.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # so this runs without installing ecmpsim

import numpy as np
import pandas as pd

from ecmpsim import ECMP, DChoice, Unit, exact, run, summarize, theory

NS = [4, 8, 16, 32, 64]                         # ECMP group sizes (paths)
LOADS = [1, 2, 4, 8, 16, 32, 64]                # flows per path, m / n
TRIALS, SEED = 1000, 3821
CSV = ROOT / "results" / "exp01_unit_accuracy.csv"
PNG = ROOT / "results" / "exp01_unit_accuracy.png"


def simulate() -> pd.DataFrame:
    rows = []
    for n in NS:
        for t in LOADS:
            m = n * t
            sim = summarize(run(n, m, Unit(), [ECMP(), DChoice(2)], trials=TRIALS, seed=SEED),
                            metrics=["max_load"]).set_index("policy")
            for policy, d in (("ecmp", 1), ("d2-bytes", 2)):
                rows.append({
                    "n": n, "m": m, "t": t, "d": d,
                    "sim_mean": sim.loc[policy, "max_load_mean"],
                    "sim_ci95": sim.loc[policy, "max_load_ci95"],
                    "exact": exact.expected_max_load(m, n) if d == 1 else np.nan,
                    "raab_steger": theory.raab_steger(m, n) if d == 1 else np.nan,
                    "gaussian": theory.gaussian_max(m, n) if d == 1 else np.nan,
                    "gonnet": theory.gonnet(n) if d == 1 and m == n else np.nan,
                    "fluid": theory.fluid_expected_max(m, n, d),
                    "bcsv": theory.heavily_loaded_dchoice(m, n, d) if d >= 2 else np.nan,
                })
            print(f"  n={n:<3} m/n={t:<3} done", flush=True)
    df = pd.DataFrame(rows)
    CSV.parent.mkdir(exist_ok=True)
    df.to_csv(CSV, index=False)
    return df


def report(df: pd.DataFrame) -> None:
    one = df[df.d == 1].copy()
    gap = one.exact - one.m / one.n
    one["rs_err_%"] = 100 * (one.raab_steger - one.m / one.n - gap) / gap
    one["gauss_err_%"] = 100 * (one.gaussian - one.m / one.n - gap) / gap
    one["sim_minus_exact"] = one.sim_mean - one.exact
    cols = ["n", "t", "exact", "sim_mean", "sim_ci95", "sim_minus_exact", "raab_steger",
            "rs_err_%", "gaussian", "gauss_err_%"]
    print("\nOne choice: E[max load]; errors are relative to the exact gap (max - m/n)")
    print(one[cols].round(3).to_string(index=False))
    two = df[df.d == 2].copy()
    print("\nTwo choices: simulated E[max load] vs the fluid-limit heuristic and m/n + lnln n/ln 2")
    print(two[["n", "t", "sim_mean", "sim_ci95", "fluid", "bcsv"]].round(3).to_string(index=False))


# -- plotting -------------------------------------------------------------------------
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {4: "#2a78d6", 16: "#eb6834", 64: "#1baf7a"}   # fixed categorical order


def plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE)
    panels = [(axes[0], 1, "One choice (ECMP)", "exact", "raab_steger",
               ["exact", "simulated, 1000 trials", "Raab–Steger leading term"]),
              (axes[1], 2, "Two choices", "sim_mean", "fluid",
               ["simulated, 1000 trials", None, "fluid-limit heuristic"])]
    for ax, d, title, solid, dashed, names in panels:
        ax.set_facecolor(SURFACE)
        sub = df[df.d == d]
        for n, color in SERIES.items():
            s = sub[sub.n == n].sort_values("t")
            ax.plot(s.t, s[solid] - s.t, color=color, lw=2, solid_capstyle="round")
            ax.plot(s.t, s[dashed] - s.t, color=color, lw=2, ls=(0, (4, 3)))
            ax.plot(s.t, s.sim_mean - s.t, ls="none", marker="o", ms=7, color=color,
                    mec=SURFACE, mew=2)
            ax.annotate(f"n = {n}", (s.t.iloc[-1], (s[solid] - s.t).iloc[-1]),
                        xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK_2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(LOADS, [str(t) for t in LOADS])
        ax.set_xlim(0.8, 110)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("flows per path, m / n")
        style = [Line2D([], [], color=INK_2, lw=2),
                 Line2D([], [], ls="none", marker="o", ms=7, color=INK_2, mec=SURFACE, mew=2),
                 Line2D([], [], color=INK_2, lw=2, ls=(0, (4, 3)))]
        if d == 2:   # simulated is drawn as line + dots
            style = [Line2D([], [], color=INK_2, lw=2, marker="o", ms=7, mec=SURFACE, mew=2),
                     style[2]]
        ax.legend(style, [nm for nm in names if nm], loc="upper left" if d == 1 else "lower right",
                  frameon=False, fontsize=9, labelcolor=INK_2)
    axes[0].set_ylabel("busiest path minus average (flows)")
    fig.suptitle("Gap between the busiest path and the average, unit flows", x=0.01,
                 ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PNG, dpi=160, facecolor=SURFACE)
    print(f"\nwrote {PNG.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rerun", action="store_true", help="re-simulate even if the CSV exists")
    args = ap.parse_args()
    df = simulate() if args.rerun or not CSV.exists() else pd.read_csv(CSV)
    report(df)
    plot(df)


if __name__ == "__main__":
    main()
