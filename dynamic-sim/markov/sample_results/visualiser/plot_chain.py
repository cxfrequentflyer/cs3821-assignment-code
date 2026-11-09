"""Draw the likeliest states of a bins Markov chain and the transitions between them.

Reads chain CSVs written by Solver::save(): each row is a state (one hex digit
per bin), then pairs of next state and transition probability. For each chain
it computes the long-run probability of every state, keeps the likeliest ones
and draws them on two ordered axes, total balls across and longest queue up.
A circle's area is the state's long-run probability and an arrow's width is the
transition's probability. Several CSVs become panels with one circle scale; each
panel's axes fit its own states unless --shared-axes gives them all the same range.

    python3 plot_chain.py ../chain_5bins_cap15_1choice.csv ../chain_5bins_cap15_2choices.csv
    python3 plot_chain.py chain.csv --top 20 --min-p 0.05 -o graph.pdf

Needs matplotlib (on Ubuntu: sudo apt install python3-matplotlib).
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as Curve
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D, ScaledTranslation

AREA = 3000   # marker area in points^2 for probability 1
RAD = 0.2     # how far arrows bend


def read_chain(path):
    """{state: [(next state, probability), ...]}"""
    chain = {}
    for row in csv.reader(open(path)):
        chain[row[0]] = [(row[i], float(row[i + 1])) for i in range(1, len(row) - 1, 2)]
    return chain


def stationary(chain, tolerance=1e-12, max_steps=100_000):
    """Long-run probability of each state, by power iteration from the uniform distribution."""
    states = list(chain)
    index = {s: i for i, s in enumerate(states)}
    moves = [[(index[t], p) for t, p in chain[s]] for s in states]
    pi = [1 / len(states)] * len(states)
    for _ in range(max_steps):
        nxt = [0.0] * len(states)
        for i, row in enumerate(moves):
            mass = pi[i]
            for j, p in row:
                nxt[j] += mass * p
        change = sum(abs(a - b) for a, b in zip(nxt, pi))
        pi = nxt
        if change < tolerance:
            break
    return dict(zip(states, pi))


def loads(state):
    return [int(c, 16) for c in state]


def positions(states):
    """Place each state by (total balls, longest queue), spreading states that share a spot."""
    cells = defaultdict(list)
    for s in states:
        cells[(sum(loads(s)), max(loads(s)))].append(s)
    return {s: (t + (i - (len(group) - 1) / 2) * 0.45, q)
            for (t, q), group in cells.items() for i, s in enumerate(group)}


def fit(states):
    """Axis ranges that just fit these states, with room for the labels under the bottom row."""
    xs = [sum(loads(s)) for s in states]
    ys = [max(loads(s)) for s in states]
    return (min(xs) - 0.7, max(xs) + 0.7), (min(ys) - 0.7, max(ys) + 0.4)


def draw_loop(ax, centre, r, p):
    """Arrow from a state back to itself: a loop over the top of its circle, sized in points."""
    s, c = math.sin(math.radians(65)), math.cos(math.radians(65))
    w, h = r + 10, r + 22
    loop = Curve([(-r * c, r * s), (-w, h), (w, h), (r * c, r * s)], [Curve.MOVETO] + [Curve.CURVE4] * 3)
    # points -> inches -> screen, then shifted to wherever the state sits on screen
    points = Affine2D().scale(1 / 72) + ax.figure.dpi_scale_trans + ScaledTranslation(*centre, ax.transData)
    ax.add_patch(FancyArrowPatch(path=loop, transform=points, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=0.4 + 5 * p, color="0.55", alpha=0.7))
    ax.annotate(f"{p:.0%}", centre, textcoords="offset points", xytext=(0, 0.23 * r * s + 0.75 * h),
                ha="center", va="center", fontsize=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def draw(ax, chain, pi, top, min_p, limits, title):
    xy = positions(top)
    radius = {s: math.sqrt(max(AREA * pi[s], 10) / math.pi) for s in top}
    ax.set(title=title, xlabel="total balls", ylabel="longest queue", xlim=limits[0], ylim=limits[1])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)

    def midpoint(a, b):
        # midpoint of matplotlib's arc3 curve, worked out in screen coordinates where the bend is defined
        (x1, y1), (x2, y2) = ax.transData.transform([xy[a], xy[b]])
        mx, my = (x1 + x2) / 2 + RAD * (y2 - y1) / 2, (y1 + y2) / 2 - RAD * (x2 - x1) / 2
        return ax.transData.inverted().transform((mx, my))

    shown = set(top)
    for a in top:
        for b, p in chain[a]:
            if b not in shown or p < min_p:
                continue
            if a == b:
                draw_loop(ax, xy[a], radius[a], p)
                continue
            ax.add_patch(FancyArrowPatch(xy[a], xy[b], connectionstyle=f"arc3,rad={RAD}", arrowstyle="-|>",
                                         mutation_scale=9, linewidth=0.4 + 5 * p, color="0.55", alpha=0.7,
                                         shrinkA=radius[a], shrinkB=radius[b] + 2))
            ax.text(*midpoint(a, b), f"{p:.0%}", fontsize=6, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for s in top:
        ax.scatter(*xy[s], s=max(AREA * pi[s], 10), color="#2a78d6", edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(s, xy[s], textcoords="offset points", xytext=(0, -radius[s] - 9), ha="center",
                    fontsize=7, family="monospace")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="+", help="chain CSVs from Solver::save()")
    ap.add_argument("--top", type=int, default=15, help="most states to draw per chain (default 15)")
    ap.add_argument("--mass", type=float, default=0.99,
                    help="stop adding states once they cover this share of the time (default 0.99)")
    ap.add_argument("--min-p", type=float, default=0.10, help="hide transitions below this probability (default 0.10)")
    ap.add_argument("--shared-axes", action="store_true",
                    help="give every panel the same axis ranges, so positions compare across panels")
    ap.add_argument("-o", "--out", help="output image, .png/.pdf/.svg (default: next to the first CSV)")
    args = ap.parse_args()

    panels = []
    for path in args.csv:
        chain = read_chain(path)
        pi = stationary(chain)
        top, covered = [], 0.0
        for s in sorted(pi, key=pi.get, reverse=True)[:args.top]:
            if covered >= args.mass:
                break
            top.append(s)
            covered += pi[s]
        title = f"{Path(path).stem}\n{len(top)} likeliest of {len(chain):,} states ({covered:.0%} of the time)"
        panels.append((chain, pi, top, title))
        print(f"{path}: {len(chain)} states; top {len(top)} hold {covered:.1%}")

    shared = fit([s for *_, top, _ in panels for s in top])
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5), squeeze=False)
    for ax, (chain, pi, top, title) in zip(axes[0], panels):
        draw(ax, chain, pi, top, args.min_p, shared if args.shared_axes else fit(top), title)
    fig.tight_layout()
    out = args.out or str(Path(args.csv[0]).with_name(Path(args.csv[0]).stem + "_graph.png"))
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
