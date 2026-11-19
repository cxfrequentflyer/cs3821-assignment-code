"""Draw a bins Markov chain's transition matrix as a heatmap, states in ascending lexicographic order.

Reads chain CSVs written by Solver::save() and writes one image per chain in
which every transition is its own pixel: row i, column j is the probability of
going from state i to state j in one turn, on a log colour scale that starts at
the smallest probability in the chains given, so even the rarest move shows.
White means the move is impossible. States are written fullest bin first, so
lexicographic order groups them by longest queue; the ticks and faint lines
mark where each group starts. --max-queue zooms into the top-left corner,
keeping only states whose fullest bin holds at most that many balls.

    python3 plot_matrix.py ../chain_5bins_cap15_1choice.csv ../chain_5bins_cap15_2choices.csv
    python3 plot_matrix.py ../*.csv --max-queue 7 --format pdf
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from plot_chain import read_chain, stationary

# from step 250, so the rarest moves still stand out against white
BLUES = LinearSegmentedColormap.from_list("blues", ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MIN_SIDE = 1600   # small matrices get several screen pixels per state, up to at least this many a side
AXIS_INCHES = 8   # the matrix's printed size; the dpi is set so that it is still one pixel per state


def entries(chain, max_queue):
    """States in lexicographic order and the (row, column, probability) of every move among them."""
    states = sorted(s for s in chain if int(s[0], 16) <= max_queue)
    index = {s: i for i, s in enumerate(states)}
    rows, cols, ps = zip(*((index[s], index[t], p) for s in states for t, p in chain[s] if t in index))
    return states, np.array(rows), np.array(cols), np.array(ps)


def draw(states, rows, cols, ps, norm, title, out):
    n = len(states)
    scale = math.ceil(MIN_SIDE / n)           # screen pixels per state
    side = n * scale
    dpi = round(side / AXIS_INCHES)
    left, right, bottom, top = (round(inches * dpi) for inches in (0.9, 1.5, 0.3, 1.4))
    width, height = left + side + right, bottom + side + top
    # the 0.01 stops the canvas being rounded down a pixel
    fig = plt.figure(figsize=((width + 0.01) / dpi, (height + 0.01) / dpi), dpi=dpi)

    starts = [i for i, s in enumerate(states) if i == 0 or s[0] != states[i - 1][0]]
    image = np.zeros((n, n, 4), dtype=np.uint8)   # transparent, so white, where the move is impossible
    for i in starts[1:]:   # faint lines where each group starts, through the empty cells only
        image[i, :] = image[:, i] = (224, 224, 224, 255)
    image[rows, cols] = BLUES(norm(ps), bytes=True)
    # stamped onto the canvas pixel for pixel, never resampled
    fig.figimage(image.repeat(scale, 0).repeat(scale, 1), xo=left, yo=bottom, origin="upper")

    # the axes over it only carry the ticks and frame
    ax = fig.add_axes([left / width, bottom / height, side / width, side / height], zorder=1)
    ax.patch.set_visible(False)
    if n <= 40:
        ticks = range(n)
    else:   # the group starts that have room for a label
        ticks = [0]
        for i in starts:
            if i - ticks[-1] >= n / 40:
                ticks.append(i)
    ax.set_xticks([i + 0.5 for i in ticks], [states[i] for i in ticks])
    ax.set_yticks([i + 0.5 for i in ticks], [states[i] for i in ticks])
    ax.tick_params(length=2, labelsize=6, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis="x", labelrotation=90)
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()] + ax.get_yticklabels(), family="monospace")
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():   # the frame sits just outside the matrix instead of over its edge cells
        spine.set_position(("outward", 2))
        spine.set_linewidth(0.5)
    ax.set(title=title, xlabel="to state", ylabel="from state", xlim=(0, n), ylim=(n, 0))

    cax = fig.add_axes([(left + side + 0.3 * dpi) / width, (bottom + side / 4) / height, 0.15 * dpi / width, side / 2 / height])
    fig.colorbar(ScalarMappable(norm, BLUES), cax=cax, label="transition probability")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="+", help="chain CSVs from Solver::save()")
    ap.add_argument("--max-queue", type=int, default=15,
                    help="only states whose fullest bin holds at most this many balls (default: all)")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="image format (default png)")
    ap.add_argument("--out-dir", help="where to write the images (default: next to each CSV)")
    args = ap.parse_args()

    chains = {path: read_chain(path) for path in args.csv}
    smallest = min(p for chain in chains.values() for row in chain.values() for _, p in row if p > 0)
    norm = LogNorm(vmin=10 ** math.floor(math.log10(smallest)), vmax=1)   # one scale for every chain given
    zoom = f"_maxq{args.max_queue}" if args.max_queue < 15 else ""
    for path, chain in chains.items():
        states, rows, cols, ps = entries(chain, args.max_queue)
        pi = stationary(chain)
        share = sum(pi[s] for s in states)
        title = (f"{Path(path).stem}\n{len(states):,} of {len(chain):,} states ({share:.1%} of the time), "
                 f"{len(ps):,} transitions")
        out = Path(args.out_dir or Path(path).parent) / f"{Path(path).stem}_matrix{zoom}.{args.format}"
        draw(states, rows, cols, ps, norm, title, out)
        print(f"{path}: {len(states)} states, {len(ps)} transitions -> {out}")


if __name__ == "__main__":
    main()
