"""ns06 (ecmp-sim exp06): what finer granularity buys, and what it costs.

ecmp-sim showed that splitting flows into smaller chunks all but removes
imbalance. Here the same ladder runs with TCP, from whole flows (ECMP) to
flowlets (gap-based; random or CONGA-like), flowcells (Presto, cell size c)
and single packets (random, round robin, DRILL), on the web-search workload.
The cost side, which the static model could not see, is reordering: the
share of flows with a segment that arrived late although it was sent only
once (overtaken on another path, so not a retransmission), and the share
with more than 3 duplicate ACKs (TCP's fast-retransmit threshold; DRILL
Fig. 11). The core is 2:1 oversubscribed so the uplinks, where the choice
is made, are the bottleneck.

    NS3_DIR=~/ns-3.49 python3 ns-3-sim/experiments/ns06_granularity.py [--quick]

Writes results/ns06_granularity.png: FCT against reordering, one point per scheme.
"""
import sys

from common import HERE, REPO, WEBSEARCH, parser, run_grid, show

LADDER = ([("ecmp (flow)", {"scheme": "ecmp"})]
          + [(f"flowlet {g}us", {"scheme": "flowlet", "flowletGap": g}) for g in (500, 100, 20)]
          + [("conga 100us", {"scheme": "conga", "flowletGap": 100})]
          + [(f"presto {c}KB", {"scheme": "presto", "cellKB": c}) for c in (1024, 256, 64, 16)]
          + [("random (pkt)", {"scheme": "random"}), ("rr (pkt)", {"scheme": "rr"}),
             ("drill(2,1) (pkt)", {"scheme": "drill", "d": 2, "memory": 1})])


def main():
    args = parser(__doc__).parse_args()
    flows = 200 if args.quick else 800
    base = {"leaves": 4, "spines": 4, "hosts": 8, "coreRate": "10Gbps", "load": 0.6,
            "sizes": WEBSEARCH, "flows": flows, "windowUs": 500, "tag": "ns06"}
    configs = [{**base, **c} for _, c in LADDER]
    df = run_grid("ns06_granularity", configs, args)
    names = {}
    from common import _key
    for (name, c), cfg in zip(LADDER, configs):
        for s in range(1, args.seeds + 1):
            names[_key({**cfg, "seed": s})] = name
    df["granularity"] = df.config_key.map(names)
    metrics = ["fct_mean_ms", "small_fct_p99_ms", "large_fct_mean_ms", "uplink_imbalance_window",
               "flows_reordered", "flows_over_3_dupacks", "retx_segments", "drops_uplink"]
    print("\nWeb search, load 0.6 of the uplinks, 4 x 4 leaf-spine, 2:1 oversubscribed core")
    show(df, "granularity", metrics)
    plot(df.groupby("granularity", sort=False)[["fct_mean_ms", "flows_reordered"]].mean())


def plot(g):
    sys.path.insert(0, str(REPO / "ecmp-sim" / "experiments"))
    import plotstyle as ps
    plt = ps.setup()
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=ps.SURFACE)
    family = lambda n: 0 if "flow)" in n else 1 if "flowlet" in n or "conga" in n else \
        2 if "presto" in n else 3
    placed = []
    for name, row in g.iterrows():
        x, y = 100 * row.flows_reordered, row.fct_mean_ms
        ax.plot(x, y, "o", ms=8, color=ps.SERIES[family(name)], mec=ps.SURFACE, mew=2)
        # push a label below its point when another label already sits close by
        crowded = any(abs(x - px) < 6 and abs(y - py) < 0.08 * g.fct_mean_ms.max()
                      for px, py in placed)
        ax.annotate(name, (x, y), xytext=(6, -12 if crowded else 3), textcoords="offset points",
                    fontsize=8.5, color=ps.INK_2)
        placed.append((x, y))
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], ls="none", marker="o", ms=8, color=ps.SERIES[i]) for i in range(4)],
              ["flow", "flowlet", "flowcell (Presto)", "packet"], frameon=False, fontsize=9,
              loc="upper right")
    ps.style(ax, "Finer granularity: faster flows, more reordering (web search, load 0.6)",
             "flows with a reordered segment (%)", "mean flow completion time (ms)")
    fig.tight_layout()
    out = HERE / "results" / "ns06_granularity.png"
    fig.savefig(out, dpi=160, facecolor=ps.SURFACE)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
