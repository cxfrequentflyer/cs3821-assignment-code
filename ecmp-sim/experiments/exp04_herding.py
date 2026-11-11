"""Idea 2a: more information can make things worse (core question 3).

DRILL (SIGCOMM 2017, §3.2.3, Fig. 2-3): a switch has F forwarding engines
deciding in parallel from slightly stale queue information. With many
engines under heavy load, extra choices (d) or memory (m) beyond DRILL(2,1)
make many engines pick the same "best" port at once: the synchronization
effect. The authors leave a model of it open ("our multiple arbiters produce
distinct behavior... a theoretical analysis may be valuable"; future work:
"the effect of delayed queue information in switches with multiple
forwarding engines").

Model (policies.py, `engines=F`): flows are decided in rounds of F; everyone
in a round reads the same snapshot of the path loads. F = 1 is the classical
model. Two sweeps:
    1. engines F, for ECMP, round robin, DRILL(2,0), DRILL(2,1), DRILL(8,1),
       greedy (d = n) with first and random tie-breaking
    2. at fixed F, d with memory 1 and memory with d = 1 (DRILL Fig. 3)

    python experiments/exp04_herding.py [--rerun] [--n 16] [--t 16]

What a static snapshot can't show: DRILL's queues drain, so a remembered
port goes stale over time; here loads only grow. Treat this as the clean
first-order version of the effect, to confirm later with dynamic-sim.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import pandas as pd

from ecmpsim import ECMP, DChoice, Exponential, Greedy, RoundRobin, run

ENGINES = [1, 2, 4, 8, 16, 32, 64]
FIXED_F = 16
TRIALS, SEED = 1000, 3821
FLOWS = Exponential()


def family(F, n):
    return {
        "ecmp": ECMP(),
        "round robin": RoundRobin(F),
        "DRILL(2,0)": DChoice(2, replace=False, engines=F),
        "DRILL(2,1)": DChoice(2, replace=False, memory=1, engines=F),
        "DRILL(8,1)": DChoice(8, replace=False, memory=1, engines=F),
        "greedy, first tie": Greedy(F),
        "greedy, random tie": Greedy(F, ties="random"),
    }


def simulate(n, m, csv) -> pd.DataFrame:
    rows = []
    for F in ENGINES:
        fam = family(F, n)
        df = run(n, m, FLOWS, list(fam.values()), trials=TRIALS, seed=SEED)
        names = {p.name: k for k, p in fam.items()}
        for name, g in df.groupby("policy", sort=False):
            rows.append({"sweep": "engines", "engines": F, "policy": names[name], "d": None,
                         "memory": None, "max_over_lb": g.max_over_lb.mean(),
                         "imbalance": g.imbalance.mean(), "p99_max_over_lb": g.max_over_lb.quantile(0.99)})
        print(f"  engines = {F} done", flush=True)
    configs = [(d, 1) for d in (1, 2, 3, 4, 8, 12, n)] + [(1, k) for k in (2, 4, 8, 12)] + [(2, 0)]
    pols = [DChoice(d, replace=False, memory=k, engines=FIXED_F) for d, k in configs]
    df = run(n, m, FLOWS, pols, trials=TRIALS, seed=SEED)
    for (d, k), p in zip(configs, pols):
        g = df[df.policy == p.name]
        rows.append({"sweep": "d,m", "engines": FIXED_F, "policy": f"DRILL({d},{k})", "d": d,
                     "memory": k, "max_over_lb": g.max_over_lb.mean(),
                     "imbalance": g.imbalance.mean(), "p99_max_over_lb": g.max_over_lb.quantile(0.99)})
    out = pd.DataFrame(rows)
    csv.parent.mkdir(exist_ok=True)
    out.to_csv(csv, index=False)
    return out


def report(df, n, m):
    e = df[df.sweep == "engines"]
    print(f"\nn = {n} paths, m = {m} exponential flows, {TRIALS} trials. "
          "Mean busiest path / lower bound (1 = optimal):")
    print(e.pivot(index="policy", columns="engines", values="max_over_lb")
          .reindex(list(family(1, n))).round(3).to_string())
    dm = df[df.sweep == "d,m"]
    print(f"\nAt {FIXED_F} engines: DRILL(d, memory)")
    print(dm[["policy", "max_over_lb", "p99_max_over_lb", "imbalance"]].round(3).to_string(index=False))


def plot(df, n, m, png):
    import plotstyle as ps
    plt = ps.setup()
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=ps.SURFACE)
    e = df[df.sweep == "engines"]
    for i, name in enumerate(family(1, n)):
        g = e[e.policy == name].sort_values("engines")
        ps.line(ax, g.engines, g.max_over_lb, i, name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ENGINES, [str(F) for F in ENGINES])
    ax.set_xlim(right=ENGINES[-1] * 5)
    ps.style(ax, f"Stale, parallel decisions: n = {n}, m = {m}", "forwarding engines F (one shared snapshot per round)",
             "busiest path / lower bound")
    fig.tight_layout()
    fig.savefig(png, dpi=160, facecolor=ps.SURFACE)
    print(f"\nfigure: {png.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--t", type=int, default=16, help="flows per path")
    args = ap.parse_args()
    n, m = args.n, args.n * args.t
    csv = ROOT / "results" / f"exp04_herding_n{n}_t{args.t}.csv"
    df = simulate(n, m, csv) if args.rerun or not csv.exists() else pd.read_csv(csv)
    report(df, n, m)
    plot(df, n, m, csv.with_suffix(".png"))


if __name__ == "__main__":
    main()
