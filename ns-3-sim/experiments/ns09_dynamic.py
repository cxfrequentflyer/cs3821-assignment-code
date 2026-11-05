"""ns09 (dynamic-sim: lab and transition-matrix): bins over time, with packets.

dynamic-sim places log-normal balls into FIFO bins with 1 or D choices, lets
bins run at different efficiencies, adds and retires bins mid-run, and counts
idle bins. The same story on one ECMP group of leaf 0's 8 uplinks (all
traffic leaf 0 -> leaf 1, log-normal sizes with sigma = 1 as dynamic-sim's
default):

    steady         8 equal uplinks
    hetero         2 of the 8 uplinks run 2x faster (dynamic-sim's `eff 6x1,2x2`)
    retire 2       two uplinks leave service halfway through (they drain first)
    add 2          two extra uplinks join halfway through

Schemes: ECMP (1 choice), per-flow 2 choices (dynamic-sim's D = 2 per ball),
and per-packet DRILL(2,0) / DRILL(2,1). Metrics: FCT before and after the
change, uplink queue length (the queueing-delay analogue), windowed
imbalance, and the share of idle uplinks per 50 us window (the
transition-matrix experiment's idle bins).

    NS3_DIR=~/ns-3.49 python3 ns-3-sim/experiments/ns09_dynamic.py [--quick]
"""
from common import parser, run_grid, table

N, MEAN, RATE = 8, 100_000, 10e9
SCHEMES = [{"scheme": "ecmp"}, {"scheme": "dchoice", "d": 2},
           {"scheme": "drill", "d": 2, "memory": 0}, {"scheme": "drill", "d": 2, "memory": 1}]


def scenarios(event_us):
    half = f"@{event_us:.0f}"
    return {"steady": {}, "hetero": {"slow": "0-6:2,0-7:2"},
            "retire 2": {"retire": f"0-6{half},0-7{half}"},
            "add 2": {"spines": N + 2, "add": f"0-{N}{half},0-{N + 1}{half}"}}


def main():
    args = parser(__doc__).parse_args()
    flows = 1000 if args.quick else 4000
    configs = []
    for load in (0.6, 0.9):
        lam = load * N * RATE / (8 * MEAN)
        event_us = 1e6 * (0.001 + 0.5 * flows / lam)          # halfway through the arrivals
        for name, extra in scenarios(event_us).items():
            for s in SCHEMES:
                configs.append({"leaves": 2, "spines": N, "hosts": N, "coreRate": "10Gbps",
                                "pattern": "pair:0:1", "sizes": "lognormal:1:100000",
                                "load": load, "flows": flows, "windowUs": 50,
                                **s, **extra, "tag": f"ns09-{name}"})
    df = run_grid("ns09_dynamic", configs, args)
    df["scenario"] = df.tag.str.replace("ns09-", "")
    df["policy"] = [f"DRILL(2,{m})" if s == "drill" else ("2 choices" if s == "dchoice" else s)
                    for s, m in zip(df.scheme, df.memory)]
    df = df.sort_values("load", kind="stable")
    for metric in ("fct_before_event_ms", "fct_after_event_ms", "fct_mean_ms", "uplink_queue_mean",
                   "uplink_imbalance_window", "uplink_idle_frac_window"):
        print(f"\n{metric}")
        print(table(df, ["load", "scenario"], "policy", metric))


if __name__ == "__main__":
    main()
