"""ns08 (ecmp-sim exp08): asymmetric fabrics.

The leaf-spine scenarios of exp08 with real traffic: symmetric, one failed
link (DRILL Fig. 4), a quarter-speed link (CONGA Fig. 2-3) and imbalanced
striping (DRILL Fig. 13). Local per-packet balancing (drill) splits flows
over asymmetric paths; DRILL's symmetric components (drill-comp) keep each
flow inside one symmetric group; CONGA uses path-wide congestion.

    NS3_DIR=~/ns-3.49 python3 ns-3-sim/experiments/ns08_asymmetry.py [--quick]
"""
from common import parser, run_grid, table

SCENARIOS = {"symmetric": {}, "1 failed link": {"fail": "0-0"},
             "slow link": {"slow": "1-1:0.25"}, "striping": {"striping": "true"}}
SCHEMES = [{"scheme": "ecmp"}, {"scheme": "wcmp"}, {"scheme": "dchoice", "d": 2},
           {"scheme": "conga"}, {"scheme": "presto"}, {"scheme": "drill", "d": 2, "memory": 1},
           {"scheme": "drill-comp", "d": 2, "memory": 1}]


def main():
    args = parser(__doc__).parse_args()
    flows = 500 if args.quick else 2500
    configs = [{**s, **extra, "leaves": 4, "spines": 4, "hosts": 8, "coreRate": "10Gbps",
                "load": 0.5, "sizes": "exp:200000", "flows": flows, "windowUs": 200,
                "tag": f"ns08-{name}"}
               for name, extra in SCENARIOS.items() for s in SCHEMES]
    df = run_grid("ns08_asymmetry", configs, args)
    df["scenario"] = df.tag.str.replace("ns08-", "")
    for metric in ("fct_mean_ms", "fct_p99_ms", "uplink_imbalance_run",
                   "downlink_imbalance_run", "flows_reordered", "drops_downlink"):
        print(f"\n{metric}")
        print(table(df, "scenario", "scheme", metric))


if __name__ == "__main__":
    main()
