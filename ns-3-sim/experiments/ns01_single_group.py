"""ns01 (ecmp-sim exp01 + exp02): one ECMP group, now with TCP and queues.

All traffic goes from leaf 0's hosts to leaf 1's, so leaf 0's n uplinks are
a single ECMP group, the setting of ecmp-sim level 1. We sweep the group
size n and the flow-size distribution (the same families as exp02, scaled to
a 100KB mean) for ECMP, per-flow 2 choices (ecmp-sim's DChoice) and DRILL.

Metrics: uplink imbalance (busiest - least loaded uplink) / mean, per 1 ms
window and over the run (the packet-level analogue of the static
imbalance), drops on the uplinks, and flow completion times.

    NS3_DIR=~/ns-3.49 python3 ns-3-sim/experiments/ns01_single_group.py [--quick]
"""
from common import parser, run_grid, table

SIZES = ["fixed:100000", "exp:100000", "lognormal:1.5:100000", "bimodal:0.05:100:100000",
         "pareto:1.5:100000"]
SCHEMES = [{"scheme": "ecmp"}, {"scheme": "dchoice", "d": 2}, {"scheme": "drill", "d": 2, "memory": 1}]


def label(c):
    return c["scheme"] + (f"(d={c['d']})" if c["scheme"] == "dchoice" else "")


def main():
    args = parser(__doc__).parse_args()
    flows = 400 if args.quick else 2000
    configs = []
    for n in (4, 8, 16):
        for sizes in SIZES:
            for s in SCHEMES:
                configs.append({**s, "leaves": 2, "spines": n, "hosts": n, "coreRate": "10Gbps",
                                "pattern": "pair:0:1", "sizes": sizes, "load": 0.7,
                                "flows": flows, "maxFlowBytes": 1e8, "tag": "ns01"})
    df = run_grid("ns01_single_group", configs, args)
    df["policy"] = df.scheme.where(df.scheme != "dchoice", "dchoice(d=2)")
    for metric in ("uplink_imbalance_window", "uplink_imbalance_run", "fct_mean_ms",
                   "small_fct_p99_ms", "drops_uplink"):
        print(f"\n{metric} (mean over seeds)")
        print(table(df, ["sizes", "spines"], "policy", metric))


if __name__ == "__main__":
    main()
