"""ns07 (ecmp-sim exp07): web search vs data mining, pFabric-style.

pFabric (§5.4) finds the more skewed data-mining workload *easier* because
large flows rarely overlap; CONGA (§6.2) finds ECMP does worse on it. ecmp-sim
reconciled the two in a snapshot; here are the flow completion times, split
as pFabric does into small (< 100KB) and large (> 10MB) flows, normalized to
the best possible time on an empty fabric, across loads.

Data-mining flows reach ~1GB, so these runs are the slowest of the set.

    NS3_DIR=~/ns-3.49 python3 ns-3-sim/experiments/ns07_workloads.py [--quick]
"""
from common import DATAMINING, WEBSEARCH, parser, run_grid, table

SCHEMES = [{"scheme": "ecmp"}, {"scheme": "conga"}, {"scheme": "presto"},
           {"scheme": "drill", "d": 2, "memory": 1}]


def main():
    args = parser(__doc__).parse_args()
    flows = {"websearch": 150 if args.quick else 800, "datamining": 150 if args.quick else 600}
    loads = (0.3, 0.6) if args.quick else (0.3, 0.5, 0.7)
    configs = [{**s, "leaves": 4, "spines": 4, "hosts": 8, "coreRate": "10Gbps", "load": load,
                "sizes": sizes, "flows": flows[name], "windowUs": 500, "maxSeconds": 20,
                "tag": f"ns07-{name}"}
               for name, sizes in (("websearch", WEBSEARCH), ("datamining", DATAMINING))
               for s in SCHEMES for load in loads]
    df = run_grid("ns07_workloads", configs, args)
    df["workload"] = df.tag.str.replace("ns07-", "")
    df = df.sort_values("load", kind="stable")
    for metric in ("nfct_mean", "small_nfct_p99", "large_nfct_mean", "unfinished"):
        print(f"\n{metric}")
        print(table(df, ["workload", "load"], "scheme", metric))


if __name__ == "__main__":
    main()
