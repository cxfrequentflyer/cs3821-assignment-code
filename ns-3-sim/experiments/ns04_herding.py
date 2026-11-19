"""ns04 (ecmp-sim exp04): DRILL's synchronization effect with packets.

A leaf has F forwarding engines (packet from input port i -> engine i mod F).
With --staleness the switch's queue-length snapshot is refreshed only every
`staleness` us; between refreshes each engine sees the snapshot plus its own
packets, not the other engines'. So F engines decide in parallel on partly
shared, stale information, the mechanism DRILL blames for the
synchronization effect (§3.2.3, Fig. 2-3) and leaves open to model.

Sweep: DRILL(d, m) for a few (d, m), engines F, staleness, at 80% load.
Metrics: DRILL's own (the spread of a leaf's uplink queue lengths), windowed
uplink imbalance, FCTs, and reordering.

    NS3_DIR=~/ns-3.49 python3 ns-3-sim/experiments/ns04_herding.py [--quick]
"""
from common import parser, run_grid, table

DM = [(2, 0), (2, 1), (4, 1), (8, 1), (1, 4), (8, 4)]
ENGINES = [1, 4, 16]
STALENESS = [0, 20, 100]


def main():
    args = parser(__doc__).parse_args()
    flows = 600 if args.quick else 3000
    base = {"leaves": 4, "spines": 8, "hosts": 16, "coreRate": "20Gbps", "load": 0.8,
            "sizes": "exp:100000", "flows": flows, "windowUs": 200, "tag": "ns04"}
    configs = [{**base, "scheme": "random"}]
    configs += [{**base, "scheme": "drill", "d": d, "memory": m, "engines": f, "staleness": st}
                for d, m in DM for f in ENGINES for st in STALENESS]
    df = run_grid("ns04_herding", configs, args)
    df["policy"] = [f"DRILL({d},{m})" if s == "drill" else s
                    for s, d, m in zip(df.scheme, df.d, df.memory)]
    rnd = df[df.scheme == "random"]
    print("\nper-packet random:",
          rnd[["uplink_queue_std", "fct_mean_ms", "small_fct_p99_ms"]].mean().round(4).to_dict())
    dr = df[df.scheme == "drill"]
    for metric in ("uplink_queue_std", "uplink_imbalance_window", "fct_mean_ms",
                   "small_fct_p99_ms", "flows_reordered"):
        print(f"\n{metric}: rows DRILL(d,m), columns (staleness us, engines)")
        print(table(dr, "policy", ["staleness_us", "engines"], metric, 4))


if __name__ == "__main__":
    main()
