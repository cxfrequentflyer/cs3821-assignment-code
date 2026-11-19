"""ns05 (ecmp-sim exp05): memory and unequal path speeds (DRILL Thm 1-2).

Every leaf's link to spine 0 runs 4x faster than the others, so spine 0
should carry 4/7 of the traffic. Memoryless d-choice can only send a packet
to spine 0 when it samples it (probability about d/n), the counting argument
of DRILL's Theorem 1; one memory slot keeps coming back to it (Theorem 2).
ECMP ignores speeds; WCMP weights by them; CONGA reads congestion.

    NS3_DIR=~/ns-3.49 python3 ns-3-sim/experiments/ns05_speeds.py [--quick]
"""
from common import parser, run_grid, table

SCHEMES = ([{"scheme": "ecmp"}, {"scheme": "wcmp"}, {"scheme": "conga"}]
           + [{"scheme": "drill", "d": d, "memory": m} for d in (1, 2, 3) for m in (0, 1)])


def main():
    args = parser(__doc__).parse_args()
    flows = 600 if args.quick else 3000
    configs = [{**s, "leaves": 4, "spines": 4, "hosts": 8, "coreRate": "10Gbps",
                "slow": "0-0:4,1-0:4,2-0:4,3-0:4", "load": load, "sizes": "exp:100000",
                "flows": flows, "windowUs": 200, "tag": "ns05"}
               for s in SCHEMES for load in (0.5, 0.8)]
    df = run_grid("ns05_speeds", configs, args)
    df["policy"] = [f"DRILL({d},{m})" if s == "drill" else s
                    for s, d, m in zip(df.scheme, df.d, df.memory)]
    for metric in ("uplink_imbalance_run", "uplink_queue_mean", "fct_mean_ms",
                   "small_fct_p99_ms", "drops_uplink"):
        print(f"\n{metric} (columns: load)")
        print(table(df, "policy", "load", metric))


if __name__ == "__main__":
    main()
