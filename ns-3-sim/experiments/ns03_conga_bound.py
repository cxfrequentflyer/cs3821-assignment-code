"""ns03 (ecmp-sim exp03): CONGA's imbalance theorem with real packets.

CONGA's Theorem 2 bounds the expected (max - min) / mean traffic over n links
by sqrt(8 n ln n (1 + CV^2) / m) for m flows. Here ECMP spreads Poisson
flow arrivals over leaf 0's n uplinks and we measure the imbalance of the
bytes each uplink carries in windows of length w; m is the expected number
of flows arriving per window. Bytes in a window come from flows that are
still sending, not from the flows that arrived in it, so this is the
dynamic analogue, not the theorem's exact setting.

    NS3_DIR=~/ns-3.49 python3 ns-3-sim/experiments/ns03_conga_bound.py [--quick]
"""
import math

from common import parser, run_grid, show

N, LOAD, MEAN, RATE = 8, 0.6, 100_000, 10e9
SIZES = {"fixed:100000": 0.0, "exp:100000": 1.0,
         "lognormal:1.5:100000": math.expm1(1.5 ** 2), "pareto:1.5:100000": math.inf}
WINDOWS = [100, 300, 1000, 3000]


def main():
    args = parser(__doc__).parse_args()
    flows = 1000 if args.quick else 4000
    configs = [{"scheme": "ecmp", "leaves": 2, "spines": N, "hosts": N, "coreRate": "10Gbps",
                "pattern": "pair:0:1", "sizes": s, "load": LOAD, "flows": flows,
                "maxFlowBytes": 1e8, "windowUs": w, "tag": "ns03"}
               for s in SIZES for w in WINDOWS]
    df = run_grid("ns03_conga_bound", configs, args)
    lam = LOAD * N * RATE / (8 * MEAN)                      # flows per second
    df["m"] = lam * df.window_us * 1e-6                     # flows arriving per window
    df["cv2"] = df.sizes.map(SIZES)
    df["conga_bound"] = [math.sqrt(8 * N * math.log(N) * (1 + c) / m) if math.isfinite(c)
                         else math.inf for c, m in zip(df.cv2, df.m)]
    print(f"\nECMP over {N} uplinks at load {LOAD}: windowed uplink imbalance vs CONGA's bound")
    df = df.sort_values(["window_us"], kind="stable")
    show(df, ["sizes", "window_us"], ["m", "uplink_imbalance_window", "conga_bound"])


if __name__ == "__main__":
    main()
