"""Real hashes: structured 5-tuples and hash polarization.

Part A, one ECMP group (ecmpsim.hashing). Path = hash(5-tuple) mod n for
CRC-16/CCITT, CRC-16/ARC, CRC-32 and a 16-bit XOR fold, against the ideal
hash, on unit flows so that only the hash matters. Traffic:

    random       uniform bits (the hash's best case)
    datacenter   1024 hosts in 10.pod.edge.host, ephemeral ports, a few services
    sequential   one client and server, source port counting up
    symmetric    source port = destination port (they cancel in an XOR)
    incast       many clients to one server and service

Reported: busiest path / mean (1 = perfectly even) and the fraction of paths
that carry nothing. A deterministic hash can be BETTER than ideal (a counter
through an XOR fold is a round robin) or far worse.

Part B, a k-ary fat-tree (ecmpsim.fattree), inter-pod traffic: what happens
when the edge and the aggregation tier hash the same way.

    same hash           every switch computes crc16(5-tuple)
    + xor seeds         each switch XORs in its own seed (= seeding the CRC)
    other polynomial    crc16 at the edge, crc16-arc at the aggregation tier
    other bits          crc16 at the edge, bits 8.. of it at the aggregation tier
    xor16, + other bits the weak hash, with the same cure
    ideal               independent uniform choices (FTECMP)

    python experiments/exp10_hashes.py [--rerun] [--k 8]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from ecmpsim import ECMP, Hash, HashECMP, Unit, run
from ecmpsim.fattree import FTECMP, FatTree, FTHashECMP, run_fattree
from ecmpsim.hashing import TRAFFIC_KINDS

SEED, TRIALS, PER_PATH = 3821, 500, 16
NS = [8, 12, 64]
HASHES = ["crc16", "crc16-arc", "crc32", "xor16"]
FT_TRIALS, FT_PER_HOST = 60, 16
FT_POLICIES = {
    "same hash": FTHashECMP("crc16"),
    "+ xor seeds": FTHashECMP("crc16", seeds="xor"),
    "other polynomial": FTHashECMP("crc16", "crc16-arc"),
    "other bits": FTHashECMP("crc16", Hash("crc16", shift=8)),
    "xor16": FTHashECMP("xor16"),
    "xor16, other bits": FTHashECMP("xor16", Hash("xor16", shift=8)),
    "ideal": FTECMP(),
}
CSV_A = ROOT / "results" / "exp10_hashes_group.csv"


def simulate_group() -> pd.DataFrame:
    rows = []
    for n in NS:
        pols = [ECMP()] + [HashECMP(h, t) for t in TRAFFIC_KINDS for h in HASHES]
        df = run(n, PER_PATH * n, Unit(), pols, trials=TRIALS, seed=SEED)
        for p in pols:
            g = df[df.policy == p.name]
            hashed = isinstance(p, HashECMP)
            rows.append({"n": n, "traffic": p.traffic if hashed else "(ideal hash)",
                         "hash": p.hash.label if hashed else "ideal",
                         "max_over_mean": g.max_over_mean.mean(), "idle_frac": g.idle_frac.mean()})
        print(f"  n = {n} done", flush=True)
    out = pd.DataFrame(rows)
    CSV_A.parent.mkdir(exist_ok=True)
    out.to_csv(CSV_A, index=False)
    return out


def simulate_fattree(k, csv) -> pd.DataFrame:
    net = FatTree.kary(k)
    parts = []
    for ports in ("ephemeral", "symmetric"):
        df = run_fattree(net, FT_PER_HOST * k ** 3 // 4, Unit(), list(FT_POLICIES.values()),
                         trials=FT_TRIALS, seed=SEED, pattern="interpod", ports=ports)
        parts.append(df.assign(ports=ports))
        print(f"  fat-tree / {ports} ports done", flush=True)
    names = {p.name: label for label, p in FT_POLICIES.items()}
    out = pd.concat(parts, ignore_index=True)
    out["scheme"] = out.policy.map(names)
    csv.parent.mkdir(exist_ok=True)
    out.to_csv(csv, index=False)
    return out


def report_group(df):
    ideal = df[df.hash == "ideal"].set_index("n")
    for n, g in df[df.hash != "ideal"].groupby("n"):
        print(f"\nOne group, n = {n} paths, {PER_PATH * n} unit flows, {TRIALS} trials. "
              f"Ideal hash: busiest/mean = {ideal.max_over_mean[n]:.3f}, "
              f"idle paths = {ideal.idle_frac[n]:.3f}")
        t = g.pivot(index="traffic", columns="hash", values="max_over_mean")
        idle = g.pivot(index="traffic", columns="hash", values="idle_frac")
        t = t.loc[list(TRAFFIC_KINDS), HASHES].round(3)
        t["xor16 idle paths"] = idle.loc[list(TRAFFIC_KINDS), "xor16"].round(3)
        print(t.to_string())


def report_fattree(df, k):
    half = k // 2
    print(f"\nFat-tree k = {k}, inter-pod unit flows, {FT_TRIALS} trials: {half * half} cores, "
          f"{half} uplinks per aggregation switch")
    for ports, g in df.groupby("ports", sort=False):
        t = (g.assign(agg_up=g.agg_up_max / g.opt, edge_up=g.edge_up_max / g.opt)
             .groupby("scheme", sort=False)[["cores_used", "edge_up", "agg_up", "over_opt"]].mean())
        print(f"\n{ports} ports (busiest link of a tier / optimum)")
        print(t.loc[list(FT_POLICIES)].round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    csv_b = ROOT / "results" / f"exp10_hashes_fattree_k{args.k}.csv"
    report_group(simulate_group() if args.rerun or not CSV_A.exists() else pd.read_csv(CSV_A))
    report_fattree(simulate_fattree(args.k, csv_b) if args.rerun or not csv_b.exists()
                   else pd.read_csv(csv_b), args.k)


if __name__ == "__main__":
    main()
