"""Shared runner for the ns-3 experiments: run lb-fabric over a grid of
configurations in parallel, cache the results, and summarize them.

Every experiment script builds a list of configurations (dicts of lb-fabric
command-line options), calls `run_grid`, and gets back one DataFrame row per
(configuration, seed). Results are cached in results/<name>.csv: rerunning a
script only simulates configurations that are missing, unless --rerun.

Finding the simulator: set NS3_DIR to your ns-3.49 directory (the one with the
`ns3` script, whose `scratch` is a link to this repo's ns-3-sim), build it
once with `./ns3 build lb-fabric`, and the scripts call the built binary
directly (much faster than `./ns3 run` for hundreds of runs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
WORKLOADS = REPO / "ecmp-sim" / "workloads"
WEBSEARCH = f"empirical:{WORKLOADS / 'websearch.txt'}"
DATAMINING = f"empirical:{WORKLOADS / 'datamining.txt'}"


def find_binary(ns3_dir: str | None = None) -> Path:
    root = Path(ns3_dir or os.environ.get("NS3_DIR", ""))
    if not root or not root.exists():
        sys.exit("Set NS3_DIR (or pass --ns3) to your ns-3.49 directory; see ns-3-sim/README.md")
    found = sorted((root / "build" / "scratch" / "lb-fabric").glob("ns3*-lb-fabric-*"),
                   key=lambda p: ("optimized" not in p.name, p.name))
    if not found:
        sys.exit(f"No lb-fabric binary under {root}/build/scratch/lb-fabric: "
                 "run `./ns3 build lb-fabric` there first")
    return found[0]


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--ns3", help="ns-3.49 directory (default: $NS3_DIR)")
    ap.add_argument("--rerun", action="store_true", help="simulate everything again")
    ap.add_argument("--seeds", type=int, default=2, help="runs per configuration")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2, help="parallel runs")
    ap.add_argument("--quick", action="store_true", help="fewer flows: a fast smoke test")
    return ap


def _key(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _run_one(binary: Path, cfg: dict) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "row.csv"
        args = [str(binary)] + [f"--{k}={v}" for k, v in cfg.items()] + [f"--out={out}"]
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"lb-fabric failed ({res.returncode}): {' '.join(args)}\n"
                               f"{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
        df = pd.read_csv(out)
    df["config_key"] = _key(cfg)
    return df


def run_grid(name: str, configs: list[dict], args) -> pd.DataFrame:
    """Run every config for seeds 1..args.seeds; returns the cached + new rows
    for exactly these configs."""
    binary = find_binary(args.ns3)
    RESULTS.mkdir(exist_ok=True)
    csv = RESULTS / f"{name}.csv"
    cached = pd.read_csv(csv) if csv.exists() and not args.rerun else pd.DataFrame()
    todo, keep = [], []
    for cfg in configs:
        for seed in range(1, args.seeds + 1):
            full = {**cfg, "seed": seed}
            k = _key(full)
            if len(cached) and (cached.get("config_key") == k).any():
                keep.append(k)
            else:
                todo.append(full)
    print(f"{name}: {len(todo)} runs to simulate, {len(keep)} cached ({binary.name}, "
          f"{args.jobs} at a time)", flush=True)
    new = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_one, binary, cfg): cfg for cfg in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            new.append(fut.result())
            if i % max(1, len(todo) // 10) == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done", flush=True)
    frames = [f for f in [cached, *new] if len(f)]
    allrows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(new):
        allrows.drop_duplicates("config_key", keep="last").to_csv(csv, index=False)
    wanted = {_key({**c, "seed": s}) for c in configs for s in range(1, args.seeds + 1)}
    return allrows[allrows.config_key.isin(wanted)].reset_index(drop=True)


def table(df: pd.DataFrame, index, columns, value: str, digits: int = 3) -> str:
    """Mean over seeds, pivoted."""
    t = df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean", sort=False)
    try:
        t = t.reindex(sorted(t.columns), axis=1)      # numeric sweeps read left to right
    except TypeError:
        pass
    return t.round(digits).to_string()


def show(df: pd.DataFrame, by, metrics, digits: int = 3) -> None:
    g = df.groupby(by, sort=False)[metrics].mean()
    print(g.round(digits).to_string())
