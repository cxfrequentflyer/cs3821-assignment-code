"""Time: flows arrive and leave, so the load on a path is a queueing process.

The static levels drop m flows onto n paths and stop. Here flows arrive one
after another (a Poisson process by default), a flow of size w put on path i
adds w bytes to that path's queue, and path i works its queue off at
speeds[i] bytes per unit time, first come first served. A flow has left when
its last byte has been sent. What a load-aware policy compares is no longer
"everything ever sent" but what is still queued when the flow arrives: the
path's backlog, measured as the time it needs to clear it (backlog / speed).

    df = run_dynamic(n=16, load=0.9, flows=Exponential(), arrivals=20_000, trials=20,
                     policies=[ECMP(), DChoice(2, replace=False),
                               DChoice(1, replace=False, memory=1)], seed=3821)
    summarize_dynamic(df)

`load` is the offered load rho: the arrival rate is rho * sum(speeds) flows
per unit time, and flows have mean size 1, so rho < 1 is "the paths could
keep up, if the traffic were split right". Whether they do is the question:
with ECMP every path is its own M/G/1 queue, so the simulator can be checked
against the Pollaczek-Khinchine formula (theory.mg1_wait); with unequal
speeds ECMP and memoryless d-choice can be unstable below rho = 1 and one
unit of memory fixes it (DRILL's Theorems 1 and 2, now over time).

The policies are the ones from policies.py, unchanged: ECMP, DChoice
(d, memory, engines, ties), Greedy, RoundRobin, HashECMP. They receive the
inter-arrival times and drain between flows (policies._place). Two things
are excluded: LPT, which needs the future, and by="count", because the
number of flows still on a path is not tracked (only bytes are).

Seeding extends runner.py's scheme: flow sizes and policy draws use the SAME
streams as the static run(), and the arrival times come from unit-rate gaps
keyed by (seed, "gaps", process, arrivals, trial), divided by the arrival
rate. So policies, loads and speeds all see the same flows in the same order,
and for a load-oblivious policy the paths are identical to the static run.

Metrics are measured at arrival instants, on what the arriving flow finds.
For Poisson arrivals that is also the time average (PASTA); for
process="constant" it is not, it is what flows experience.

Modeling choices to state: a path is a single FIFO queue with no buffer
limit (nothing is dropped, so an overloaded path just grows); a flow's bytes
are all there at its arrival (no TCP, no rate); every system starts empty
and the first `warmup` arrivals are placed but not measured.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .runner import _stack, rng_for

PROCESSES = ("poisson", "constant")


def unit_gaps(rng: np.random.Generator, m: int, process: str = "poisson") -> np.ndarray:
    """m inter-arrival times with mean 1 (the first is the time to the first flow)."""
    if process == "poisson":
        return rng.exponential(1.0, m)
    if process == "constant":
        return np.ones(m)
    raise ValueError(f"unknown arrival process {process!r}")


def queue_metrics(paths: np.ndarray, w: np.ndarray, gaps: np.ndarray, n: int, speeds=None,
                  warmup: int = 0) -> dict[str, np.ndarray]:
    """Replay a placement over time. One value per trial for each metric,
    over the arrivals from `warmup` on.

    wait_mean, wait_p99   time a flow spends queued behind earlier flows: the
                    clearing time of its path when it arrives
    fct_mean        flow completion time: wait + size / speed of its path
    slowdown_mean   mean of fct / size: 1 / speed would be an empty network
    queue_mean      clearing time averaged over paths and arrivals
    queue_max       clearing time of the busiest path, averaged over arrivals
    idle_frac       fraction of paths with nothing queued
    backlog_mean    total queued bytes, averaged over arrivals
    growth          how fast the total backlog grows between the first and
                    the last third of the measured arrivals, as a fraction of
                    the offered byte rate: about 0 when the system is stable,
                    and the share of the traffic that is piling up when not
    """
    T, m = w.shape
    if not 0 <= warmup < m:
        raise ValueError("warmup must be smaller than the number of arrivals")
    rows = np.arange(T)
    s = np.ones(n) if speeds is None else np.asarray(speeds, dtype=float)
    q = np.zeros((T, n))                              # clearing time of each path
    wait = np.empty((T, m))
    backlog = np.empty((T, m))
    q_mean, q_max, idle = np.zeros(T), np.zeros(T), np.zeros(T)
    for j in range(m):
        q -= gaps[:, j, None]
        np.maximum(q, 0.0, out=q)
        p = paths[:, j]
        wait[:, j] = q[rows, p]
        backlog[:, j] = q @ s
        if j >= warmup:
            q_mean += q.mean(axis=1)
            q_max += q.max(axis=1)
            idle += (q == 0).mean(axis=1)
        q[rows, p] += w[:, j] / s[p]
    k = m - warmup
    service = w / s[paths]
    fct = wait + service
    wt, ft, sz, bl = wait[:, warmup:], fct[:, warmup:], w[:, warmup:], backlog[:, warmup:]
    with np.errstate(divide="ignore", invalid="ignore"):
        slowdown = np.nanmean(np.where(sz > 0, ft / sz, np.nan), axis=1)
    third = max(k // 3, 1)
    when = np.cumsum(gaps, axis=1)[:, warmup:]
    span = when[:, -third:].mean(axis=1) - when[:, :third].mean(axis=1)
    offered = sz.sum(axis=1) / np.maximum(when[:, -1] - when[:, 0], 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = (bl[:, -third:].mean(axis=1) - bl[:, :third].mean(axis=1)) / span / offered
    return {
        "wait_mean": wt.mean(axis=1),
        "wait_p99": np.quantile(wt, 0.99, axis=1),
        "fct_mean": ft.mean(axis=1),
        "slowdown_mean": slowdown,
        "queue_mean": q_mean / k,
        "queue_max": q_max / k,
        "idle_frac": idle / k,
        "backlog_mean": bl.mean(axis=1),
        "growth": growth,
    }


def run_dynamic(n: int, load: float, flows, policies: Sequence, arrivals: int = 20_000,
                warmup: int | None = None, trials: int = 20, seed: int = 0, speeds=None,
                process: str = "poisson", batch_size: int | None = None) -> pd.DataFrame:
    """Simulate `trials` independent runs of `arrivals` flows through one ECMP
    group of n draining paths and score every policy on each of them.

    load     offered load rho = arrival rate * mean flow size / sum(speeds)
    warmup   arrivals to place before measuring (default: a quarter). Systems
             start empty, so at high load make this, and `arrivals`, larger:
             a queue at load rho forgets its start after roughly
             1 / (1 - rho)^2 arrivals per path
    speeds   optional (n,) path speeds (default all 1)
    process  "poisson" or "constant" inter-arrival times

    Returns one row per (policy, trial): the configuration columns followed by
    the metrics from queue_metrics.
    """
    if len(set(p.name for p in policies)) != len(policies):
        raise ValueError("policy names must be distinct")
    for p in policies:
        if getattr(p, "offline", False):
            raise ValueError(f"{p.name} needs every flow up front and cannot run over time")
        if getattr(p, "by", "bytes") != "bytes":
            raise ValueError(f"{p.name}: by='count' is not modeled over time, only bytes drain")
        if getattr(p, "uses_flow", False):
            raise ValueError(f"{p.name} works on chunks, which run_dynamic does not model")
    if load <= 0:
        raise ValueError("load must be positive")
    if speeds is not None:
        speeds = np.asarray(speeds, dtype=float)
        if speeds.shape != (n,) or np.any(speeds <= 0):
            raise ValueError("speeds must be n positive numbers")
    if warmup is None:
        warmup = arrivals // 4
    m = arrivals
    rate = load * (n if speeds is None else speeds.sum())
    if batch_size is None:
        batch_size = max(1, min(trials, 4_000_000 // m))
    frames = []
    for start in range(0, trials, batch_size):
        ids = range(start, min(start + batch_size, trials))
        w = np.stack([flows.sample(rng_for(seed, "flows", flows.label, m, t), m) for t in ids])
        gaps = np.stack([unit_gaps(rng_for(seed, "gaps", process, m, t), m, process)
                         for t in ids]) / rate
        for pol in policies:
            key = getattr(pol, "seed_name", pol.name)
            draws = _stack([pol.draw(rng_for(seed, "policy", key, n, m, t), m, n) for t in ids])
            paths = pol.assign(w, draws, n, speeds=speeds, gaps=gaps)
            frame = pd.DataFrame(queue_metrics(paths, w, gaps, n, speeds, warmup))
            frame.insert(0, "trial", np.arange(ids.start, ids.stop))
            frame.insert(0, "policy", pol.name)
            frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    order = {p.name: i for i, p in enumerate(policies)}
    df = df.sort_values(["policy", "trial"], key=lambda c: c.map(order) if c.name == "policy" else c,
                        kind="stable").reset_index(drop=True)
    for i, (col, val) in enumerate([("flows", flows.label), ("n", n), ("load", float(load)),
                                    ("arrivals", m), ("process", process), ("seed", seed)]):
        df.insert(i, col, val)
    return df


def summarize_dynamic(df: pd.DataFrame,
                      metrics: Sequence[str] = ("wait_mean", "wait_p99", "queue_max",
                                                "idle_frac", "growth"),
                      by: Sequence[str] = ("flows", "n", "load", "policy")) -> pd.DataFrame:
    """Mean over trials of each metric, with a 95% CI half-width (trials are
    independent runs, so this is the honest way to get an error bar out of
    autocorrelated queues)."""
    groups = df.groupby(list(by), sort=False)
    out = {}
    for col in metrics:
        s = groups[col]
        out[col] = s.mean()
        out[f"{col}_ci95"] = 1.96 * s.std(ddof=1) / np.sqrt(s.count())
    return pd.DataFrame(out).reset_index()
