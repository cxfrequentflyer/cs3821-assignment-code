"""Run seeded, reproducible experiments and summarize them.

Seeding scheme. Every random stream is keyed by *what it is for*, never by the
order in which code happens to run:

    flow sizes     (seed, "flows",  flows.label, m, trial)
    policy draws   (seed, "policy", policy.name, n, m, trial)

So every policy sees exactly the same flows (paired comparisons), the same
flow sets are reused across n and across policies, adding or removing a
policy changes nothing else, and batch_size only affects speed and memory.
String keys go through CRC-32 rather than Python's hash(), which is salted
differently in every process.
"""
from __future__ import annotations

import zlib
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import trial_metrics


def _key(x) -> int:
    if isinstance(x, (int, np.integer)) and x >= 0:
        return int(x)
    return zlib.crc32(str(x).encode())


def rng_for(seed: int, *keys) -> np.random.Generator:
    """An independent generator for one purpose, e.g. rng_for(3821, "flows", "unit", 64, 7)."""
    return np.random.default_rng(
        np.random.SeedSequence(_key(seed), spawn_key=tuple(_key(k) for k in keys)))


def run(n: int, m: int, flows, policies: Sequence, trials: int = 1000, seed: int = 0,
        capacity: float | None = None, batch_size: int | None = None) -> pd.DataFrame:
    """Simulate `trials` independent flow sets on one ECMP group of n paths and
    score every policy on each of them.

    Returns a long-format DataFrame with one row per (policy, trial): the
    configuration columns followed by the metrics from trial_metrics.
    """
    if batch_size is None:
        batch_size = max(1, min(trials, 2_000_000 // max(m, 1)))
    frames = []
    for start in range(0, trials, batch_size):
        ids = range(start, min(start + batch_size, trials))
        w = np.stack([flows.sample(rng_for(seed, "flows", flows.label, m, t), m) for t in ids])
        for pol in policies:
            per_trial = [pol.draw(rng_for(seed, "policy", pol.name, n, m, t), m, n) for t in ids]
            draws = None if per_trial[0] is None else np.stack(per_trial)
            met = trial_metrics(pol.assign(w, draws, n), w, n, capacity)
            frame = pd.DataFrame(met)
            frame.insert(0, "trial", np.arange(ids.start, ids.stop))
            frame.insert(0, "policy", pol.name)
            frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    order = {p.name: i for i, p in enumerate(policies)}
    df = df.sort_values(["policy", "trial"], key=lambda s: s.map(order) if s.name == "policy" else s,
                        kind="stable").reset_index(drop=True)
    for i, (col, val) in enumerate([("flows", flows.label), ("n", n), ("m", m),
                                    ("t", m / n), ("seed", seed)]):
        df.insert(i, col, val)
    return df


def sweep(configs: Iterable[dict], **common) -> pd.DataFrame:
    """Run many configurations. Each config is a dict of run() arguments
    (typically n, m, flows); `common` supplies the rest (policies, trials, seed...)."""
    return pd.concat([run(**{**common, **c}) for c in configs], ignore_index=True)


def summarize(df: pd.DataFrame,
              metrics: Sequence[str] = ("max_load", "gap", "max_over_lb", "idle_frac"),
              by: Sequence[str] = ("flows", "n", "m", "policy"),
              quantiles: Sequence[float] = (0.5, 0.99)) -> pd.DataFrame:
    """Mean with a normal-approximation 95% CI, plus quantiles, per group.

    The CI is only trustworthy when the metric has finite variance. Raw loads
    under Pareto with alpha < 2 do not: use max_over_lb, the quantiles, or
    bootstrap_ci instead.
    """
    groups = df.groupby(list(by), sort=False)
    out = {}
    for col in metrics:
        if col not in df:
            continue
        s = groups[col]
        out[f"{col}_mean"] = s.mean()
        out[f"{col}_ci95"] = 1.96 * s.std(ddof=1) / np.sqrt(s.count())
        for q in quantiles:
            out[f"{col}_p{round(100 * q)}"] = s.quantile(q)
    return pd.DataFrame(out).reset_index()


def paired_diff(df: pd.DataFrame, metric: str, baseline: str = "ecmp",
                by: Sequence[str] = ("flows", "n", "m")) -> pd.DataFrame:
    """Mean of (policy - baseline) over identical trials, with a 95% CI.

    Every policy saw the same flows, so the per-trial difference is far less
    noisy than either policy on its own. Use this for "is d = 2 better, and by
    how much?" rather than comparing two separate confidence intervals.
    """
    wide = df.pivot_table(index=[*by, "trial"], columns="policy", values=metric)
    parts = []
    for pol in wide.columns:
        if pol == baseline:
            continue
        diff = (wide[pol] - wide[baseline]).groupby(level=list(by)).agg(["mean", "std", "count"])
        diff["ci95"] = 1.96 * diff["std"] / np.sqrt(diff["count"])
        diff["policy"] = pol
        parts.append(diff.reset_index())
    return pd.concat(parts, ignore_index=True)[[*by, "policy", "mean", "ci95", "count"]]


def bootstrap_ci(x, stat=np.median, reps: int = 2000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap: returns (estimate, low, high).

    `stat` must accept an `axis` argument (np.median, np.mean, or e.g.
    lambda a, axis: np.quantile(a, 0.99, axis=axis)). Use it for medians,
    tail quantiles and anything whose variance may be infinite.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = stat(x[rng.integers(0, len(x), size=(reps, len(x)))], axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return float(stat(x, axis=0)), float(lo), float(hi)
