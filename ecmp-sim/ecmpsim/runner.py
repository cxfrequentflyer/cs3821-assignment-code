"""Run seeded, reproducible experiments and summarize them.

Seeding scheme. Every random stream is keyed by *what it is for*, never by the
order in which code happens to run:

    flow sizes     (seed, "flows",  flows.label, m, trial)
    policy draws   (seed, "policy", policy.seed_name or name, n, m, trial[, "chunk", c])

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

from .chunking import chunk_flows
from .metrics import trial_metrics


def _key(x) -> int:
    if isinstance(x, (int, np.integer)) and x >= 0:
        return int(x)
    return zlib.crc32(str(x).encode())


def rng_for(seed: int, *keys) -> np.random.Generator:
    """An independent generator for one purpose, e.g. rng_for(3821, "flows", "unit", 64, 7)."""
    return np.random.default_rng(
        np.random.SeedSequence(_key(seed), spawn_key=tuple(_key(k) for k in keys)))


def _stack(per_trial):
    """Stack one draw per trial into a batch, padding ragged draws with zeros
    (chunked trials have different numbers of chunks). A tuple of draws is
    stacked component by component."""
    if per_trial[0] is None:
        return None
    if isinstance(per_trial[0], tuple):
        return tuple(_stack([d[i] for d in per_trial]) for i in range(len(per_trial[0])))
    size = max(len(d) for d in per_trial)
    if all(len(d) == size for d in per_trial):
        return np.stack(per_trial)
    out = np.zeros((len(per_trial), size, *per_trial[0].shape[1:]), dtype=per_trial[0].dtype)
    for i, d in enumerate(per_trial):
        out[i, :len(d)] = d
    return out


def _pad(rows, fill):
    out = np.full((len(rows), max(len(r) for r in rows)), fill, dtype=rows[0].dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def run(n: int, m: int, flows, policies: Sequence, trials: int = 1000, seed: int = 0,
        capacity: float | None = None, batch_size: int | None = None,
        speeds=None, chunk: float | None = None, chunk_order: str = "interleave") -> pd.DataFrame:
    """Simulate `trials` independent flow sets on one ECMP group of n paths and
    score every policy on each of them.

    speeds  optional (n,) relative path capacities: load-aware policies and the
            utilization metrics use load / speed (DRILL's unequal service rates).
    chunk   optional chunk size c (in mean flow sizes): every flow is split into
            chunks (chunking.chunk_flows) and the chunks are what get placed.
            The flow sets are the same as without chunking, so results pair up
            across chunk sizes. Padding is masked out of the count metrics, and
            a `chunks` column records how many chunks each trial had.
            The lower bound in max_over_lb then uses the largest *chunk*; to
            compare with unchunked runs use max_over_mean or imbalance.

    Returns a long-format DataFrame with one row per (policy, trial): the
    configuration columns followed by the metrics from trial_metrics.
    """
    if len(set(p.name for p in policies)) != len(policies):
        raise ValueError("policy names must be distinct")
    if speeds is not None:
        speeds = np.asarray(speeds, dtype=float)
        if speeds.shape != (n,) or np.any(speeds <= 0):
            raise ValueError("speeds must be n positive numbers")
    per_flow = 1 if chunk is None else 1 + 1 / chunk          # rough balls per flow
    if batch_size is None:
        batch_size = max(1, min(trials, int(2_000_000 // max(m * per_flow, 1))))
    extra = () if chunk is None else ("chunk", repr(float(chunk)))
    frames = []
    for start in range(0, trials, batch_size):
        ids = range(start, min(start + batch_size, trials))
        w = np.stack([flows.sample(rng_for(seed, "flows", flows.label, m, t), m) for t in ids])
        context, mask, sizes = {}, None, [m] * len(ids)
        if chunk is not None:
            parts = [chunk_flows(row, chunk, chunk_order) for row in w]
            sizes = [len(p[0]) for p in parts]
            w = _pad([p[0] for p in parts], 0.0)
            context = {"flow": _pad([p[1] for p in parts], -1), "rank": _pad([p[2] for p in parts], 0)}
            mask = w > 0
        if speeds is not None:
            context["speeds"] = speeds
        for pol in policies:
            uses_flow = getattr(pol, "uses_flow", False)
            key = getattr(pol, "seed_name", pol.name)
            draws = _stack([pol.draw(rng_for(seed, "policy", key, n, m, t, *extra),
                                     m if uses_flow else size, n)
                            for t, size in zip(ids, sizes)])
            kw = {k: v for k, v in context.items() if uses_flow or k == "speeds"}
            met = trial_metrics(pol.assign(w, draws, n, **kw), w, n, capacity, speeds, mask)
            frame = pd.DataFrame(met)
            if chunk is not None:
                frame["chunks"] = sizes
            frame.insert(0, "trial", np.arange(ids.start, ids.stop))
            frame.insert(0, "policy", pol.name)
            frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    order = {p.name: i for i, p in enumerate(policies)}
    df = df.sort_values(["policy", "trial"], key=lambda s: s.map(order) if s.name == "policy" else s,
                        kind="stable").reset_index(drop=True)
    config = [("flows", flows.label), ("n", n), ("m", m), ("t", m / n), ("seed", seed)]
    if chunk is not None:
        config.append(("chunk", float(chunk)))
    for i, (col, val) in enumerate(config):
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
