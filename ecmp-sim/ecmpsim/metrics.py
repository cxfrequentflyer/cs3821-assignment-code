"""Turn path assignments into numbers.

All functions work on a batch of T trials at once: `paths` and `w` are (T, m).
"""
from __future__ import annotations

import numpy as np


def path_loads(paths: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """(T, n) total weight on each path."""
    T = paths.shape[0]
    flat = (paths + n * np.arange(T)[:, None]).ravel()
    return np.bincount(flat, weights=w.ravel(), minlength=T * n).reshape(T, n)


def path_counts(paths: np.ndarray, n: int, mask: np.ndarray | None = None) -> np.ndarray:
    """(T, n) number of flows on each path. `mask` (T, m) marks the real flows
    when the batch was padded (chunked runs pad with zero-weight entries)."""
    T = paths.shape[0]
    flat = (paths + n * np.arange(T)[:, None]).ravel()
    weights = None if mask is None else mask.ravel().astype(float)
    return np.bincount(flat, weights=weights, minlength=T * n).reshape(T, n).astype(np.int64)


def opt_lower_bound(w: np.ndarray, n: int, speeds=None) -> np.ndarray:
    """No placement of unsplittable flows can beat max(total / n, largest flow).

    With path speeds this is a bound on the busiest *utilization* (load /
    speed): max(total / sum(speeds), largest flow / fastest speed)."""
    if speeds is None:
        return np.maximum(w.sum(axis=1) / n, w.max(axis=1))
    speeds = np.asarray(speeds, dtype=float)
    return np.maximum(w.sum(axis=1) / speeds.sum(), w.max(axis=1) / speeds.max())


def trial_metrics(paths: np.ndarray, w: np.ndarray, n: int, capacity: float | None = None,
                  speeds=None, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """One value per trial for each metric.

    max_load        weight on the busiest path
    gap             max_load - total/n: what balls-into-bins theorems describe
    max_over_mean   max_load / (total/n): unitless imbalance
    imbalance       (busiest - least loaded) / (total/n): CONGA's chi(t), the
                    quantity its Theorem 2 bounds
    max_over_lb     max_load / max(total/n, largest flow): at least 1 and at
                    most n, so it stays well behaved under heavy tails; 1 means
                    no placement could have done better
    lb_is_elephant  True when one flow is bigger than a fair share, so some
                    imbalance is unavoidable whatever the policy does
    idle_frac       fraction of paths carrying no flow
    max_count       flows on the most crowded path
    throughput      only when `capacity` is given: weights are rate demands,
                    a path delivers min(capacity, its load) (any work-conserving
                    sharing, e.g. max-min fair), relative to the splittable
                    optimum min(n * capacity, total)

    With `speeds` (relative path capacities), max_over_mean, imbalance and
    max_over_lb are computed on utilization, load / speed, against the ideal
    total / sum(speeds); with equal speeds of 1 nothing changes. Path i's
    capacity for `throughput` becomes capacity * speeds[i].
    """
    load = path_loads(paths, w, n)
    count = path_counts(paths, n, mask)
    total = w.sum(axis=1)
    mean = total / n
    max_load = load.max(axis=1)
    if speeds is None:
        util, ideal = load, mean
    else:
        speeds = np.asarray(speeds, dtype=float)
        util, ideal = load / speeds, total / speeds.sum()
    lb = opt_lower_bound(w, n, speeds)
    out = {
        "max_load": max_load,
        "gap": max_load - mean,
        "max_over_mean": util.max(axis=1) / ideal,
        "imbalance": (util.max(axis=1) - util.min(axis=1)) / ideal,
        "max_over_lb": util.max(axis=1) / lb,
        "lb_is_elephant": lb > ideal,
        "idle_frac": (count == 0).mean(axis=1),
        "max_count": count.max(axis=1),
    }
    if capacity is not None:
        cap = capacity * (np.ones(n) if speeds is None else speeds)
        out["throughput"] = (np.minimum(load, cap).sum(axis=1)
                             / np.minimum(cap.sum(), total))
    return out
