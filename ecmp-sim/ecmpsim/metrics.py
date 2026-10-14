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


def path_counts(paths: np.ndarray, n: int) -> np.ndarray:
    """(T, n) number of flows on each path."""
    T = paths.shape[0]
    flat = (paths + n * np.arange(T)[:, None]).ravel()
    return np.bincount(flat, minlength=T * n).reshape(T, n)


def opt_lower_bound(w: np.ndarray, n: int) -> np.ndarray:
    """No placement of unsplittable flows can beat max(total / n, largest flow)."""
    return np.maximum(w.sum(axis=1) / n, w.max(axis=1))


def trial_metrics(paths: np.ndarray, w: np.ndarray, n: int,
                  capacity: float | None = None) -> dict[str, np.ndarray]:
    """One value per trial for each metric.

    max_load        weight on the busiest path
    gap             max_load - total/n: what balls-into-bins theorems describe
    max_over_mean   max_load / (total/n): unitless imbalance
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
    """
    load = path_loads(paths, w, n)
    count = path_counts(paths, n)
    total = w.sum(axis=1)
    mean = total / n
    max_load = load.max(axis=1)
    out = {
        "max_load": max_load,
        "gap": max_load - mean,
        "max_over_mean": max_load / mean,
        "max_over_lb": max_load / opt_lower_bound(w, n),
        "lb_is_elephant": w.max(axis=1) > mean,
        "idle_frac": (count == 0).mean(axis=1),
        "max_count": count.max(axis=1),
    }
    if capacity is not None:
        out["throughput"] = (np.minimum(load, capacity).sum(axis=1)
                             / np.minimum(n * capacity, total))
    return out
