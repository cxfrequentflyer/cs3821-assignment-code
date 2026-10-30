"""Split flows into chunks: placing traffic at a finer granularity.

The papers place traffic in very different units: ECMP whole flows, CONGA
flowlets, Presto 64KB flowcells, DRILL and pFabric single packets. In
balls-into-bins terms, cutting a flow into chunks of size c replaces one
heavy ball with many light ones. That lowers the coefficient of variation of
the ball sizes, which is what drives CONGA's Theorem 2 (theory.conga_imbalance_bound).

Sizes here are in units of the mean flow size (flows have mean 1), so c = 0.1
means a chunk is a tenth of an average flow. A flow of size s becomes
ceil(s / c) chunks: full chunks of size c and one remainder.

What this does not model: the cost of splitting (packet reordering, which the
ns-3 track can measure), and flowlets, whose sizes depend on gaps in a flow's
sending rather than on a fixed size.
"""
from __future__ import annotations

import numpy as np


def chunk_flows(w: np.ndarray, c: float, order: str = "interleave"):
    """Split ONE trial's flows (shape (m,)) into chunks of size at most c.

    Returns (sizes, flow, rank): each chunk's weight, the index of the flow it
    came from, and its position inside that flow (0, 1, ...).

    order="interleave"  every flow's first chunk, then every flow's second
                        chunk, ...: flows send at the same time, as they do in
                        a snapshot. The default.
    order="flow"        all of flow 0's chunks, then all of flow 1's, ...
    Load-oblivious policies don't care; load-aware ones do, a little.
    """
    if c <= 0:
        raise ValueError("chunk size must be positive")
    if order not in ("interleave", "flow"):
        raise ValueError("order must be 'interleave' or 'flow'")
    w = np.asarray(w, dtype=float)
    # chunks per flow; the 1e-9 stops rounding error from adding an empty last chunk
    k = np.maximum(1, np.ceil(w / c - 1e-9)).astype(np.int64)
    flow = np.repeat(np.arange(len(w)), k)
    first = np.cumsum(k) - k
    rank = np.arange(k.sum()) - np.repeat(first, k)
    sizes = np.full(k.sum(), float(c))
    sizes[first + k - 1] = w - (k - 1) * c                    # the remainder (0 < r <= c)
    if order == "interleave":
        perm = np.lexsort((flow, rank))
        sizes, flow, rank = sizes[perm], flow[perm], rank[perm]
    return sizes, flow, rank


def chunk_cv2(w: np.ndarray, c: float) -> float:
    """Squared coefficient of variation of the chunk sizes of one flow set."""
    sizes, _, _ = chunk_flows(w, c, order="flow")
    return float(sizes.var() / sizes.mean() ** 2)
