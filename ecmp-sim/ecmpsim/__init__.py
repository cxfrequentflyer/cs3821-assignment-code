"""ecmpsim: a small balls-into-bins simulator for one ECMP group, a two-tier
leaf-spine fabric and a three-tier fat-tree for when one group is not enough,
real hash functions for when the ideal one is not, and draining queues for
when a snapshot is not.

Flows are balls, equal-cost paths are bins, and a flow's size is its weight.
See README.md for the layout, the seeding scheme and the validation story.
"""
from . import dynamic, exact, fattree, hashing, leafspine, theory
from .chunking import chunk_cv2, chunk_flows
from .dynamic import queue_metrics, run_dynamic, summarize_dynamic
from .flows import Bimodal, Empirical, Exponential, LogNormal, Pareto, Unit
from .hashing import Hash, HashECMP, five_tuples
from .metrics import opt_lower_bound, path_counts, path_loads, trial_metrics
from .policies import ECMP, LPT, DChoice, Greedy, RoundRobin, Spray, sample_distinct
from .runner import bootstrap_ci, paired_diff, rng_for, run, summarize, sweep

__all__ = [
    "exact", "leafspine", "fattree", "hashing", "dynamic", "theory",
    "Unit", "Exponential", "Pareto", "LogNormal", "Bimodal", "Empirical",
    "ECMP", "DChoice", "Greedy", "LPT", "RoundRobin", "Spray", "sample_distinct",
    "Hash", "HashECMP", "five_tuples",
    "run_dynamic", "summarize_dynamic", "queue_metrics",
    "chunk_flows", "chunk_cv2",
    "path_loads", "path_counts", "opt_lower_bound", "trial_metrics",
    "run", "sweep", "summarize", "paired_diff", "bootstrap_ci", "rng_for",
]
