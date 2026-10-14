"""ecmpsim: a small balls-into-bins simulator for one ECMP group.

Flows are balls, equal-cost paths are bins, and a flow's size is its weight.
See README.md for the layout, the seeding scheme and the validation story.
"""
from . import exact, theory
from .flows import Bimodal, Empirical, Exponential, LogNormal, Pareto, Unit
from .metrics import opt_lower_bound, path_counts, path_loads, trial_metrics
from .policies import ECMP, LPT, DChoice, Greedy, sample_distinct
from .runner import bootstrap_ci, paired_diff, rng_for, run, summarize, sweep

__all__ = [
    "exact", "theory",
    "Unit", "Exponential", "Pareto", "LogNormal", "Bimodal", "Empirical",
    "ECMP", "DChoice", "Greedy", "LPT", "sample_distinct",
    "path_loads", "path_counts", "opt_lower_bound", "trial_metrics",
    "run", "sweep", "summarize", "paired_diff", "bootstrap_ci", "rng_for",
]
