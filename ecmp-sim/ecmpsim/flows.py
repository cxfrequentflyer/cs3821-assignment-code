"""Flow-size distributions: the weights of the balls.

Every distribution here is scaled to have mean 1. Then m/n is always the
expected average load per path in units of "one average flow", so results for
different distributions share an axis.

What a weight *means* is a modeling choice to state in the report. In a
snapshot of the network it is a flow's rate demand (what it would send with a
link to itself). Over a time window it is the number of bytes the flow sends.

Every class has a `label` (used in result tables and for seeding) and a
`sample(rng, size)` method returning an array of positive weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Unit:
    """Every flow has weight 1: the classical, unweighted balls-into-bins model."""

    @property
    def label(self) -> str:
        return "unit"

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        return np.ones(size)


@dataclass(frozen=True)
class Exponential:
    """Exponential sizes with mean 1: light-tailed, but no two flows are equal."""

    @property
    def label(self) -> str:
        return "exp"

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        return rng.exponential(1.0, size)


@dataclass(frozen=True)
class Pareto:
    """Pareto (type I) sizes: P(X > x) = (x_min / x) ** alpha for x >= x_min.

    `alpha` is the tail index (the shape parameter). The mean is finite only
    for alpha > 1 and the variance only for alpha > 2, so alpha = 1.5 gives a
    finite mean with infinite variance. x_min = (alpha - 1) / alpha makes the
    mean exactly 1.
    """

    alpha: float = 1.5

    def __post_init__(self):
        if self.alpha <= 1:
            raise ValueError("alpha must be > 1, otherwise the mean is infinite "
                             "and the sizes cannot be scaled to mean 1")

    @property
    def label(self) -> str:
        return f"pareto(a={self.alpha:g})"

    @property
    def x_min(self) -> float:
        return (self.alpha - 1) / self.alpha

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        u = rng.random(size)                                   # in [0, 1)
        return self.x_min * (1.0 - u) ** (-1.0 / self.alpha)   # inverse CDF


@dataclass(frozen=True)
class LogNormal:
    """Log-normal sizes with mean 1. Heavier than exponential; every moment finite."""

    sigma: float = 1.0

    @property
    def label(self) -> str:
        return f"lognormal(s={self.sigma:g})"

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        return rng.lognormal(-0.5 * self.sigma ** 2, self.sigma, size)


@dataclass(frozen=True)
class Bimodal:
    """Mice and elephants: a fraction `p` of flows is `ratio` times larger than the rest.

    A blunt but very readable heavy-hitter model. With p = 0.05 and ratio = 100,
    elephants are 5% of the flows but about 84% of the bytes.
    """

    p: float = 0.05
    ratio: float = 100.0

    def __post_init__(self):
        if not 0 < self.p < 1 or self.ratio < 1:
            raise ValueError("need 0 < p < 1 and ratio >= 1")

    @property
    def label(self) -> str:
        return f"bimodal(p={self.p:g},r={self.ratio:g})"

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        small = 1.0 / (1 - self.p + self.p * self.ratio)
        return np.where(rng.random(size) < self.p, self.ratio * small, small)


@dataclass(frozen=True)
class Empirical:
    """A flow-size CDF given as points (size, cumulative probability).

    The CDF is linear between points, which is how data center simulators
    usually read published workloads (for example the web-search and
    data-mining CDFs that came out of the DCTCP and VL2 measurements). A first
    CDF value above 0 is a point mass at the smallest size. Sizes are rescaled
    to mean 1, so it doesn't matter whether the file is in bytes or packets.
    """

    sizes: tuple
    cdf: tuple
    name: str = "empirical"

    def __post_init__(self):
        x = np.asarray(self.sizes, dtype=float)
        F = np.asarray(self.cdf, dtype=float)
        if x.ndim != 1 or x.shape != F.shape or len(x) < 2:
            raise ValueError("sizes and cdf must be 1-D and the same length (at least 2)")
        if np.any(np.diff(x) < 0) or np.any(np.diff(F) < 0):
            raise ValueError("sizes and cdf must both be non-decreasing")
        if x[0] <= 0:
            raise ValueError("sizes must be positive")
        if F[0] < 0 or abs(F[-1] - 1) > 1e-9:
            raise ValueError("cdf must start at >= 0 and end at 1")

    @property
    def label(self) -> str:
        return self.name

    @property
    def raw_mean(self) -> float:
        """Mean of the piecewise-linear distribution, in the file's own units."""
        x = np.asarray(self.sizes, dtype=float)
        F = np.asarray(self.cdf, dtype=float)
        return float(F[0] * x[0] + np.sum(np.diff(F) * (x[:-1] + x[1:]) / 2))

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        u = rng.random(size)
        return np.interp(u, self.cdf, self.sizes) / self.raw_mean

    @classmethod
    def from_csv(cls, path, name: str | None = None) -> "Empirical":
        """Read a CDF file: first column = size, last column = cumulative probability.

        Commas or whitespace separate columns, so both "size,cdf" and the common
        three-column "size  x  cdf" layout work. Blank lines, lines starting with
        '#' and non-numeric header lines are skipped. A CDF given in percent
        (ending at 100) is converted to a fraction.
        """
        sizes, cdf = [], []
        for line in Path(path).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            try:
                s, F = float(parts[0]), float(parts[-1])
            except ValueError:
                continue                                       # header line
            sizes.append(s)
            cdf.append(F)
        if cdf and abs(cdf[-1] - 100) < 1e-6:
            cdf = [F / 100 for F in cdf]
        return cls(tuple(sizes), tuple(cdf), name or Path(path).stem)
