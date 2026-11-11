"""The bundled flow-size CDFs reproduce the summaries pFabric §5 gives for them."""
from pathlib import Path

import numpy as np
import pytest

from ecmpsim import Empirical

WORKLOADS = Path(__file__).resolve().parents[1] / "workloads"
PKT = 1460                                         # bytes per packet in the files


def sample_bytes(name, size=2_000_000):
    f = Empirical.from_csv(WORKLOADS / f"{name}.txt")
    return f, f.sample(np.random.default_rng(0), size) * f.raw_mean * PKT


def test_web_search_summary():
    # "over 95% of all bytes are from the 30% of the flows that are 1-20MB"
    f, x = sample_bytes("websearch")
    big = x >= 667 * PKT                           # ~1MB, the CDF's 70% point
    assert big.mean() == pytest.approx(0.30, abs=0.005)
    assert x[big].sum() / x.sum() == pytest.approx(0.95, abs=0.01)


def test_data_mining_summary():
    # "more than 80% of the flows are less than 10KB and 95% of all bytes are in
    #  the ~3.6% flows that are larger than 35MB"
    f, x = sample_bytes("datamining")
    assert (x <= 10 * 1024).mean() == pytest.approx(0.80, abs=0.005)
    big = x > 35e6
    assert big.mean() == pytest.approx(0.036, abs=0.002)
    assert x[big].sum() / x.sum() == pytest.approx(0.95, abs=0.01)


def test_data_mining_is_more_skewed():
    web = Empirical.from_csv(WORKLOADS / "websearch.txt")
    mining = Empirical.from_csv(WORKLOADS / "datamining.txt")
    assert mining.cv2 > 10 * web.cv2
