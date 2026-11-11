"""CPU tests for latency statistics (methodology §7)."""

import numpy as np
import pytest

from infer_toolkit.benchmark.stats import (
    compare,
    speedup,
    summarize_latency,
    trial_spread,
)


def test_percentiles_use_linear_method():
    x = np.arange(1, 1001, dtype=float)  # 1..1000 ms
    s = summarize_latency(x)
    assert s.n == 1000
    assert s.p50_ms == pytest.approx(500.5)
    assert s.p95_ms == pytest.approx(np.percentile(x, 95, method="linear"))
    assert s.p99_ms == pytest.approx(990.01)
    assert s.min_ms == 1 and s.max_ms == 1000
    assert s.cv == pytest.approx(x.std(ddof=1) / x.mean())


def test_p99_withheld_below_1000_samples():
    assert summarize_latency(np.ones(999)).p99_ms is None


@pytest.mark.parametrize("bad", [[], [1.0, -1.0], [1.0, float("nan")], [[1.0, 2.0]]])
def test_rejects_bad_samples(bad):
    with pytest.raises(ValueError):
        summarize_latency(bad)


def test_trial_spread():
    assert trial_spread([10.0, 10.5, 9.5]) == pytest.approx(0.1)


def test_compare_requires_non_overlapping_ranges():
    assert compare([1.0, 1.1], [1.2, 1.3]) == "faster"
    assert compare([1.2, 1.3], [1.0, 1.1]) == "slower"
    assert compare([1.0, 1.25], [1.2, 1.3]) == "no_difference"


def test_speedup_is_ratio_of_median_p50s():
    assert speedup([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == pytest.approx(2.0)
