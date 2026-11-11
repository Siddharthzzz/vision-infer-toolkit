"""Latency statistics exactly as defined in docs/methodology.md §7."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

MIN_SAMPLES_FOR_P99 = 1000
PERCENTILE_METHOD = "linear"  # numpy default, Hyndman & Fan type 7


@dataclass(frozen=True)
class LatencySummary:
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float | None  # None when n < MIN_SAMPLES_FOR_P99
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    cv: float

    def as_row(self) -> dict[str, float | int | None]:
        return asdict(self)


def summarize_latency(samples_ms: Sequence[float] | np.ndarray) -> LatencySummary:
    x = np.asarray(samples_ms, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("samples_ms must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("samples_ms must be finite and > 0")
    p50, p95, p99 = np.percentile(x, [50, 95, 99], method=PERCENTILE_METHOD)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return LatencySummary(
        n=int(x.size),
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99) if x.size >= MIN_SAMPLES_FOR_P99 else None,
        mean_ms=mean,
        std_ms=std,
        min_ms=float(x.min()),
        max_ms=float(x.max()),
        cv=std / mean,
    )


def trial_spread(trial_p50s: Sequence[float]) -> float:
    """(max - min) / median of per-trial p50s. The pilot gate in §9 is 0.05."""
    x = np.asarray(trial_p50s, dtype=np.float64)
    if x.size < 2:
        raise ValueError("need at least 2 trials")
    return float((x.max() - x.min()) / np.median(x))


def compare(a_trial_p50s: Sequence[float], b_trial_p50s: Sequence[float]) -> str:
    """'faster' if A's trial p50 range lies entirely below B's, 'slower' if entirely above,
    otherwise 'no_difference' (§7)."""
    a = np.asarray(a_trial_p50s, dtype=np.float64)
    b = np.asarray(b_trial_p50s, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("both sides need at least one trial")
    if a.max() < b.min():
        return "faster"
    if a.min() > b.max():
        return "slower"
    return "no_difference"


def speedup(a_trial_p50s: Sequence[float], b_trial_p50s: Sequence[float]) -> float:
    """median p50(B) / median p50(A): >1 means A is faster."""
    return float(np.median(b_trial_p50s) / np.median(a_trial_p50s))
