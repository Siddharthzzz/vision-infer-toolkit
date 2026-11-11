"""GPU latency timing (docs/methodology.md §3.2-3.3).

>>> YOU IMPLEMENT `time_gpu` (teaching mode). Spec below; tests in tests/python/test_timing.py.
Concepts: docs/notes/gpu-timing.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimingResult:
    samples_ms: np.ndarray  # shape (iters,), float64, one GPU-time sample per timed iteration
    warmup_iters: int  # warm-up iterations actually run
    warmup_seconds: float  # host wall time spent in warm-up


def time_gpu(
    fn: Callable[[], object],
    *,
    iters: int,
    warmup_iters: int = 50,
    warmup_seconds: float = 2.0,
    stream=None,  # torch.cuda.Stream | None
) -> TimingResult:
    """Time `fn` in **latency mode**: one isolated GPU-time sample per iteration.

    Spec
    ----
    1. Validation: raise ValueError if iters < 1, warmup_iters < 0 or warmup_seconds < 0.
    2. Stream: if `stream` is None use the current CUDA stream. `fn` must run with that stream
       as the *current* stream (so the work it enqueues lands there), and every timing event
       is recorded on that same stream.
    3. Warm-up: call `fn` repeatedly until BOTH at least `warmup_iters` calls have been made
       AND at least `warmup_seconds` of host wall time have passed. Never sampled.
       Report how many calls and how long (host clock) in the result.
    4. Timed loop: for each of `iters` iterations, measure the GPU time of exactly one `fn()`
       call with CUDA events and make sure the GPU has finished that iteration before
       starting the next one (latency mode: iterations must not overlap).
    5. Return samples in milliseconds as a float64 numpy array of length `iters`.
    6. Total calls to `fn` == result.warmup_iters + iters.
    7. Do not put host timers or allocations (e.g. creating events) inside the timed region.
    """
    raise NotImplementedError("time_gpu: your implementation goes here")
