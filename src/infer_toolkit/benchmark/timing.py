"""GPU latency timing (docs/methodology.md §3.2-3.3). Concepts: docs/notes/gpu-timing.md."""

from __future__ import annotations

import time
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
    2. Stream: if `stream` is None use the current CUDA stream. `fn` runs with that stream
       as the *current* stream (so the work it enqueues lands there), and every timing event
       is recorded on that same stream.
    3. Warm-up: call `fn` until BOTH at least `warmup_iters` calls have been made AND at
       least `warmup_seconds` of host wall time have passed. Never sampled. The result
       reports how many calls were made and how long they took (host clock).
    4. Timed loop: each of `iters` iterations measures the GPU time of exactly one `fn()`
       call with CUDA events, and the GPU finishes that iteration before the next starts
       (latency mode: iterations never overlap).
    5. Samples are returned in milliseconds as a float64 numpy array of length `iters`.
    6. Total calls to `fn` == result.warmup_iters + iters.
    7. No host timers or allocations (e.g. creating events) inside the timed region.
    """
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")
    if warmup_seconds < 0:
        raise ValueError(f"warmup_seconds must be >= 0, got {warmup_seconds}")

    import torch  # imported here so CPU-only environments (CI) can import this module

    if stream is None:
        stream = torch.cuda.current_stream()

    # Created once, outside the timed region (rule 7). Because every iteration synchronises
    # on `end` before the next `start.record`, one reusable pair is enough.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = np.empty(iters, dtype=np.float64)

    # Make `stream` current so fn's kernels are enqueued on it (rule 2).
    with torch.cuda.stream(stream):
        # --- warm-up: both minimums must be met (rule 3) ---------------------------------
        t0 = time.perf_counter()
        n_warm = 0
        while n_warm < warmup_iters or time.perf_counter() - t0 < warmup_seconds:
            fn()
            n_warm += 1
        # Drain warm-up work so none of it is still running when the first sample starts.
        stream.synchronize()
        warm_s = time.perf_counter() - t0

        # --- timed loop: record, call, record, wait (rule 4) -----------------------------
        for i in range(iters):
            start.record(stream)
            fn()
            end.record(stream)
            end.synchronize()  # latency mode: this iteration is done before the next starts
            samples[i] = start.elapsed_time(end)  # ms, measured by the GPU

    return TimingResult(samples_ms=samples, warmup_iters=n_warm, warmup_seconds=warm_s)
