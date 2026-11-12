"""Acceptance tests for time_gpu (spec in timing.py docstring).

Uses torch.cuda._sleep(cycles): a kernel that spins the GPU for a number of clock cycles.
Its wall duration depends on the current clock, so tests check *ratios*, not absolute ms.
"""

import time

import numpy as np
import pytest

from infer_toolkit.benchmark.timing import TimingResult, time_gpu

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

CYCLES = 2_000_000  # roughly 1 ms at ~2 GHz; only ratios matter


def gpu_sleep(cycles):
    return lambda: torch.cuda._sleep(cycles)


def ramp_clocks(seconds=0.5):
    """Bring the GPU out of idle clocks without filling the launch queue (sync each kernel).
    Without this, the first measurement runs at a lower SM clock than the next one."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        torch.cuda._sleep(CYCLES)
        torch.cuda.synchronize()


def test_argument_validation():
    f = gpu_sleep(10)
    for kw in (dict(iters=0), dict(iters=1, warmup_iters=-1), dict(iters=1, warmup_seconds=-1)):
        with pytest.raises(ValueError):
            time_gpu(f, **kw)


def test_one_positive_float64_sample_per_iteration():
    r = time_gpu(gpu_sleep(CYCLES), iters=20, warmup_iters=3, warmup_seconds=0.0)
    assert isinstance(r, TimingResult)
    assert r.samples_ms.shape == (20,)
    assert r.samples_ms.dtype == np.float64
    assert np.all(r.samples_ms > 0)


@pytest.mark.parametrize("use_side_stream", [False, True], ids=["current", "side_stream"])
def test_measures_gpu_time_not_launch_time(use_side_stream):
    # Host-only timing (no events/sync) would see ~equal launch costs -> ratio ~1.
    # Events recorded on a different stream than the work would see ~0 -> ratio ~1 or noise.
    # warmup_seconds=0 on purpose: a timed warm-up loop can enqueue enough kernels to fill
    # the CUDA launch queue, after which launches block and a host timer *accidentally*
    # tracks GPU time (see docs/notes/gpu-timing.md §7).
    # Short/long/short and the mean of the shorts cancel residual clock drift.
    stream = torch.cuda.Stream() if use_side_stream else None
    kw = dict(iters=30, warmup_iters=5, warmup_seconds=0.0, stream=stream)
    ramp_clocks()
    s1 = np.median(time_gpu(gpu_sleep(CYCLES), **kw).samples_ms)
    long = np.median(time_gpu(gpu_sleep(4 * CYCLES), **kw).samples_ms)
    s2 = np.median(time_gpu(gpu_sleep(CYCLES), **kw).samples_ms)
    ratio = long / ((s1 + s2) / 2)
    assert 2.5 < ratio < 5.5, f"short={s1:.3f}/{s2:.3f} ms long={long:.3f} ms ratio={ratio:.2f}"


def test_fn_runs_on_the_requested_stream():
    s = torch.cuda.Stream()
    seen = []

    def fn():
        seen.append(torch.cuda.current_stream())
        torch.cuda._sleep(1000)

    time_gpu(fn, iters=5, warmup_iters=2, warmup_seconds=0.0, stream=s)
    assert len(seen) == 7
    assert all(x == s for x in seen)


def test_call_count_and_warmup_iteration_minimum():
    calls = 0

    def fn():
        nonlocal calls
        calls += 1
        torch.cuda._sleep(1000)

    r = time_gpu(fn, iters=10, warmup_iters=5, warmup_seconds=0.0)
    assert r.warmup_iters == 5
    assert calls == r.warmup_iters + 10


def test_warmup_time_minimum():
    t0 = time.perf_counter()
    r = time_gpu(gpu_sleep(1000), iters=5, warmup_iters=1, warmup_seconds=0.3)
    assert r.warmup_seconds >= 0.3
    assert r.warmup_iters > 1  # a fast fn must be called more than once to fill 0.3 s
    assert time.perf_counter() - t0 >= 0.3


def test_warmup_is_not_sampled():
    first = True

    def fn():
        nonlocal first
        torch.cuda._sleep(50 * CYCLES if first else CYCLES)
        first = False

    r = time_gpu(fn, iters=20, warmup_iters=1, warmup_seconds=0.0)
    assert r.samples_ms.max() < 10 * np.median(r.samples_ms)


def test_iterations_do_not_overlap():
    # Latency mode: each sample is one isolated request, so host-side work inside fn
    # (2 ms here) must show up in every sample. If iterations overlap (no per-iteration
    # sync), that host time hides behind the previous iteration's long kernel and vanishes.
    long_kernel = 8 * CYCLES

    def with_host_work():
        time.sleep(0.002)
        torch.cuda._sleep(long_kernel)

    kw = dict(iters=30, warmup_iters=3, warmup_seconds=0.0)
    ramp_clocks()
    base = np.median(time_gpu(gpu_sleep(long_kernel), **kw).samples_ms)
    with_host = np.median(time_gpu(with_host_work, **kw).samples_ms)
    assert with_host - base > 1.5, f"base={base:.3f} ms with_host={with_host:.3f} ms"


def test_stream_idle_on_return():
    s = torch.cuda.Stream()
    time_gpu(gpu_sleep(CYCLES), iters=5, warmup_iters=0, warmup_seconds=0.0, stream=s)
    assert s.query(), "stream still busy after time_gpu returned"


def test_throughput_scales_with_k():
    from infer_toolkit.benchmark.timing import time_gpu_throughput

    ramp_clocks()
    kw = dict(warmup_iters=3, warmup_seconds=0.0)
    t10 = time_gpu_throughput(gpu_sleep(CYCLES), k=10, **kw)
    t40 = time_gpu_throughput(gpu_sleep(CYCLES), k=40, **kw)
    t10b = time_gpu_throughput(gpu_sleep(CYCLES), k=10, **kw)
    assert 2.5 < t40 / ((t10 + t10b) / 2) < 5.5
