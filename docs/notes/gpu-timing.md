# Note: timing GPU work correctly

Background for `time_gpu` in `src/infer_toolkit/benchmark/timing.py`.

## 1. The GPU runs asynchronously

When Python calls `model(x)` on a CUDA tensor, the CPU does not run the convolutions. It
**enqueues** kernels onto a CUDA *stream*, a FIFO queue the GPU drains in order, and returns
almost immediately. The GPU finishes the work later.

```
CPU:  [launch k1][launch k2][launch k3] -> returns         (fast, ~tens of µs)
GPU:        [====== k1 ======][=== k2 ===][===== k3 =====]  (the actual work)
```

So this measures **launch time, not compute time**:

```python
t0 = time.perf_counter()
y = model(x)  # returns before the GPU is done
t1 = time.perf_counter()  # WRONG: GPU still busy
```

Adding `torch.cuda.synchronize()` before `t1` makes the result *correct but coarse*. It
measures CPU→GPU round-trip time too, including the cost of the sync call, and under WSL2 that
goes through an extra paravirtualisation layer.

## 2. CUDA events

A CUDA event is a marker **placed into the stream**. When the GPU reaches it, the GPU writes a
timestamp. Two events give the GPU-side elapsed time between two points in the queue:

```
stream: ... [start] [k1][k2][k3] [end] ...
                 └──── elapsed_time ────┘   measured by the GPU itself, ~0.5 µs resolution
```

The parts you need (PyTorch):

- `torch.cuda.Event(enable_timing=True)` creates an event. Timing is off by default.
- `ev.record(stream)` enqueues the marker on that stream. Recording on the wrong stream is a
  classic bug: the event then times a different queue.
- `ev.synchronize()` blocks the CPU until the GPU has passed that event.
- `start.elapsed_time(end)` returns milliseconds as a float. It is only valid after `end` has
  completed, so calling it too early raises an error.

The C++ equivalents (used in M6) are `cudaEventCreate`, `cudaEventRecord(ev, stream)`,
`cudaEventSynchronize` and `cudaEventElapsedTime`.

**What events include:** everything on the stream between the markers. If the CPU is slow to
launch kernels (a small model, Python overhead), the GPU sits idle *between* kernels, and that
idle gap **is** counted. That's correct, because it's real latency a user would see, and it's
the overhead CUDA Graphs remove.

## 3. Latency vs throughput

- **Latency mode:** synchronise after every iteration. Each sample is one isolated request.
  Without the sync, the CPU queues iteration *i+1* while the GPU is still on *i*, and
  per-iteration times start measuring queueing.
- **Throughput mode:** enqueue K iterations back to back between one pair of events and divide.
  Overlapping CPU launches with GPU work is the point here.

## 4. Why warm-up

The first iterations are slow for reasons that have nothing to do with steady-state speed:

- lazy CUDA context and module loading
- cuDNN/cuBLAS heuristics and autotuning, and TensorRT tactic caches
- the caching allocator requesting memory from the driver for the first time
- **GPU clocks ramping up** from an idle power state (P5/P8 → P0)

Warm-up has to cover both an iteration count and a minimum wall time, because clock ramp-up
is time-based, not iteration-based. The M2 pilot checks whether our budget is enough by
plotting the first 2000 raw samples.

## 5. Why percentiles, not the mean

Latency distributions are right-skewed: most iterations are fast, and a few hit a context
switch, a clock dip or WSL scheduling. The mean is pulled by that tail and describes no actual
request. p50 is the typical request, and p95/p99 are what a user sees on a bad frame. A p99
from N samples rests on about N/100 points, which is why we time 1000 iterations per trial.

## 6. Checklist for `time_gpu`

- [ ] work and both events are on the **same** stream
- [ ] warm-up excluded from the samples, satisfying both the count and the time minimum
- [ ] one sample per iteration, in ms, with the end event synchronised before reading it
- [ ] no host-side timing inside the timed region
- [ ] events created once, outside the loop (creating them is not free)

## 7. Two traps found while writing the tests

Both showed up while validating `tests/python/test_timing.py` against deliberately wrong
reference implementations (not committed).

- **A full launch queue hides a broken timer.** A host-timer-only implementation, with no
  events and no sync, *passed* the first version of the tests. A timed warm-up loop had
  enqueued thousands of kernels. Once the CUDA launch queue is full, every launch blocks until
  a slot frees, so `perf_counter` around a launch accidentally measured GPU time. In a real
  model the queue is usually not full, and the same timer reports launch time. The test now
  avoids saturating the queue, so this implementation fails as it should.
- **Clock ramp-up skews back-to-back comparisons.** With no warm-up, the first measurement
  ran at a visibly lower SM clock (cycles per ms) than the one right after it, so a 4× kernel
  measured well under 4× slower. That is the reason for the time-based warm-up in methodology
  §3.3. The test ramps the clocks first and brackets the long run between two short runs.
