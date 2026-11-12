# Benchmark methodology

Written in M1, **before any benchmark number was collected**. Every result in this repo must
follow this document. Changes made after data exists go in [Amendments](#10-amendments), dated,
with the reason. Earlier text is never silently rewritten.

## 0. Principles

1. **Measured, not estimated.** Every reported number comes from a run whose raw rows are in
   `results/*.csv`. Anything not measured is labelled *not measured*.
2. **Same boundary, same rules for every backend.** A latency is only compared with a latency
   measured across the same boundary (§2), batch size, input shape and machine state (§5).
3. **Distributions, not single numbers.** We report p50/p95/p99 within a trial and the spread
   across trials. "A is faster than B" needs non-overlapping trial ranges (§7).
4. **Nothing is dropped silently.** A trial that breaks a rule is flagged in the CSV and
   re-run. It is not deleted.
5. **This is a WSL2 laptop.** All numbers describe an RTX 4070 Laptop GPU under WSL2 on
   Windows 11, with the power state recorded per run. We make no claims about native Linux,
   desktop GPUs or other power modes.

## 1. What is measured

| Quantity | Unit | How | Section |
|---|---|---|---|
| Model latency | ms | CUDA events, per iteration, batch 1 and above | §2, §3 |
| End-to-end latency | ms | host monotonic clock, after a device sync | §2, §3 |
| Throughput | images/s | CUDA events around K back-to-back iterations | §3.4 |
| Accuracy | COCO AP, ADE20K mIoU | full validation sets | §6 |
| Peak GPU memory | MiB | allocator stats, NVML device delta, TensorRT API | §4 |
| Build/compile time | s | host clock, reported separately, never inside latency | §3.5 |

## 2. Latency boundaries

| Name | Starts | Ends | Includes | Excludes |
|---|---|---|---|---|
| `model` | Pre-processed input already on the GPU in the backend's expected layout and dtype | Raw output tensors on the GPU | Every kernel the backend runs, plus launch overhead on the timed stream | H2D/D2H copies, pre/post-processing, JPEG decode |
| `e2e` | Decoded `uint8` HWC image in host RAM | Final results in host RAM (boxes after decode and NMS, or class map) | H2D, pre-process, model, post-process, D2H | JPEG decode, file I/O, drawing/display |

- `model` is the cross-backend number used for PyTorch eager, AMP, `torch.compile`, ONNX
  Runtime and TensorRT.
- ONNX Runtime is timed with **IOBinding** and device-resident inputs and outputs. Without it,
  ORT adds copies, and its `model` latency would not be comparable.
- AMP FP16 takes an FP32 input. The casts autocast inserts are part of its `model` latency.
- `e2e` is reported for the PyTorch baseline (M2) and the C++ runtime (M6). It is never mixed
  with `model` numbers in one comparison.

## 3. Timing procedure

### 3.1 Clocks

- **GPU work (`model`)** uses `cudaEvent` pairs recorded on the stream the work is enqueued on
  (`torch.cuda.Event(enable_timing=True)` in Python, `cudaEventRecord` in C++). Host timers are
  wrong for GPU work because kernel launches are asynchronous; see
  [notes/gpu-timing.md](notes/gpu-timing.md).
- **`e2e`** uses `time.perf_counter_ns()` or `std::chrono::steady_clock`, starting before the
  H2D copy and ending after the final device-to-host sync.

### 3.2 Latency mode (the headline numbers)

Each iteration is isolated: record `start` → run → record `end` → `end.synchronize()`.
Synchronising every iteration means iterations never overlap or queue, so each sample is the
latency of one request. This is the batch-1 serving case. Per-iteration samples are kept so
percentiles can be computed.

### 3.3 Per-trial protocol

| Parameter | Value | Rationale |
|---|---|---|
| Warm-up | until **both** ≥ 50 iterations **and** ≥ 2.0 s have elapsed | Covers lazy CUDA/cuDNN init, autotuning, allocator growth and clock ramp-up. Checked in the pilot (§9). |
| Timed iterations `N` | 1000 | Puts about 10 samples beyond p99. Fewer than 1000 means p99 is not reported. |
| Trials `R` per configuration | 5 | Captures trial-to-trial (thermal/clock) variance |
| Trial order | **interleaved round-robin** across all configurations of one model (A B C A B C …) | Slow drift (heat soak) spreads evenly instead of penalising whichever config runs last |
| Inputs | 64 real validation images, pre-processed once, cycled | Data-dependent work (NMS, top-k) behaves realistically, with no I/O in the loop |
| Cool-down before each trial | wait until GPU temperature ≤ session idle baseline + 5 °C, with a 300 s timeout (amended by A9: ≤ max(60 °C, idle + 5 °C)) | Every trial starts from a comparable thermal state. A timeout sets a flag. |

The idle baseline is the median GPU temperature over 60 s of idle at session start, recorded
in the CSV.

### 3.4 Throughput mode

Batch sizes are **{1, 8, 16}** for every model and backend, where they fit in memory (§4.1). An
optional sweep {1, 2, 4, 8, 16, 32} is run for the best engine of each model in M4.
Throughput = `batch × K / elapsed`, with K = 200 back-to-back iterations between one pair of
events, using the same warm-up, trials and interleaving as above. This measures pipelined
throughput, not per-request latency. The two are never reported in the same column.

### 3.5 Excluded one-off costs

`torch.compile` compilation, TensorRT engine build and ORT session creation are timed
separately (`build_s` column) and never appear in latency. `torch.compile` uses
`mode="max-autotune"`, which on GPU includes CUDA Graphs. The C++ TensorRT runtime gets CUDA
Graphs in M6, and the comparison table states which rows use graphs.

### 3.6 Shapes

Inputs are fixed-size per model, and TensorRT engines are built with **static shapes**, one
engine per batch size.

| Model | Input | Note |
|---|---|---|
| YOLOX-S | 640×640, letterbox | matches official eval |
| RT-DETR-R18 | 640×640, resize | matches official eval |
| SegFormer-B0 | 512×512, whole-image resize | the official ADE20K eval differs (keeps aspect ratio, short side 512), so our mIoU is reported next to the published one with that difference stated |

## 4. Memory

Under WSL2, NVML does **not** report per-process GPU memory (checked in M1: the process list
is empty). The following definitions are therefore used.

| Column | Definition | Applies to |
|---|---|---|
| `peak_alloc_mib` | `torch.cuda.max_memory_allocated()`, which counts tensors only | PyTorch |
| `peak_reserved_mib` | `torch.cuda.max_memory_reserved()`, which includes the caching allocator | PyTorch |
| `trt_activation_mib` | `engine.device_memory_size_v2` | TensorRT |
| `trt_engine_mib` | serialized engine size, which approximates weight memory | TensorRT |
| `peak_device_delta_mib` | peak NVML device `used`, sampled at 10 Hz during the trial, minus the device `used` measured **before this process created a CUDA context** | all backends |

- `peak_device_delta_mib` includes the CUDA context, library workspaces (cuDNN, cuBLAS,
  TensorRT) and the allocator cache. It is the honest "what does this cost on the card"
  number.
- 10 Hz sampling can miss short spikes, so it is a lower bound on the true peak.
- **Other GPU users contaminate it.** Close GPU-heavy Windows apps (browser video, games).
  The preflight records the baseline and flags it above 1024 MiB.

### 4.1 The 8 GB budget

A configuration that runs out of memory is recorded as `status=OOM` with its batch size. It is
not silently skipped. Before any step that might not fit, an alternative is proposed first
(project rule).

## 5. Machine state (laptop variance)

The M0 snapshot (`results/env.csv`) recorded an enforced power limit of **42 W**, against a
55 W default and a 115 W maximum. A later NVML probe showed the limit had changed, without
any configuration change in this repo. The power state therefore has to be **checked and
recorded on every trial**, not assumed.

### 5.1 Required state (preflight, run before each session)

| Check | Source | On failure |
|---|---|---|
| Running on AC power | NVML power source | **refuse to run** |
| Enforced power limit | NVML | record it. If it changes during a session, flag every later trial. |
| Windows power mode | `powercfg /getactivescheme` (via `powershell.exe`) | record it |
| Vendor performance profile | not readable programmatically; set by hand in `configs/bench_state.yaml` | record it |
| Device memory in use before start | NVML | record it, flag above 1024 MiB |
| Idle temperature baseline | NVML, 60 s median | record it |

Clocks are **not locked**. Application clocks are reported as deprecated on this driver, and
locked clocks would need Windows admin rights and would not reflect how the laptop actually
runs. Variance is handled by monitoring, interleaving and repetition instead.

### 5.2 Per-trial telemetry

A background NVML sampler runs at 10 Hz during every timed trial. Its cost is checked in the
pilot (§9). Each trial row records the following:

- SM clock (median and minimum), memory clock (median)
- temperature (start and maximum), power (median), enforced limit, power source
- fraction of samples with each clock-event reason: `sw_power_cap`, `sw_thermal`,
  `hw_slowdown`, `hw_thermal`, `hw_power_brake`

### 5.3 Flags

A flagged trial stays in the CSV (`flags` column, `;`-separated) and is re-run. Headline
numbers use unflagged trials only, and say how many trials were flagged.

| Flag | Condition |
|---|---|
| `battery` | power source was not AC at any sample |
| `power_limit_changed` | enforced limit differs from the session start |
| `hw_throttle` | any sample has `hw_slowdown`, `hw_thermal` or `hw_power_brake` |
| `sw_thermal` | `sw_thermal` set in more than 10 % of samples |
| `clock_outlier` | the trial's median SM clock differs by more than 5 % from the median of the same configuration's other trials |
| `cooldown_timeout` | the cool-down did not reach its target in 300 s |
| `busy_baseline` | device memory in use at start was above 1024 MiB |

`sw_power_cap` is **recorded but not flagged**. Under sustained load a laptop GPU normally runs
against its power cap, so that is the regime being measured.

## 6. Accuracy

| Model | Dataset | Metric | Protocol |
|---|---|---|---|
| YOLOX-S, RT-DETR-R18 | COCO val2017, all 5000 images | `pycocotools` COCOeval bbox AP@[.5:.95] (headline), AP50, AP_S/M/L | maximum 100 detections per image. Score threshold and NMS parameters come from each model's official eval config, stored in `configs/`. |
| SegFormer-B0 | ADE20K val, all 2000 images | mIoU over 150 classes | intersection and union accumulated over the whole set (not a mean of per-image IoU). Labels reduced by one, background = 255 ignored. Single scale, no flip. |

- One pre-processing pipeline per model, shared by every precision and backend. Only the
  inference engine changes between rows.
- Accuracy deltas are reported relative to **our** FP32 eager baseline. Published numbers are
  shown next to ours with a citation and are never substituted for a measurement.
- **Calibration data:** 1000 images from COCO train2017 (detection) and 1000 from ADE20K train
  (segmentation). They are sampled with a fixed seed, and the file list and its SHA-256 are
  committed to `configs/`. They are disjoint from validation because they come from the train
  split, and a unit test checks that none of the file names appear in val.
- Calibration-size ablation (M4): {100, 250, 500, 1000} images, each drawn as 3 independent
  seeded subsets so the spread caused by the choice of subset is visible.

## 7. Statistics and reporting

- **Percentiles** use `numpy.percentile(..., method="linear")` (type 7) over the N samples
  of one trial.
- **Per configuration** we report the median across unflagged trials of p50, p95, p99 and
  mean, with the **min–max across trials** in brackets, plus `n_trials` and `n_flagged`.
- **Comparisons:** "A is faster than B" only when A's trial p50 range lies entirely below B's.
  Otherwise the result is "no measurable difference in this setup". Speedup = median p50(B) /
  median p50(A).
- **Coefficient of variation** (std/mean) is reported per trial, to diagnose noisy
  configurations.
- Targets (for example a latency or accuracy budget) are set only **after** M2 baselines
  exist.

## 8. Results files and traceability

- Raw rows are written to `results/<milestone>_<kind>.csv`, for example
  `results/m2_latency.csv`, one row per trial. Summary tables are derived from them by a
  script, never typed by hand.
- Every row carries `git_commit`, `run_id` and `env_id`. `env_id` is the SHA-256 prefix of
  the `check_env.py` snapshot taken in the same session.
- Every row in the README results table links to the one command that reproduces it
  (`scripts/reproduce_*.sh`).

### Latency CSV columns

`run_id, git_commit, env_id, timestamp_utc, model, backend, precision, batch, input_hw,
boundary, mode, trial, n_iters, warmup_iters, warmup_s, p50_ms, p95_ms, p99_ms, mean_ms,
std_ms, min_ms, max_ms, cv, sm_clock_med_mhz, sm_clock_min_mhz, mem_clock_med_mhz,
temp_start_c, temp_max_c, power_med_w, power_limit_w, power_source, frac_sw_power_cap,
frac_sw_thermal, frac_hw_throttle, idle_temp_c, baseline_used_mib, windows_power_mode,
vendor_profile, status, flags`

## 9. Pilot (first step of M2, before any baseline numbers)

On YOLOX-S, PyTorch eager FP32, batch 1:

1. **Warm-up adequacy.** Record 2000 iterations from a cold start with no warm-up. Check that
   the per-iteration latency has levelled off within the §3.3 warm-up budget. If it hasn't,
   increase the budget here.
2. **Sampler overhead.** Compare 5 trials with the 10 Hz NVML sampler against 5 without, all
   interleaved. If the p50 ranges don't overlap, lower the sampling rate or move the sampler
   out of process.
3. **Trial spread.** Calculate (max − min) / median of trial p50s. If it is above 5 %,
   investigate the power and thermal flags before collecting baselines.

Pilot rows go to `results/m2_pilot.csv`. Any methodology change they cause is written into §10
before baselines are collected.

## 10. Amendments

A1–A8 are dated 2026-11-12, made after the M2 pilot and before any latency baseline was
collected. A9 was made later the same day, during the baseline run (see its entry).
Pilot data: `results/m2_pilot.csv`, `m2_pilot_coldstart.csv`, `m2_pilot_telemetry.csv`,
`m2_pilot_summary.csv` (YOLOX-S, eager FP32, batch 1).

**A1. Warm-up check: the pre-registered rule failed, and the rule was at fault.** §9 asked
whether the per-iteration latency levels off within the warm-up budget, and `pilot.py` set the
test in advance: "settled" = the 25-iteration rolling median stays within ±2 % of the
reference (median of iterations 1000–1999) *for the rest of the run*. That gave iteration 1967
of 2000, against a budget ending at iteration 252, so the check **failed as written**.

The data shows why. Iteration 0 took 207 ms (lazy initialisation), iterations 1–2 about 8 ms,
and the rolling median first came within ±2 % at iteration 32 (about 0.45 s of GPU time). After
iteration 250, though, 37.6 % of rolling windows were outside ±2 % and 15.3 % outside ±3 %:
**steady-state jitter is larger than the tolerance**, so "stays within forever" detects jitter,
not the end of warm-up. Decision: the warm-up budget (≥ 50 iterations and ≥ 2.0 s) is
**unchanged**, because the measured transient ends well inside it. The analysis now also
reports the end of the transient and the first post-budget window (`a1_*` columns) next to the
original rule's result, which stays in the summary. Steady-state jitter is covered by per-trial
percentiles and the trial spread, not by warm-up.

**A2. `power_limit_changed` flag removed. The enforced limit is recorded as min and max.** On
this laptop the enforced power limit is dynamic, consistent with NVIDIA Dynamic Boost. Within
single pilot trials it ranged from 75.0 to 103.9 W (25–33 distinct values in about 97 samples
per trial). Meanwhile, the GPU drew a median of 61–64 W, below the lowest limit; the SM clock
median was 2550 MHz in all 5 sampled trials; and `sw_power_cap` was set in ≤ 1 % of samples.
"Any change" therefore flagged every trial for something with no measurable effect. The
latency CSV column `power_limit_w` becomes `power_limit_min_w` and `power_limit_max_w`.
Performance-relevant slowdowns are still caught by `clock_outlier`, `sw_thermal` and
`hw_throttle`. The env fingerprint (`env_id`) excludes the enforced limit for the same reason.

**A3. Sampler overhead: no change.** The 10 Hz sampler showed no measurable difference: the
trial-p50 ranges with and without it overlap (§7 rule). It stays on for all trials.

**A4. Trial spread: passes.** (max − min) / median of trial p50s was 2.26 % with the sampler
and 2.22 % without, under the 5 % gate.

**A5. `busy_baseline`: rule unchanged, but every pilot trial was flagged.** Device memory in
use before the pilot was 1612 MiB (1360 MiB measured at idle later the same day), against
about 240 MiB in M0, so a Windows-side process is holding GPU memory. The 1024 MiB threshold is
**not** raised to make the flag go away. Baselines are collected only after the source is closed
and the preflight baseline is back under the threshold.

**A6. Memory is measured in a separate process per configuration** (clarifying §4). Latency
trials are interleaved in one process per model (§3.3), and a shared process would contaminate
the "before this process created a CUDA context" baseline. `benchmark/memory.py` runs each
configuration in a fresh subprocess.

**A7. `torch.compile(mode="max-autotune")` does not autotune GEMMs on this GPU.** Inductor
logs `Not enough SMs to use max_autotune_gemm mode`: the RTX 4070 Laptop has 36 SMs, below
Inductor's threshold. The other max-autotune behaviour, including CUDA Graphs, still applies.
Compile rows are labelled accordingly.

**A8. Accuracy is evaluated at batch 1** for every backend. Batch size does not change these
models' eval-mode outputs, and batch 1 avoids extra `torch.compile` recompiles for a smaller
final batch.

**A9. Cool-down target: 60 °C floor** (2026-11-12, made *during* the M2 baseline run, after
YOLOX-S latency was collected). The §3.3 target
(idle + 5 °C) is replaced by **max(60 °C, idle + 5 °C)**. Reason: wall-clock time. Cooling
from about 82 °C under load back to idle + 5 °C before every trial made the run too long for
the machine owner. The `max` keeps the original rule when the idle is hot, so the target is
never unreachably close to idle.

Rows affected, identifiable by `git_commit`:

| Phase | Idle (`idle_temp_c`) | Target used |
|---|---|---|
| YOLOX-S latency | 54 °C | 59 °C (original rule) |
| everything after YOLOX-S latency | — | max(60 °C, idle + 5 °C) |

An RT-DETR-R18 latency run under the original rule (56 °C target) was stopped partway at the
owner's request. Rows are written only when a phase completes, so none of its trials were
saved, and that phase re-runs under A9.

Within each (model, mode) phase every trial uses the same target, so backend and batch
comparisons inside a phase are unaffected. Comparisons between YOLOX-S latency and other
phases mix targets about 1 °C apart. Whether that matters was **not measured**. The actual start temperature of every trial
is in `temp_start_c`, so it can be checked against latency afterwards. `cooldown_timeout` keeps
its meaning (target not reached within 300 s). The pilot (`pilot.py`) is unchanged, because it
documents the rule that was in force when it ran.
