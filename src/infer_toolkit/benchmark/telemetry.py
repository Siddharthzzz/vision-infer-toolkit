"""GPU telemetry sampling and trial flags (docs/methodology.md §5).

`GpuMonitor` samples NVML in a background thread while a trial runs. `trial_flags` is a pure
function so the flag rules are unit-testable without a GPU.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

# NVML clock-event reason bits (nvml.h: nvmlClocksEventReason*)
REASON_GPU_IDLE = 0x1
REASON_SW_POWER_CAP = 0x4
REASON_HW_SLOWDOWN = 0x8
REASON_SW_THERMAL = 0x20
REASON_HW_THERMAL = 0x40
REASON_HW_POWER_BRAKE = 0x80
HW_THROTTLE_MASK = REASON_HW_SLOWDOWN | REASON_HW_THERMAL | REASON_HW_POWER_BRAKE

POWER_SOURCE_AC = 0  # nvml.h: NVML_POWER_SOURCE_AC

# Flag thresholds (§5.3)
SW_THERMAL_MAX_FRAC = 0.10
CLOCK_OUTLIER_FRAC = 0.05
BUSY_BASELINE_MIB = 1024


@dataclass
class Sample:
    t: float
    sm_mhz: int
    mem_mhz: int
    temp_c: int
    power_w: float
    limit_w: float
    reasons: int
    power_source: int
    used_mib: float


@dataclass
class TelemetrySummary:
    n_samples: int
    sm_clock_med_mhz: float
    sm_clock_min_mhz: float
    mem_clock_med_mhz: float
    temp_start_c: float
    temp_max_c: float
    power_med_w: float
    power_limit_w: float
    power_limit_changed: bool
    on_ac_all: bool
    frac_sw_power_cap: float
    frac_sw_thermal: float
    frac_hw_throttle: float
    peak_used_mib: float


def summarize_samples(samples: list[Sample]) -> TelemetrySummary:
    if not samples:
        raise ValueError("no telemetry samples")

    def arr(name: str) -> np.ndarray:
        return np.array([getattr(s, name) for s in samples], dtype=np.float64)

    reasons = np.array([s.reasons for s in samples], dtype=np.int64)
    limits = arr("limit_w")
    return TelemetrySummary(
        n_samples=len(samples),
        sm_clock_med_mhz=float(np.median(arr("sm_mhz"))),
        sm_clock_min_mhz=float(arr("sm_mhz").min()),
        mem_clock_med_mhz=float(np.median(arr("mem_mhz"))),
        temp_start_c=float(samples[0].temp_c),
        temp_max_c=float(arr("temp_c").max()),
        power_med_w=float(np.median(arr("power_w"))),
        power_limit_w=float(limits[0]),
        power_limit_changed=bool(np.any(limits != limits[0])),
        on_ac_all=all(s.power_source == POWER_SOURCE_AC for s in samples),
        frac_sw_power_cap=float(np.mean((reasons & REASON_SW_POWER_CAP) != 0)),
        frac_sw_thermal=float(np.mean((reasons & REASON_SW_THERMAL) != 0)),
        frac_hw_throttle=float(np.mean((reasons & HW_THROTTLE_MASK) != 0)),
        peak_used_mib=float(arr("used_mib").max()),
    )


@dataclass
class SessionState:
    """Values captured once at session start by the preflight."""

    power_limit_w: float
    baseline_used_mib: float
    idle_temp_c: float
    peer_sm_clock_meds: list[float] = field(default_factory=list)  # same config, other trials


def trial_flags(
    t: TelemetrySummary, session: SessionState, cooldown_timed_out: bool = False
) -> list[str]:
    flags = []
    if not t.on_ac_all:
        flags.append("battery")
    if t.power_limit_changed or t.power_limit_w != session.power_limit_w:
        flags.append("power_limit_changed")
    if t.frac_hw_throttle > 0:
        flags.append("hw_throttle")
    if t.frac_sw_thermal > SW_THERMAL_MAX_FRAC:
        flags.append("sw_thermal")
    if session.peer_sm_clock_meds:
        ref = float(np.median(session.peer_sm_clock_meds))
        if abs(t.sm_clock_med_mhz - ref) / ref > CLOCK_OUTLIER_FRAC:
            flags.append("clock_outlier")
    if cooldown_timed_out:
        flags.append("cooldown_timeout")
    if session.baseline_used_mib > BUSY_BASELINE_MIB:
        flags.append("busy_baseline")
    return flags


class GpuMonitor:
    """Context manager: samples NVML at `hz` in a daemon thread.

    with GpuMonitor(hz=10) as mon:
        run_trial()
    summary = mon.summary()
    """

    def __init__(self, device_index: int = 0, hz: float = 10.0):
        self.device_index = device_index
        self.period = 1.0 / hz
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read(self) -> Sample:
        import pynvml as n

        h = self._handle
        return Sample(
            t=time.perf_counter(),
            sm_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
            mem_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
            temp_c=n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
            power_w=n.nvmlDeviceGetPowerUsage(h) / 1000.0,
            limit_w=n.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
            reasons=n.nvmlDeviceGetCurrentClocksEventReasons(h),
            power_source=n.nvmlDeviceGetPowerSource(h),
            used_mib=n.nvmlDeviceGetMemoryInfo(h).used / 2**20,
        )

    def _loop(self) -> None:
        # __enter__ already took the first sample, so wait one period before each read.
        nxt = time.perf_counter() + self.period
        while not self._stop.wait(max(0.0, nxt - time.perf_counter())):
            self.samples.append(self._read())
            nxt += self.period

    def __enter__(self) -> GpuMonitor:
        import pynvml as n

        n.nvmlInit()
        self._handle = n.nvmlDeviceGetHandleByIndex(self.device_index)
        self.samples.append(self._read())  # guarantees >= 1 sample, captures start temp
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.samples.append(self._read())  # guarantees an end sample
        import pynvml as n

        n.nvmlShutdown()

    def summary(self) -> TelemetrySummary:
        return summarize_samples(self.samples)
