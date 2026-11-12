"""Session preflight and cool-down (methodology §3.3, §5.1).

`preflight()` must run BEFORE the process creates a CUDA context, because it records the
device memory baseline that `peak_device_delta_mib` is measured against.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..runinfo import REPO_ROOT
from .telemetry import POWER_SOURCE_AC

BENCH_STATE = REPO_ROOT / "configs/bench_state.yaml"
COOLDOWN_MARGIN_C = 5.0
COOLDOWN_FLOOR_C = 60.0  # amendment A9
COOLDOWN_TIMEOUT_S = 300.0

# Windows 11 power-mode overlay GUIDs (ActiveOverlayAcPowerScheme)
OVERLAYS = {
    "00000000-0000-0000-0000-000000000000": "balanced_overlay",
    "ded574b5-45a0-4f42-8737-46345c09c238": "best_performance",
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "best_power_efficiency",
}


class PreflightError(RuntimeError):
    pass


@dataclass
class Session:
    power_limit_w: float
    baseline_used_mib: float
    idle_temp_c: float
    windows_power_mode: str
    vendor_profile: str

    def as_row(self) -> dict:
        return asdict(self)


def _nvml():
    import pynvml as n

    n.nvmlInit()
    return n, n.nvmlDeviceGetHandleByIndex(0)


def read_temp_c() -> float:
    n, h = _nvml()
    try:
        return float(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
    finally:
        n.nvmlShutdown()


# PowerGetEffectiveOverlayScheme = the overlay Windows actually applies. The slider value
# (PowerGetActualOverlayScheme / registry) is ignored under non-Balanced plans such as
# "High performance", so reading only the slider misreports the state (found 2026-11-12).
_DLL = '[DllImport("powrprof.dll")] public static extern uint '
_OVERLAY_PS = (
    "$sig = '"
    + _DLL
    + "PowerGetEffectiveOverlayScheme(out Guid g); "
    + _DLL
    + "PowerGetActualOverlayScheme(out Guid g);'\n"
    "$t = Add-Type -MemberDefinition $sig -Name P -Namespace W -PassThru\n"
    "$e = [Guid]::Empty; [void]$t::PowerGetEffectiveOverlayScheme([ref]$e)\n"
    "$a = [Guid]::Empty; [void]$t::PowerGetActualOverlayScheme([ref]$a)\n"
    '"$e $a"\n'
)


def windows_power_mode() -> str:
    """'<plan>/effective=<overlay>/slider=<overlay>' via WSL interop; 'unknown' parts if not
    readable. The effective overlay is what is in force."""

    def run(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd="/mnt/c")
            return r.stdout.replace("\r", "").strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    scheme = run(["powershell.exe", "-NoProfile", "-Command", "powercfg /getactivescheme"])
    plan = scheme.split("(")[-1].split(")")[0].strip() if "(" in scheme else "unknown"
    parts = run(["powershell.exe", "-NoProfile", "-Command", _OVERLAY_PS]).split()
    eff, slider = (parts + ["", ""])[:2] if len(parts) >= 2 else ("", "")
    name = {**OVERLAYS, "00000000-0000-0000-0000-000000000000": "none"}
    return (
        f"{plan.replace(' ', '_').lower()}"
        f"/effective={name.get(eff, eff or 'unknown')}"
        f"/slider={OVERLAYS.get(slider, slider or 'unknown')}"
    )


def vendor_profile() -> str:
    """Hand-set in configs/bench_state.yaml (not readable programmatically)."""
    import yaml

    if not BENCH_STATE.exists():
        return "unset"
    return str((yaml.safe_load(BENCH_STATE.read_text()) or {}).get("vendor_profile", "unset"))


def preflight(idle_seconds: float = 60.0, hz: float = 2.0) -> Session:
    """Refuses on battery; records everything else. No CUDA context may exist yet."""
    import sys

    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_initialized():
            raise PreflightError("CUDA already initialised: device-memory baseline would be wrong")
    n, h = _nvml()
    try:
        if n.nvmlDeviceGetPowerSource(h) != POWER_SOURCE_AC:
            raise PreflightError("not on AC power: methodology §5.1 forbids benchmarking")
        limit_w = n.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
        used = n.nvmlDeviceGetMemoryInfo(h).used / 2**20
        temps = []
        t_end = time.perf_counter() + idle_seconds
        while True:
            temps.append(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
            if time.perf_counter() >= t_end:
                break
            time.sleep(1.0 / hz)
    finally:
        n.nvmlShutdown()
    return Session(
        power_limit_w=limit_w,
        baseline_used_mib=round(used, 1),
        idle_temp_c=float(np.median(temps)),
        windows_power_mode=windows_power_mode(),
        vendor_profile=vendor_profile(),
    )


def cooldown_target(idle_temp_c: float) -> float:
    """Per-trial cool-down target (methodology §3.3 as amended by A9): a fixed 60 °C floor,
    never below idle + 5 °C, so a hot idle cannot make every cool-down time out."""
    return max(COOLDOWN_FLOOR_C, idle_temp_c + COOLDOWN_MARGIN_C)


def cooldown(target_c: float, timeout_s: float = COOLDOWN_TIMEOUT_S) -> tuple[bool, float]:
    """Wait until GPU temp <= target_c. Returns (timed_out, seconds_waited)."""
    t0 = time.perf_counter()
    while read_temp_c() > target_c:
        if time.perf_counter() - t0 > timeout_s:
            return True, time.perf_counter() - t0
        time.sleep(1.0)
    return False, time.perf_counter() - t0
