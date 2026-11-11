"""Telemetry flag rules (CPU) and the NVML sampler (GPU)."""

import time

import pytest

from infer_toolkit.benchmark.telemetry import (
    REASON_HW_THERMAL,
    REASON_SW_POWER_CAP,
    REASON_SW_THERMAL,
    Sample,
    SessionState,
    summarize_samples,
    trial_flags,
)


def _s(sm=2000, temp=60, limit=115.0, reasons=0, src=0, used=1000.0):
    return Sample(0.0, sm, 8000, temp, 80.0, limit, reasons, src, used)


def _session(**kw):
    base = dict(power_limit_w=115.0, baseline_used_mib=300.0, idle_temp_c=45.0)
    base.update(kw)
    return SessionState(**base)


def test_clean_trial_has_no_flags():
    t = summarize_samples([_s(), _s(reasons=REASON_SW_POWER_CAP)])
    assert t.frac_sw_power_cap == 0.5  # recorded...
    assert trial_flags(t, _session()) == []  # ...but not flagged (§5.3)


def test_each_flag_rule():
    assert "battery" in trial_flags(summarize_samples([_s(), _s(src=1)]), _session())
    assert "power_limit_changed" in trial_flags(
        summarize_samples([_s(), _s(limit=42.0)]), _session()
    )
    assert "power_limit_changed" in trial_flags(summarize_samples([_s(limit=42.0)]), _session())
    assert "hw_throttle" in trial_flags(
        summarize_samples([_s(reasons=REASON_HW_THERMAL)] + [_s()] * 99), _session()
    )
    hot = [_s(reasons=REASON_SW_THERMAL)] * 2 + [_s()] * 8  # 20 % > 10 %
    assert "sw_thermal" in trial_flags(summarize_samples(hot), _session())
    ok = [_s(reasons=REASON_SW_THERMAL)] + [_s()] * 9  # 10 % is not > 10 %
    assert "sw_thermal" not in trial_flags(summarize_samples(ok), _session())
    slow = summarize_samples([_s(sm=1800)])
    assert "clock_outlier" in trial_flags(slow, _session(peer_sm_clock_meds=[2000.0, 2010.0]))
    assert "cooldown_timeout" in trial_flags(slow, _session(), cooldown_timed_out=True)
    assert "busy_baseline" in trial_flags(slow, _session(baseline_used_mib=2048.0))


def test_summary_fields():
    t = summarize_samples([_s(sm=1000, temp=50, used=500), _s(sm=3000, temp=70, used=900)])
    assert t.sm_clock_med_mhz == 2000 and t.sm_clock_min_mhz == 1000
    assert t.temp_start_c == 50 and t.temp_max_c == 70
    assert t.peak_used_mib == 900


@pytest.mark.gpu
def test_gpu_monitor_samples_real_device():
    pytest.importorskip("pynvml")
    from infer_toolkit.benchmark.telemetry import GpuMonitor

    with GpuMonitor(hz=20) as mon:
        time.sleep(0.5)
    s = mon.summary()
    assert 8 <= s.n_samples <= 14  # 1 start + ~10 periodic + 1 end
    assert s.sm_clock_med_mhz > 0 and s.power_limit_w > 0
    assert 0 < s.temp_max_c < 110
    assert s.on_ac_all  # methodology requires AC; fails loudly on battery
