"""CPU tests: summary tables follow methodology §7 (median of unflagged trials, min-max)."""

import pandas as pd
import pytest

from infer_toolkit.benchmark.summarize import summarize_latency


def _row(trial, p50, flags="", status="ok"):
    return dict(
        model="m",
        backend="b",
        precision="fp32",
        batch=1,
        input_hw="640x640",
        trial=trial,
        p50_ms=p50,
        p95_ms=None if p50 is None else p50 + 1,
        p99_ms=None if p50 is None else p50 + 2,
        mean_ms=p50,
        cv=0.01,
        sm_clock_med_mhz=2500,
        temp_max_c=60,
        build_s=0.0,
        status=status,
        flags=flags,
        run_id="r1",
        git_commit="abc",
    )


def test_flagged_trials_excluded_but_counted():
    df = pd.DataFrame(
        [_row(0, 10.0), _row(1, 11.0), _row(2, 50.0, flags="hw_throttle"), _row(3, 12.0)]
    )
    (s,) = summarize_latency(df).to_dict("records")
    assert s["n_trials"] == 4 and s["n_flagged"] == 1
    assert s["p50_ms"] == 11.0  # median of 10, 11, 12 (flagged 50 excluded)
    assert (s["p50_ms_min"], s["p50_ms_max"]) == (10.0, 12.0)
    assert s["trial_spread"] == pytest.approx(2 / 11)


def test_oom_config_reported_without_numbers():
    df = pd.DataFrame([_row(None, None, status="OOM")])
    (s,) = summarize_latency(df).to_dict("records")
    assert s["status"] == "OOM" and s["n_trials"] == 0 and "p50_ms" not in s
