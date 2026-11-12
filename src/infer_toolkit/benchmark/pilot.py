"""M2 pilot (methodology §9): validates warm-up budget, sampler overhead, trial spread, and
measures how the laptop's enforced power limit behaves under load.

Analysis rules were fixed before any pilot data existed:
- settle: reference = median of iterations 1000..1999 of a cold start; settled at the first
  iteration after which the rolling median (window 25) stays within +/-2 % of the reference.
  The warm-up budget is adequate if settling happens before the §3.3 budget ends
  (>= 50 iterations AND >= 2.0 s).
- sampler overhead: compare trial-p50 ranges with/without the 10 Hz NVML sampler
  (stats.compare: overlapping ranges = no measurable difference).
- spread: (max - min) / median of trial p50s, gate 0.05.

python -m infer_toolkit.benchmark.pilot cold      # run in a fresh process
python -m infer_toolkit.benchmark.pilot trials
python -m infer_toolkit.benchmark.pilot analyze
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ..runinfo import RESULTS, append_rows, new_run

MODEL, BACKEND, BATCH = "yolox_s", "eager_fp32", 1
COLD_ITERS = 2000
SETTLE_WINDOW, SETTLE_TOL = 25, 0.02
TRIALS, ITERS = 5, 1000


def _setup():
    from ..backends import make_runner
    from ..models import get_adapter
    from .latency import load_inputs, make_fn

    ad = get_adapter(MODEL)
    runner, _ = make_runner(ad.build("cuda"), BACKEND)
    return make_fn(runner, load_inputs(ad), BATCH)


def _telemetry_rows(prov, part, trial, sampler_on, mon):
    t0 = mon.samples[0].t
    return [
        {
            **prov,
            "part": part,
            "trial": trial,
            "sampler": sampler_on,
            "t_s": round(s.t - t0, 3),
            "sm_mhz": s.sm_mhz,
            "temp_c": s.temp_c,
            "power_w": round(s.power_w, 2),
            "limit_w": round(s.limit_w, 2),
            "reasons": hex(s.reasons),
            "power_source": s.power_source,
        }
        for s in mon.samples
    ]


def cold() -> None:
    from .session import COOLDOWN_MARGIN_C, cooldown, preflight
    from .telemetry import GpuMonitor
    from .timing import time_gpu

    sess = preflight()
    timed_out, waited = cooldown(sess.idle_temp_c + COOLDOWN_MARGIN_C)
    prov = new_run()
    fn = _setup()  # model build + input upload: CUDA context now exists, nothing timed yet
    with GpuMonitor(hz=10) as mon:
        r = time_gpu(fn, iters=COLD_ITERS, warmup_iters=0, warmup_seconds=0.0)
    cum = np.cumsum(r.samples_ms)
    append_rows(
        RESULTS / "m2_pilot_coldstart.csv",
        [
            {
                **prov,
                "iter": i,
                "ms": round(float(v), 5),
                "cum_gpu_ms": round(float(c), 3),
                "cooldown_timeout": timed_out,
                **sess.as_row(),
            }
            for i, (v, c) in enumerate(zip(r.samples_ms, cum, strict=True))
        ],
    )
    append_rows(RESULTS / "m2_pilot_telemetry.csv", _telemetry_rows(prov, "cold", 0, True, mon))
    print(f"cold start recorded ({COLD_ITERS} iters), cooldown waited {waited:.0f}s")


def trials() -> None:
    from dataclasses import asdict

    from .session import COOLDOWN_MARGIN_C, cooldown, preflight
    from .stats import summarize_latency
    from .telemetry import GpuMonitor, SessionState, summarize_samples, trial_flags
    from .timing import time_gpu

    sess = preflight()
    prov = new_run()
    fn = _setup()
    recs = []
    for trial in range(TRIALS):
        for sampler_on in (True, False):  # interleaved: on, off, on, off, ...
            timed_out, _ = cooldown(sess.idle_temp_c + COOLDOWN_MARGIN_C)
            mon = GpuMonitor(hz=10) if sampler_on else None
            if mon:
                mon.__enter__()
            try:
                r = time_gpu(fn, iters=ITERS, warmup_iters=50, warmup_seconds=2.0)
            finally:
                if mon:
                    mon.__exit__(None, None, None)
            recs.append((trial, sampler_on, r, mon, timed_out))
            print(f"trial {trial} sampler={sampler_on} p50={np.median(r.samples_ms):.4f} ms")

    rows, tel_rows = [], []
    for trial, on, r, mon, timed_out in recs:
        s = summarize_latency(r.samples_ms)
        row = {
            **prov,
            "model": MODEL,
            "backend": BACKEND,
            "batch": BATCH,
            "trial": trial,
            "sampler": on,
            **{k: v for k, v in asdict(s).items()},
            "cooldown_timeout": timed_out,
            **sess.as_row(),
        }
        if mon:
            t = summarize_samples(mon.samples)
            peers = [
                summarize_samples(m.samples).sm_clock_med_mhz
                for tr, o, _, m, _ in recs
                if o and tr != trial
            ]
            st = SessionState(sess.power_limit_w, sess.baseline_used_mib, sess.idle_temp_c, peers)
            row.update(
                sm_clock_med_mhz=t.sm_clock_med_mhz,
                temp_max_c=t.temp_max_c,
                limit_min_w=min(x.limit_w for x in mon.samples),
                limit_max_w=max(x.limit_w for x in mon.samples),
                frac_sw_power_cap=t.frac_sw_power_cap,
                frac_sw_thermal=t.frac_sw_thermal,
                frac_hw_throttle=t.frac_hw_throttle,
                flags=";".join(trial_flags(t, st, timed_out)),
            )
            tel_rows += _telemetry_rows(prov, "trials", trial, True, mon)
        rows.append(row)
    cols = list(rows[0].keys())  # first row is sampler-on, so it carries every column
    append_rows(RESULTS / "m2_pilot.csv", [{c: r.get(c) for c in cols} for r in rows])
    append_rows(RESULTS / "m2_pilot_telemetry.csv", tel_rows)


def analyze() -> dict:
    import pandas as pd

    from .stats import compare, trial_spread

    cs = pd.read_csv(RESULTS / "m2_pilot_coldstart.csv")
    cs = cs[cs.run_id == cs.run_id.iloc[-1]].sort_values("iter")  # latest cold run
    ms = cs.ms.to_numpy()
    ref = float(np.median(ms[1000:2000]))
    roll = pd.Series(ms).rolling(SETTLE_WINDOW).median().to_numpy()
    ok = np.abs(roll / ref - 1) <= SETTLE_TOL
    ok[: SETTLE_WINDOW - 1] = False
    bad = np.where(~ok)[0]
    settle = int(bad[-1] + 1) if len(bad) else 0
    settle_ms = float(cs.cum_gpu_ms.iloc[settle]) if settle < len(cs) else float("nan")
    iters_in_2s = int(np.searchsorted(cs.cum_gpu_ms.to_numpy(), 2000.0))
    budget_end = max(50, iters_in_2s)

    pt = pd.read_csv(RESULTS / "m2_pilot.csv")
    pt = pt[pt.run_id == pt.run_id.iloc[-1]]
    on, off = pt[pt.sampler].p50_ms.to_list(), pt[~pt.sampler].p50_ms.to_list()
    tel = pd.read_csv(RESULTS / "m2_pilot_telemetry.csv")
    tl = tel[(tel.part == "trials") & (tel.run_id == pt.run_id.iloc[-1])]

    out = {
        "cold_ref_p50_ms": round(ref, 4),
        "cold_first_iter_ms": round(float(ms[0]), 3),
        "settle_iter": settle,
        "settle_cum_gpu_ms": round(settle_ms, 1),
        "warmup_budget_end_iter": budget_end,
        "warmup_budget_adequate": settle <= budget_end,
        "sampler_on_vs_off": compare(on, off),
        "spread_sampler_on": round(trial_spread(on), 4),
        "spread_sampler_off": round(trial_spread(off), 4),
        "spread_gate_0.05_pass": max(trial_spread(on), trial_spread(off)) <= 0.05,
        "limit_w_min": float(tl.limit_w.min()),
        "limit_w_median": float(tl.limit_w.median()),
        "limit_w_max": float(tl.limit_w.max()),
        "flagged_trials": int((pt["flags"].fillna("") != "").sum()),
        "flags_seen": ";".join(
            sorted({f for x in pt["flags"].dropna() for f in x.split(";") if f})
        ),
    }
    append_rows(RESULTS / "m2_pilot_summary.csv", [{**new_run(), **{k: out[k] for k in out}}])
    for k, v in out.items():
        print(f"{k:28s} {v}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["cold", "trials", "analyze"])
    a = ap.parse_args()
    {"cold": cold, "trials": trials, "analyze": analyze}[a.part]()
    sys.exit(0)
