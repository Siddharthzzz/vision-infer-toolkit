"""Latency / throughput benchmark for one model across backends x batch sizes
(methodology §3-§5). All configs of the model live in one process so trials can be
interleaved round-robin. Memory is measured separately (benchmark/memory.py).

python -m infer_toolkit.benchmark.latency --model yolox_s --mode latency
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict

from ..backends import BACKENDS, make_runner, warm_compile
from ..models import REGISTRY, get_adapter
from ..runinfo import RESULTS, append_rows, new_run
from .session import COOLDOWN_MARGIN_C, cooldown, preflight
from .stats import summarize_latency
from .telemetry import GpuMonitor, SessionState, trial_flags
from .timing import time_gpu, time_gpu_throughput

N_INPUT_IMAGES = 64  # §3.3: real val images, pre-processed once, cycled
WARMUP_ITERS, WARMUP_S = 50, 2.0  # §3.3
THROUGHPUT_K = 200  # §3.4
MONITOR_HZ = 10.0  # §5.2

_COMMON = [
    "run_id",
    "git_commit",
    "env_id",
    "timestamp_utc",
    "model",
    "backend",
    "precision",
    "batch",
    "input_hw",
    "boundary",
    "mode",
    "trial",
]
_TELEMETRY = [
    "sm_clock_med_mhz",
    "sm_clock_min_mhz",
    "mem_clock_med_mhz",
    "temp_start_c",
    "temp_max_c",
    "power_med_w",
    "power_limit_w",
    "power_source",
    "frac_sw_power_cap",
    "frac_sw_thermal",
    "frac_hw_throttle",
    "idle_temp_c",
    "baseline_used_mib",
    "windows_power_mode",
    "vendor_profile",
    "build_s",
    "sampler",
    "status",
    "flags",
]
COLUMNS = {  # methodology §8 (one fixed column set per results file)
    "latency": _COMMON
    + [
        "n_iters",
        "warmup_iters",
        "warmup_s",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "cv",
    ]
    + _TELEMETRY,
    "throughput": _COMMON + ["k_iters", "elapsed_ms", "images_per_s"] + _TELEMETRY,
}


def load_inputs(adapter, n: int = N_INPUT_IMAGES):
    """First n validation images, pre-processed with the adapter's reference pipeline."""
    import torch

    from ..evaluate.datasets import ade_val, coco_val

    paths = [p for _, p in coco_val(n)[0]] if adapter.task == "det" else [p for p, _ in ade_val(n)]
    xs = [torch.from_numpy(adapter.preprocess(p)[0]) for p in paths]
    return torch.stack(xs).cuda()


def make_fn(runner, images, batch: int):
    """Closure cycling through contiguous (view) batches of the pre-loaded images."""
    n_batches = images.shape[0] // batch
    batches = [images[i * batch : (i + 1) * batch] for i in range(n_batches)]
    it = itertools.count()

    def fn():
        return runner(batches[next(it) % n_batches])

    return fn


def run(
    model_name,
    backends,
    batches,
    mode,
    trials,
    iters,
    out,
    sampler=True,
    idle_seconds=60.0,
    extra_cols=None,
):
    import torch

    session = preflight(idle_seconds=idle_seconds)  # before any CUDA context
    print(f"preflight: {session}")
    prov = new_run()
    ad = get_adapter(model_name)
    model = ad.build("cuda")
    images = load_inputs(ad)
    H, W = ad.input_hw

    # ---- build every config once (compile time measured here, never in latency) ----------
    configs = []
    for backend, batch in itertools.product(backends, batches):
        cfg = {"backend": backend, "batch": batch, "status": "ok", "build_s": 0.0}
        try:
            runner, info = make_runner(model, backend)
            fn = make_fn(runner, images, batch)
            warm_compile(lambda _x, f=fn: f(), None, info)  # fn cycles its own inputs
            cfg.update(fn=fn, precision=info["precision"], build_s=info["build_s"])
        except torch.OutOfMemoryError:
            cfg.update(status="OOM", precision=BACKENDS[backend][0], fn=None)
            torch.cuda.empty_cache()
        configs.append(cfg)
        print(f"built {backend} b{batch}: {cfg['status']} build_s={cfg['build_s']:.1f}")

    # ---- interleaved trials ----------------------------------------------------------------
    records = []  # (cfg, trial, result, telemetry, cooldown_timeout)
    target = session.idle_temp_c + COOLDOWN_MARGIN_C
    for trial in range(trials):
        for cfg in configs:
            if cfg["status"] != "ok":
                continue
            timed_out, _ = cooldown(target)
            mon = GpuMonitor(hz=MONITOR_HZ) if sampler else None
            if mon:
                mon.__enter__()
            try:
                if mode == "latency":
                    res = time_gpu(
                        cfg["fn"], iters=iters, warmup_iters=WARMUP_ITERS, warmup_seconds=WARMUP_S
                    )
                else:
                    res = time_gpu_throughput(
                        cfg["fn"],
                        k=THROUGHPUT_K,
                        warmup_iters=WARMUP_ITERS,
                        warmup_seconds=WARMUP_S,
                    )
            finally:
                if mon:
                    mon.__exit__(None, None, None)
            tel = mon.summary() if mon else None
            records.append((cfg, trial, res, tel, timed_out))
            print(f"trial {trial} {cfg['backend']} b{cfg['batch']} done")

    # ---- flags need the same config's other trials (clock_outlier) -------------------------
    rows = []
    sess_row = session.as_row()
    for cfg, trial, res, tel, timed_out in records:
        peers = [
            t.sm_clock_med_mhz
            for c, tr, _, t, _ in records
            if c is cfg and tr != trial and t is not None
        ]
        flags = []
        if tel is not None:
            st = SessionState(
                session.power_limit_w, session.baseline_used_mib, session.idle_temp_c, peers
            )
            flags = trial_flags(tel, st, cooldown_timed_out=timed_out)
        elif timed_out:
            flags = ["cooldown_timeout"]
        row = {
            **prov,
            "model": model_name,
            "backend": cfg["backend"],
            "precision": cfg["precision"],
            "batch": cfg["batch"],
            "input_hw": f"{H}x{W}",
            "boundary": "model",
            "mode": mode,
            "trial": trial,
        }
        if mode == "latency":
            s = summarize_latency(res.samples_ms)
            row.update(
                n_iters=s.n,
                warmup_iters=res.warmup_iters,
                warmup_s=round(res.warmup_seconds, 3),
                **{
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in asdict(s).items()
                    if k != "n"
                },
            )
        else:
            row.update(
                k_iters=THROUGHPUT_K,
                elapsed_ms=round(res, 3),
                images_per_s=round(cfg["batch"] * THROUGHPUT_K / (res / 1e3), 2),
            )
        t = asdict(tel) if tel else {}
        row.update(
            sm_clock_med_mhz=t.get("sm_clock_med_mhz"),
            sm_clock_min_mhz=t.get("sm_clock_min_mhz"),
            mem_clock_med_mhz=t.get("mem_clock_med_mhz"),
            temp_start_c=t.get("temp_start_c"),
            temp_max_c=t.get("temp_max_c"),
            power_med_w=t.get("power_med_w"),
            power_limit_w=t.get("power_limit_w"),
            power_source="ac" if t.get("on_ac_all", True) else "battery",
            frac_sw_power_cap=t.get("frac_sw_power_cap"),
            frac_sw_thermal=t.get("frac_sw_thermal"),
            frac_hw_throttle=t.get("frac_hw_throttle"),
            idle_temp_c=sess_row["idle_temp_c"],
            baseline_used_mib=sess_row["baseline_used_mib"],
            windows_power_mode=sess_row["windows_power_mode"],
            vendor_profile=sess_row["vendor_profile"],
            build_s=round(cfg["build_s"], 1),
            sampler=sampler,
            status=cfg["status"],
            flags=";".join(flags),
        )
        row.update(extra_cols or {})
        rows.append(row)
    for cfg in configs:  # OOM configs still get a row (methodology §4.1)
        if cfg["status"] != "ok":
            rows.append(
                {
                    **prov,
                    "model": model_name,
                    "backend": cfg["backend"],
                    "precision": cfg["precision"],
                    "batch": cfg["batch"],
                    "input_hw": f"{H}x{W}",
                    "boundary": "model",
                    "mode": mode,
                    "status": cfg["status"],
                    "sampler": sampler,
                    **sess_row,
                    **(extra_cols or {}),
                }
            )
    cols = COLUMNS[mode] + list((extra_cols or {}).keys())
    rows = [{c: r.get(c) for c in cols} for r in rows]
    append_rows(RESULTS / f"{out}.csv", rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=REGISTRY)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--batches", default="1,8,16")
    ap.add_argument("--mode", choices=["latency", "throughput"], default="latency")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"m2_{a.mode}"
    run(
        a.model,
        a.backends.split(","),
        [int(b) for b in a.batches.split(",")],
        a.mode,
        a.trials,
        a.iters,
        out,
    )


if __name__ == "__main__":
    main()
