"""Peak GPU memory per (model, backend, batch), one fresh process per config (methodology §4).

A dedicated process is required: `peak_device_delta_mib` is measured against the NVML device
`used` value read before this process created a CUDA context, and other configs' tensors or
allocator caches in the same process would contaminate it.

python -m infer_toolkit.benchmark.memory --model yolox_s
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from ..backends import BACKENDS
from ..models import REGISTRY
from ..runinfo import RESULTS, append_rows, new_run

RUN_SECONDS = 3.0  # timed region per config; >= 30 samples at 10 Hz
MONITOR_HZ = 10.0
COLUMNS = [
    "run_id",
    "git_commit",
    "env_id",
    "timestamp_utc",
    "model",
    "backend",
    "precision",
    "batch",
    "input_hw",
    "status",
    "peak_alloc_mib",
    "peak_reserved_mib",
    "peak_device_delta_mib",
    "baseline_used_mib",
    "n_samples",
    "error",
]


def child(model_name: str, backend: str, batch: int) -> dict:
    # ---- baseline BEFORE torch touches CUDA ----
    import pynvml as n

    n.nvmlInit()
    h = n.nvmlDeviceGetHandleByIndex(0)
    baseline = n.nvmlDeviceGetMemoryInfo(h).used / 2**20
    n.nvmlShutdown()

    import torch

    from ..backends import make_runner, warm_compile
    from ..models import get_adapter
    from .latency import load_inputs, make_fn
    from .telemetry import GpuMonitor

    ad = get_adapter(model_name)
    H, W = ad.input_hw
    out = {
        "model": model_name,
        "backend": backend,
        "precision": BACKENDS[backend][0],
        "batch": batch,
        "input_hw": f"{H}x{W}",
        "baseline_used_mib": round(baseline, 1),
    }
    try:
        with GpuMonitor(hz=MONITOR_HZ) as mon:  # covers build + warm-up + run
            model = ad.build("cuda")
            runner, info = make_runner(model, backend)
            fn = make_fn(runner, load_inputs(ad), batch)
            torch.cuda.reset_peak_memory_stats()
            warm_compile(lambda _x: fn(), None, info)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < RUN_SECONDS:
                fn()
                torch.cuda.synchronize()
        tel = mon.summary()
        out.update(
            status="ok",
            peak_alloc_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
            peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1),
            peak_device_delta_mib=round(tel.peak_used_mib - baseline, 1),
            n_samples=tel.n_samples,
        )
    except torch.OutOfMemoryError as e:
        out.update(status="OOM", error=str(e).splitlines()[0][:200])
    return out


def run(model_name: str, backends: list[str], batches: list[int], out: str) -> list[dict]:
    prov = new_run()
    rows = []
    for backend in backends:
        for batch in batches:
            cmd = [
                sys.executable,
                "-m",
                "infer_toolkit.benchmark.memory",
                "--child",
                "--model",
                model_name,
                "--backend",
                backend,
                "--batch",
                str(batch),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            if r.returncode == 0 and lines:
                res = json.loads(lines[-1])
            else:  # a crash is recorded, never skipped
                res = {
                    "model": model_name,
                    "backend": backend,
                    "batch": batch,
                    "status": "error",
                    "error": (r.stderr.strip().splitlines() or ["?"])[-1][:200],
                }
            rows.append({c: {**prov, **res}.get(c) for c in COLUMNS})
            print(
                f"{backend} b{batch}: {res.get('status')} "
                f"delta={res.get('peak_device_delta_mib')} MiB"
            )
    append_rows(RESULTS / f"{out}.csv", rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=REGISTRY)
    ap.add_argument("--backend", choices=list(BACKENDS))
    ap.add_argument("--batch", type=int)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--batches", default="1,8,16")
    ap.add_argument("--out", default="m2_memory")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        print(json.dumps(child(a.model, a.backend, a.batch)))
    else:
        run(a.model, a.backends.split(","), [int(b) for b in a.batches.split(",")], a.out)


if __name__ == "__main__":
    main()
