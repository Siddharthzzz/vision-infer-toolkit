"""PyTorch execution backends used by M2 (accuracy and latency share these exactly)."""

from __future__ import annotations

import time
from collections.abc import Callable

BACKENDS = {
    # name: (precision label, description)
    "eager_fp32": ("fp32", "PyTorch eager, FP32"),
    "eager_amp": ("fp16_amp", "PyTorch eager, torch.autocast(float16)"),
    "compile_fp32": ("fp32", 'torch.compile(mode="max-autotune"), FP32'),
}
COMPILE_MODE = "max-autotune"  # methodology §3.5 (includes CUDA Graphs on GPU)


def make_runner(model, backend: str) -> tuple[Callable, dict]:
    """Return (fn, info). fn(x) runs one forward under inference_mode and returns raw outputs.

    For compile_fp32 the compilation is NOT triggered here; call `warm_compile` once with a
    real input of the final shape so compile time is measured separately (build_s).
    """
    import torch

    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; choose from {list(BACKENDS)}")
    info = {"backend": backend, "precision": BACKENDS[backend][0], "build_s": 0.0}

    if backend == "eager_fp32":

        def fn(x):
            with torch.inference_mode():
                return model(x)

    elif backend == "eager_amp":

        def fn(x):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                return model(x)

    else:
        compiled = torch.compile(model, mode=COMPILE_MODE, dynamic=False)

        def fn(x):
            with torch.inference_mode():
                return compiled(x)

    return fn, info


def warm_compile(fn: Callable, x, info: dict) -> None:
    """Trigger (and time) torch.compile for this input shape. No-op cost for eager."""
    import torch

    t0 = time.perf_counter()
    for _ in range(3):  # max-autotune + cudagraphs record on the first few calls
        fn(x)
    torch.cuda.synchronize()
    if info["backend"].startswith("compile"):
        info["build_s"] = time.perf_counter() - t0
