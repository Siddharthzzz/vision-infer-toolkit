"""Collect a reproducible snapshot of the software/hardware environment.

Every results CSV in this repo is tied back to one of these snapshots, so a
number can always be traced to the exact driver, power limit and library versions
it was measured under.
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path

GPU_QUERY_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "compute_cap",
    "enforced.power.limit",  # "power.limit" reports [N/A] on this laptop GPU
    "power.default_limit",
    "power.max_limit",
    "pstate",
    "clocks.max.sm",
    "clocks.max.mem",
    "temperature.gpu",
)

# import name -> distribution name (versions read from package metadata)
PY_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime-gpu",
    "tensorrt": "tensorrt-cu13",
    "modelopt": "nvidia-modelopt",
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "pycocotools": "pycocotools",
}


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def parse_nvidia_smi_csv(text: str, fields: tuple[str, ...] = GPU_QUERY_FIELDS) -> dict[str, str]:
    """Parse `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` for GPU 0."""
    first = text.strip().splitlines()[0]
    values = [v.strip() for v in first.split(",")]
    if len(values) != len(fields):
        raise ValueError(f"expected {len(fields)} fields, got {len(values)}: {first!r}")
    return {f"gpu.{k}": v for k, v in zip(fields, values, strict=True)}


def parse_nvcc_version(text: str) -> str | None:
    """Extract 'X.Y' from `nvcc --version` output."""
    m = re.search(r"release (\d+\.\d+)", text)
    return m.group(1) if m else None


def parse_trt_header_version(text: str) -> str | None:
    """Extract 'MAJOR.MINOR.PATCH.BUILD' from NvInferVersion.h contents."""
    parts = []
    for key in ("MAJOR", "MINOR", "PATCH", "BUILD"):
        m = re.search(rf"#define\s+(?:NV_)?TRT_{key}_ENTERPRISE\s+(\d+)", text) or re.search(
            rf"#define\s+NV_TENSORRT_{key}\s+(\d+)", text
        )
        if not m:
            return None
        parts.append(m.group(1))
    return ".".join(parts)


def _py_versions() -> dict[str, str]:
    out = {}
    for name, dist in PY_PACKAGES.items():
        try:
            importlib.import_module(name)
            out[f"py.{name}"] = metadata.version(dist)
        except Exception as e:  # noqa: BLE001 - record any import/metadata failure verbatim
            out[f"py.{name}"] = f"MISSING ({type(e).__name__})"
    return out


def _torch_cuda() -> dict[str, str]:
    try:
        import torch
    except ImportError:
        return {}
    return {
        "torch.cuda_build": str(torch.version.cuda),
        "torch.cuda_available": str(torch.cuda.is_available()),
        "torch.cudnn": str(torch.backends.cudnn.version()),
    }


def collect() -> dict[str, str]:
    info: dict[str, str] = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "os.kernel": platform.release(),
        "python": platform.python_version(),
        "git.commit": _run(["git", "rev-parse", "--short", "HEAD"]) or "none",
    }
    smi = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if smi:
        info.update(parse_nvidia_smi_csv(smi))
    nvcc = _run(["nvcc", "--version"]) or _run(["/usr/local/cuda/bin/nvcc", "--version"])
    info["cuda.nvcc"] = (parse_nvcc_version(nvcc) if nvcc else None) or "MISSING"
    hdr = Path("/usr/include/x86_64-linux-gnu/NvInferVersion.h")
    info["trt.cpp_headers"] = (
        parse_trt_header_version(hdr.read_text()) if hdr.exists() else None
    ) or "MISSING"
    info.update(_py_versions())
    info.update(_torch_cuda())
    try:
        import onnxruntime as ort

        info["ort.providers"] = ";".join(ort.get_available_providers())
    except ImportError:
        pass
    return info


def write_csv(info: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(sorted(info.items()))
