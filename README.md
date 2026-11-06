# Vision Inference Optimization Toolkit

Three vision models (YOLOX-S, RT-DETR-R18, SegFormer-B0) taken from PyTorch → ONNX → TensorRT
(FP32 / FP16 / INT8 PTQ / INT8 QAT), served from a C++ runtime with CUDA pre/post-processing,
and benchmarked for accuracy, latency and memory.

> Status: **M0 (setup)**. No results yet. Every number in this README will come from a CSV in `results/`.

## Environment (pinned)

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 on WSL2, Windows NVIDIA driver (CUDA 13.x capable) |
| CUDA toolkit | 13.0 |
| Python | 3.12 |
| PyTorch | 2.14.1+cu130 |
| TensorRT | 10.13.3.9 (Python wheel + C++ apt packages) |
| ONNX Runtime GPU | 1.30.0 |
| NVIDIA Model Optimizer | 0.48.0 |
| ONNX | 1.21.x |

The exact resolved Python set is in `requirements.lock`. `results/env.csv` records the snapshot
of the machine each set of results was produced on.

## Setup

```bash
bash scripts/setup_system.sh            # needs sudo: build tools, CUDA 13.0, TensorRT 10.13.3.9, Nsight Systems
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock && pip install -e . --no-deps
pre-commit install
python scripts/check_env.py             # writes results/env.csv
pytest                                  # CPU + GPU tests
cmake -S cpp -B cpp/build -G Ninja && cmake --build cpp/build && ctest --test-dir cpp/build
```

## Licences

Code: Apache-2.0. Model licences are listed per model once they're added (SegFormer weights are
under NVIDIA's own licence, which needs checking before redistribution).
