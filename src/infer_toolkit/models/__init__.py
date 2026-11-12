"""Model adapters: one uniform interface per model (docs/design.md).

Each adapter exposes
- `build(device)`: eval-mode FP32 `nn.Module`. Its forward takes the pre-processed batch
  (B, 3, H, W) float32 and returns raw output tensor(s). This module is what gets timed
  (`model` boundary), exported to ONNX and compiled to TensorRT.
- `preprocess(path)`: CPU reference pre-processing copied from each model's official eval
  pipeline. Returns (CHW float32 array, meta).
- `postprocess(outputs, metas)`: raw outputs to per-image results in original image
  coordinates (detection) or a label map at original resolution (segmentation).
"""

from __future__ import annotations

from .base import ModelAdapter

REGISTRY = ("yolox_s", "rtdetr_r18", "segformer_b0")


def get_adapter(name: str, repo_root=None) -> ModelAdapter:
    if name == "yolox_s":
        from .yolox import YoloxS

        return YoloxS(repo_root)
    if name == "rtdetr_r18":
        from .rtdetr import RtDetrR18

        return RtDetrR18(repo_root)
    if name == "segformer_b0":
        from .segformer import SegformerB0

        return SegformerB0(repo_root)
    raise KeyError(f"unknown model {name!r}; choose from {REGISTRY}")
