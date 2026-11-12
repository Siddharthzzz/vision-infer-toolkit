"""Accuracy metrics (methodology §6)."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

ADE_NUM_CLASSES = 150
IGNORE = 255


def to_coco_dets(result: dict, image_id: int, class_ids: list[int] | None) -> list[dict]:
    """Per-image detections -> COCO results json entries (bbox as x, y, w, h)."""
    b = result["boxes_xyxy"]
    cats = (
        result["category_ids"]
        if "category_ids" in result
        else np.asarray(class_ids)[result["labels"].astype(np.int64)]
    )
    out = []
    for (x1, y1, x2, y2), s, c in zip(
        b.tolist(), result["scores"].tolist(), cats.tolist(), strict=True
    ):
        out.append(
            {
                "image_id": image_id,
                "category_id": int(c),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(s),
            }
        )
    return out


def coco_bbox_ap(ann_file: Path, dets: list[dict], image_ids: list[int]) -> dict[str, float]:
    """Standard pycocotools COCOeval (bbox, maxDets 100) restricted to `image_ids`."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_file))
        dt = gt.loadRes(dets) if dets else None
    if dt is None:
        return dict.fromkeys(["ap", "ap50", "ap75", "ap_s", "ap_m", "ap_l"], 0.0)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = sorted(image_ids)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"ap": s[0], "ap50": s[1], "ap75": s[2], "ap_s": s[3], "ap_m": s[4], "ap_l": s[5]}


def ade_reduce_labels(label: np.ndarray) -> np.ndarray:
    """ADE20K png: 0 = 'other' (ignored), 1..150 = classes -> 0..149, ignore = 255."""
    lab = label.astype(np.int64) - 1
    lab[label == 0] = IGNORE
    return lab


class MeanIoU:
    """Accumulates intersection/union over the whole dataset (not a mean of per-image IoU)."""

    def __init__(self, num_classes: int = ADE_NUM_CLASSES):
        self.n = num_classes
        self.conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, label: np.ndarray) -> None:
        if pred.shape != label.shape:
            raise ValueError(f"shape mismatch pred {pred.shape} vs label {label.shape}")
        valid = label != IGNORE
        idx = label[valid].astype(np.int64) * self.n + pred[valid].astype(np.int64)
        self.conf += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def compute(self) -> dict[str, float]:
        inter = np.diag(self.conf).astype(np.float64)
        union = self.conf.sum(0) + self.conf.sum(1) - np.diag(self.conf)
        present = union > 0
        iou = inter[present] / union[present]
        return {
            "miou": float(iou.mean()),
            "aacc": float(inter.sum() / max(self.conf.sum(), 1)),
            "n_classes_present": int(present.sum()),
        }
