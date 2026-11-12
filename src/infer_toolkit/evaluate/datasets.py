"""Validation-set listings and a DataLoader wrapper around a model adapter's preprocess."""

from __future__ import annotations

import json

import numpy as np

from ..runinfo import REPO_ROOT

COCO_DIR = REPO_ROOT / "data/coco"
COCO_ANN = COCO_DIR / "annotations/instances_val2017.json"
ADE_DIR = REPO_ROOT / "data/ade20k/ADEChallengeData2016"


def coco_val(limit: int | None = None) -> tuple[list[tuple[int, str]], list[int]]:
    """[(image_id, path)] sorted by id, and the sorted COCO category ids (80 entries).
    YOLOX's contiguous label i maps to category id class_ids[i]."""
    ann = json.loads(COCO_ANN.read_text())
    items = sorted((im["id"], str(COCO_DIR / "val2017" / im["file_name"])) for im in ann["images"])
    class_ids = sorted(c["id"] for c in ann["categories"])
    return (items[:limit] if limit else items), class_ids


def ade_val(limit: int | None = None) -> list[tuple[str, str]]:
    """[(image_path, annotation_png_path)] sorted by name."""
    imgs = sorted((ADE_DIR / "images/validation").glob("*.jpg"))
    items = [(str(p), str(ADE_DIR / "annotations/validation" / (p.stem + ".png"))) for p in imgs]
    return items[:limit] if limit else items


class AdapterDataset:
    """Map-style dataset: item -> (CHW float32 tensor, meta, key). Picklable for workers."""

    def __init__(self, adapter, items: list[tuple], path_index: int):
        self.adapter, self.items, self.path_index = adapter, items, path_index

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        import torch

        item = self.items[i]
        x, meta = self.adapter.preprocess(item[self.path_index])
        return torch.from_numpy(np.ascontiguousarray(x)), meta, item


def collate(batch):
    import torch

    xs, metas, keys = zip(*batch, strict=True)
    return torch.stack(xs), list(metas), list(keys)
