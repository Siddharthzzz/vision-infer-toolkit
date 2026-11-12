"""Accuracy run for one (model, backend): full val set -> one row in results/<out>.csv.

python -m infer_toolkit.evaluate.accuracy --model yolox_s --backend eager_fp32
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..backends import BACKENDS, make_runner, warm_compile
from ..models import REGISTRY, get_adapter
from ..runinfo import RESULTS, append_rows, new_run
from .datasets import COCO_ANN, AdapterDataset, ade_val, coco_val, collate
from .metrics import MeanIoU, ade_reduce_labels, coco_bbox_ap, to_coco_dets


def run(
    model_name: str,
    backend: str,
    batch: int = 1,
    limit: int | None = None,
    workers: int = 6,
    out: str = "m2_accuracy",
) -> dict:
    import torch
    from PIL import Image

    prov = new_run()
    ad = get_adapter(model_name)
    model = ad.build("cuda")
    fn, info = make_runner(model, backend)

    if ad.task == "det":
        items, class_ids = coco_val(limit)
        ds = AdapterDataset(ad, items, path_index=1)
    else:
        items, class_ids = ade_val(limit), None
        ds = AdapterDataset(ad, items, path_index=0)
    # drop_last=False: a smaller final batch is fine for eager; for compile it triggers one
    # extra compilation (dynamic=False), which only costs time, not accuracy.
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch, num_workers=workers, collate_fn=collate, pin_memory=True
    )

    H, W = ad.input_hw
    warm_compile(fn, torch.zeros(batch, 3, H, W, device="cuda"), info)

    dets, miou = [], MeanIoU()
    t0 = time.perf_counter()
    for x, metas, keys in dl:
        outs = fn(x.cuda(non_blocking=True))
        results = ad.postprocess(outs, metas)
        for r, k in zip(results, keys, strict=True):
            if ad.task == "det":
                dets.extend(to_coco_dets(r, k[0], class_ids))
            else:
                miou.update(r, ade_reduce_labels(np.array(Image.open(k[1]))))
    wall = time.perf_counter() - t0

    if ad.task == "det":
        metrics = coco_bbox_ap(COCO_ANN, dets, [k for k, _ in items])
        primary = ("coco_ap", metrics["ap"])
    else:
        metrics = miou.compute()
        primary = ("ade_miou", metrics["miou"])

    row = {
        **prov,
        "model": model_name,
        "backend": backend,
        "precision": info["precision"],
        "batch": batch,
        "input_hw": f"{H}x{W}",
        "n_images": len(items),
        "metric": primary[0],
        "value": round(primary[1], 5),
        "details": ";".join(
            f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
        ),
        "build_s": round(info["build_s"], 1),
        "eval_wall_s": round(wall, 1),
    }
    append_rows(RESULTS / f"{out}.csv", [row])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=REGISTRY)
    ap.add_argument("--backend", required=True, choices=list(BACKENDS))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="debug only: first N images")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="m2_accuracy")
    a = ap.parse_args()
    if a.limit and a.out == "m2_accuracy":
        a.out = "debug_accuracy"  # partial-set numbers never land in the real results file
    print(run(a.model, a.backend, a.batch, a.limit, a.workers, a.out))


if __name__ == "__main__":
    main()
