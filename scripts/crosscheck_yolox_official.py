"""Cross-check: evaluate yolox_s.pth with YOLOX's OWN dataset + evaluator code (no adapter code
from this repo), to tell a bug in our pipeline apart from an environment difference.

Uses third_party/YOLOX: COCODataset + ValTransform(legacy=False) + COCOEvaluator with the
eval defaults from yolox/exp/yolox_base.py (test_size 640, test_conf 0.01, nmsthre 0.65).
Appends one row to results/m2_accuracy_crosscheck.csv.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/YOLOX"))

from yolox.data import COCODataset, ValTransform  # noqa: E402
from yolox.evaluators import COCOEvaluator  # noqa: E402

from infer_toolkit.models import get_adapter  # noqa: E402  (model build only: weights/arch)
from infer_toolkit.runinfo import RESULTS, append_rows, new_run  # noqa: E402

prov = new_run()
ds = COCODataset(
    data_dir=str(ROOT / "data/coco"),
    json_file="instances_val2017.json",
    name="val2017",
    img_size=(640, 640),
    preproc=ValTransform(legacy=False),
)
dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=6, shuffle=False)
ev = COCOEvaluator(dl, img_size=(640, 640), confthre=0.01, nmsthre=0.65, num_classes=80)
model = get_adapter("yolox_s").build("cuda")
ap50_95, ap50, summary = ev.evaluate(model, distributed=False, half=False)
print(summary)
append_rows(
    RESULTS / "m2_accuracy_crosscheck.csv",
    [
        {
            **prov,
            "model": "yolox_s",
            "pipeline": "yolox_official_COCOEvaluator",
            "batch": 8,
            "n_images": len(ds),
            "coco_ap": round(float(ap50_95), 5),
            "ap50": round(float(ap50), 5),
        }
    ],
)
print(f"official-pipeline AP={ap50_95:.5f} AP50={ap50:.5f}")
