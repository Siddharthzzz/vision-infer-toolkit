"""RT-DETR-R18 (lyuwenyu/RT-DETR, Apache-2.0): v1 config + v1 COCO weights, loaded through the
same repo's rtdetrv2_pytorch code (the v1 rtdetr_pytorch code needs torchvision<=0.15).
Mirrors configs/rtdetr/include/dataloader.yml (val) and src/zoo/rtdetr/rtdetr_postprocessor.py.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import ModelAdapter, add_path

CONFIG = "third_party/RT-DETR/rtdetrv2_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml"


class RtDetrR18(ModelAdapter):
    name = "rtdetr_r18"
    task = "det"
    input_hw = (640, 640)
    weights = "weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"

    def _cfg(self):
        add_path(self.root / "third_party/RT-DETR/rtdetrv2_pytorch")
        from src.core import YAMLConfig

        cfg = YAMLConfig(str(self.root / CONFIG))
        cfg.yaml_cfg["PResNet"]["pretrained"] = False  # full COCO weights are loaded below
        return cfg

    def build(self, device: str = "cuda"):
        import torch

        cfg = self._cfg()
        m = cfg.model
        ck = torch.load(self.root / self.weights, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["ema"]["module"], strict=True)
        m = m.deploy()

        class RawOut(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                o = self.inner(x)
                return o["pred_logits"], o["pred_boxes"]

        self._post = cfg.postprocessor  # remap_mscoco_category=True, top-300, sigmoid scores
        return RawOut(m).eval().to(device)

    def preprocess(self, path: str):
        import torch
        from torchvision.transforms import v2 as T

        img = Image.open(path).convert("RGB")
        w, h = img.size
        x = T.Resize(list(self.input_hw))(img)  # {type: Resize, size: [640, 640]}
        x = T.functional.pil_to_tensor(x).to(torch.float32) / 255.0  # ConvertPILImage scale
        return x.numpy(), {"orig_hw": (h, w)}

    def postprocess(self, outputs, metas):
        import torch

        if not hasattr(self, "_post"):
            self._post = self._cfg().postprocessor
        logits, boxes = (o.float() for o in outputs)
        sizes = torch.tensor(
            [[m["orig_hw"][1], m["orig_hw"][0]] for m in metas], device=logits.device
        )  # (w, h), as the repo's orig_target_sizes
        res = self._post({"pred_logits": logits, "pred_boxes": boxes}, sizes)
        return [
            {
                "boxes_xyxy": r["boxes"].cpu().numpy(),
                "scores": r["scores"].cpu().numpy(),
                "category_ids": r["labels"].cpu().numpy().astype(np.int64),  # already COCO ids
            }
            for r in res
        ]
