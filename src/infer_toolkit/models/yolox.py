"""YOLOX-S (Megvii, Apache-2.0). Mirrors third_party/YOLOX eval:
yolox/data/data_augment.py::preproc, yolox/utils/boxes.py::postprocess,
yolox/evaluators/coco_evaluator.py::convert_to_coco_format, exps/default/yolox_s.py."""

from __future__ import annotations

import cv2
import numpy as np

from .base import ModelAdapter, add_path

DEPTH, WIDTH = 0.33, 0.50  # exps/default/yolox_s.py
TEST_CONF, NMS_THRE = 0.01, 0.65  # yolox/exp/yolox_base.py (eval defaults)
PAD_VALUE = 114
NUM_CLASSES = 80


class YoloxS(ModelAdapter):
    name = "yolox_s"
    task = "det"
    input_hw = (640, 640)
    weights = "weights/yolox_s.pth"

    def build(self, device: str = "cuda"):
        import torch

        add_path(self.root / "third_party/YOLOX")
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

        m = YOLOX(YOLOPAFPN(DEPTH, WIDTH, act="silu"), YOLOXHead(NUM_CLASSES, WIDTH, act="silu"))
        ck = torch.load(self.root / self.weights, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"], strict=True)
        m.head.decode_in_inference = True  # forward returns (B, N, 4+1+80) decoded cxcywh
        return m.eval().to(device)

    def preprocess(self, path: str):
        img = cv2.imread(path)  # BGR uint8, as in YOLOX's COCODataset
        h, w = img.shape[:2]
        H, W = self.input_hw
        r = min(H / h, W / w)
        nh, nw = int(h * r), int(w * r)
        out = np.full((H, W, 3), PAD_VALUE, dtype=np.uint8)
        out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        chw = np.ascontiguousarray(out.transpose(2, 0, 1), dtype=np.float32)  # 0..255, no norm
        return chw, {"orig_hw": (h, w), "ratio": r}

    def postprocess(self, outputs, metas):
        import torch
        import torchvision

        pred = outputs.float()
        boxes = torch.empty_like(pred[..., :4])
        boxes[..., 0] = pred[..., 0] - pred[..., 2] / 2
        boxes[..., 1] = pred[..., 1] - pred[..., 3] / 2
        boxes[..., 2] = pred[..., 0] + pred[..., 2] / 2
        boxes[..., 3] = pred[..., 1] + pred[..., 3] / 2
        results = []
        for b, meta in enumerate(metas):
            cls_conf, cls_id = pred[b, :, 5 : 5 + NUM_CLASSES].max(dim=1)
            score = pred[b, :, 4] * cls_conf
            keep = score >= TEST_CONF
            bx, sc, cl = boxes[b][keep], score[keep], cls_id[keep]
            idx = torchvision.ops.batched_nms(bx, sc, cl, NMS_THRE)
            results.append(
                {
                    "boxes_xyxy": (bx[idx] / meta["ratio"]).cpu().numpy(),
                    "scores": sc[idx].cpu().numpy(),
                    "labels": cl[idx].cpu().numpy(),  # contiguous 0..79
                }
            )
        return results
