"""SegFormer-B0 fine-tuned on ADE20K (nvidia/segformer-b0-finetuned-ade-512-512).
Licence: NVIDIA Source Code License for SegFormer (non-commercial). Pre-processing is the
model's own Hugging Face image processor (512x512 resize, /255, ImageNet mean/std).
Eval: logits bilinearly upsampled to the label resolution, then argmax (methodology §6).
"""

from __future__ import annotations

from PIL import Image

from .base import ModelAdapter

HF_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
HF_REV = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"


class SegformerB0(ModelAdapter):
    name = "segformer_b0"
    task = "seg"
    input_hw = (512, 512)

    def _processor(self):
        if not hasattr(self, "_proc"):
            from transformers import SegformerImageProcessor

            self._proc = SegformerImageProcessor.from_pretrained(
                HF_ID, revision=HF_REV, cache_dir=str(self.root / "weights/hf")
            )
        return self._proc

    def build(self, device: str = "cuda"):
        import torch
        from transformers import SegformerForSemanticSegmentation

        hf = SegformerForSemanticSegmentation.from_pretrained(
            HF_ID, revision=HF_REV, cache_dir=str(self.root / "weights/hf"), use_safetensors=True
        )

        class Logits(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(pixel_values=x).logits  # (B, 150, H/4, W/4)

        return Logits(hf).eval().to(device)

    def preprocess(self, path: str):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        x = self._processor()(images=img, return_tensors="pt")["pixel_values"][0]
        return x.numpy(), {"orig_hw": (h, w)}

    def postprocess(self, outputs, metas):
        import torch.nn.functional as F

        logits = outputs.float()
        out = []
        for b, m in enumerate(metas):
            up = F.interpolate(logits[b : b + 1], size=m["orig_hw"], mode="bilinear")
            out.append(up.argmax(dim=1)[0].to("cpu").numpy().astype("uint8"))  # 0..149
        return out
