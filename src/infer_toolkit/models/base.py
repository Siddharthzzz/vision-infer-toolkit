from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def add_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


class ModelAdapter:
    name: str
    task: str  # "det" | "seg"
    input_hw: tuple[int, int]

    def __init__(self, repo_root: Path | None = None):
        self.root = Path(repo_root) if repo_root else REPO_ROOT

    def build(self, device: str = "cuda"):
        raise NotImplementedError

    def preprocess(self, path: str) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def postprocess(self, outputs, metas: list[dict[str, Any]]) -> list[Any]:
        raise NotImplementedError
