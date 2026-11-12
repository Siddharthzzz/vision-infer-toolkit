"""CPU tests for accuracy metrics and results-file handling."""

import numpy as np
import pytest

from infer_toolkit.evaluate.metrics import MeanIoU, ade_reduce_labels, to_coco_dets
from infer_toolkit.runinfo import append_rows


def test_coco_dets_xyxy_to_xywh_and_yolox_label_mapping():
    r = {
        "boxes_xyxy": np.array([[10.0, 20.0, 30.0, 60.0]]),
        "scores": np.array([0.9]),
        "labels": np.array([2]),  # contiguous index
    }
    class_ids = [1, 2, 3, 4]  # COCO category ids are not contiguous in reality
    (d,) = to_coco_dets(r, image_id=42, class_ids=class_ids)
    assert d == {"image_id": 42, "category_id": 3, "bbox": [10.0, 20.0, 20.0, 40.0], "score": 0.9}


def test_coco_dets_with_category_ids_passthrough():
    r = {
        "boxes_xyxy": np.zeros((1, 4)),
        "scores": np.array([0.5]),
        "category_ids": np.array([90]),
    }
    assert to_coco_dets(r, 1, None)[0]["category_id"] == 90


def test_ade_reduce_labels():
    lab = np.array([[0, 1], [150, 7]], dtype=np.uint8)
    np.testing.assert_array_equal(ade_reduce_labels(lab), [[255, 0], [149, 6]])


def test_miou_accumulates_over_dataset_and_ignores_255():
    m = MeanIoU(num_classes=3)
    # image 1: class 0 perfect, class 1 predicted as 2 once
    m.update(np.array([0, 0, 2, 1]), np.array([0, 0, 1, 1]))
    # image 2: ignored pixel must not count
    m.update(np.array([2, 1]), np.array([2, 255]))
    r = m.compute()
    # class0: I=2 U=2 ->1 ; class1: I=1 U=2 ->0.5 ; class2: I=1 U=2 -> 0.5
    assert r["miou"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert r["aacc"] == pytest.approx(4 / 5)
    assert r["n_classes_present"] == 3


def test_miou_skips_classes_absent_from_both():
    m = MeanIoU(num_classes=4)
    m.update(np.array([0, 1]), np.array([0, 1]))
    assert m.compute()["miou"] == 1.0
    assert m.compute()["n_classes_present"] == 2


def test_miou_shape_mismatch():
    with pytest.raises(ValueError):
        MeanIoU(3).update(np.zeros((2, 2)), np.zeros((2, 3)))


def test_append_rows_refuses_column_change(tmp_path):
    p = tmp_path / "r.csv"
    append_rows(p, [{"a": 1, "b": 2}])
    append_rows(p, [{"a": 3, "b": 4}])
    assert p.read_text().splitlines() == ["a,b", "1,2", "3,4"]
    with pytest.raises(ValueError):
        append_rows(p, [{"a": 1, "c": 2}])
