"""Tests for detector availability check and VOC ground truth bounding box rendering."""
import pytest
from PIL import Image

from traffic_dip.detection import is_detector_available
from traffic_dip.visibility import draw_ground_truth_boxes


def test_is_detector_available_returns_bool_and_string():
    available, message = is_detector_available()
    assert isinstance(available, bool)
    assert isinstance(message, str)
    assert len(message) > 0


def test_draw_ground_truth_boxes_with_empty_annotation():
    img = Image.new("RGB", (100, 100), (40, 50, 60))
    res = draw_ground_truth_boxes(img, None)
    assert res.size == (100, 100)
    res_empty = draw_ground_truth_boxes(img, {"annotations": []})
    assert res_empty.size == (100, 100)


def test_draw_ground_truth_boxes_draws_valid_boxes():
    img = Image.new("RGB", (200, 200), (30, 30, 30))
    ann = {
        "width": 200,
        "height": 200,
        "filename": "test_traffic.jpg",
        "annotations": [
            {"class_name": "car", "bbox_xyxy": [20.0, 30.0, 80.0, 90.0], "difficult": False},
            {"class_name": "truck", "bbox_xyxy": [100.0, 110.0, 180.0, 170.0], "difficult": False},
            {"class_name": "dog", "bbox_xyxy": [5.0, 5.0, 15.0, 15.0]},  # non-traffic class should be ignored
        ]
    }
    annotated = draw_ground_truth_boxes(img, ann)
    assert annotated.size == (200, 200)
    assert annotated.mode == "RGB"
    # Pixels should be modified by the bounding boxes
    assert annotated.tobytes() != img.tobytes()
