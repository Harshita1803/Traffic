"""Tests for image resizing preserving aspect ratio and decode validations."""
from io import BytesIO
import numpy as np
import pytest
from PIL import Image

from dip.spatial import resize_preserving_aspect_ratio
from traffic_dip.batch import decode_image


def test_resize_preserving_aspect_ratio_max_dim():
    # 1600x900 image (16:9 ratio)
    img = Image.new("RGB", (1600, 900), color=(100, 150, 200))
    img.info["custom_tag"] = "test_meta"

    resized = resize_preserving_aspect_ratio(img, max_dim=800)
    assert resized.size == (800, 450)
    assert resized.info.get("custom_tag") == "test_meta"


def test_resize_preserving_aspect_ratio_smaller_image_unchanged():
    img = Image.new("RGB", (640, 480))
    resized = resize_preserving_aspect_ratio(img, max_dim=1000)
    assert resized.size == (640, 480)


def test_resize_preserving_aspect_ratio_target_size():
    img = Image.new("RGB", (1200, 800))  # 3:2
    resized = resize_preserving_aspect_ratio(img, target_size=(600, 600))
    # Fitting inside 600x600 should yield 600x400
    assert resized.size == (600, 400)


def test_resize_validates_inputs():
    img = Image.new("RGB", (100, 100))
    with pytest.raises(TypeError):
        resize_preserving_aspect_ratio("not_an_image")
    with pytest.raises(ValueError, match="positive integer"):
        resize_preserving_aspect_ratio(img, max_dim=0)
    with pytest.raises(ValueError, match="positive integer"):
        resize_preserving_aspect_ratio(img, max_dim=-50)
    with pytest.raises(ValueError, match="target_size dimensions must be positive"):
        resize_preserving_aspect_ratio(img, target_size=(0, 100))


def test_decode_image_supports_modes_and_max_dim():
    # Test RGBA image
    rgba = Image.new("RGBA", (800, 600), (50, 100, 150, 200))
    stream = BytesIO()
    rgba.save(stream, format="PNG")
    decoded = decode_image(stream.getvalue(), max_dim=400)
    assert decoded.mode == "RGB"
    assert decoded.size == (400, 300)
    assert decoded.info.get("source_mode") == "RGBA"

    # Test Grayscale image
    gray = Image.new("L", (500, 500), 128)
    stream = BytesIO()
    gray.save(stream, format="PNG")
    decoded_gray = decode_image(stream.getvalue())
    assert decoded_gray.mode == "RGB"
    assert decoded_gray.size == (500, 500)
    assert decoded_gray.info.get("source_mode") == "L"


def test_decode_image_rejects_empty_and_corrupt():
    with pytest.raises(ValueError, match="empty"):
        decode_image(b"")
    with pytest.raises(ValueError, match="Unsupported or invalid image"):
        decode_image(b"this is completely corrupted not an image data at all")
