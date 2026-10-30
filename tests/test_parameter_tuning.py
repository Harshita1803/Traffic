"""Tests for custom parameter overrides in the adaptive and fixed pipelines."""
import numpy as np
import pytest
from PIL import Image

from dip.adaptive import build_pipeline


def test_custom_parameters_override_defaults():
    # Create dark image that triggers gamma correction and CLAHE
    arr = np.tile(np.arange(32, dtype=np.uint8), (24, 1)) // 3
    img = Image.fromarray(arr)

    custom = {
        "CLAHE on luminance": {"clip_limit": 3.75, "tile_size": 32},
        "Robust luminance normalization": {"low_percentile": 2.0, "high_percentile": 98.0},
    }

    result = build_pipeline(img, custom_parameters=custom)
    clahe_stage = next((s for s in result["stages"] if s["name"] == "CLAHE on luminance"), None)
    if clahe_stage is not None:
        assert clahe_stage["parameters"]["clip_limit"] == 3.75
        assert clahe_stage["parameters"]["tile_size"] == 32


def test_custom_parameters_in_fixed_pipeline():
    y, x = np.mgrid[:32, :32]
    img = Image.fromarray(np.clip(50 + 4 * x + 3 * y, 0, 255).astype(np.uint8))
    custom = {
        "Luminance unsharp mask": {"radius": 2.5, "percent": 180, "threshold": 2},
    }
    result = build_pipeline(
        img,
        selected_operations=["Luminance unsharp mask"],
        custom_parameters=custom
    )
    assert len(result["stages"]) == 1
    stage = result["stages"][0]
    assert stage["parameters"]["radius"] == 2.5
    assert stage["parameters"]["percent"] == 180
    assert stage["parameters"]["threshold"] == 2
