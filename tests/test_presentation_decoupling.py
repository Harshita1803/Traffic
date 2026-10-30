"""Tests confirming presentation module operates independently of Streamlit."""
import numpy as np
import pytest
from PIL import Image

from traffic_dip.presentation import (
    TECHNIQUE_GROUPS,
    conceptual_stage_cards,
    photo_stages,
    rgb_histograms_32,
    stage_impact_data,
    stage_metric_rows,
    stage_output_caption,
    status_text,
    technique_group_status,
)


def sample_stage_record():
    im1 = Image.new("RGB", (64, 64), (100, 100, 100))
    im2 = Image.new("RGB", (64, 64), (120, 110, 105))
    metrics1 = {"mean_brightness": 100.0, "contrast_stddev": 15.0, "local_contrast_stddev": 12.0,
                "sharpness_laplacian_variance": 40.0, "noise_estimate": 4.5}
    metrics2 = {"mean_brightness": 115.0, "contrast_stddev": 22.0, "local_contrast_stddev": 18.0,
                "sharpness_laplacian_variance": 55.0, "noise_estimate": 4.2}
    return {
        "index": 1,
        "name": "CLAHE on luminance",
        "algorithm_id": "clahe_on_luminance",
        "explanation": "Local contrast equalization",
        "reason": "Low local contrast",
        "input_image": im1,
        "image": im2,
        "output_image": im2,
        "before": metrics1,
        "after": metrics2,
        "changed_pixel_fraction": 1.0,
        "changed_pixel_count": 64 * 64,
        "parameters": {"tile_size": 64, "clip_limit": 2.0},
        "processing_ms": 12.4,
        "output_kind": "photograph",
    }


def test_photo_stages_filters_diagnostics():
    stg = sample_stage_record()
    diag = dict(stg, output_kind="diagnostic")
    result = {"stages": [stg, diag]}
    photos = photo_stages(result)
    assert len(photos) == 1
    assert photos[0]["name"] == "CLAHE on luminance"


def test_stage_output_caption():
    stg = sample_stage_record()
    caption = stage_output_caption(stg)
    assert "clahe_on_luminance" in caption
    assert "100.0% of pixels changed" in caption


def test_stage_metric_rows():
    stg = sample_stage_record()
    rows = stage_metric_rows(stg)
    assert len(rows) == 5
    brightness_row = next(r for r in rows if r["Measurement"] == "Mean luminance")
    assert "100.000" in brightness_row["Before"]
    assert "115.000" in brightness_row["After"]
    assert "+15.000" in brightness_row["Absolute Δ"]


def test_rgb_histograms_32():
    im = Image.new("RGB", (32, 32), (50, 100, 150))
    hists = rgb_histograms_32(im)
    assert set(hists.keys()) == {"Red", "Green", "Blue", "Luminance"}
    assert sum(hists["Red"]) == 32 * 32
    assert sum(hists["Green"]) == 32 * 32
    assert sum(hists["Blue"]) == 32 * 32


def test_technique_group_status():
    result = {
        "plan": {"actions": [{"operation": "CLAHE on luminance", "status": "planned"}]},
        "stages": [{"name": "CLAHE on luminance", "output_kind": "photograph"}]
    }
    status, status_class = technique_group_status(result, ("Global histogram equalization", "CLAHE on luminance"))
    assert status == "Executed"
    assert status_class == ""
