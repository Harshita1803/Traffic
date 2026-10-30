import pathlib
import pytest

from traffic_dip.research_evaluation import (
    aggregate_category_statistics,
    create_run_manifest,
    evaluate_single_image,
    per_image_results_csv_bytes,
    research_report_json_bytes,
)

# Helper to get a sample image path from the bundled DAWN dataset
def get_sample_image_path(category="fog"):
    samples_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "samples" / category
    for p in sorted(samples_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            return p
    raise FileNotFoundError(f"No sample image found in {samples_dir}")

def test_create_run_manifest_structure():
    manifest = create_run_manifest(
        dataset_name="DAWN Test",
        pipeline_mode="Adaptive",
        algorithm_settings={"test": True},
        notes="unit test",
    )
    assert isinstance(manifest, dict)
    for key in ["run_id", "created_at_utc", "platform", "packages", "dataset", "pipeline_configuration"]:
        assert key in manifest

def test_evaluate_single_image_basic():
    img_path = get_sample_image_path()
    result = evaluate_single_image(
        img_path,
        filename=img_path.name,
        category="fog",
        reference_image=None,
        pipeline_mode="Adaptive",
        max_dim=800,
    )
    # Basic sanity checks on result structure
    assert isinstance(result, dict)
    for key in ["original_metrics", "final_metrics", "deltas", "processing_ms", "throughput_mp_per_sec"]:
        assert key in result
    # Ensure metrics contain expected fields
    for metric_dict in (result["original_metrics"], result["final_metrics"]):
        for metric_key in ["mean_brightness", "contrast_stddev", "sharpness_laplacian_variance", "noise_estimate"]:
            assert metric_key in metric_dict

def test_aggregate_category_statistics_and_exports():
    # Create two dummy per-image records with minimal required fields
    rec1 = {
        "category": "fog",
        "processing_ms": 100.0,
        "throughput_mp_per_sec": 2.0,
        "stage_count": 5,
        "original_metrics": {"mean_brightness": 0.1, "contrast_stddev": 0.2, "sharpness_laplacian_variance": 0.3, "noise_estimate": 0.4},
        "final_metrics": {"mean_brightness": 0.15, "contrast_stddev": 0.25, "sharpness_laplacian_variance": 0.35, "noise_estimate": 0.45},
        "deltas": {"delta_brightness": 0.05, "delta_contrast": 0.05, "delta_sharpness": 0.05, "delta_noise_proxy": 0.05},
    }
    rec2 = rec1.copy()
    rec2["category"] = "rain"
    rec2["processing_ms"] = 150.0
    rec2["throughput_mp_per_sec"] = 1.5
    # Insert a changed pixel fraction for completeness
    rec1["changed_pixel_fraction"] = 0.01
    rec2["changed_pixel_fraction"] = 0.02

    summaries = aggregate_category_statistics([rec1, rec2])
    assert isinstance(summaries, dict)
    assert set(summaries.keys()) == {"fog", "rain"}
    # Verify CSV export functions return bytes
    csv_bytes = per_image_results_csv_bytes([rec1, rec2])
    assert isinstance(csv_bytes, (bytes, bytearray))
    json_bytes = research_report_json_bytes(create_run_manifest("DAWN", "Adaptive"), [rec1, rec2], summaries)
    assert isinstance(json_bytes, (bytes, bytearray))
