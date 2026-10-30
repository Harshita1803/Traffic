"""Tests for the Multi-Algorithm Comparative Benchmark Engine."""
import numpy as np
import pytest
from PIL import Image

from dip.benchmark import (
    AVAILABLE_BENCHMARK_OPERATIONS,
    run_algorithm_benchmark,
)
from traffic_dip.presentation import (
    benchmark_comparison_table,
    benchmark_csv_bytes,
)


def sample_benchmark_scene():
    y, x = np.mgrid[:64, :64]
    rgb = np.stack([
        50 + (x // 8) * 10,
        60 + (y // 8) * 8,
        70 + ((x + y) // 16) * 12
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def test_benchmark_runs_subset_of_operations():
    img = sample_benchmark_scene()
    candidates = [
        "Robust Normalization",
        "CLAHE (Adaptive Equalization)",
        "Luminance Unsharp Masking",
        "Otsu Automatic Binarization",
    ]
    bench_data = run_algorithm_benchmark(img, candidate_operations=candidates)

    assert bench_data["original_image"].size == img.size
    results = bench_data["benchmark_results"]
    # 1 baseline + 4 candidates
    assert len(results) == 5
    assert results[0]["name"] == "Original (Baseline)"
    assert results[0]["processing_ms"] == 0.0

    for item in results:
        assert item["image"].size == img.size
        assert "contrast_stddev" in item["metrics"]
        assert 0.0 <= item["changed_pixel_fraction"] <= 1.0


def test_benchmark_table_and_csv_formatting():
    img = sample_benchmark_scene()
    bench_data = run_algorithm_benchmark(img, candidate_operations=["Robust Normalization", "CLAHE (Adaptive Equalization)"])
    results = bench_data["benchmark_results"]

    table = benchmark_comparison_table(results)
    assert len(table) == len(results)
    assert all("Method" in row and "Time (ms)" in row and "Contrast σ" in row for row in table)

    csv_data = benchmark_csv_bytes(results)
    assert isinstance(csv_data, (bytes, bytearray))
    csv_str = csv_data.decode("utf-8-sig")
    assert "Robust Normalization" in csv_str
    assert "CLAHE (Adaptive Equalization)" in csv_str
    assert "contrast_stddev" in csv_str


def test_benchmark_custom_params():
    img = sample_benchmark_scene()
    custom = {
        "CLAHE (Adaptive Equalization)": {"clip_limit": 4.5, "tile_size": 16},
    }
    bench_data = run_algorithm_benchmark(img, candidate_operations=["CLAHE (Adaptive Equalization)"], custom_params=custom)
    clahe_res = next(r for r in bench_data["benchmark_results"] if r["name"] == "CLAHE (Adaptive Equalization)")
    assert "clip=4.5" in clahe_res["description"]
