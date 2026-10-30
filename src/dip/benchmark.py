"""Standardized Multi-Algorithm Comparative Benchmark Engine for Traffic DIP.

Executes candidate classical digital image processing algorithms against the
same input image, measures execution time with microsecond resolution, and
computes comparative image quality and visibility statistics.
"""
import time
import numpy as np
from PIL import Image

from analysis.quality import calculate_quality
from .dehaze import dark_channel_prior
from .frequency import gaussian_lowpass_luminance
from .spatial import (
    adaptive_threshold,
    as_rgb,
    bilateral_denoise,
    clahe_luminance,
    edge_enhance,
    equalize_luminance,
    gamma_correct,
    gaussian_denoise,
    median_denoise,
    normalize_luminance,
    otsu_threshold,
    unsharp_luminance,
)


AVAILABLE_BENCHMARK_OPERATIONS = [
    "Robust Normalization",
    "Global Histogram Equalization",
    "CLAHE (Adaptive Equalization)",
    "Dark-Channel Dehazing",
    "Gamma Correction (Low-Light Boost)",
    "Bilateral Denoising",
    "Median Denoising",
    "Gaussian Denoising",
    "FFT Low-Pass Filter",
    "Luminance Unsharp Masking",
    "Sobel Edge Enhancement",
    "Otsu Automatic Binarization",
    "Adaptive Local Thresholding",
]


def run_algorithm_benchmark(image, candidate_operations=None, custom_params=None):
    """Run candidate algorithms against the same image and return structured comparisons.

    Parameters
    ----------
    image : PIL.Image
        Input image.
    candidate_operations : list of str, optional
        Subset of operations from AVAILABLE_BENCHMARK_OPERATIONS. If None, runs
        a recommended suite of 8 core traffic enhancement methods.
    custom_params : dict, optional
        Per-operation custom parameter overrides.

    Returns
    -------
    dict
        {
            "original_image": PIL.Image,
            "original_metrics": dict,
            "benchmark_results": list of dict,
            "total_benchmark_ms": float,
        }
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image")
    rgb = as_rgb(image)
    orig_metrics = calculate_quality(rgb)
    custom_params = custom_params or {}

    default_suite = [
        "Robust Normalization",
        "Global Histogram Equalization",
        "CLAHE (Adaptive Equalization)",
        "Dark-Channel Dehazing",
        "Bilateral Denoising",
        "Median Denoising",
        "Luminance Unsharp Masking",
        "Sobel Edge Enhancement",
        "Otsu Automatic Binarization",
    ]
    ops = candidate_operations if candidate_operations is not None else default_suite

    results = []
    # Always include baseline
    results.append({
        "name": "Original (Baseline)",
        "category": "Baseline",
        "image": rgb.copy(),
        "processing_ms": 0.0,
        "metrics": orig_metrics,
        "delta_contrast": 0.0,
        "delta_sharpness": 0.0,
        "changed_pixel_fraction": 0.0,
        "description": "Unmodified original input.",
    })

    benchmark_start = time.perf_counter()

    for op in ops:
        params = custom_params.get(op, {})
        op_start = time.perf_counter()
        out_image = None
        category = "General"
        desc = ""

        try:
            if op == "Robust Normalization":
                category = "Illumination & Contrast"
                low_p = params.get("low_percentile", 1.0)
                high_p = params.get("high_percentile", 99.0)
                out_image = normalize_luminance(rgb, low_percentile=low_p, high_percentile=high_p)
                desc = f"Percentile stretch [{low_p}%, {high_p}%] on luminance."

            elif op == "Global Histogram Equalization":
                category = "Illumination & Contrast"
                out_image = equalize_luminance(rgb)
                desc = "Global redistribution of luminance histogram."

            elif op == "CLAHE (Adaptive Equalization)":
                category = "Illumination & Contrast"
                tile_size = params.get("tile_size", 64)
                clip_limit = params.get("clip_limit", 2.0)
                out_image = clahe_luminance(rgb, tile_size=tile_size, clip_limit=clip_limit)
                desc = f"Local tile contrast equalization (tile={tile_size}, clip={clip_limit})."

            elif op == "Dark-Channel Dehazing":
                category = "Atmospheric Restoration"
                window = params.get("window", 15)
                omega = params.get("omega", 0.70)
                floor = params.get("transmission_floor", 0.55)
                strength = params.get("strength", 0.75)
                out_image, _ = dark_channel_prior(rgb, window=window, omega=omega,
                                                  transmission_floor=floor, strength=strength)
                desc = f"Dark channel scattering inversion (omega={omega}, floor={floor})."

            elif op == "Gamma Correction (Low-Light Boost)":
                category = "Illumination & Contrast"
                gamma = params.get("gamma", 0.75)
                out_image = gamma_correct(rgb, gamma=gamma)
                desc = f"Power-law luminance mapping with gamma={gamma}."

            elif op == "Bilateral Denoising":
                category = "Noise Reduction"
                diameter = params.get("diameter", 5)
                sigma_c = params.get("sigma_color", 25.0)
                sigma_s = params.get("sigma_space", 2.0)
                out_image = bilateral_denoise(rgb, diameter=diameter, sigma_color=sigma_c, sigma_space=sigma_s)
                desc = f"Edge-preserving bilateral smoothing (d={diameter}, σ_color={sigma_c})."

            elif op == "Median Denoising":
                category = "Noise Reduction"
                size = params.get("kernel_size", 3)
                out_image = median_denoise(rgb, size=size)
                desc = f"{size}x{size} median filter for impulse noise reduction."

            elif op == "Gaussian Denoising":
                category = "Noise Reduction"
                radius = params.get("radius", 1.0)
                out_image = gaussian_denoise(rgb, radius=radius)
                desc = f"Gaussian blur smoothing with radius={radius}."

            elif op == "FFT Low-Pass Filter":
                category = "Frequency Domain"
                sigma = params.get("sigma", 0.8)
                out_image = gaussian_lowpass_luminance(rgb, sigma=sigma)
                desc = f"2D Fourier low-pass Gaussian filter (sigma={sigma})."

            elif op == "Luminance Unsharp Masking":
                category = "Detail & Sharpening"
                radius = params.get("radius", 1.5)
                percent = params.get("percent", 100)
                threshold = params.get("threshold", 4)
                out_image = unsharp_luminance(rgb, radius=radius, percent=percent, threshold=threshold)
                desc = f"High-frequency detail emphasis (radius={radius}, percent={percent}%)."

            elif op == "Sobel Edge Enhancement":
                category = "Detail & Sharpening"
                strength = params.get("strength", 0.25)
                out_image = edge_enhance(rgb, strength=strength, operator="sobel")
                desc = f"Bounded Sobel gradient emphasis (strength={strength})."

            elif op == "Otsu Automatic Binarization":
                category = "Segmentation & Thresholding"
                thresh_val, mask = otsu_threshold(rgb)
                out_image = mask.convert("RGB")
                desc = f"Optimal inter-class variance threshold T={thresh_val}."

            elif op == "Adaptive Local Thresholding":
                category = "Segmentation & Thresholding"
                block_size = params.get("block_size", 31)
                c_val = params.get("c", 5.0)
                mask = adaptive_threshold(rgb, block_size=block_size, c=c_val)
                out_image = mask.convert("RGB")
                desc = f"Local adaptive mean thresholding (block={block_size}, C={c_val})."

            else:
                continue

            elapsed_ms = (time.perf_counter() - op_start) * 1000.0
            out_rgb = out_image.convert("RGB")
            after_metrics = calculate_quality(out_rgb)

            # Changed pixel fraction
            a = np.asarray(rgb, dtype=np.uint8)
            b = np.asarray(out_rgb, dtype=np.uint8)
            changed = float(np.any(a != b, axis=2).mean())

            results.append({
                "name": op,
                "category": category,
                "image": out_rgb,
                "processing_ms": elapsed_ms,
                "metrics": after_metrics,
                "delta_contrast": after_metrics["contrast_stddev"] - orig_metrics["contrast_stddev"],
                "delta_sharpness": after_metrics["sharpness_laplacian_variance"] - orig_metrics["sharpness_laplacian_variance"],
                "changed_pixel_fraction": changed,
                "description": desc,
            })
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - op_start) * 1000.0
            results.append({
                "name": op,
                "category": category,
                "image": rgb.copy(),
                "processing_ms": elapsed_ms,
                "metrics": orig_metrics,
                "delta_contrast": 0.0,
                "delta_sharpness": 0.0,
                "changed_pixel_fraction": 0.0,
                "error": str(exc),
                "description": f"Failed: {exc}",
            })

    total_benchmark_ms = (time.perf_counter() - benchmark_start) * 1000.0
    return {
        "original_image": rgb,
        "original_metrics": orig_metrics,
        "benchmark_results": results,
        "total_benchmark_ms": total_benchmark_ms,
    }
