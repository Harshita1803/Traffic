
"""Research evaluation and benchmarking module for adverse-weather traffic surveillance.

Supports batch evaluation across weather categories (fog, rain, snow, sandstorm),
per-image and per-category statistical aggregations (mean, median, std dev, min, max),
processing latency, throughput (megapixels/sec), paired PSNR/SSIM evaluation when
references are present, and reproducible run manifest generation.
"""

import csv
from io import StringIO
import json
import math
import platform
import statistics
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from analysis.quality import calculate_quality, evaluate_paired_reference
from dip.adaptive import build_pipeline
from .batch import decode_image


def create_run_manifest(
    dataset_name="DAWN",
    pipeline_mode="Adaptive",
    algorithm_settings=None,
    notes=None,
):
    """Generate a reproducible experiment run manifest."""
    import PIL
    import streamlit

    packages = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "streamlit": streamlit.__version__,
    }

    try:
        import scipy
        packages["scipy"] = scipy.__version__
    except ImportError:
        pass

    try:
        import torch
        packages["torch"] = torch.__version__
    except ImportError:
        pass

    return {
        "run_id": str(uuid.uuid4()),
        "created_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "dataset": {
            "name": dataset_name,
            "description": "Adverse-weather traffic surveillance evaluation",
        },
        "pipeline_configuration": {
            "mode": pipeline_mode,
            "settings": algorithm_settings or {},
        },
        "notes": notes or "Automated research evaluation run.",
    }


def _decode_image_input(file_data, max_dim=None):
    """Decode image bytes or a filesystem path into an RGB PIL image."""

    if isinstance(file_data, (str, Path)):
        path = Path(file_data)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image file does not exist: {path}"
            )

        file_data = path.read_bytes()

    if not isinstance(file_data, (bytes, bytearray)) or not file_data:
        raise ValueError("The image input is empty or invalid.")

    image = decode_image(file_data, max_dim=max_dim).convert("RGB")
    return image


def evaluate_single_image(
    file_data,
    filename,
    category="unspecified",
    reference_data=None,
    pipeline_kwargs=None,
    reference_image=None,
    pipeline_mode="Adaptive",
    max_dim=None,
):
    """Evaluate one image using the existing adaptive DIP pipeline.

    Supports filesystem paths, uploaded image bytes, optional resizing,
    reference-image evaluation, processing-time measurement, and
    image-quality metric comparisons.
    """

    pipeline_kwargs = dict(pipeline_kwargs or {})

    if str(pipeline_mode).strip().lower() != "adaptive":
        raise ValueError(
            "Unsupported pipeline_mode. This evaluation module currently "
            "supports the Adaptive pipeline."
        )

    if max_dim is not None:
        if isinstance(max_dim, bool) or not isinstance(max_dim, (int, float)):
            raise ValueError("max_dim must be a positive number or None.")

        if not math.isfinite(float(max_dim)) or max_dim <= 0:
            raise ValueError("max_dim must be greater than zero.")

        max_dim = int(max_dim)

        if max_dim <= 0:
            raise ValueError("max_dim must be at least 1.")

    start_time = time.perf_counter()

    image = _decode_image_input(file_data, max_dim=max_dim)

    # Resize proportionally if the image exceeds the requested maximum.
    if max_dim is not None and max(image.size) > max_dim:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_dim, max_dim), resampling)

    w, h = image.size
    megapixels = (w * h) / 1_000_000.0

    orig_metrics = calculate_quality(image)

    result = build_pipeline(image, **pipeline_kwargs)

    final_img = result["final"]
    final_metrics = calculate_quality(final_img)

    total_elapsed_ms = result.get(
        "total_processing_ms",
        (time.perf_counter() - start_time) * 1000.0,
    )

    throughput = (
        megapixels / (total_elapsed_ms / 1000.0)
        if total_elapsed_ms > 0
        else 0.0
    )

    # Support both reference_image and the older reference_data argument.
    ref_input = (
        reference_image
        if reference_image is not None
        else reference_data
    )

    if ref_input is not None:
        try:
            ref_img = _decode_image_input(ref_input)

            if ref_img.size != final_img.size:
                ref_img = ref_img.resize(
                    final_img.size,
                    getattr(Image, "Resampling", Image).LANCZOS,
                )

            ref_eval = evaluate_paired_reference(final_img, ref_img)

        except Exception as exc:
            ref_eval = {
                "psnr": None,
                "ssim": None,
                "reference_available": False,
                "status": f"reference_error: {exc}",
            }
    else:
        ref_eval = evaluate_paired_reference(final_img, None)

    # Calculate changed pixels.
    original_array = np.asarray(
        image.convert("RGB"), dtype=np.uint8
    )
    final_array = np.asarray(
        final_img.convert("RGB"), dtype=np.uint8
    )

    if original_array.shape != final_array.shape:
        raise ValueError(
            "The pipeline output dimensions do not match the input dimensions."
        )

    diff = np.any(original_array != final_array, axis=2)

    changed_count = int(diff.sum())
    changed_fraction = float(diff.mean())

    # Calculate metric differences.
    deltas = {
        "delta_brightness": (
            final_metrics["mean_brightness"]
            - orig_metrics["mean_brightness"]
        ),
        "delta_contrast": (
            final_metrics["contrast_stddev"]
            - orig_metrics["contrast_stddev"]
        ),
        "delta_local_contrast": (
            final_metrics["local_contrast_stddev"]
            - orig_metrics["local_contrast_stddev"]
        ),
        "delta_sharpness": (
            final_metrics["sharpness_laplacian_variance"]
            - orig_metrics["sharpness_laplacian_variance"]
        ),
        "delta_noise_proxy": (
            final_metrics["noise_estimate"]
            - orig_metrics["noise_estimate"]
        ),
        "delta_dynamic_range": (
            final_metrics["robust_dynamic_range"]
            - orig_metrics["robust_dynamic_range"]
        ),
    }

    return {
        "filename": filename,
        "category": category,
        "width": w,
        "height": h,
        "megapixels": megapixels,
        "processing_ms": total_elapsed_ms,
        "throughput_mp_per_sec": throughput,
        "stage_count": len(result.get("stages", [])),
        "executed_stages": [
            stage["name"]
            for stage in result.get("stages", [])
        ],
        "original_metrics": orig_metrics,
        "final_metrics": final_metrics,
        "deltas": deltas,
        "changed_pixel_count": changed_count,
        "changed_pixel_fraction": changed_fraction,
        "reference_metrics": ref_eval,
        "status": "success",
    }


def _metric_distribution_stats(values):
    """Compute mean, median, standard deviation, min, and max."""

    clean = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]

    if not clean:
        return {
            "mean": None,
            "median": None,
            "std_dev": None,
            "min": None,
            "max": None,
            "count": 0,
        }

    return {
        "mean": float(statistics.mean(clean)),
        "median": float(statistics.median(clean)),
        "std_dev": (
            float(statistics.stdev(clean))
            if len(clean) > 1
            else 0.0
        ),
        "min": float(min(clean)),
        "max": float(max(clean)),
        "count": len(clean),
    }


def aggregate_category_statistics(eval_records):
    """Group evaluation records by weather category."""

    categories = sorted({
        record.get("category", "unspecified")
        for record in eval_records
    })

    aggregates = {}

    def extract_stats(subset):
        latencies = [
            record.get("processing_ms")
            for record in subset
        ]

        throughputs = [
            record.get("throughput_mp_per_sec")
            for record in subset
        ]

        contrast_deltas = [
            record.get("deltas", {}).get("delta_contrast")
            for record in subset
        ]

        sharpness_deltas = [
            record.get("deltas", {}).get("delta_sharpness")
            for record in subset
        ]

        brightness_deltas = [
            record.get("deltas", {}).get("delta_brightness")
            for record in subset
        ]

        noise_deltas = [
            record.get("deltas", {}).get("delta_noise_proxy")
            for record in subset
        ]

        final_contrasts = [
            record.get("final_metrics", {}).get("contrast_stddev")
            for record in subset
        ]

        stage_counts = [
            record.get("stage_count")
            for record in subset
        ]

        changed_fractions = [
            record.get("changed_pixel_fraction")
            for record in subset
        ]

        return {
            "image_count": len(subset),
            "successful_count": sum(
                record.get("status", "success") == "success"
                for record in subset
            ),
            "failure_count": sum(
                record.get("status", "success") != "success"
                for record in subset
            ),
            "latency_ms": _metric_distribution_stats(latencies),
            "throughput_mp_per_sec": _metric_distribution_stats(throughputs),
            "delta_contrast": _metric_distribution_stats(contrast_deltas),
            "delta_sharpness": _metric_distribution_stats(sharpness_deltas),
            "delta_brightness": _metric_distribution_stats(brightness_deltas),
            "delta_noise_proxy": _metric_distribution_stats(noise_deltas),
            "final_contrast": _metric_distribution_stats(final_contrasts),
            "stage_count": _metric_distribution_stats(stage_counts),
            "changed_pixel_fraction": _metric_distribution_stats(
                changed_fractions
            ),
        }

    for category in categories:
        subset = [
            record
            for record in eval_records
            if record.get("category", "unspecified") == category
        ]

        aggregates[category] = extract_stats(subset)

    # Do not add an extra "overall" category.
    return aggregates


def research_report_json_bytes(
    manifest,
    category_aggregates,
    per_image_records,
):
    """Serialize the complete research evaluation report to JSON."""

    payload = {
        "manifest": manifest,
        "category_aggregates": category_aggregates,
        "per_image_records": per_image_records,
        "methodology": {
            "luminance_space": (
                "YCbCr luminance channel (Y) processing with chroma preservation"
            ),
            "contrast_measurement": (
                "Global standard deviation and 64x64 tile local standard deviation"
            ),
            "sharpness_measurement": (
                "Four-neighbour discrete Laplacian variance"
            ),
            "noise_proxy_note": (
                "Median-residual MAD proxy; sensitive to fine road texture and edges"
            ),
            "reference_metrics_note": (
                "PSNR and SSIM are reported only when aligned clear references are provided"
            ),
        },
    }

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")


def category_summary_csv_bytes(category_aggregates):
    """Serialize category aggregate distributions to CSV bytes."""

    stream = StringIO(newline="")

    fieldnames = [
        "category",
        "image_count",
        "success_count",
        "failure_count",
        "mean_latency_ms",
        "median_latency_ms",
        "std_latency_ms",
        "mean_throughput_mp_s",
        "mean_delta_contrast",
        "median_delta_contrast",
        "std_delta_contrast",
        "mean_delta_sharpness",
        "median_delta_sharpness",
        "mean_delta_brightness",
        "mean_delta_noise_proxy",
        "mean_stage_count",
        "mean_changed_pixel_fraction",
    ]

    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()

    def formatted(stats, key, format_string):
        value = stats.get(key, {}).get("mean")
        return format_string.format(value) if value is not None else ""

    for category, stats in category_aggregates.items():
        latency = stats.get("latency_ms", {})
        contrast = stats.get("delta_contrast", {})
        sharpness = stats.get("delta_sharpness", {})

        writer.writerow({
            "category": category,
            "image_count": stats.get("image_count", 0),
            "success_count": stats.get("successful_count", 0),
            "failure_count": stats.get("failure_count", 0),
            "mean_latency_ms": formatted(stats, "latency_ms", "{:.2f}"),
            "median_latency_ms": (
                f"{latency['median']:.2f}"
                if latency.get("median") is not None else ""
            ),
            "std_latency_ms": (
                f"{latency['std_dev']:.2f}"
                if latency.get("std_dev") is not None else ""
            ),
            "mean_throughput_mp_s": formatted(
                stats, "throughput_mp_per_sec", "{:.2f}"
            ),
            "mean_delta_contrast": formatted(
                stats, "delta_contrast", "{:+.2f}"
            ),
            "median_delta_contrast": (
                f"{contrast['median']:+.2f}"
                if contrast.get("median") is not None else ""
            ),
            "std_delta_contrast": (
                f"{contrast['std_dev']:.2f}"
                if contrast.get("std_dev") is not None else ""
            ),
            "mean_delta_sharpness": formatted(
                stats, "delta_sharpness", "{:+.1f}"
            ),
            "median_delta_sharpness": (
                f"{sharpness['median']:+.1f}"
                if sharpness.get("median") is not None else ""
            ),
            "mean_delta_brightness": formatted(
                stats, "delta_brightness", "{:+.2f}"
            ),
            "mean_delta_noise_proxy": formatted(
                stats, "delta_noise_proxy", "{:+.2f}"
            ),
            "mean_stage_count": formatted(
                stats, "stage_count", "{:.1f}"
            ),
            "mean_changed_pixel_fraction": (
                f"{stats['changed_pixel_fraction']['mean']:.1%}"
                if stats.get("changed_pixel_fraction", {}).get("mean") is not None
                else ""
            ),
        })

    return stream.getvalue().encode("utf-8-sig")


def per_image_results_csv_bytes(per_image_records):
    """Serialize per-image evaluation results to CSV bytes.

    Missing optional fields in partial/test records are exported as blank
    or safe default values instead of raising KeyError.
    """

    stream = StringIO(newline="")

    fieldnames = [
        "filename",
        "category",
        "width",
        "height",
        "megapixels",
        "processing_ms",
        "throughput_mp_per_sec",
        "stage_count",
        "executed_stages",
        "orig_brightness",
        "final_brightness",
        "delta_brightness",
        "orig_contrast",
        "final_contrast",
        "delta_contrast",
        "orig_sharpness",
        "final_sharpness",
        "delta_sharpness",
        "orig_noise_proxy",
        "final_noise_proxy",
        "delta_noise_proxy",
        "changed_pixel_fraction",
        "psnr",
        "ssim",
        "reference_status",
    ]

    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()

    def number(value, format_string):
        if value is None:
            return ""

        try:
            return format_string.format(float(value))
        except (TypeError, ValueError, OverflowError):
            return ""

    for record in per_image_records:
        original = record.get("original_metrics", {}) or {}
        final = record.get("final_metrics", {}) or {}
        deltas = record.get("deltas", {}) or {}
        reference = record.get("reference_metrics", {}) or {}

        executed_stages = record.get("executed_stages", []) or []

        writer.writerow({
            "filename": record.get("filename", ""),
            "category": record.get("category", "unspecified"),
            "width": record.get("width", ""),
            "height": record.get("height", ""),
            "megapixels": number(record.get("megapixels"), "{:.2f}"),
            "processing_ms": number(
                record.get("processing_ms"), "{:.2f}"
            ),
            "throughput_mp_per_sec": number(
                record.get("throughput_mp_per_sec"), "{:.2f}"
            ),
            "stage_count": record.get("stage_count", ""),
            "executed_stages": " -> ".join(
                str(stage) for stage in executed_stages
            ),
            "orig_brightness": number(
                original.get("mean_brightness"), "{:.2f}"
            ),
            "final_brightness": number(
                final.get("mean_brightness"), "{:.2f}"
            ),
            "delta_brightness": number(
                deltas.get("delta_brightness"), "{:+.2f}"
            ),
            "orig_contrast": number(
                original.get("contrast_stddev"), "{:.2f}"
            ),
            "final_contrast": number(
                final.get("contrast_stddev"), "{:.2f}"
            ),
            "delta_contrast": number(
                deltas.get("delta_contrast"), "{:+.2f}"
            ),
            "orig_sharpness": number(
                original.get("sharpness_laplacian_variance"), "{:.1f}"
            ),
            "final_sharpness": number(
                final.get("sharpness_laplacian_variance"), "{:.1f}"
            ),
            "delta_sharpness": number(
                deltas.get("delta_sharpness"), "{:+.1f}"
            ),
            "orig_noise_proxy": number(
                original.get("noise_estimate"), "{:.2f}"
            ),
            "final_noise_proxy": number(
                final.get("noise_estimate"), "{:.2f}"
            ),
            "delta_noise_proxy": number(
                deltas.get("delta_noise_proxy"), "{:+.2f}"
            ),
            "changed_pixel_fraction": number(
                record.get("changed_pixel_fraction"), "{:.2%}"
            ),
            "psnr": (
                number(reference.get("psnr"), "{:.2f}")
                if reference.get("psnr") is not None
                else "unavailable"
            ),
            "ssim": (
                number(reference.get("ssim"), "{:.4f}")
                if reference.get("ssim") is not None
                else "unavailable"
            ),
            "reference_status": reference.get("status", "unpaired"),
        })

    return stream.getvalue().encode("utf-8-sig")
