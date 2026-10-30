"""Presentation, stage data aggregation, and reporting helpers for DIP workbenches."""
import csv
from io import StringIO
import json
import numpy as np
from PIL import Image
from analysis.quality import calculate_quality

TECHNIQUE_GROUPS = (
    ("Illumination correction", "Robust luminance normalization, Gamma correction",
     ("Robust luminance normalization", "Gamma correction"),
     "Adjusts luminance while retaining chroma where the selected method supports it."),
    ("Noise reduction", "Median, Gaussian, bilateral or FFT filtering",
     ("Median denoising", "Gaussian denoising", "Bilateral denoising", "FFT Gaussian low-pass", "Denoising"),
     "Selected from the noise proxy or explicit user settings; texture may also be smoothed."),
    ("Local contrast", "Global histogram equalization or CLAHE",
     ("Global histogram equalization", "CLAHE on luminance"),
     "Redistributes luminance; stronger contrast is not proof of better visibility."),
    ("Dehazing", "Dark-channel-prior atmospheric model",
     ("Dark-channel-prior dehazing",),
     "A classical heuristic selected only by measured evidence or explicit user choice."),
    ("Detail enhancement", "Unsharp masking or bounded Sobel emphasis",
     ("Luminance unsharp mask", "Sobel edge enhancement"),
     "Can emphasize edges, noise or compression artifacts; at most one is normally selected."),
    ("Structural detail analysis", "Sobel and morphology diagnostics",
     ("Sobel edge diagnostic", "Morphological diagnostic"),
     "Auxiliary outputs only; diagnostic images do not replace the traffic photograph."),
)


def photo_stages(result):
    """Return only executed photograph stages; diagnostics stay out of the main path."""
    return [stage for stage in result.get("stages", []) if stage.get("output_kind") != "diagnostic"]


def conceptual_stage_cards(original, stages):
    """Map N executed filters to N+1 cards with Stage 1 as the immutable source."""
    cards = [{"step": 1, "name": "Original", "image": original, "stage": None,
              "before_image": None, "status": "original"}]
    for number, stage in enumerate(stages, start=2):
        cards.append({"step": number, "name": stage["name"], "image": stage["image"],
                      "stage": stage, "before_image": stage["input_image"], "status": "completed"})
    return cards


def stage_output_caption(stage):
    """Format the canonical executed-stage identity and measured pixel delta."""
    return (f"Computed output of {stage['algorithm_id']} · "
            f"{stage['changed_pixel_fraction']:.1%} of pixels changed")


def status_text(result):
    """Summarize executed and failed actions for the current result."""
    failed = sum(a.get("status") == "failed" for a in result.get("plan", {}).get("actions", []))
    photo_count = sum(stage.get("output_kind") != "diagnostic" for stage in result.get("stages", []))
    return f"{photo_count} filter(s) executed" + (f" · {failed} failed" if failed else "")


def technique_group_status(result, operation_names):
    """Summarize only operations recorded by the real plan and stage result."""
    if result is None:
        return "Available", ""
    actions = [action for action in result.get("plan", {}).get("actions", [])
               if action.get("operation") in operation_names]
    stages = [stage for stage in result.get("stages", [])
              if stage.get("name") in operation_names]
    if any(stage.get("output_kind") == "diagnostic" for stage in stages):
        return "Diagnostic output", "diagnostic"
    if stages:
        return "Executed", ""
    if any(action.get("status") == "failed" for action in actions):
        return "Failed", "skipped"
    if actions and all(action.get("status") == "unavailable" for action in actions):
        return "Unavailable", "skipped"
    if actions:
        return "Skipped in this run", "skipped"
    return "Available", ""


def stage_metric_rows(stage):
    """Build compact comparisons from this operation's saved before/after metrics."""
    definitions = (
        ("Mean luminance", "mean_brightness", "luminance"),
        ("Global contrast · σ", "contrast_stddev", "luminance σ"),
        ("Local contrast · σ", "local_contrast_stddev", "luminance σ"),
        ("Sharpness · Laplacian variance", "sharpness_laplacian_variance", "variance"),
        ("Noise proxy · median-residual MAD", "noise_estimate", "proxy"),
    )
    rows = []
    for label, key, unit in definitions:
        before = stage.get("before", {}).get(key)
        after = stage.get("after", {}).get(key)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        delta = float(after) - float(before)
        relative = "n/a · zero baseline" if before == 0 else f"{delta / abs(float(before)):+.1%}"
        rows.append({"Measurement": label, "Before": f"{before:.3f} {unit}",
                     "After": f"{after:.3f} {unit}", "Absolute Δ": f"{delta:+.3f} {unit}",
                     "Relative change": relative})
    return rows


def stage_impact_data(stage):
    """Return panel data from exactly one stage's saved input/output pair."""
    before, after = stage["input_image"], stage["image"]
    if before.size != after.size:
        changed_count, changed_fraction = None, None
    else:
        a = np.asarray(before.convert("RGB"), dtype=np.uint8)
        b = np.asarray(after.convert("RGB"), dtype=np.uint8)
        changed_count = int(np.any(a != b, axis=2).sum())
        changed_fraction = changed_count / (before.width * before.height)
    return {
        "name": stage["name"], "purpose": stage.get("explanation", ""),
        "reason": stage.get("reason", ""), "before_image": before, "after_image": after,
        "before_metrics": stage.get("before", {}), "after_metrics": stage.get("after", {}),
        "metric_rows": stage_metric_rows(stage), "changed_pixel_count": changed_count,
        "changed_pixel_fraction": changed_fraction, "parameters": stage.get("parameters", {}),
        "processing_ms": stage.get("processing_ms"), "limitation": stage.get("limitation", "")
    }


def rgb_histograms_32(image):
    """Return 32-bin histograms for R, G, B channels and grayscale luminance."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    bins = 32
    r_hist, _ = np.histogram(rgb[:, :, 0], bins=bins, range=(0, 256))
    g_hist, _ = np.histogram(rgb[:, :, 1], bins=bins, range=(0, 256))
    b_hist, _ = np.histogram(rgb[:, :, 2], bins=bins, range=(0, 256))
    lum_hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
    return {
        "Red": r_hist.astype(int).tolist(),
        "Green": g_hist.astype(int).tolist(),
        "Blue": b_hist.astype(int).tolist(),
        "Luminance": lum_hist.astype(int).tolist(),
    }


def benchmark_comparison_table(benchmark_results):
    """Format benchmark output list into a clean table of comparative statistics."""
    rows = []
    for item in benchmark_results:
        m = item.get("metrics", {})
        delta_contrast = item.get("delta_contrast", 0.0)
        delta_sharpness = item.get("delta_sharpness", 0.0)
        rows.append({
            "Method": item["name"],
            "Category": item.get("category", "General"),
            "Time (ms)": f"{item.get('processing_ms', 0.0):.2f}",
            "Luminance": f"{m.get('mean_brightness', 0.0):.1f}",
            "Contrast σ": f"{m.get('contrast_stddev', 0.0):.1f} ({delta_contrast:+.1f})",
            "Local Contrast": f"{m.get('local_contrast_stddev', 0.0):.1f}",
            "Laplacian Sharpness": f"{m.get('sharpness_laplacian_variance', 0.0):.1f} ({delta_sharpness:+.1f})",
            "Dynamic Range": f"{m.get('robust_dynamic_range', 0.0):.1f}",
            "Pixels Changed": f"{item.get('changed_pixel_fraction', 0.0):.1%}",
        })
    return rows


def benchmark_csv_bytes(benchmark_results):
    """Serialize benchmark comparison items to CSV bytes."""
    stream = StringIO(newline="")
    fieldnames = [
        "name", "category", "processing_ms", "changed_pixel_fraction",
        "mean_brightness", "contrast_stddev", "delta_contrast",
        "local_contrast_stddev", "sharpness_laplacian_variance",
        "delta_sharpness", "robust_dynamic_range", "noise_estimate",
        "clipped_shadow_fraction", "clipped_highlight_fraction"
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for item in benchmark_results:
        m = item.get("metrics", {})
        row = {
            "name": item["name"],
            "category": item.get("category", ""),
            "processing_ms": item.get("processing_ms", 0.0),
            "changed_pixel_fraction": item.get("changed_pixel_fraction", 0.0),
            "mean_brightness": m.get("mean_brightness", 0.0),
            "contrast_stddev": m.get("contrast_stddev", 0.0),
            "delta_contrast": item.get("delta_contrast", 0.0),
            "local_contrast_stddev": m.get("local_contrast_stddev", 0.0),
            "sharpness_laplacian_variance": m.get("sharpness_laplacian_variance", 0.0),
            "delta_sharpness": item.get("delta_sharpness", 0.0),
            "robust_dynamic_range": m.get("robust_dynamic_range", 0.0),
            "noise_estimate": m.get("noise_estimate", 0.0),
            "clipped_shadow_fraction": m.get("clipped_shadow_fraction", 0.0),
            "clipped_highlight_fraction": m.get("clipped_highlight_fraction", 0.0),
        }
        writer.writerow(row)
    return stream.getvalue().encode("utf-8-sig")
