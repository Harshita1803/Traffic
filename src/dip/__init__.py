"""Classical spatial- and frequency-domain processing algorithms."""
from .spatial import (
    adaptive_threshold,
    as_rgb,
    bilateral_denoise,
    clahe_luminance,
    edge_enhance,
    edge_map,
    equalize_luminance,
    gamma_correct,
    gaussian_denoise,
    median_denoise,
    morphology,
    normalize_luminance,
    otsu_threshold,
    resize_preserving_aspect_ratio,
    unsharp_luminance,
)
from .dehaze import dark_channel_prior
from .frequency import gaussian_lowpass_luminance, spectrum_image
from .adaptive import analyze_and_plan, build_pipeline, normalize_pipeline_result, stage_view

__all__ = [
    "adaptive_threshold",
    "analyze_and_plan",
    "as_rgb",
    "bilateral_denoise",
    "build_pipeline",
    "clahe_luminance",
    "dark_channel_prior",
    "edge_enhance",
    "edge_map",
    "equalize_luminance",
    "gamma_correct",
    "gaussian_denoise",
    "gaussian_lowpass_luminance",
    "median_denoise",
    "morphology",
    "normalize_luminance",
    "normalize_pipeline_result",
    "otsu_threshold",
    "resize_preserving_aspect_ratio",
    "spectrum_image",
    "stage_view",
    "unsharp_luminance",
]
