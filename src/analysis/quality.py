"""Pixel-derived quality and degradation indicators.

These measurements describe image statistics; none is a perceptual-quality
score or a definitive weather classifier.
"""
import numpy as np
from PIL import Image, ImageFilter


def _gray_array(image):
    return np.asarray(image.convert("L"), dtype=np.float32)


def _local_contrast(gray, tile=64):
    values = []
    for y in range(0, gray.shape[0], tile):
        for x in range(0, gray.shape[1], tile):
            patch = gray[y:y + tile, x:x + tile]
            if patch.size:
                values.append(float(patch.std()))
    return float(np.mean(values)) if values else 0.0


def _edge_stats(gray):
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]
          - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]) / 8.0
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
          - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]) / 8.0
    magnitude = np.hypot(gx, gy)
    return (float(magnitude.mean()), float((magnitude >= 4).mean()),
            float((magnitude >= 12).mean()))


def calculate_quality(image):
    """Return brightness, contrast, sharpness, histogram, clipping, color, edges.

    Sharpness is four-neighbour Laplacian variance. It is content and resolution
    dependent; higher values can also reflect noise or oversharpening.
    """
    gray = _gray_array(image)
    if gray.size == 0:
        raise ValueError("Image contains no pixels")
    p = np.pad(gray, 1, mode="edge")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * gray
    # Keep the full RGB image in uint8: converting it to float32 here would
    # allocate four times the pixel storage without improving these statistics.
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    channel_means = {name: float(rgb[:, :, i].mean()) for i, name in enumerate(("red", "green", "blue"))}
    channel_stddevs = {name: float(rgb[:, :, i].std()) for i, name in enumerate(("red", "green", "blue"))}
    max_c = rgb.max(axis=2)
    min_c = rgb.min(axis=2)
    max_float = max_c.astype(np.float32)
    spread = max_float - min_c.astype(np.float32)
    saturation = np.divide(spread, max_float, out=np.zeros_like(max_float), where=max_c > 0)
    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    luminance_p1, luminance_p99 = np.percentile(gray, [1, 99])
    edge_mean, edge_fraction_4, edge_fraction_12 = _edge_stats(gray)
    noise_proxy = estimate_noise(image)
    low_contrast_evidence = float(np.clip((60.0 - gray.std()) / 60.0, 0.0, 1.0))
    min_channel = min_c
    win = min(15, min(image.width, image.height))
    if win % 2 == 0:
        win -= 1
    if win < 1:
        win = 1
    dark = Image.fromarray(min_channel, mode="L")
    if win > 1:
        dark = dark.filter(ImageFilter.MinFilter(win))
    dark_channel_mean = float(np.asarray(dark, dtype=np.float32).mean() / 255.0)
    local_contrast = _local_contrast(gray)
    low_local_contrast_evidence = float(np.clip((30.0 - local_contrast) / 30.0, 0.0, 1.0))
    # Heuristic evidence combines dark-channel lift, global/local contrast, and
    # low saturation. It is not a probability or weather label.
    haze_index = float(np.clip(0.35 * dark_channel_mean + 0.20 * low_contrast_evidence
                               + 0.25 * low_local_contrast_evidence
                               + 0.20 * (1.0 - float(saturation.mean())), 0.0, 1.0))
    return {
        "mean_brightness": float(gray.mean()),
        "contrast_stddev": float(gray.std()),
        "luminance_p1": float(luminance_p1), "luminance_p99": float(luminance_p99),
        "robust_dynamic_range": float(luminance_p99 - luminance_p1),
        "sharpness_laplacian_variance": float(lap.var()),
        "noise_estimate": noise_proxy,
        "width": int(image.width), "height": int(image.height),
        "mode": image.info.get("source_mode", image.mode),
        "channel_means_rgb": channel_means, "channel_stddevs_rgb": channel_stddevs,
        "clipped_shadow_fraction": float((gray <= 5).mean()),
        "clipped_highlight_fraction": float((gray >= 250).mean()),
        "local_contrast_stddev": local_contrast,
        "edge_gradient_mean": edge_mean,
        "edge_fraction_gradient_ge_4": edge_fraction_4,
        "edge_fraction_gradient_ge_12": edge_fraction_12,
        "mean_saturation": float(saturation.mean()),
        "dark_channel_mean": dark_channel_mean,
        "haze_like_indicator": haze_index,
        "haze_local_contrast_evidence": low_local_contrast_evidence,
        "haze_indicator_note": "Heuristic index using dark-channel lift, global/local contrast and low saturation; not a probability or weather classification.",
        "luminance_histogram_32": hist.astype(int).tolist(),
    }


def estimate_noise(image):
    """Robust fine-grain noise proxy from median-filter residual MAD.

    Edges and texture also contribute to this estimate, especially at small sizes.
    """
    gray = image.convert("L")
    smooth = np.asarray(gray.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    residual = np.asarray(gray, dtype=np.float32) - smooth
    med = np.median(residual)
    return float(1.4826 * np.median(np.abs(residual - med)))


def calculate_psnr(image_a, image_b):
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB between two aligned images.

    Returns float('inf') for identical images. Raises ValueError if dimensions differ.
    """
    if not isinstance(image_a, Image.Image) or not isinstance(image_b, Image.Image):
        raise TypeError("Both inputs must be PIL images")
    if image_a.size != image_b.size:
        raise ValueError(f"Image dimension mismatch for PSNR: {image_a.size} vs {image_b.size}")

    a = np.asarray(image_a.convert("RGB"), dtype=np.float64)
    b = np.asarray(image_b.convert("RGB"), dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def calculate_ssim(image_a, image_b, window_size=11):
    """Compute mean Structural Similarity Index (SSIM) between two aligned images on luminance.

    Follows Wang et al. (2004) standard definition with dynamic range L=255,
    k1=0.01, k2=0.03.
    """
    if not isinstance(image_a, Image.Image) or not isinstance(image_b, Image.Image):
        raise TypeError("Both inputs must be PIL images")
    if image_a.size != image_b.size:
        raise ValueError(f"Image dimension mismatch for SSIM: {image_a.size} vs {image_b.size}")

    gray_a = np.asarray(image_a.convert("L"), dtype=np.float64)
    gray_b = np.asarray(image_b.convert("L"), dtype=np.float64)

    if np.array_equal(gray_a, gray_b):
        return 1.0

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    h, w = gray_a.shape
    if min(h, w) < window_size:
        window_size = max(3, min(h, w) if min(h, w) % 2 == 1 else min(h, w) - 1)

    radius = window_size // 2
    im_a = Image.fromarray(gray_a.astype(np.float32), mode="F")
    im_b = Image.fromarray(gray_b.astype(np.float32), mode="F")

    mu_a = np.asarray(im_a.filter(ImageFilter.BoxBlur(radius)), dtype=np.float64)
    mu_b = np.asarray(im_b.filter(ImageFilter.BoxBlur(radius)), dtype=np.float64)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = np.asarray(Image.fromarray((gray_a ** 2).astype(np.float32), mode="F").filter(ImageFilter.BoxBlur(radius)), dtype=np.float64) - mu_a_sq
    sigma_b_sq = np.asarray(Image.fromarray((gray_b ** 2).astype(np.float32), mode="F").filter(ImageFilter.BoxBlur(radius)), dtype=np.float64) - mu_b_sq
    sigma_ab = np.asarray(Image.fromarray((gray_a * gray_b).astype(np.float32), mode="F").filter(ImageFilter.BoxBlur(radius)), dtype=np.float64) - mu_ab

    sigma_a_sq = np.maximum(sigma_a_sq, 0.0)
    sigma_b_sq = np.maximum(sigma_b_sq, 0.0)

    num = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    den = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.clip(ssim_map.mean(), -1.0, 1.0))


def evaluate_paired_reference(enhanced_image, reference_image=None):
    """Evaluate PSNR and SSIM only when a valid paired reference image is supplied.

    For unpaired adverse-weather datasets (e.g. DAWN), marks metrics as explicitly
    unavailable rather than inventing values or reporting zeros.
    """
    if reference_image is None:
        return {
            "psnr": None,
            "ssim": None,
            "reference_available": False,
            "status": "unavailable (unpaired adverse-weather scene; no aligned clear reference)",
        }
    return {
        "psnr": calculate_psnr(enhanced_image, reference_image),
        "ssim": calculate_ssim(enhanced_image, reference_image),
        "reference_available": True,
        "status": "evaluated",
    }
