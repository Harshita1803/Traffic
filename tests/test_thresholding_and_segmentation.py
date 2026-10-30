"""Tests for Otsu automatic thresholding, adaptive thresholding, and morphology."""
import numpy as np
import pytest
from PIL import Image

from dip.spatial import adaptive_threshold, morphology, otsu_threshold


def bimodal_image():
    """Create a synthetic bimodal image (dark road background with bright lane markings)."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:32, :] = 40   # background class
    img[32:, :] = 200  # foreground class
    return Image.fromarray(img, mode="L")


def test_otsu_threshold_finds_optimal_threshold_on_bimodal():
    image = bimodal_image()
    threshold, mask = otsu_threshold(image)
    # The optimal threshold between peaks at 40 and 200 should lie between them
    assert 40 <= threshold <= 200
    assert mask.mode == "L"
    assert mask.size == image.size
    mask_arr = np.asarray(mask)
    # Background should be 0, foreground 255
    assert np.all(mask_arr[:32, :] == 0)
    assert np.all(mask_arr[32:, :] == 255)


def test_otsu_threshold_handles_uniform_image():
    uniform = Image.fromarray(np.full((32, 32), 128, dtype=np.uint8), mode="L")
    threshold, mask = otsu_threshold(uniform)
    assert 0 <= threshold <= 255
    assert mask.size == uniform.size


def test_otsu_threshold_rejects_empty():
    with pytest.raises(ValueError, match="has no pixels"):
        otsu_threshold(Image.new("L", (0, 0)))


def test_adaptive_threshold_validates_block_size():
    img = bimodal_image()
    with pytest.raises(ValueError, match="odd integer >= 3"):
        adaptive_threshold(img, block_size=2)
    with pytest.raises(ValueError, match="odd integer >= 3"):
        adaptive_threshold(img, block_size=4)


def test_adaptive_threshold_produces_binary_mask():
    img = bimodal_image()
    mask = adaptive_threshold(img, block_size=15, c=2.0)
    assert mask.mode == "L"
    assert mask.size == img.size
    unique = set(np.unique(np.asarray(mask)))
    assert unique.issubset({0, 255})


def test_morphology_gradient_tophat_blackhat():
    # Create an image with a central bright square
    arr = np.full((32, 32), 50, dtype=np.uint8)
    arr[12:20, 12:20] = 220
    img = Image.fromarray(arr, mode="L")

    # Morphological gradient = dilation - erosion (boundary of the square)
    grad = morphology(img, operation="gradient", size=3)
    assert grad.mode == "L"
    assert grad.size == img.size
    grad_arr = np.asarray(grad)
    assert grad_arr.max() > 0
    # Inside the uniform 50 region, gradient should be 0
    assert grad_arr[0, 0] == 0

    # Top-hat = original - opening (extracts small bright elements)
    tophat = morphology(img, operation="tophat", size=3)
    assert tophat.mode == "L"
    assert tophat.size == img.size

    # Black-hat = closing - original
    blackhat = morphology(img, operation="blackhat", size=3)
    assert blackhat.mode == "L"
    assert blackhat.size == img.size


def test_morphology_rejects_invalid_ops_and_even_sizes():
    img = bimodal_image()
    with pytest.raises(ValueError, match="unsupported morphology operation"):
        morphology(img, operation="invalid_op")
    with pytest.raises(ValueError, match="odd integer >= 3"):
        morphology(img, operation="opening", size=4)
    with pytest.raises(ValueError, match="odd integer >= 3"):
        morphology(img, operation="erosion", size=1)
