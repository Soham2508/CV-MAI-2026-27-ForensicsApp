"""Shared ORB/SIFT keypoint detection and description.

Used by Descriptor extraction, Image matching and Best image match in directory,
so validation, minimum sizes and empty-image handling behave identically.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from skimage.feature import ORB, SIFT

# Smallest image side each method can describe in this app.
MIN_IMAGE_SIZE = {"ORB": 32, "SIFT": 12}
DESCRIPTOR_LAYOUT = {"ORB": (256, np.bool_), "SIFT": (128, np.uint8)}


def detect_and_describe(
    image: Image.Image,
    method: str = "ORB",
    *,
    n_keypoints: int = 500,
    fast_threshold: float = 0.08,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect keypoints and compute descriptors with ORB or SIFT.

    Returns ``(keypoints, descriptors)``: keypoints are (row, col) with shape
    (N, 2); descriptors are (N, 256) booleans for ORB or (N, 128) uint8 for
    SIFT. A textureless image gives empty arrays rather than an error.
    """
    method_upper = method.upper() if isinstance(method, str) else ""
    if method_upper not in ("ORB", "SIFT"):
        raise ValueError("Choose ORB or SIFT.")

    if method_upper == "ORB":
        if type(n_keypoints) is not int or n_keypoints < 1:
            raise ValueError("ORB keypoint count must be a positive integer.")
        if not math.isfinite(fast_threshold) or not 0 <= fast_threshold <= 1:
            raise ValueError("FAST threshold must be between 0 and 1.")
        extractor = ORB(n_keypoints=n_keypoints, fast_threshold=fast_threshold)
    else:
        if type(n_octaves) is not int or n_octaves < 1:
            raise ValueError("SIFT octave count must be a positive integer.")
        if type(n_scales) is not int or n_scales < 1:
            raise ValueError("SIFT scale count must be a positive integer.")
        # With upsampling=2 and sigma_in=0.5, sigma_min must be at least 1.
        if not math.isfinite(sigma_min) or sigma_min < 1.0:
            raise ValueError("SIFT sigma_min must be finite and at least 1.0.")
        extractor = SIFT(
            upsampling=2,
            n_octaves=n_octaves,
            n_scales=n_scales,
            sigma_min=sigma_min,
        )

    minimum_size = MIN_IMAGE_SIZE[method_upper]
    if min(image.size) < minimum_size:
        raise ValueError(
            f"{method_upper} needs an image at least "
            f"{minimum_size} by {minimum_size} pixels for this tool."
        )

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    try:
        extractor.detect_and_extract(gray)
    except RuntimeError as error:
        # A textureless image is a valid empty result, not a tool failure.
        if not str(error).startswith(f"{method_upper} found no features."):
            raise
        width, dtype = DESCRIPTOR_LAYOUT[method_upper]
        return np.empty((0, 2), dtype=np.float64), np.empty((0, width), dtype=dtype)
    return extractor.keypoints.copy(), extractor.descriptors.copy()
