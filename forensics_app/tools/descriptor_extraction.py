"""Configurable ORB/SIFT extraction and keypoint visualization."""

from __future__ import annotations

import math
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import ORB, SIFT

from forensics_app.core import ImageDocument
from forensics_app.utils.parameter_dialog import ParameterField, ask_parameters

from .base import ForensicsTool, ToolResult


def extract_descriptors(
    image: Image.Image,
    method: str = "ORB",
    *,
    n_keypoints: int = 200,
    fast_threshold: float = 0.08,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, Image.Image]:

    if method not in ("ORB", "SIFT"):
        raise ValueError("Choose ORB or SIFT.")

    if method == "ORB":
        if type(n_keypoints) is not int or n_keypoints < 1:
            raise ValueError("ORB keypoint count must be a positive integer.")
        if not math.isfinite(fast_threshold) or not 0 <= fast_threshold <= 1:
            raise ValueError("FAST threshold must be between 0 and 1.")
        extractor = ORB(n_keypoints=n_keypoints, fast_threshold=fast_threshold)
        descriptor_width, descriptor_dtype = 256, np.bool_
        minimum_size = 32
    else:
        if type(n_octaves) is not int or n_octaves < 1:
            raise ValueError("SIFT octave count must be a positive integer.")
        if type(n_scales) is not int or n_scales < 1:
            raise ValueError("SIFT scale count must be a positive integer.")
        # With upsampling=2 and sigma_in=0.5, sigma_min must be at least 1.
        if not math.isfinite(sigma_min) or sigma_min < 1:
            raise ValueError("SIFT sigma_min must be finite and at least 1.0.")
        extractor = SIFT(
            upsampling=2,
            n_octaves=n_octaves,
            n_scales=n_scales,
            sigma_min=sigma_min,
        )
        descriptor_width, descriptor_dtype = 128, np.uint8
        minimum_size = 12

    if min(image.size) < minimum_size:
        raise ValueError(
            f"{method} needs an image at least "
            f"{minimum_size} by {minimum_size} pixels for this tool."
        )

    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    try:
        extractor.detect_and_extract(gray)
    except RuntimeError as error:
        # A textureless image is a valid empty result, not a tool failure.
        if not str(error).startswith(f"{method} found no features."):
            raise
        keypoints = np.empty((0, 2), dtype=np.float64)
        descriptors = np.empty((0, descriptor_width), dtype=descriptor_dtype)
    else:
        keypoints = extractor.keypoints.copy()
        descriptors = extractor.descriptors.copy()

    output = gray_image.convert("RGB")
    drawing = ImageDraw.Draw(output)
    for row, column in keypoints:
        x, y = float(column), float(row)
        drawing.ellipse((x - 3, y - 3, x + 3, y + 3), outline="red", width=1)

    return keypoints, descriptors, output

def extract_features(
    image: Image.Image,
    mode: str = "Both",
    *,
    orb_n_keypoints: int = 200,
    orb_fast_threshold: float = 0.08,
    sift_n_octaves: int = 8,
    sift_n_scales: int = 3,
    sift_sigma_min: float = 1.6,
) -> tuple[dict[str, dict[str, np.ndarray]], Image.Image]:
    """Extract ORB, SIFT, or both with independent settings."""
    if mode not in ("ORB", "SIFT", "Both"):
        raise ValueError("Choose ORB, SIFT, or Both.")

    settings = {}
    if mode in ("ORB", "Both"):
        settings["ORB"] = {
            "n_keypoints": orb_n_keypoints,
            "fast_threshold": orb_fast_threshold,
        }
    if mode in ("SIFT", "Both"):
        settings["SIFT"] = {
            "n_octaves": sift_n_octaves,
            "n_scales": sift_n_scales,
            "sigma_min": sift_sigma_min,
        }

    features = {}
    output = image.convert("L").convert("RGB")
    drawing = ImageDraw.Draw(output)

    for method, options in settings.items():
        keypoints, descriptors, _ = extract_descriptors(image, method, **options)
        features[method] = {
            "keypoints": keypoints,
            "descriptors": descriptors,
        }

        for row, column in keypoints:
            x, y = float(column), float(row)
            if method == "ORB":
                drawing.ellipse(
                    (x - 3, y - 3, x + 3, y + 3), outline="red", width=1
                )
            else:
                drawing.line((x - 4, y, x + 4, y), fill="#0080ff", width=1)
                drawing.line((x, y - 4, x, y + 4), fill="#0080ff", width=1)

    return features, output

class DescriptorExtractionTool(ForensicsTool):
    tool_id = "descriptor_extraction"
    title = "Descriptor extraction"
    category = "Matching & Descriptors"
    description = "Extract ORB, SIFT, or both with independent settings."
    menu_options = ("ORB", "SIFT", "Both")

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        return self.run_action(parent, document, "ORB")

    def run_action(
        self, parent: tk.Misc, document: ImageDocument, action: str
    ) -> ToolResult | None:
        if action not in self.menu_options:
            raise ValueError("Choose ORB, SIFT, or Both.")
        assert document.current is not None

        fields = ()
        integer_fields = ()

        if action in ("ORB", "Both"):
            fields += (
                ParameterField(
                    "orb_n_keypoints", "ORB: maximum keypoints", 1, 5000, 1, 200
                ),
                ParameterField(
                    "orb_fast_threshold", "ORB: FAST threshold", 0, 1, 0.01, 0.08
                ),
            )
            integer_fields += ("orb_n_keypoints",)

        if action in ("SIFT", "Both"):
            fields += (
                ParameterField("sift_n_octaves", "SIFT: maximum octaves", 1, 8, 1, 8),
                ParameterField("sift_n_scales", "SIFT: scales per octave", 1, 8, 1, 3),
                ParameterField("sift_sigma_min", "SIFT: initial sigma", 1, 3, 0.1, 1.6),
            )
            integer_fields += ("sift_n_octaves", "sift_n_scales")

        parameters = ask_parameters(
            parent,
            f"{self.title}: {action}",
            fields,
            lambda values: (
                "Counts must be whole numbers."
                if any(not values[key].is_integer() for key in integer_fields)
                else None
            ),
        )
        if parameters is None:
            return None

        settings = {
            key: int(value) if key in integer_fields else value
            for key, value in parameters.items()
        }
        features, output = extract_features(
            document.current, mode=action, **settings
        )

        details = {"Mode": action}
        summaries = []

        for method, result in features.items():
            keypoints = result["keypoints"]
            descriptors = result["descriptors"]

            details[f"{method} keypoints"] = len(keypoints)
            details[f"{method} descriptor matrix"] = (
                f"{descriptors.shape[0]} x {descriptors.shape[1]}"
            )
            details[f"{method} descriptor dtype"] = str(descriptors.dtype)
            details[f"{method} markers"] = (
                "Red circles" if method == "ORB" else "Blue crosses"
            )
            summaries.append(f"{method}: {len(keypoints)} keypoints")

        details.update({field.label: settings[field.key] for field in fields})
        message = "; ".join(summaries) + "."

        if all(len(result["keypoints"]) == 0 for result in features.values()):
            message += " No keypoints found. Try a more textured image."

        return ToolResult(image=output, message=message, details=details)