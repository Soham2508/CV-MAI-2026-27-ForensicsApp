"""FAST and Harris corner detection with adjustable peak selection."""

from __future__ import annotations

import math
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw
from skimage.feature import corner_fast, corner_harris, corner_peaks

from forensics_app.core import ImageDocument
from forensics_app.utils.parameter_dialog import ParameterField, ask_parameters

from .base import ForensicsTool, ToolResult


def detect_corners(
    image: Image.Image,
    method: str = "FAST",
    threshold_rel: float = 0.1,
    min_distance: int = 5,
    fast_threshold: float = 0.15,
    harris_sigma: float = 1.0,
) -> tuple[np.ndarray, Image.Image]:
    """Return (row, column) corner coordinates and a grayscale overlay."""
    if method not in ("FAST", "Harris"):
        raise ValueError("Choose FAST or Harris.")
    if not math.isfinite(threshold_rel) or not 0 <= threshold_rel <= 1:
        raise ValueError("Peak threshold must be between 0 and 1.")
    if type(min_distance) is not int or min_distance < 1:
        raise ValueError("Minimum distance must be a positive integer.")
    if not math.isfinite(fast_threshold) or not 0 <= fast_threshold <= 1:
        raise ValueError("FAST intensity threshold must be between 0 and 1.")
    if not math.isfinite(harris_sigma) or harris_sigma <= 0:
        raise ValueError("Harris sigma must be a finite positive number.")
    if min(image.size) < 1:
        raise ValueError("The image must not be empty.")
    if method == "FAST" and min(image.size) < 7:
        raise ValueError("FAST needs an image at least 7 by 7 pixels.")

    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0

    if method == "FAST":
        response = corner_fast(gray, n=12, threshold=fast_threshold)
    else:
        # Extend the boundary so Harris does not see an artificial black frame.
        padding = math.ceil(4 * harris_sigma) + 1
        padded = np.pad(gray, padding, mode="edge")
        response = corner_harris(padded, method="k", k=0.05, sigma=harris_sigma)
        response = response[padding:-padding, padding:-padding]

    corners = corner_peaks(
        response,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=0.0,
        exclude_border=False,
    )

    output = gray_image.convert("RGB")
    drawing = ImageDraw.Draw(output)
    for row, column in corners:
        x, y = int(column), int(row)

        if method == "FAST":
            drawing.ellipse(
                (x - 3, y - 3, x + 3, y + 3),
                outline="red",
                width=1,
            )
        else:
            drawing.line(
                (x - 4, y, x + 4, y),
                fill="#0080ff",
                width=1,
            )
            drawing.line(
                (x, y - 4, x, y + 4),
                fill="#0080ff",
                width=1,
            )

    return corners, output


class CornerDetectionTool(ForensicsTool):
    tool_id = "corner_detection"
    title = "Corner detection"
    category = "Matching & Descriptors"
    description = "Detect FAST or Harris corners with an adjustable peak threshold."
    menu_options = ("FAST", "Harris")

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        return self.run_action(parent, document, "FAST")

    def run_action(
        self, parent: tk.Misc, document: ImageDocument, action: str
    ) -> ToolResult | None:
        if action not in self.menu_options:
            raise ValueError("Choose FAST or Harris.")
        assert document.current is not None

        fields = (
            ParameterField(
                "threshold_rel", "Relative peak threshold", 0.0, 1.0, 0.01, 0.1
            ),
            ParameterField(
                "min_distance",
                "Minimum peak distance (px)",
                1.0,
                50.0,
                1.0,
                5.0,
                show_slider=False,
                no_threshold=True,
            ),
        )
        if action == "FAST":
            fields += (
                ParameterField(
                    "fast_threshold",
                    "FAST intensity threshold",
                    0.0,
                    1.0,
                    0.01,
                    0.15,
                ),
            )
        else:
            fields += (
                ParameterField(
                    "harris_sigma",
                    "Harris sigma",
                    0.1,
                    5.0,
                    0.1,
                    1.0,
                    show_slider=False,
                    no_threshold=True,
                ),
            )

        parameters = ask_parameters(
            parent,
            f"{self.title}: {action}",
            fields,
            lambda values: (
                "Minimum peak distance must be a whole number."
                if not values["min_distance"].is_integer()
                else None
            ),
        )
        if parameters is None:
            return None

        min_distance = int(parameters["min_distance"])
        fast_threshold = parameters.get("fast_threshold", 0.15)
        harris_sigma = parameters.get("harris_sigma", 1.0)
        corners, output = detect_corners(
            document.current,
            method=action,
            threshold_rel=parameters["threshold_rel"],
            min_distance=min_distance,
            fast_threshold=fast_threshold,
            harris_sigma=harris_sigma,
        )

        details = {
            "Method": action,
            "Corners detected": len(corners),
            "Relative peak threshold": parameters["threshold_rel"],
            "Minimum peak distance": f"{min_distance} px",
        }
        if action == "FAST":
            details.update({"FAST intensity threshold": fast_threshold, "FAST n": 12})
        else:
            details.update({"Harris sigma": harris_sigma, "Harris k": 0.05})

        return ToolResult(
            image=output,
            message=f"{action}: detected {len(corners)} corners.",
            details=details,
        )
