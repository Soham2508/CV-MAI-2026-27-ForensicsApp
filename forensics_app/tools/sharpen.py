"""Unsharp-mask image sharpening."""

from __future__ import annotations

import math
import tkinter as tk

import numpy as np
from PIL import Image
from skimage.filters import gaussian

from forensics_app.core import ImageDocument
from forensics_app.utils.parameter_dialog import ParameterField, ask_parameters

from .base import ForensicsTool, ToolResult


def sharpen_image(
    image: Image.Image, radius: float = 2.0, amount: float = 1.5
) -> Image.Image:
    """Sharpen using original + amount * (original - Gaussian blur)."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be a finite number greater than zero.")
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Amount must be a finite number greater than or equal to zero.")

    grayscale = image.mode in ("1", "L", "I", "F")
    if grayscale:
        original = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        blurred = gaussian(original, sigma=radius)
    else:
        original = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        blurred = gaussian(original, sigma=radius, channel_axis=-1)

    detail = original.copy() - blurred
    sharpened = np.clip(original + amount * detail, 0.0, 1.0)
    return Image.fromarray(np.round(sharpened * 255).astype(np.uint8))


class SharpenTool(ForensicsTool):
    tool_id = "sharpen"
    title = "Sharpen image"
    category = "Filtering & Edges"
    description = "Sharpen the image with adjustable blur radius and amount."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        parameters = ask_parameters(
            parent,
            self.title,
            (
                ParameterField(
                    "radius",
                    "Gaussian blur radius (sigma)",
                    0.1,
                    10.0,
                    0.1,
                    2.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ("amount", "Sharpening amount", 0.0, 30.0, 0.1, 1.5),
            ),
        )
        if parameters is None:
            return None

        radius = parameters["radius"]
        amount = parameters["amount"]
        output = sharpen_image(document.current, radius, amount)
        return ToolResult(
            image=output,
            message=f"Sharpened the image (radius={radius:g}, amount={amount:g}).",
            details={
                "Operation": "Unsharp mask",
                "Blur radius (sigma)": radius,
                "Sharpening amount": amount,
            },
        )
