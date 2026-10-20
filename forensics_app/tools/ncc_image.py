"""Find a template in the current image using normalized cross-correlation."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image
from skimage.feature import match_template
from skimage.transform import rotate

from forensics_app.core import ImageDocument
from forensics_app.utils import ask_parameters

from .base import ForensicsTool, ToolResult

IMAGE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def normalized_cross_correlation(
    image: Image.Image,
    template: Image.Image,
    angle: float = 0.0,
) -> Image.Image:
    """Return a grayscale visualization of the image's normalized correlation map."""
    if not math.isfinite(angle) or not -180 <= angle <= 180:
        raise ValueError("Rotation angle must be between -180 and 180 degrees.")

    image_array = np.asarray(image.convert("L"))
    template_array = rotate(
        np.asarray(template.convert("L")),
        angle=angle,
        preserve_range=True,
    )
    
    if (
        template_array.shape[0] > image_array.shape[0]
        or template_array.shape[1] > image_array.shape[1]
    ):
        raise ValueError("The template must not be larger than the original image.")

    correlation = match_template(image_array, template_array, pad_input=True)
    visualization = np.rint(np.clip((correlation + 1) * 127.5, 0, 255)).astype(np.uint8)
    return Image.fromarray(visualization)


class NormalizedCrossCorrelationTool(ForensicsTool):
    tool_id = "normalized_cross_correlation"
    title = "Normalized cross-correlation"
    category = "Template Matching"
    description = "Rotate a template and find it in the currently selected image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        template_path = filedialog.askopenfilename(
            parent=parent,
            title="Select template image",
            filetypes=IMAGE_TYPES,
        )
        if not template_path:
            return None
        with Image.open(template_path) as image:
            template = image.copy()

        parameters = ask_parameters(
            parent,
            "Template rotation",
            (("angle", "Rotation angle (degrees)", -180, 180, 1, 0),),
        )
        if parameters is None:
            return None
        angle = parameters["angle"]

        original = document.current
        output = normalized_cross_correlation(original, template, angle)
        return ToolResult(
            image=output,
            message="Calculated the normalized cross-correlation image.",
            details={
                "Original image size": f"{original.width} × {original.height}",
                "Template size": f"{template.width} × {template.height}",
                "Rotation angle": f"{angle:g}°",
                "Output size": f"{output.width} × {output.height}",
            },
        )
