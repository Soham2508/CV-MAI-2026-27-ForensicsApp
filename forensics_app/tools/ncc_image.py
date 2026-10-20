"""Find a template in the current image using normalized cross-correlation."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image
from skimage.feature import match_template

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

IMAGE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def normalized_cross_correlation(
    image: Image.Image,
    template: Image.Image,
) -> Image.Image:
    """Return a grayscale visualization of the image's normalized correlation map."""
    image_array = np.asarray(image.convert("L"))
    template_array = np.asarray(template.convert("L"))
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
    description = "Find a template image in the currently selected image."

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

        original = document.current
        output = normalized_cross_correlation(original, template)
        return ToolResult(
            image=output,
            message="Calculated the normalized cross-correlation image.",
            details={
                "Original image size": f"{original.width} × {original.height}",
                "Template size": f"{template.width} × {template.height}",
                "Output size": f"{output.width} × {output.height}",
            },
        )
