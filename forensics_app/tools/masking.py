"""Apply a mask-derived texture layer over a selected base image."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


IMAGE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def _resize_array(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an image array using nearest-neighbor index mapping."""
    target_width, target_height = size
    source_height, source_width = array.shape[:2]
    rows = np.minimum(
        np.arange(target_height) * source_height // target_height,
        source_height - 1,
    )
    columns = np.minimum(
        np.arange(target_width) * source_width // target_width,
        source_width - 1,
    )
    return array[rows[:, np.newaxis], columns]


def _mask_array(image: Image.Image, threshold: int) -> np.ndarray:
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be between 0 and 255.")

    mask = np.any(np.asarray(image.convert("RGB")) > threshold, axis=2)
    return mask


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Apply textured mask"
    category = "Image analysis"
    description = (
        "Use the open image as a mask source, then choose a base image and texture."
    )

    def composite_masked_texture(
        self,
        mask_source: Image.Image,
        base_image: Image.Image,
        texture_image: Image.Image,
        threshold: int = 10,
    ) -> Image.Image:
        """Place texture over base wherever the mask source exceeds threshold."""
        width, height = base_image.size
        base = np.asarray(base_image.convert("RGB"))
        texture = np.asarray(texture_image.convert("RGB"))
        mask = _mask_array(mask_source, threshold)

        texture = _resize_array(texture, (width, height))
        mask = _resize_array(mask, (width, height))
        composite = np.where(mask[:, :, np.newaxis], texture, base)
        return Image.fromarray(composite)

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window

        base_path = filedialog.askopenfilename(
            parent=parent,
            title="Select base image",
            filetypes=IMAGE_TYPES,
        )
        if not base_path:
            return None
        with Image.open(base_path) as image:
            base_image = image.copy()

        texture_path = filedialog.askopenfilename(
            parent=parent,
            title="Select texture image",
            filetypes=IMAGE_TYPES,
        )
        if not texture_path:
            return None
        with Image.open(texture_path) as image:
            texture_image = image.copy()

        threshold = 10
        mask_source = document.current
        output = self.composite_masked_texture(
            mask_source,
            base_image,
            texture_image,
            threshold,
        )
        return ToolResult(
            image=output,
            message="Applied the texture through the mask over the base image.",
            details={
                "Mask source size": f"{mask_source.width} × {mask_source.height}",
                "Base image size": f"{base_image.width} × {base_image.height}",
                "Texture size": f"{texture_image.width} × {texture_image.height}",
                "Output size": f"{output.width} × {output.height}",
                "Threshold": threshold,
            },
        )
