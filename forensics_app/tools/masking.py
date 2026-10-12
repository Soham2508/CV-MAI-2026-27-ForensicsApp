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


def _rgb_array(image: Image.Image) -> np.ndarray:
    """Represent common Pillow image modes as an RGB NumPy array."""
    pixels = np.asarray(image)
    if image.mode == "1":
        pixels = pixels.astype(np.uint8) * 255
    elif image.mode == "P":
        palette = image.getpalette()
        if palette is not None:
            colors = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
            pixels = colors[pixels]

    if pixels.ndim == 2:
        pixels = np.repeat(pixels[:, :, np.newaxis], 3, axis=2)
    elif pixels.ndim == 3 and pixels.shape[2] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    elif pixels.ndim == 3 and pixels.shape[2] == 2:
        pixels = np.repeat(pixels[:, :, :1], 3, axis=2)
    elif pixels.ndim == 3 and pixels.shape[2] >= 3:
        pixels = pixels[:, :, :3]
    else:
        raise ValueError(f"Unsupported image mode for compositing: {image.mode}")

    return np.clip(pixels, 0, 255).astype(np.uint8)


def _luminance_array(image: Image.Image) -> np.ndarray:
    """Calculate 8-bit luminance from an image using NumPy."""
    rgb = _rgb_array(image).astype(np.float32)
    luminance = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )
    return luminance.astype(np.uint8)


def _binary_mask_array(image: Image.Image, threshold: int) -> np.ndarray:
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be between 0 and 255.")

    luminance = _luminance_array(image)
    return np.where(luminance > threshold, 255, 0).astype(np.uint8)


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Apply textured mask"
    category = "Image analysis"
    description = (
        "Use the open image as a mask source, then choose a base image and texture."
    )

    def binary_masking_function(
        self, image: Image.Image, threshold: int = 10
    ) -> Image.Image:
        """Return a binary luminance mask; brighter-than-threshold pixels are white."""
        return Image.fromarray(_binary_mask_array(image, threshold))

    def composite_masked_texture(
        self,
        mask_source: Image.Image,
        base_image: Image.Image,
        texture_image: Image.Image,
        threshold: int = 10,
    ) -> Image.Image:
        """Place texture over base wherever the mask source exceeds threshold."""
        width, height = base_image.size
        base = _rgb_array(base_image)
        texture = _rgb_array(texture_image)
        mask = _binary_mask_array(mask_source, threshold)

        texture = _resize_array(texture, (width, height))
        mask = _resize_array(mask, (width, height)) > 0
        composite = np.where(mask[:, :, np.newaxis], texture, base).astype(np.uint8)
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
