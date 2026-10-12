"""Contrast stretching and histogram-equalization image tools."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage import exposure

from forensics_app.core import ImageDocument
from forensics_app.utils import ask_parameters
from .base import ForensicsTool, ToolResult


def contrast_stretch_image(
    image: Image.Image, low_percentile: float = 2, high_percentile: float = 98
) -> Image.Image:
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100.")
    rgb = np.asarray(image.convert("RGB"))
    low, high = np.percentile(rgb, (low_percentile, high_percentile))
    stretched = exposure.rescale_intensity(
        rgb, in_range=(low, high), out_range=(0, 255)
    )
    return Image.fromarray(stretched.astype(np.uint8))


def equalize_histogram(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    equalized = np.stack(
        [exposure.equalize_hist(rgb[:, :, channel]) for channel in range(3)], axis=2
    )
    return Image.fromarray((equalized * 255).astype(np.uint8))


def adaptive_histogram_equalization(
    image: Image.Image, clip_limit: float = 0.03
) -> Image.Image:
    if not 0 < clip_limit <= 1:
        raise ValueError("Clip limit must be greater than 0 and at most 1.")
    rgb = np.asarray(image.convert("RGB"))
    equalized = exposure.equalize_adapthist(rgb, clip_limit=clip_limit)
    return Image.fromarray((equalized * 255).astype(np.uint8))


class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast stretching"
    category = "Image enhancement"
    description = "Stretch RGB intensity values between adjustable percentiles."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        parameters = ask_parameters(
            parent,
            self.title,
            (
                ("low", "Lower percentile", 0, 49, 1, 2),
                ("high", "Upper percentile", 51, 100, 1, 98),
            ),
            lambda values: (
                "The lower percentile must be less than the upper percentile."
                if values["low"] >= values["high"]
                else None
            ),
        )
        if parameters is None:
            return None
        image = contrast_stretch_image(
            document.current, parameters["low"], parameters["high"]
        )
        return ToolResult(
            image=image,
            message="Stretched image contrast.",
            details={
                "Operation": self.title,
                "Lower percentile": f"{parameters['low']:g}",
                "Upper percentile": f"{parameters['high']:g}",
            },
        )


class HistogramEqualizationTool(ForensicsTool):
    tool_id = "histogram_equalization"
    title = "Histogram equalization"
    category = "Image enhancement"
    description = "Equalize the image histogram."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        return ToolResult(
            image=equalize_histogram(document.current),
            message="Equalized the image histogram.",
            details={"Operation": self.title},
        )


class CLAHETool(ForensicsTool):
    tool_id = "clahe"
    title = "Adaptive histogram equalization (CLAHE)"
    category = "Image enhancement"
    description = "Apply CLAHE with an adjustable clip limit."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        parameters = ask_parameters(
            parent,
            self.title,
            (("clip_limit", "Clip limit", 0.001, 1.0, 0.001, 0.03),),
        )
        if parameters is None:
            return None
        clip_limit = parameters["clip_limit"]
        return ToolResult(
            image=adaptive_histogram_equalization(document.current, clip_limit),
            message="Applied adaptive histogram equalization.",
            details={"Operation": "CLAHE", "Clip limit": f"{clip_limit:g}"},
        )
