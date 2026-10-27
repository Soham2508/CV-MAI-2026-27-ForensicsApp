"""Configurable Histogram of Oriented Gradients extraction."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image
from skimage.feature import hog

from forensics_app.core import ImageDocument
from forensics_app.utils.parameter_dialog import ParameterField, ask_parameters

from .base import ForensicsTool, ToolResult


def compute_hog_features(
    gray_image: np.ndarray,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
) -> np.ndarray:
    """Return a one-dimensional HOG descriptor for a grayscale float image."""
    return hog(
        gray_image,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        visualize=False,
        feature_vector=True,
    )


def hog_visualization(
    image: Image.Image,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
) -> tuple[np.ndarray, Image.Image]:
    """Return the descriptor and a visible HOG image."""
    if min(orientations, pixels_per_cell, cells_per_block) < 1:
        raise ValueError("All HOG settings must be positive integers.")

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    minimum_size = pixels_per_cell * cells_per_block
    if gray.shape[0] < minimum_size or gray.shape[1] < minimum_size:
        raise ValueError(
            f"Image must be at least {minimum_size} × {minimum_size} pixels "
            "for the selected HOG settings."
        )

    descriptor, visualization = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        visualize=True,
        feature_vector=True,
    )

    maximum = float(visualization.max())
    if maximum > 0:
        visualization = visualization / maximum
    output = Image.fromarray(
        np.clip(visualization * 255, 0, 255).astype(np.uint8),
        mode="L",
    )
    return descriptor, output


class HOGVisualizationTool(ForensicsTool):
    tool_id = "hog_visualization"
    title = "HOG visualization"
    category = "Matching & Descriptors"
    description = "Extract and display configurable Histogram of Oriented Gradients."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        parameters = ask_parameters(
            parent,
            self.title,
            (
                ParameterField(
                    "orientations",
                    "Orientations",
                    1.0,
                    18.0,
                    1.0,
                    9.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ParameterField(
                    "pixels_per_cell",
                    "Pixels per cell",
                    1.0,
                    32.0,
                    1.0,
                    8.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ParameterField(
                    "cells_per_block",
                    "Cells per block",
                    1.0,
                    4.0,
                    1.0,
                    2.0,
                    show_slider=False,
                    no_threshold=True,
                ),
            ),
        )
        if parameters is None:
            return None

        orientations = int(parameters["orientations"])
        pixels_per_cell = int(parameters["pixels_per_cell"])
        cells_per_block = int(parameters["cells_per_block"])

        descriptor, output = hog_visualization(
            document.current,
            orientations,
            pixels_per_cell,
            cells_per_block,
        )

        return ToolResult(
            image=output,
            message="Extracted HOG descriptor and visualization.",
            details={
                "Operation": "Histogram of Oriented Gradients",
                "Orientations": orientations,
                "Pixels per cell": f"{pixels_per_cell} × {pixels_per_cell}",
                "Cells per block": f"{cells_per_block} × {cells_per_block}",
                "Descriptor length": len(descriptor),
            },
        )
