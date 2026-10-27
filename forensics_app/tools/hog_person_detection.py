"""Educational fixed-scale person detection using sliding-window HOG."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from forensics_app.utils.parameter_dialog import ParameterField, ask_parameters

from .base import ForensicsTool, ToolResult
from .hog_extraction_visualization import compute_hog_features

IMAGE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
BOX_COLORS = ("Red", "Green", "Blue", "Yellow", "Cyan", "Magenta", "White")
EUCLIDEAN_BOX_COLOR = "Red"
COSINE_BOX_COLOR = "Blue"


def _window_starts(length: int, window: int, step: int) -> list[int]:
    starts = list(range(0, length - window + 1, step))
    final_start = length - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def detect_with_hog(
    scene: Image.Image,
    template: Image.Image,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
    step_size: int,
) -> dict[str, object]:
    """Return the best sliding-window locations using two HOG similarity scores."""
    scene_gray = np.asarray(scene.convert("L"), dtype=np.float32) / 255.0
    template_gray = np.asarray(template.convert("L"), dtype=np.float32) / 255.0

    window_height, window_width = template_gray.shape
    scene_height, scene_width = scene_gray.shape

    if window_height > scene_height or window_width > scene_width:
        raise ValueError("The person template must fit inside the scene image.")

    template_features = compute_hog_features(
        template_gray,
        orientations,
        pixels_per_cell,
        cells_per_block,
    )

    locations: list[tuple[int, int]] = []
    distances: list[float] = []
    similarities: list[float] = []

    for top in _window_starts(scene_height, window_height, step_size):
        for left in _window_starts(scene_width, window_width, step_size):
            patch = scene_gray[top : top + window_height, left : left + window_width]
            patch_features = compute_hog_features(
                patch,
                orientations,
                pixels_per_cell,
                cells_per_block,
            )

            locations.append((left, top))
            distances.append(float(np.linalg.norm(template_features - patch_features)))

            denominator = np.linalg.norm(template_features) * np.linalg.norm(
                patch_features
            )
            similarities.append(
                float(np.dot(template_features, patch_features) / denominator)
                if denominator > 0
                else -1.0
            )

    euclidean_index = int(np.argmin(distances))
    cosine_index = int(np.argmax(similarities))

    return {
        "template_size": (window_width, window_height),
        "euclidean_location": locations[euclidean_index],
        "euclidean_distance": distances[euclidean_index],
        "cosine_location": locations[cosine_index],
        "cosine_similarity": similarities[cosine_index],
        "windows_checked": len(locations),
    }


def ask_box_colors(parent: tk.Misc) -> tuple[str, str] | None:
    """Prompt for the Euclidean and cosine match box colors."""
    dialog = tk.Toplevel(parent)
    dialog.title("HOG detection box colors")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    euclidean_color = tk.StringVar(value=EUCLIDEAN_BOX_COLOR)
    cosine_color = tk.StringVar(value=COSINE_BOX_COLOR)
    for row, (label, variable) in enumerate(
        (
            ("Euclidean-distance box", euclidean_color),
            ("Cosine-similarity box", cosine_color),
        )
    ):
        ttk.Label(dialog, text=label).grid(
            row=row, column=0, sticky="w", padx=12, pady=8
        )
        ttk.Combobox(
            dialog,
            textvariable=variable,
            values=BOX_COLORS,
            state="readonly",
            width=12,
        ).grid(row=row, column=1, padx=(8, 12), pady=8)

    result: tuple[str, str] | None = None

    def apply() -> None:
        nonlocal result
        result = (euclidean_color.get(), cosine_color.get())
        dialog.destroy()

    buttons = ttk.Frame(dialog)
    buttons.grid(row=2, column=0, columnspan=2, sticky="e", padx=12, pady=12)
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Apply", command=apply).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


def draw_detection_boxes(
    image: Image.Image,
    result: dict[str, object],
    euclidean_color: str = EUCLIDEAN_BOX_COLOR,
    cosine_color: str = COSINE_BOX_COLOR,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)

    width, height = result["template_size"]
    left, top = result["euclidean_location"]
    draw.rectangle(
        (left, top, left + width, top + height),
        outline=euclidean_color,
        width=3,
    )

    left, top = result["cosine_location"]
    draw.rectangle(
        (left, top, left + width, top + height),
        outline=cosine_color,
        width=3,
    )

    return output


class HOGPersonDetectionTool(ForensicsTool):
    tool_id = "hog_person_detection"
    title = "Person detection with HOG"
    category = "Matching & Descriptors"
    description = "Find a person-template sized object with sliding-window HOG."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        template_path = filedialog.askopenfilename(
            parent=parent,
            title="Select cropped person template",
            filetypes=IMAGE_TYPES,
        )
        if not template_path:
            return None

        with Image.open(template_path) as template_file:
            template = template_file.copy()

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
                ParameterField(
                    "step_size",
                    "Sliding-window step (pixels)",
                    1.0,
                    64.0,
                    1.0,
                    8.0,
                    show_slider=False,
                    no_threshold=True,
                ),
            ),
        )
        if parameters is None:
            return None

        box_colors = ask_box_colors(parent)
        if box_colors is None:
            return None
        euclidean_color, cosine_color = box_colors

        result = detect_with_hog(
            document.current,
            template,
            int(parameters["orientations"]),
            int(parameters["pixels_per_cell"]),
            int(parameters["cells_per_block"]),
            int(parameters["step_size"]),
        )
        output = draw_detection_boxes(
            document.current, result, euclidean_color, cosine_color
        )

        euclidean_x, euclidean_y = result["euclidean_location"]
        cosine_x, cosine_y = result["cosine_location"]

        return ToolResult(
            image=output,
            message="Completed HOG sliding-window detection.",
            details={
                "Operation": "HOG person detection",
                "Template size": (
                    f"{result['template_size'][0]} × {result['template_size'][1]}"
                ),
                "Windows checked": result["windows_checked"],
                "Euclidean match": f"x={euclidean_x}, y={euclidean_y}",
                "Euclidean distance": f"{result['euclidean_distance']:.4f}",
                "Euclidean box color": euclidean_color,
                "Cosine match": f"x={cosine_x}, y={cosine_y}",
                "Cosine similarity": f"{result['cosine_similarity']:.4f}",
                "Cosine box color": cosine_color,
            },
        )
