"""Contrast stretching and histogram-equalization image tools."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

import numpy as np
from PIL import Image
from skimage import exposure

from forensics_app.core import ImageDocument
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


def _ask_parameters(
    parent: tk.Misc,
    title: str,
    fields: tuple[tuple[str, str, float, float, float, float], ...],
    validator: Callable[[dict[str, float]], str | None] | None = None,
) -> dict[str, float] | None:
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()
    controls: dict[str, ttk.Spinbox] = {}

    for row, (key, label, minimum, maximum, step, initial) in enumerate(fields):
        variable = tk.DoubleVar(value=initial)
        ttk.Label(dialog, text=label).grid(
            row=row * 2, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 0)
        )
        tk.Scale(
            dialog,
            from_=minimum,
            to=maximum,
            resolution=step,
            orient="horizontal",
            variable=variable,
            showvalue=False,
            length=260,
        ).grid(row=row * 2 + 1, column=0, sticky="ew", padx=(12, 4))
        control = ttk.Spinbox(
            dialog, from_=minimum, to=maximum, increment=step,
            textvariable=variable, width=8,
        )
        control.grid(row=row * 2 + 1, column=1, padx=(4, 12))
        controls[key] = control

    result: dict[str, float] | None = None

    def apply() -> None:
        nonlocal result
        try:
            values = {key: float(control.get()) for key, control in controls.items()}
        except ValueError:
            messagebox.showerror(title, "Enter a valid number.", parent=dialog)
            return
        if any(not np.isfinite(value) for value in values.values()):
            messagebox.showerror(title, "Enter a finite number.", parent=dialog)
            return
        error = validator(values) if validator else None
        if error:
            messagebox.showerror(title, error, parent=dialog)
            return
        result = values
        dialog.destroy()

    buttons = ttk.Frame(dialog)
    buttons.grid(row=len(fields) * 2, column=0, columnspan=2, sticky="e", padx=12, pady=12)
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Apply", command=apply).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast stretching"
    category = "Image enhancement"
    description = "Stretch RGB intensity values between adjustable percentiles."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        parameters = _ask_parameters(
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
        parameters = _ask_parameters(
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
