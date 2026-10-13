"""Overlay detected edges on the source image."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from skimage.feature import canny

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

EDGE_COLORS = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "White": (255, 255, 255),
}
LINE_WIDTH_RANGE = (1, 10)
CANNY_SIGMA_DEFAULT = 1.0
CANNY_THRESHOLD_DEFAULTS = (0.1, 0.2)


def overlay_edges(
    image: Image.Image,
    color: str = "Red",
    line_width: int = 2,
    sigma: float = CANNY_SIGMA_DEFAULT,
    low_threshold: float = CANNY_THRESHOLD_DEFAULTS[0],
    high_threshold: float = CANNY_THRESHOLD_DEFAULTS[1],
) -> Image.Image:
    """Return the source image with Canny edges drawn in the selected color."""
    if color not in EDGE_COLORS:
        raise ValueError(f"Unsupported edge color: {color}")
    if (
        type(line_width) is not int
        or not LINE_WIDTH_RANGE[0] <= line_width <= LINE_WIDTH_RANGE[1]
    ):
        raise ValueError(
            f"Line width must be an integer from {LINE_WIDTH_RANGE[0]} "
            f"to {LINE_WIDTH_RANGE[1]}."
        )
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Sigma must be a finite number greater than or equal to zero.")
    if (
        not math.isfinite(low_threshold)
        or not math.isfinite(high_threshold)
        or not 0 <= low_threshold <= high_threshold <= 1
    ):
        raise ValueError(
            "Thresholds must be between 0 and 1, with low not exceeding high."
        )

    rgb = np.asarray(image.convert("RGB"))
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    edges = canny(
        grayscale,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    if line_width > 1:
        edges = maximum_filter(edges, size=line_width, mode="constant", cval=0)

    output = np.array(rgb, copy=True)
    output[edges] = EDGE_COLORS[color]
    return Image.fromarray(output)


def ask_edge_options(parent: tk.Misc) -> tuple[str, int, float, float, float] | None:
    """Prompt for edge display and Canny options, or return None when cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.title("Edge overlay")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    color_choice = tk.StringVar(value="Red")
    ttk.Label(dialog, text="Edge color").grid(
        row=0, column=0, sticky="w", padx=12, pady=(12, 4)
    )
    ttk.Combobox(
        dialog,
        textvariable=color_choice,
        values=tuple(EDGE_COLORS),
        state="readonly",
        width=16,
    ).grid(row=0, column=1, padx=(8, 12), pady=(12, 4))

    sigma_value = tk.DoubleVar(value=CANNY_SIGMA_DEFAULT)
    ttk.Label(dialog, text="Sigma").grid(row=1, column=0, sticky="w", padx=12, pady=4)
    ttk.Spinbox(
        dialog,
        textvariable=sigma_value,
        from_=0.0,
        increment=0.1,
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=(8, 12), pady=4)

    low_value = tk.DoubleVar(value=CANNY_THRESHOLD_DEFAULTS[0])
    ttk.Label(dialog, text="Low threshold").grid(
        row=2, column=0, sticky="w", padx=12, pady=4
    )
    tk.Scale(
        dialog,
        from_=0.0,
        to=1.0,
        resolution=0.01,
        orient="horizontal",
        variable=low_value,
        showvalue=True,
        length=200,
    ).grid(row=2, column=1, padx=(8, 12), pady=4)

    high_value = tk.DoubleVar(value=CANNY_THRESHOLD_DEFAULTS[1])
    ttk.Label(dialog, text="High threshold").grid(
        row=3, column=0, sticky="w", padx=12, pady=4
    )
    tk.Scale(
        dialog,
        from_=0.0,
        to=1.0,
        resolution=0.01,
        orient="horizontal",
        variable=high_value,
        showvalue=True,
        length=200,
    ).grid(row=3, column=1, padx=(8, 12), pady=4)

    width_value = tk.IntVar(value=2)
    ttk.Label(dialog, text="Line width").grid(
        row=4, column=0, sticky="w", padx=12, pady=4
    )
    tk.Scale(
        dialog,
        from_=LINE_WIDTH_RANGE[0],
        to=LINE_WIDTH_RANGE[1],
        resolution=1,
        orient="horizontal",
        variable=width_value,
        showvalue=True,
        length=200,
    ).grid(row=4, column=1, padx=(8, 12), pady=4)

    result: tuple[str, int, float, float, float] | None = None

    def apply() -> None:
        nonlocal result
        try:
            sigma = sigma_value.get()
            low_threshold = low_value.get()
            high_threshold = high_value.get()
        except (ValueError, tk.TclError):
            messagebox.showerror(
                "Edge overlay", "Enter valid Canny parameter values.", parent=dialog
            )
            return
        if not math.isfinite(sigma) or sigma < 0:
            messagebox.showerror(
                "Edge overlay",
                "Sigma must be a finite number greater than or equal to zero.",
                parent=dialog,
            )
            return
        if (
            not math.isfinite(low_threshold)
            or not math.isfinite(high_threshold)
            or not 0 <= low_threshold <= high_threshold <= 1
        ):
            messagebox.showerror(
                "Edge overlay",
                "Thresholds must be between 0 and 1, with low not exceeding high.",
                parent=dialog,
            )
            return
        result = (
            color_choice.get(),
            width_value.get(),
            sigma,
            low_threshold,
            high_threshold,
        )
        dialog.destroy()

    buttons = ttk.Frame(dialog)
    buttons.grid(row=5, column=0, columnspan=2, sticky="e", padx=12, pady=12)
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Apply", command=apply).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class EdgeOverlayTool(ForensicsTool):
    tool_id = "edge_overlay"
    title = "Edge overlay"
    category = "Filtering & Edges"
    description = "Draw colored Canny edges over the original image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        options = ask_edge_options(parent)
        if options is None:
            return None

        color, line_width, sigma, low_threshold, high_threshold = options
        output = overlay_edges(
            document.current,
            color,
            line_width,
            sigma,
            low_threshold,
            high_threshold,
        )
        return ToolResult(
            image=output,
            message=f"Overlaid {color.lower()} edges at {line_width}px width.",
            details={
                "Operation": "Edge overlay",
                "Edge color": color,
                "Line width": f"{line_width} px",
                "Sigma": sigma,
                "Low threshold": low_threshold,
                "High threshold": high_threshold,
            },
        )
