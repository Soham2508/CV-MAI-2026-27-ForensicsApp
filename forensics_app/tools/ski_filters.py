"""Spatial filters implemented with scikit-image."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image
from skimage.filters import gaussian, median, prewitt, scharr, sobel, unsharp_mask

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

FILTER_NAMES = (
    "Gaussian",
    "Prewitt",
    "Sobel",
    "Scharr",
    "Median",
    "Unsharp mask",
)
FILTER_MODES = ("Per-channel", "Direct RGB array")


def apply_ski_filter(
    image: Image.Image,
    filter_name: str,
    sigma: float = 1.0,
    mode: str = "per-channel",
) -> Image.Image:
    """Apply a selected filter to an RGB image and return an RGB PIL image.

    Per-channel mode filters each color plane independently. Direct mode passes
    the complete RGB array to the selected scikit-image function.
    """
    rgb = np.asarray(image.convert("RGB"))[:, :, :3]
    name = filter_name.lower().replace(" ", "_")

    if mode not in ("per-channel", "direct"):
        raise ValueError(f"Unsupported filter mode: {mode}")

    if name == "gaussian":
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Sigma must be a finite number greater than zero.")
        if mode == "direct":
            filtered = gaussian(rgb, sigma=sigma)
        else:
            filtered = gaussian(rgb, sigma=sigma, channel_axis=-1)
    elif name == "unsharp_mask":
        if mode == "direct":
            filtered = unsharp_mask(rgb)
        else:
            filtered = unsharp_mask(rgb, channel_axis=-1)
    elif name in ("prewitt", "sobel", "scharr", "median"):
        filter_function = {
            "prewitt": prewitt,
            "sobel": sobel,
            "scharr": scharr,
            "median": median,
        }[name]
        if mode == "direct":
            filtered = filter_function(rgb)
        else:
            filtered = np.stack(
                [filter_function(rgb[:, :, channel]) for channel in range(3)],
                axis=2,
            )
    else:
        raise ValueError(f"Unsupported filter: {filter_name}")

    if np.issubdtype(filtered.dtype, np.floating):
        filtered = np.clip(filtered * 255, 0, 255)
    return Image.fromarray(filtered.astype(np.uint8))


def ask_filter_options(parent: tk.Misc) -> tuple[str, float, str] | None:
    """Show filter options and return them, or None if the dialog is cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.title("Spatial filters")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    filter_choice = tk.StringVar(value=FILTER_NAMES[0])
    mode_choice = tk.StringVar(value=FILTER_MODES[0])
    sigma_value = tk.StringVar(value="1.0")
    ttk.Label(dialog, text="Filter").grid(
        row=0, column=0, sticky="w", padx=12, pady=(12, 4)
    )
    filter_menu = ttk.Combobox(
        dialog,
        textvariable=filter_choice,
        values=FILTER_NAMES,
        state="readonly",
        width=20,
    )
    filter_menu.grid(row=0, column=1, padx=(8, 12), pady=(12, 4))
    ttk.Label(dialog, text="Mode").grid(row=1, column=0, sticky="w", padx=12, pady=4)
    mode_menu = ttk.Combobox(
        dialog,
        textvariable=mode_choice,
        values=FILTER_MODES,
        state="readonly",
        width=20,
    )
    mode_menu.grid(row=1, column=1, padx=(8, 12), pady=4)

    sigma_label = ttk.Label(dialog, text="Sigma")
    sigma_entry = ttk.Entry(dialog, textvariable=sigma_value, width=22)

    def update_sigma_visibility(event: tk.Event | None = None) -> None:
        if event is not None and event.widget is not filter_menu:
            return
        if filter_choice.get() == "Gaussian":
            sigma_label.grid(row=2, column=0, sticky="w", padx=12, pady=4)
            sigma_entry.grid(row=2, column=1, padx=(8, 12), pady=4)
        else:
            sigma_label.grid_remove()
            sigma_entry.grid_remove()

    filter_menu.bind("<<ComboboxSelected>>", update_sigma_visibility)
    update_sigma_visibility()

    result: tuple[str, float, str] | None = None

    def apply() -> None:
        nonlocal result
        name = filter_choice.get()
        sigma = 1.0
        if name == "Gaussian":
            try:
                sigma = float(sigma_value.get())
            except ValueError:
                messagebox.showerror(
                    "Spatial filters",
                    "Enter a valid sigma value.",
                    parent=dialog,
                )
                return
            if not math.isfinite(sigma) or sigma <= 0:
                messagebox.showerror(
                    "Spatial filters",
                    "Sigma must be a finite number greater than zero.",
                    parent=dialog,
                )
                return
        mode = "per-channel" if mode_choice.get() == "Per-channel" else "direct"
        result = (name.lower().replace(" ", "_"), sigma, mode)
        dialog.destroy()

    buttons = ttk.Frame(dialog)
    buttons.grid(row=3, column=0, columnspan=2, sticky="e", padx=12, pady=12)
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Apply", command=apply).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class SkiFiltersTool(ForensicsTool):
    tool_id = "ski_filters"
    title = "Spatial filters"
    category = "Image enhancement"
    description = "Apply a Gaussian, Prewitt, Sobel, Scharr, median, or unsharp mask."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        options = ask_filter_options(parent)
        if options is None:
            return None
        filter_name, sigma, mode = options
        output = apply_ski_filter(document.current, filter_name, sigma=sigma, mode=mode)
        details = {
            "Operation": filter_name.replace("_", " ").title(),
            "Mode": FILTER_MODES[0] if mode == "per-channel" else FILTER_MODES[1],
        }
        if filter_name == "gaussian":
            details["Sigma"] = f"{sigma:g}"
        return ToolResult(
            image=output,
            message=f"Applied the {filter_name.replace('_', ' ').title()} filter.",
            details=details,
        )
