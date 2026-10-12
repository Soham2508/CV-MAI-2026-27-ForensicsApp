from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image
from scipy import ndimage

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

KERNEL_DIRECTIONS = ("Horizontal", "Vertical", "2D")


def create_kernel(kernel_size: int, direction: str) -> np.ndarray:
    """Create a normalized box kernel in the selected direction."""
    if kernel_size < 1:
        raise ValueError("Kernel size must be a positive integer.")

    direction = direction.strip().lower()
    shapes = {
        "horizontal": (1, kernel_size),
        "vertical": (kernel_size, 1),
        "2d": (kernel_size, kernel_size),
    }
    if direction not in shapes:
        raise ValueError(f"Unsupported kernel direction: {direction}")

    shape = shapes[direction]
    return np.ones(shape, dtype=float) / (shape[0] * shape[1])


def apply_convolution(
    image: Image.Image,
    kernel_size: int,
    direction: str,
) -> Image.Image:
    """Apply a normalized directional box kernel to a grayscale image."""
    pixels = np.asarray(image.convert("L"), dtype=float)
    kernel = create_kernel(kernel_size, direction)
    result = ndimage.convolve(pixels, kernel)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def ask_convolution_options(parent: tk.Misc) -> tuple[int, str] | None:
    """Ask for kernel size and direction, or return None when cancelled."""
    dialog = tk.Toplevel(parent)
    dialog.title("Convolution")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    size_value = tk.StringVar(value="3")
    direction_value = tk.StringVar(value=KERNEL_DIRECTIONS[0])
    ttk.Label(dialog, text="Kernel size").grid(
        row=0, column=0, sticky="w", padx=12, pady=(12, 4)
    )
    ttk.Entry(dialog, textvariable=size_value, width=22).grid(
        row=0, column=1, padx=(8, 12), pady=(12, 4)
    )
    ttk.Label(dialog, text="Direction").grid(
        row=1, column=0, sticky="w", padx=12, pady=4
    )
    ttk.Combobox(
        dialog,
        textvariable=direction_value,
        values=KERNEL_DIRECTIONS,
        state="readonly",
        width=20,
    ).grid(row=1, column=1, padx=(8, 12), pady=4)

    result: tuple[int, str] | None = None

    def apply() -> None:
        nonlocal result
        try:
            kernel_size = int(size_value.get())
        except ValueError:
            messagebox.showerror(
                "Convolution",
                "Enter a positive whole-number kernel size.",
                parent=dialog,
            )
            return
        if kernel_size < 1:
            messagebox.showerror(
                "Convolution",
                "Kernel size must be a positive integer.",
                parent=dialog,
            )
            return
        result = (kernel_size, direction_value.get().lower())
        dialog.destroy()

    buttons = ttk.Frame(dialog)
    buttons.grid(row=2, column=0, columnspan=2, sticky="e", padx=12, pady=12)
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Apply", command=apply).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution"
    category = "Filtering"
    description = "Apply a directional mean kernel."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None
        options = ask_convolution_options(parent)
        if options is None:
            return None
        kernel_size, direction = options
        output = apply_convolution(document.current, kernel_size, direction)
        return ToolResult(
            image=output,
            message=f"Applied a size-{kernel_size} {direction} convolution.",
            details={
                "Operation": "Convolution",
                "Kernel size": str(kernel_size),
                "Direction": direction.title() if direction != "2d" else "2D",
            },
        )
