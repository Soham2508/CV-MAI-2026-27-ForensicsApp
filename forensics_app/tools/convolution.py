from tkinter import simpledialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def convolve_2d(image_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D convolution using edge padding."""
    kernel_height, kernel_width = kernel.shape

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = np.pad(
        image_array,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="edge",
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kernel_height, kernel_width),
    )

    # Flip the kernel because convolution reverses it before applying it.
    flipped_kernel = np.flip(kernel)
    return np.einsum("ijkl,kl->ij", windows, flipped_kernel)


def parse_kernel(text: str) -> np.ndarray:
    """Read rows of numbers separated by semicolons and commas."""
    rows = [
        [float(value.strip()) for value in row.split(",")]
        for row in text.split(";")
    ]

    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Kernel rows must all have the same number of values.")

    kernel = np.array(rows, dtype=float)

    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Use an odd number of kernel rows and columns.")

    if not np.isfinite(kernel).all():
        raise ValueError("Kernel values must be finite numbers.")

    return kernel

def convolve_image(image: Image.Image, kernel: np.ndarray) -> Image.Image:
    """Convolve a grayscale copy of the image and scale it for display."""
    gray = image.convert("L")
    pixels = np.asarray(gray, dtype=float)

    response = convolve_2d(pixels, kernel)
    response = np.abs(response)

    largest = response.max()
    if largest > 0:
        response = response / largest * 255

    return Image.fromarray(response.astype(np.uint8))

class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution"
    category = "Filtering"
    description = "Apply a Sobel preset or a custom directional kernel."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        choice = simpledialog.askstring(
            "Convolution",
            "Enter horizontal, vertical, or custom:",
            parent=parent,
        )

        if choice is None:
            return None

        choice = choice.strip().lower()

        if choice == "horizontal":
            kernel = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ], dtype=float)

        elif choice == "vertical":
            kernel = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ], dtype=float)

        elif choice == "custom":
            kernel_text = simpledialog.askstring(
                "Custom kernel",
                "Enter rows separated by semicolons and values by commas.\n"
                "Example: -1,0,1; -2,0,2; -1,0,1",
                parent=parent,
            )

            if kernel_text is None:
                return None

            kernel = parse_kernel(kernel_text)

            direction = simpledialog.askstring(
                "Kernel direction",
                "Enter horizontal or vertical:",
                parent=parent,
            )

            if direction is None:
                return None

            direction = direction.strip().lower()

            if direction == "vertical":
                kernel = kernel.T
            elif direction != "horizontal":
                raise ValueError("Direction must be horizontal or vertical.")

        else:
            raise ValueError("Enter horizontal, vertical, or custom.")

        assert document.current is not None

        gray = document.current.convert("L")
        image_array = np.asarray(gray, dtype=float)

        response = convolve_2d(image_array, kernel)

        # Edge filters can produce negative values. Use the absolute response
        # and scale it to 0-255 so the result can be displayed as an image.
        response = np.abs(response)
        largest = response.max()

        if largest > 0:
            response = response / largest * 255

        output = Image.fromarray(response.astype(np.uint8))

        return ToolResult(
            image=output,
            message=f"Applied {choice} convolution.",
            details={
                "Operation": "Convolution",
                "Choice": choice,
                "Kernel": str(kernel.tolist()),
            },
        )