from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage.feature import canny

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class CannyTool(ForensicsTool):
    tool_id = "canny"
    title = "Canny edge detection"
    category = "Filtering & Edges"
    description = "Detect edges with the Canny method."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        sigma = simpledialog.askfloat(
            "Canny edge detection",
            "Sigma (try 1.0):",
            parent=parent,
            minvalue=0.0,
            initialvalue=1.0,
        )

        if sigma is None:
            return None  # The user cancelled the dialog.

        assert document.current is not None

        # Canny expects a single-channel image.
        gray = document.current.convert("L")
        pixels = np.asarray(gray, dtype=np.float32) / 255.0

        # The result is a Boolean array: True where an edge was found.
        edges = canny(pixels, sigma=sigma)

        # Convert True/False to white/black pixels for the app display.
        output = Image.fromarray(edges.astype(np.uint8) * 255)

        return ToolResult(
            image=output,
            message=f"Canny edge detection completed (sigma={sigma:g}).",
            details={
                "Operation": "Canny edge detection",
                "Sigma": sigma,
                "Edge pixels": int(edges.sum()),
            },
        )
