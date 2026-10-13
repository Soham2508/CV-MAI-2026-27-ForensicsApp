"""A deliberately small example feature for students to imitate."""

from __future__ import annotations

import tkinter as tk

from PIL import ImageOps

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult


class GrayscaleTool(ForensicsTool):
    tool_id = "grayscale"
    title = "Convert to grayscale"
    category = "Color & Tone"
    description = "Convert the working image to an 8-bit grayscale image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        output = ImageOps.grayscale(document.current)
        return ToolResult(
            image=output,
            message="Converted the image to grayscale.",
            details={"Operation": "Grayscale", "Output mode": output.mode},
        )
