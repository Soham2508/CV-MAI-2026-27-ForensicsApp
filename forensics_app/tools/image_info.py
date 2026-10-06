"""Example of an analysis tool that does not alter the image."""

from __future__ import annotations

import tkinter as tk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ImageInfoTool(ForensicsTool):
    tool_id = "image_info"
    title = "Inspect image"
    category = "Starter tools"
    description = "Report basic properties without changing the image."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None
        image = document.current
        return ToolResult(
            message="Image properties updated in the Results panel.",
            details={
                "Width": f"{image.width} px",
                "Height": f"{image.height} px",
                "Mode": image.mode,
                "Format": document.path.suffix.upper().lstrip(".") if document.path else "Unknown",
            },
        )

#hello