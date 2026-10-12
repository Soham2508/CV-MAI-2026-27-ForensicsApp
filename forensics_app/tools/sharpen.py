from tkinter import simpledialog

from PIL import ImageFilter

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class SharpenTool(ForensicsTool):
    tool_id = "sharpen"
    title = "Sharpen image"
    category = "Sharpening"
    description = "Sharpen the image with a configurable radius."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        radius = simpledialog.askfloat(
            "Sharpen image",
            "Radius (try 2.0):",
            parent=parent,
            minvalue=0.1,
            initialvalue=2.0,
        )

        if radius is None:
            return None  # The user cancelled the dialog.

        assert document.current is not None

        output = document.current.filter(
            ImageFilter.UnsharpMask(
                radius=radius,
                percent=150,
                threshold=3,
            )
        )

        return ToolResult(
            image=output,
            message=f"Sharpened the image (radius={radius:g}).",
            details={
                "Operation": "Unsharp Mask",
                "Radius": radius,
                "Strength": "150%",
                "Threshold": 3,
            },
        )
