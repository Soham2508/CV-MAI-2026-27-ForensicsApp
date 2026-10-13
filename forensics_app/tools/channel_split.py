"""Display selected RGB channels as grayscale planes."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

CHANNEL_OPTIONS = ("R", "G", "B", "All three")
CHANNEL_DETAILS = {
    "R": ("Red", 0),
    "G": ("Green", 1),
    "B": ("Blue", 2),
}


def split_channels(
    image: Image.Image, selection: str = "All three"
) -> tuple[Image.Image, tuple[str, ...]]:
    """Return the selected channel plane(s) as a grayscale PIL image."""
    if selection not in CHANNEL_OPTIONS:
        raise ValueError(f"Unsupported channel selection: {selection}")

    rgb = np.asarray(image.convert("RGB"))
    if selection == "All three":
        channel_names = ("Red", "Green", "Blue")
        output_array = np.concatenate(
            (rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]), axis=0
        )
    else:
        channel_name, channel_index = CHANNEL_DETAILS[selection]
        channel_names = (channel_name,)
        output_array = rgb[:, :, channel_index]

    return Image.fromarray(output_array), channel_names


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Split color channels"
    category = "Inspect & Analyze"
    description = "View one RGB channel or all three stacked top to bottom."
    menu_options = CHANNEL_OPTIONS

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None  # guarded by the main window
        return self._split_selected(document, "All three")

    def run_action(
        self, parent: tk.Misc, document: ImageDocument, action: str
    ) -> ToolResult:
        assert document.current is not None
        if action not in self.menu_options:
            raise ValueError(f"Unknown channel selection: {action}")
        return self._split_selected(document, action)

    @staticmethod
    def _split_selected(document: ImageDocument, selection: str) -> ToolResult:
        assert document.current is not None
        output, channel_names = split_channels(document.current, selection)
        display = (
            "Grayscale planes stacked top to bottom"
            if selection == "All three"
            else "Single grayscale plane"
        )
        return ToolResult(
            image=output,
            message=f"Displayed the {', '.join(channel_names)} channel(s).",
            details={
                "Channels": ", ".join(channel_names),
                "Input size": f"{document.current.width} × {document.current.height}",
                "Display": display,
            },
        )
