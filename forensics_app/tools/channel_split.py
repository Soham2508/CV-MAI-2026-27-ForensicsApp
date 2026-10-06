"""Display an image's color channels as stacked grayscale planes."""

from __future__ import annotations

import tkinter as tk

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def split_channels(image: Image.Image) -> tuple[Image.Image, tuple[str, ...]]:
    """Return a vertical grayscale contact sheet and its channel names."""
    if image.mode in ("1", "L", "I", "F"):
        channel_names = ("Intensity",)
        channels = (image.convert("L"),)
    elif image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        channel_names = ("Red", "Green", "Blue", "Alpha")
        channels = (
            rgba.getchannel("R"),
            rgba.getchannel("G"),
            rgba.getchannel("B"),
            rgba.getchannel("A"),
        )
    else:
        rgb = image.convert("RGB")
        channel_names = ("Red", "Green", "Blue")
        channels = (
            rgb.getchannel("R"),
            rgb.getchannel("G"),
            rgb.getchannel("B"),
        )

    width, height = image.size
    output = Image.new("L", (width, height * len(channels)))
    for index, channel in enumerate(channels):
        output.paste(channel, (0, index * height))

    return output, channel_names


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Split color channels"
    category = "Image analysis"
    description = "View each color channel as a stacked grayscale plane."

    def run(self, _parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None  # guarded by the main window
        output, channel_names = split_channels(document.current)
        return ToolResult(
            image=output,
            message="Displayed the image channels as grayscale planes.",
            details={
                "Channels": ", ".join(channel_names),
                "Input size": f"{document.current.width} × {document.current.height}",
                "Display": "Grayscale planes stacked top to bottom",
            },
        )
