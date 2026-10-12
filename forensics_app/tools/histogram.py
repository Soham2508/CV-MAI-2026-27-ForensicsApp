"""Histogram tools for grayscale and RGB image channels."""

from __future__ import annotations

import tkinter as tk

import matplotlib.pyplot as plt

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

HISTOGRAM_OPTIONS = ("Greyscale", "R", "G", "B", "RGB combined")
CHANNEL_COLORS = {"R": "red", "G": "green", "B": "blue"}


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram"
    category = "Image analysis"
    description = "Choose a grayscale or color-channel histogram to display."
    menu_options = HISTOGRAM_OPTIONS

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        assert document.current is not None

        return self._show_histogram(document, "Greyscale")

    def run_action(
        self, parent: tk.Misc, document: ImageDocument, action: str
    ) -> ToolResult:
        assert document.current is not None
        if action not in self.menu_options:
            raise ValueError(f"Unknown histogram option: {action}")
        return self._show_histogram(document, action)

    @staticmethod
    def _show_histogram(document: ImageDocument, option: str) -> ToolResult:
        image = document.current
        assert image is not None

        figure, axis = plt.subplots()
        if option == "Greyscale":
            axis.plot(range(256), image.convert("L").histogram(), color="gray")
            title = "Greyscale histogram"
        else:
            channels = image.convert("RGB").split()
            channel_options = ("R", "G", "B")
            selected_channels = (
                channel_options if option == "RGB combined" else (option,)
            )
            for channel_name in selected_channels:
                channel = channels[channel_options.index(channel_name)]
                axis.plot(
                    range(256),
                    channel.histogram(),
                    color=CHANNEL_COLORS[channel_name],
                    label=channel_name,
                )
            title = (
                "RGB combined histogram"
                if option == "RGB combined"
                else f"{option} channel histogram"
            )
            if option == "RGB combined":
                axis.legend()

        axis.set_title(title)
        axis.set_xlabel("Pixel intensity (0–255)")
        axis.set_ylabel("Number of pixels")
        axis.set_xlim(0, 255)
        figure.tight_layout()
        plt.show()

        return ToolResult(
            message=f"Displayed the {option} histogram.",
            details={"Histogram": option},
        )
