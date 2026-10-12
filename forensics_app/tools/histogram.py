"""Histogram tools for grayscale and RGB images."""

import matplotlib.pyplot as plt

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class GrayscaleHistogramTool(ForensicsTool):
    tool_id = "grayscale_histogram"
    title = "Grayscale histogram"
    category = "Image analysis"
    description = "Show the image's grayscale intensity histogram."

    def run(self, parent, document: ImageDocument) -> ToolResult:
        assert document.current is not None

        gray_image = document.current.convert("L")
        pixel_counts = gray_image.histogram()

        figure, axis = plt.subplots()
        axis.plot(range(256), pixel_counts, color="gray")
        axis.set_title("Grayscale histogram")
        axis.set_xlabel("Pixel intensity (0–255)")
        axis.set_ylabel("Number of pixels")
        axis.set_xlim(0, 255)

        figure.tight_layout()
        plt.show()

        return ToolResult(
            message="Displayed the grayscale histogram.",
            details={"Histogram": "Grayscale"},
        )


class RGBHistogramsTool(ForensicsTool):
    tool_id = "rgb_histograms"
    title = "RGB histograms"
    category = "Image analysis"
    description = "Show separate red, green, and blue histograms plus an overlay."

    def run(self, parent, document: ImageDocument) -> ToolResult:
        assert document.current is not None

        red_channel, green_channel, blue_channel = (
            document.current.convert("RGB").split()
        )

        red_counts = red_channel.histogram()
        green_counts = green_channel.histogram()
        blue_counts = blue_channel.histogram()
        intensities = range(256)

        figure, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Three separate plots
        channel_plots = [
            (axes[0, 0], red_counts, "Red histogram", "red"),
            (axes[0, 1], green_counts, "Green histogram", "green"),
            (axes[1, 0], blue_counts, "Blue histogram", "blue"),
        ]

        for axis, counts, title, color in channel_plots:
            axis.plot(intensities, counts, color=color)
            axis.set_title(title)
            axis.set_xlabel("Pixel intensity (0–255)")
            axis.set_ylabel("Number of pixels")
            axis.set_xlim(0, 255)

        # Fourth plot: all three channels overlaid
        overlay_axis = axes[1, 1]
        overlay_axis.plot(
            intensities, red_counts, color="red", label="Red"
        )
        overlay_axis.plot(
            intensities, green_counts, color="green", label="Green"
        )
        overlay_axis.plot(
            intensities, blue_counts, color="blue", label="Blue"
        )
        overlay_axis.set_title("RGB overlay")
        overlay_axis.set_xlabel("Pixel intensity (0–255)")
        overlay_axis.set_ylabel("Number of pixels")
        overlay_axis.set_xlim(0, 255)
        overlay_axis.legend()

        figure.tight_layout()
        plt.show()

        return ToolResult(
            message="Displayed separate RGB histograms and their overlay.",
            details={"Histogram": "Separate channels and RGB overlay"},
        )