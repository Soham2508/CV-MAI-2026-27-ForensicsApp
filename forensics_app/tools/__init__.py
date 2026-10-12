"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretch import CLAHETool, ContrastStretchTool, HistogramEqualizationTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry
from .convolution import ConvolutionTool
from .ski_filters import SkiFiltersTool
from .sharpen import SharpenTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            ChannelSplitTool(),
            GrayscaleTool(),
            MaskingTool(),
            ContrastStretchTool(),
            HistogramEqualizationTool(),
            CLAHETool(),
            SkiFiltersTool(),
            ChannelSwapTool(),
            ConvolutionTool(),
            HistogramTool(),
            SharpenTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
