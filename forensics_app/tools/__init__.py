"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretch import CLAHETool, ContrastStretchTool, HistogramEqualizationTool
from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry
from .histogram import GrayscaleHistogramTool, RGBHistogramsTool
from .ski_filters import SkiFiltersTool


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
            GrayscaleHistogramTool(),
            RGBHistogramsTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
