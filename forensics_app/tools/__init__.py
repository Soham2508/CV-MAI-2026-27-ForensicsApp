"""Register course functionality here so it appears in the sidebar."""

from .canny import CannyTool
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretch import CLAHETool, ContrastStretchTool, HistogramEqualizationTool
from .convolution import ConvolutionTool
from .edge_overlay import EdgeOverlayTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry
from .sharpen import SharpenTool
from .ski_filters import SkiFiltersTool
from .hog_extraction_visualization import HOGVisualizationTool

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
            CannyTool(),
            EdgeOverlayTool(),
            HOGVisualizationTool(),
            HistogramTool(),
            SharpenTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
