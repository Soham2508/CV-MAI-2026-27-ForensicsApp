"""Register course functionality here so it appears in the sidebar."""

from .canny import CannyTool
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretch import CLAHETool, ContrastStretchTool, HistogramEqualizationTool
from .convolution import ConvolutionTool
from .edge_overlay import EdgeOverlayTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .hog_extraction_visualization import HOGVisualizationTool
from .hog_person_detection import IMAGE_TYPES, HOGPersonDetectionTool, detect_with_hog
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .ncc_image import NormalizedCrossCorrelationTool
from .registry import ToolRegistry
from .sharpen import SharpenTool
from .ski_filters import SkiFiltersTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            ChannelSplitTool(),
            GrayscaleTool(),
            MaskingTool(),
            NormalizedCrossCorrelationTool(),
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
            HOGPersonDetectionTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
