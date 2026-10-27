"""Register course functionality here so it appears in the sidebar."""

from .best_image_match_in_dir import BestImageMatchInDirTool
from .canny import CannyTool
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretch import CLAHETool, ContrastStretchTool, HistogramEqualizationTool
from .convolution import ConvolutionTool
from .corner_detection import CornerDetectionTool
from .descriptor_extraction import DescriptorExtractionTool
from .edge_overlay import EdgeOverlayTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .hog_extraction_visualization import HOGVisualizationTool
from .hog_person_detection import IMAGE_TYPES, HOGPersonDetectionTool, detect_with_hog
from .image_info import ImageInfoTool
from .image_matching import ImageMatchingTool
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
            CornerDetectionTool(),
            DescriptorExtractionTool(),
            ImageMatchingTool(),
            BestImageMatchInDirTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
