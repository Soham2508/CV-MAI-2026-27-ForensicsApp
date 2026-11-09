"""Register course functionality here so it appears in the sidebar."""

from .best_image_match_in_dir import BestImageMatchInDirTool
from .canny import CannyTool
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .codebook_construction import CodebookConstructionTool
from .contrast_stretch import CLAHETool, ContrastStretchTool, HistogramEqualizationTool
from .convolution import ConvolutionTool
from .corner_detection import CornerDetectionTool
from .descriptor_extraction import DescriptorExtractionTool
from .edge_overlay import EdgeOverlayTool
from .eigenface_trainer import EigenfaceTrainerTool
from .face_detection import FaceDetectionTool
from .face_recognition import FaceRecognitionTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .histogram_construction import HistogramConstructionTool
from .hog_extraction_visualization import HOGVisualizationTool
from .hog_person_detection import IMAGE_TYPES, HOGPersonDetectionTool, detect_with_hog
from .image_classification import ImageClassificationTool
from .image_info import ImageInfoTool
from .image_matching import ImageMatchingTool
from .masking import MaskingTool
from .ncc_image import NormalizedCrossCorrelationTool
from .registry import ToolRegistry
from .sharpen import SharpenTool
from .ski_filters import SkiFiltersTool
from .viola_jones_trainer import ViolaJonesTrainerTool
from .webcam_detection import WebcamFaceDetectionTool


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
            ViolaJonesTrainerTool(),
            FaceDetectionTool(),
            WebcamFaceDetectionTool(),
            EigenfaceTrainerTool(),
            FaceRecognitionTool(),
            CodebookConstructionTool(),
            HistogramConstructionTool(),
            ImageClassificationTool(),
        ]
    )


__all__ = [
    "IMAGE_TYPES",
    "ToolRegistry",
    "build_tool_registry",
    "detect_with_hog",
    "ViolaJonesTrainerTool",
    "FaceDetectionTool",
    "WebcamFaceDetectionTool",
    "EigenfaceTrainerTool",
    "FaceRecognitionTool",
    "CodebookConstructionTool",
    "HistogramConstructionTool",
    "ImageClassificationTool",
]
