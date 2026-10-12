"""Register course functionality here so it appears in the sidebar."""

from .masking import MaskingTool

from .channel_split import ChannelSplitTool
from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            ChannelSplitTool(),
            GrayscaleTool(),
            MaskingTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
