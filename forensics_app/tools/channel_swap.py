from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class _ChannelSwapTool(ForensicsTool):
    category = "Color Swap"
    first_channel = 0
    second_channel = 1
    channel_names = ""

    def run(self, parent, document: ImageDocument) -> ToolResult:
        assert document.current is not None

        channels = list(document.current.convert("RGB").split())
        channels[self.first_channel], channels[self.second_channel] = (
            channels[self.second_channel],
            channels[self.first_channel],
        )

        output = Image.merge("RGB", channels)

        return ToolResult(
            image=output,
            message=f"Swapped {self.channel_names}.",
            details={"Operation": "Channel swap", "Channels": self.channel_names},
        )


class SwapRedBlueTool(_ChannelSwapTool):
    tool_id = "swap_red_blue"
    title = "Swap R-B"
    first_channel = 0
    second_channel = 2
    channel_names = "R-B"


class SwapRedGreenTool(_ChannelSwapTool):
    tool_id = "swap_red_green"
    title = "Swap R-G"
    first_channel = 0
    second_channel = 1
    channel_names = "R-G"


class SwapGreenBlueTool(_ChannelSwapTool):
    tool_id = "swap_green_blue"
    title = "Swap G-B"
    first_channel = 1
    second_channel = 2
    channel_names = "G-B"
