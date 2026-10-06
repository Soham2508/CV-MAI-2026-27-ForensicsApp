import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.channel_split import ChannelSplitTool, split_channels
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.registry import ToolRegistry


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        """Test that the GrayscaleTool returns a new image and does not change the document's current image."""
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])

    def test_channel_split_stacks_rgb_planes_without_mutating_input(self) -> None:
        image = Image.new("RGB", (2, 1))
        image.putdata([(10, 20, 30), (40, 50, 60)])

        output, names = split_channels(image)

        self.assertEqual(names, ("Red", "Green", "Blue"))
        self.assertEqual(output.mode, "L")
        self.assertEqual(output.size, (2, 3))
        self.assertEqual(list(output.getdata()), [10, 40, 20, 50, 30, 60])
        self.assertEqual(image.mode, "RGB")

    def test_channel_split_includes_alpha_and_grayscale_intensity(self) -> None:
        rgba = Image.new("RGBA", (1, 1), (10, 20, 30, 40))
        output, names = split_channels(rgba)
        self.assertEqual(names, ("Red", "Green", "Blue", "Alpha"))
        self.assertEqual(list(output.getdata()), [10, 20, 30, 40])

        grayscale = Image.new("L", (1, 1), 73)
        output, names = split_channels(grayscale)
        self.assertEqual(names, ("Intensity",))
        self.assertEqual(output.getpixel((0, 0)), 73)

    def test_channel_split_tool_is_registered_by_default(self) -> None:
        tool_ids = {tool.tool_id for tool in build_tool_registry().all()}
        self.assertIn(ChannelSplitTool.tool_id, tool_ids)


if __name__ == "__main__":
    unittest.main()
