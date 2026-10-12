import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.channel_split import ChannelSplitTool, split_channels
from forensics_app.tools.contrast_stretch import (
    CLAHETool,
    ContrastStretchTool,
    HistogramEqualizationTool,
    adaptive_histogram_equalization,
    contrast_stretch_image,
    equalize_histogram,
)
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.masking import MaskingTool
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

    def test_enhancement_tools_are_registered_by_default(self) -> None:
        tool_ids = {tool.tool_id for tool in build_tool_registry().all()}
        self.assertTrue(
            {
                ContrastStretchTool.tool_id,
                HistogramEqualizationTool.tool_id,
                CLAHETool.tool_id,
            }.issubset(tool_ids)
        )

    def test_contrast_stretch_uses_percentiles_and_preserves_input(self) -> None:
        image = Image.new("L", (4, 1))
        image.putdata([0, 64, 128, 255])

        output = contrast_stretch_image(image, 25, 75)

        self.assertEqual(
            list(output.getdata()),
            [(0, 0, 0), (36, 36, 36), (182, 182, 182), (255, 255, 255)],
        )
        self.assertEqual(list(image.getdata()), [0, 64, 128, 255])

    def test_contrast_stretch_rejects_invalid_percentile_order(self) -> None:
        with self.assertRaises(ValueError):
            contrast_stretch_image(Image.new("L", (1, 1)), 98, 2)

    def test_histogram_equalization_spreads_a_grayscale_ramp(self) -> None:
        image = Image.new("L", (4, 1))
        image.putdata([0, 64, 128, 255])

        output = equalize_histogram(image)

        self.assertEqual(
            list(output.getdata()),
            [(63, 63, 63), (127, 127, 127), (191, 191, 191), (255, 255, 255)],
        )
        self.assertEqual(list(image.getdata()), [0, 64, 128, 255])

    def test_histogram_equalization_preserves_constant_images(self) -> None:
        image = Image.new("L", (3, 2), 73)
        self.assertEqual(
            list(equalize_histogram(image).getdata()), [(255, 255, 255)] * 6
        )

    def test_histogram_equalization_applies_to_rgb_channels(self) -> None:
        image = Image.new("RGB", (2, 2), "red")

        output = equalize_histogram(image)

        self.assertEqual(output.mode, "RGB")
        red, green, blue = output.getpixel((0, 0))
        self.assertGreater(red, 240)
        self.assertGreater(green, 240)
        self.assertGreater(blue, 240)

    def test_clahe_converts_input_to_rgb(self) -> None:
        image = Image.new("RGBA", (16, 16))
        image.putdata(
            [(x * 16, y * 16, 100, 127) for y in range(16) for x in range(16)]
        )

        output = adaptive_histogram_equalization(image, clip_limit=0.03)

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 100, 127))

    def test_clahe_rejects_invalid_clip_limit(self) -> None:
        with self.assertRaises(ValueError):
            adaptive_histogram_equalization(Image.new("L", (8, 8)), clip_limit=0)

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

    def test_masking_keeps_only_pixels_brighter_than_threshold(self) -> None:
        image = Image.new("L", (3, 1))
        image.putdata([50, 135, 200])
        base = Image.new("RGB", (3, 1), "red")
        texture = Image.new("RGB", (3, 1), "blue")

        output = MaskingTool().composite_masked_texture(
            image, base, texture, threshold=135
        )

        self.assertEqual(
            list(output.getdata()),
            [(255, 0, 0), (255, 0, 0), (0, 0, 255)],
        )
        self.assertEqual(list(image.getdata()), [50, 135, 200])

    def test_masking_composites_texture_over_base_with_mask(self) -> None:
        mask_source = Image.new("L", (2, 1))
        mask_source.putdata([0, 255])
        base = Image.new("RGB", (2, 1), "red")
        texture = Image.new("RGB", (1, 1), "blue")

        output = MaskingTool().composite_masked_texture(
            mask_source, base, texture
        )

        self.assertEqual(output.size, base.size)
        self.assertEqual(list(output.getdata()), [(255, 0, 0), (0, 0, 255)])


if __name__ == "__main__":
    unittest.main()
