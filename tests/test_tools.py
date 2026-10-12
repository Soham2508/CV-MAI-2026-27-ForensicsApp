from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.channel_split import ChannelSplitTool, split_channels
from forensics_app.tools.channel_swap import ChannelSwapTool, swap_channels
from forensics_app.tools.contrast_stretch import (
    CLAHETool,
    ContrastStretchTool,
    HistogramEqualizationTool,
    adaptive_histogram_equalization,
    contrast_stretch_image,
    equalize_histogram,
)
from forensics_app.tools.convolution import (
    ConvolutionTool,
    apply_convolution,
    create_kernel,
)
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.histogram import HISTOGRAM_OPTIONS, HistogramTool
from forensics_app.tools.masking import MaskingTool
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.ski_filters import SkiFiltersTool, apply_ski_filter


class ToolTests(unittest.TestCase):
    def test_channel_swap_exchanges_any_rgb_pair_without_mutating_input(self) -> None:
        image = Image.new("RGB", (2, 1))
        image.putdata([(10, 20, 30), (40, 50, 60)])

        output = swap_channels(image, "Green", "Blue")

        self.assertEqual(list(output.getdata()), [(10, 30, 20), (40, 60, 50)])
        self.assertEqual(list(image.getdata()), [(10, 20, 30), (40, 50, 60)])

    def test_channel_swap_rejects_invalid_or_identical_channels(self) -> None:
        image = Image.new("RGB", (1, 1))
        with self.assertRaises(ValueError):
            swap_channels(image, "Alpha", "Red")
        with self.assertRaises(ValueError):
            swap_channels(image, "Red", "Red")

    def test_channel_swap_tool_prompts_for_channels_and_reports_selection(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), (10, 20, 30))

        with patch(
            "forensics_app.tools.channel_swap.ask_channel_pair",
            return_value=("Red", "Green"),
        ):
            result = ChannelSwapTool().run(None, document)

        self.assertEqual(result.image.getpixel((0, 0)), (20, 10, 30))
        self.assertEqual(result.details["Channels"], "Red-Green")

    def test_channel_swap_tool_returns_none_when_dialog_is_cancelled(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1))

        with patch(
            "forensics_app.tools.channel_swap.ask_channel_pair", return_value=None
        ):
            result = ChannelSwapTool().run(None, document)

        self.assertIsNone(result)

    def test_registry_registers_one_channel_swap_tool(self) -> None:
        tools = build_tool_registry().all()
        swaps = [tool for tool in tools if tool.category == "Color Swap"]
        self.assertEqual(len(swaps), 1)
        self.assertIsInstance(swaps[0], ChannelSwapTool)

    def test_registry_registers_one_histogram_tool_with_all_options(self) -> None:
        tools = build_tool_registry().all()
        histograms = [tool for tool in tools if isinstance(tool, HistogramTool)]

        self.assertEqual(len(histograms), 1)
        self.assertEqual(histograms[0].menu_options, HISTOGRAM_OPTIONS)

    def test_histogram_actions_plot_selected_channels(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (2, 1))
        document.current.putdata([(10, 20, 30), (40, 50, 60)])
        figure = MagicMock()
        axis = MagicMock()

        with (
            patch(
                "forensics_app.tools.histogram.plt.subplots",
                return_value=(figure, axis),
            ),
            patch("forensics_app.tools.histogram.plt.show"),
        ):
            for option, expected_colors in (
                ("Greyscale", ["gray"]),
                ("R", ["red"]),
                ("G", ["green"]),
                ("B", ["blue"]),
                ("RGB combined", ["red", "green", "blue"]),
            ):
                with self.subTest(option=option):
                    axis.reset_mock()
                    result = HistogramTool().run_action(None, document, option)

                    self.assertEqual(result.details["Histogram"], option)
                    self.assertEqual(
                        [call.kwargs["color"] for call in axis.plot.call_args_list],
                        expected_colors,
                    )
                    if option == "RGB combined":
                        axis.legend.assert_called_once()
                    else:
                        axis.legend.assert_not_called()

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

    def test_spatial_filter_tool_is_registered_by_default(self) -> None:
        tool_ids = {tool.tool_id for tool in build_tool_registry().all()}
        self.assertIn(SkiFiltersTool.tool_id, tool_ids)

    def test_spatial_filters_return_rgb_images_without_mutating_input(self) -> None:
        image = Image.new("RGB", (5, 5), (32, 96, 160))
        image.putpixel((2, 2), (255, 0, 64))

        for filter_name in (
            "gaussian",
            "prewitt",
            "sobel",
            "scharr",
            "median",
            "unsharp_mask",
        ):
            with self.subTest(filter_name=filter_name):
                output = apply_ski_filter(image, filter_name, sigma=1.5)
                self.assertEqual(output.mode, "RGB")
                self.assertEqual(output.size, image.size)
                self.assertEqual(image.getpixel((2, 2)), (255, 0, 64))

    def test_spatial_filters_support_direct_rgb_mode(self) -> None:
        image = Image.new("RGB", (5, 5), (32, 96, 160))
        image.putpixel((2, 2), (255, 0, 64))

        for filter_name in (
            "gaussian",
            "prewitt",
            "sobel",
            "scharr",
            "median",
            "unsharp_mask",
        ):
            with self.subTest(filter_name=filter_name):
                output = apply_ski_filter(image, filter_name, mode="direct")
                self.assertEqual(output.mode, "RGB")
                self.assertEqual(output.size, image.size)

    def test_spatial_filter_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            apply_ski_filter(Image.new("RGB", (2, 2)), "sobel", mode="unknown")

    def test_convolution_kernels_match_direction_and_are_normalized(self) -> None:
        expected_shapes = {
            "horizontal": (1, 3),
            "vertical": (3, 1),
            "2d": (3, 3),
        }
        for direction, shape in expected_shapes.items():
            with self.subTest(direction=direction):
                kernel = create_kernel(3, direction)
                self.assertEqual(kernel.shape, shape)
                self.assertAlmostEqual(float(kernel.sum()), 1.0)

    def test_convolution_rejects_invalid_kernel_options(self) -> None:
        with self.assertRaises(ValueError):
            create_kernel(0, "horizontal")
        with self.assertRaises(ValueError):
            create_kernel(3, "diagonal")

    def test_convolution_preserves_constant_grayscale_intensity(self) -> None:
        image = Image.new("RGB", (5, 4), (73, 73, 73))

        output = apply_convolution(image, 3, "2d")

        self.assertEqual(output.mode, "L")
        self.assertEqual(output.size, image.size)
        self.assertEqual(list(output.getdata()), [73] * 20)

    def test_convolution_tool_uses_selected_options(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (3, 3), 90)
        with patch(
            "forensics_app.tools.convolution.ask_convolution_options",
            return_value=(5, "vertical"),
        ) as ask_options:
            result = ConvolutionTool().run(None, document)

        ask_options.assert_called_once_with(None)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.details["Kernel size"], "5")
        self.assertEqual(result.details["Direction"], "Vertical")

    def test_convolution_tool_returns_none_when_dialog_is_cancelled(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (2, 2))
        with patch(
            "forensics_app.tools.convolution.ask_convolution_options",
            return_value=None,
        ):
            result = ConvolutionTool().run(None, document)

        self.assertIsNone(result)

    def test_gaussian_filter_rejects_nonpositive_sigma(self) -> None:
        with self.assertRaises(ValueError):
            apply_ski_filter(Image.new("RGB", (2, 2)), "gaussian", sigma=0)

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

        output = MaskingTool().composite_masked_texture(mask_source, base, texture)

        self.assertEqual(output.size, base.size)
        self.assertEqual(list(output.getdata()), [(255, 0, 0), (0, 0, 255)])

    def test_masking_tool_uses_adjustable_threshold(self) -> None:
        with TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.png"
            texture_path = Path(directory) / "texture.png"
            Image.new("RGB", (3, 1), "red").save(base_path)
            Image.new("RGB", (3, 1), "blue").save(texture_path)
            document = ImageDocument()
            document.current = Image.new("L", (3, 1))
            document.current.putdata([50, 135, 200])

            with (
                patch(
                    "forensics_app.tools.masking.filedialog.askopenfilename",
                    side_effect=[str(base_path), str(texture_path)],
                ),
                patch(
                    "forensics_app.tools.masking.ask_parameters",
                    return_value={"threshold": 135.0},
                ) as ask_parameters,
            ):
                result = MaskingTool().run(None, document)

        self.assertEqual(
            list(result.image.getdata()),
            [(255, 0, 0), (255, 0, 0), (0, 0, 255)],
        )
        ask_parameters.assert_called_once_with(
            None,
            "Mask threshold",
            (("threshold", "Threshold", 0, 255, 1, 10),),
        )


if __name__ == "__main__":
    unittest.main()
