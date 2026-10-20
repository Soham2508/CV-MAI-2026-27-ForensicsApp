from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from skimage.transform import rotate as skimage_rotate

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.canny import CannyTool
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
from forensics_app.tools.edge_overlay import EdgeOverlayTool, overlay_edges
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.histogram import HISTOGRAM_OPTIONS, HistogramTool
from forensics_app.tools.masking import MaskingTool
from forensics_app.tools.ncc_image import (
    NormalizedCrossCorrelationTool,
    normalized_cross_correlation,
)
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.sharpen import SharpenTool, sharpen_image
from forensics_app.tools.ski_filters import SkiFiltersTool, apply_ski_filter
from forensics_app.utils.parameter_dialog import ParameterField


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

    def test_canny_tool_uses_configured_thresholds(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (2, 2), "white")
        edges = np.array([[True, False], [False, True]])

        with (
            patch(
                "forensics_app.tools.canny.ask_parameters",
                return_value={"sigma": 18.0, "low": 0.15, "high": 0.45},
            ) as ask_parameters,
            patch("forensics_app.tools.canny.canny", return_value=edges) as canny,
        ):
            result = CannyTool().run(None, document)

        ask_parameters.assert_called_once()
        _, title, fields, validator = ask_parameters.call_args.args
        self.assertEqual(title, "Canny edge detection")
        self.assertEqual(
            fields,
            (
                ParameterField(
                    "sigma",
                    "Sigma",
                    0.0,
                    10.0,
                    0.1,
                    1.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ParameterField(
                    "low",
                    "Low threshold",
                    0.0,
                    1.0,
                    0.01,
                    0.1,
                    show_input=True,
                ),
                ParameterField(
                    "high",
                    "High threshold",
                    0.0,
                    1.0,
                    0.01,
                    0.2,
                    show_input=True,
                ),
            ),
        )
        self.assertIsNone(validator({"low": 0.2, "high": 0.2}))
        self.assertIsNotNone(validator({"low": 0.3, "high": 0.2}))
        canny.assert_called_once()
        self.assertEqual(canny.call_args.kwargs["sigma"], 18.0)
        self.assertEqual(canny.call_args.kwargs["low_threshold"], 0.15)
        self.assertEqual(canny.call_args.kwargs["high_threshold"], 0.45)
        self.assertEqual(result.details["Low threshold"], 0.15)
        self.assertEqual(result.details["High threshold"], 0.45)

    def test_canny_tool_returns_none_when_threshold_dialog_is_cancelled(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1))

        with (
            patch("forensics_app.tools.canny.ask_parameters", return_value=None),
            patch("forensics_app.tools.canny.canny") as canny,
        ):
            result = CannyTool().run(None, document)

        self.assertIsNone(result)
        canny.assert_not_called()

    def test_edge_overlay_draws_selected_color_over_original(self) -> None:
        image = Image.new("RGB", (5, 5), "white")
        edges = np.zeros((5, 5), dtype=bool)
        edges[2, 2] = True

        with patch(
            "forensics_app.tools.edge_overlay.canny", return_value=edges
        ) as canny:
            output = overlay_edges(
                image,
                "Magenta",
                sigma=1.5,
                low_threshold=0.15,
                high_threshold=0.45,
            )

        self.assertEqual(output.size, image.size)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.getpixel((2, 2)), (255, 0, 255))
        self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(image.getpixel((2, 2)), (255, 255, 255))
        self.assertEqual(canny.call_args.kwargs["sigma"], 1.5)
        self.assertEqual(canny.call_args.kwargs["low_threshold"], 0.15)
        self.assertEqual(canny.call_args.kwargs["high_threshold"], 0.45)

    def test_edge_overlay_expands_edges_to_selected_line_width(self) -> None:
        image = Image.new("RGB", (5, 5), "white")
        edges = np.zeros((5, 5), dtype=bool)
        edges[2, 2] = True

        with patch("forensics_app.tools.edge_overlay.canny", return_value=edges):
            output = overlay_edges(image, "Red", line_width=3)

        self.assertEqual(output.getpixel((1, 1)), (255, 0, 0))
        self.assertEqual(output.getpixel((3, 3)), (255, 0, 0))
        self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))

    def test_edge_overlay_rejects_invalid_color_and_line_width(self) -> None:
        image = Image.new("RGB", (1, 1))
        with self.assertRaises(ValueError):
            overlay_edges(image, "Orange")
        with self.assertRaises(ValueError):
            overlay_edges(image, "Red", line_width=0)
        with self.assertRaises(ValueError):
            overlay_edges(image, "Red", sigma=-1)
        with self.assertRaises(ValueError):
            overlay_edges(image, "Red", low_threshold=0.8, high_threshold=0.2)

    def test_edge_overlay_tool_uses_configured_canny_options(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), "white")

        with (
            patch(
                "forensics_app.tools.edge_overlay.ask_edge_options",
                return_value=("Blue", 4, 2.5, 0.15, 0.45),
            ),
            patch(
                "forensics_app.tools.edge_overlay.overlay_edges",
                return_value=document.current,
            ) as apply_overlay,
        ):
            result = EdgeOverlayTool().run(None, document)

        apply_overlay.assert_called_once_with(
            document.current, "Blue", 4, 2.5, 0.15, 0.45
        )
        self.assertEqual(result.details["Edge color"], "Blue")
        self.assertEqual(result.details["Line width"], "4 px")
        self.assertEqual(result.details["Sigma"], 2.5)
        self.assertEqual(result.details["Low threshold"], 0.15)
        self.assertEqual(result.details["High threshold"], 0.45)

    def test_edge_overlay_tool_returns_none_when_cancelled(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1))

        with patch(
            "forensics_app.tools.edge_overlay.ask_edge_options", return_value=None
        ):
            result = EdgeOverlayTool().run(None, document)

        self.assertIsNone(result)

    def test_edge_overlay_tool_is_registered_by_default(self) -> None:
        tool_ids = {tool.tool_id for tool in build_tool_registry().all()}
        self.assertIn(EdgeOverlayTool.tool_id, tool_ids)

    def test_registry_registers_one_channel_swap_tool(self) -> None:
        tools = build_tool_registry().all()
        swaps = [tool for tool in tools if isinstance(tool, ChannelSwapTool)]
        self.assertEqual(len(swaps), 1)
        self.assertEqual(swaps[0].category, "Color & Tone")
        self.assertIsInstance(swaps[0], ChannelSwapTool)

    def test_registry_registers_one_histogram_tool_with_all_options(self) -> None:
        tools = build_tool_registry().all()
        histograms = [tool for tool in tools if isinstance(tool, HistogramTool)]

        self.assertEqual(len(histograms), 1)
        self.assertEqual(histograms[0].menu_options, HISTOGRAM_OPTIONS)

    def test_normalized_cross_correlation_returns_same_size_grayscale_image(
        self,
    ) -> None:
        pixels = np.arange(12 * 12, dtype=np.uint8).reshape((12, 12))
        original = Image.fromarray(pixels)
        template = Image.fromarray(pixels[4:8, 5:9])

        output = normalized_cross_correlation(original, template)

        self.assertEqual(output.size, original.size)
        self.assertEqual(output.mode, "L")
        self.assertEqual(np.asarray(output).max(), 255)

    def test_normalized_cross_correlation_rejects_larger_template(self) -> None:
        original = Image.new("L", (2, 2))
        template = Image.new("L", (3, 2))

        with self.assertRaisesRegex(ValueError, "must not be larger"):
            normalized_cross_correlation(original, template)

    def test_normalized_cross_correlation_rejects_invalid_rotation_angle(self) -> None:
        image = Image.new("L", (4, 4))

        for angle in (-181, 181, float("nan")):
            with (
                self.subTest(angle=angle),
                self.assertRaisesRegex(
                    ValueError,
                    "Rotation angle",
                ),
            ):
                normalized_cross_correlation(image, image, angle)

    def test_normalized_cross_correlation_tool_selects_template_image(self) -> None:
        with TemporaryDirectory() as directory:
            template_path = Path(directory) / "template.png"
            pixels = np.arange(64, dtype=np.uint8).reshape((8, 8))
            Image.fromarray(pixels[2:5, 3:6]).save(template_path)
            document = ImageDocument()
            document.current = Image.fromarray(pixels)

            with (
                patch(
                    "forensics_app.tools.ncc_image.filedialog.askopenfilename",
                    return_value=str(template_path),
                ) as askopenfilename,
                patch(
                    "forensics_app.tools.ncc_image.ask_parameters",
                    return_value={"angle": 45.0},
                ) as ask_parameters,
                patch(
                    "forensics_app.tools.ncc_image.rotate",
                    wraps=skimage_rotate,
                ) as rotate_template,
            ):
                result = NormalizedCrossCorrelationTool().run(None, document)

        askopenfilename.assert_called_once()
        self.assertEqual(
            askopenfilename.call_args.kwargs["title"],
            "Select template image",
        )
        ask_parameters.assert_called_once_with(
            None,
            "Template rotation",
            (("angle", "Rotation angle (degrees)", -180, 180, 1, 0),),
        )
        self.assertEqual(rotate_template.call_args.kwargs["angle"], 45.0)
        self.assertEqual(result.image.size, document.current.size)
        self.assertEqual(result.details["Template size"], "3 × 3")
        self.assertEqual(result.details["Rotation angle"], "45°")

    def test_normalized_cross_correlation_tool_returns_none_when_rotation_is_cancelled(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            template_path = Path(directory) / "template.png"
            Image.new("L", (2, 2), 20).save(template_path)
            document = ImageDocument()
            document.current = Image.new("L", (4, 4), 20)

            with (
                patch(
                    "forensics_app.tools.ncc_image.filedialog.askopenfilename",
                    return_value=str(template_path),
                ),
                patch(
                    "forensics_app.tools.ncc_image.ask_parameters",
                    return_value=None,
                ),
            ):
                result = NormalizedCrossCorrelationTool().run(None, document)

        self.assertIsNone(result)

    def test_normalized_cross_correlation_tool_returns_none_when_template_is_cancelled(
        self,
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 4))

        with patch(
            "forensics_app.tools.ncc_image.filedialog.askopenfilename",
            return_value="",
        ):
            result = NormalizedCrossCorrelationTool().run(None, document)

        self.assertIsNone(result)

    def test_normalized_cross_correlation_tool_is_registered_in_template_matching(
        self,
    ) -> None:
        tools = build_tool_registry().all()
        matching_tools = [
            tool for tool in tools if isinstance(tool, NormalizedCrossCorrelationTool)
        ]

        self.assertEqual(len(matching_tools), 1)
        self.assertEqual(matching_tools[0].category, "Template Matching")

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

    def test_channel_split_returns_selected_channel_as_grayscale(self) -> None:
        image = Image.new("RGB", (2, 1))
        image.putdata([(10, 20, 30), (40, 50, 60)])

        for selection, expected_name, expected_values in (
            ("R", "Red", [10, 40]),
            ("G", "Green", [20, 50]),
            ("B", "Blue", [30, 60]),
        ):
            with self.subTest(selection=selection):
                output, names = split_channels(image, selection)
                self.assertEqual(names, (expected_name,))
                self.assertEqual(output.mode, "L")
                self.assertEqual(output.size, (2, 1))
                self.assertEqual(list(output.getdata()), expected_values)

    def test_channel_split_ignores_alpha_and_handles_grayscale(self) -> None:
        rgba = Image.new("RGBA", (1, 1), (10, 20, 30, 40))
        output, names = split_channels(rgba)
        self.assertEqual(names, ("Red", "Green", "Blue"))
        self.assertEqual(list(output.getdata()), [10, 20, 30])

        grayscale = Image.new("L", (1, 1), 73)
        output, names = split_channels(grayscale)
        self.assertEqual(names, ("Red", "Green", "Blue"))
        self.assertEqual(output.getpixel((0, 0)), 73)
        self.assertEqual(output.size, (1, 3))

    def test_channel_split_rejects_unsupported_selection(self) -> None:
        with self.assertRaises(ValueError):
            split_channels(Image.new("RGB", (1, 1)), "Alpha")

    def test_channel_split_tool_exposes_histogram_style_menu_options(self) -> None:
        self.assertEqual(ChannelSplitTool.menu_options, ("R", "G", "B", "All three"))

    def test_channel_split_tool_uses_selected_channel_action(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), (10, 20, 30))

        result = ChannelSplitTool().run_action(None, document, "G")

        self.assertEqual(result.image.getpixel((0, 0)), 20)
        self.assertEqual(result.details["Channels"], "Green")
        self.assertEqual(result.details["Display"], "Single grayscale plane")

    def test_channel_split_tool_defaults_to_all_channels(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1), (10, 20, 30))

        result = ChannelSplitTool().run(None, document)

        self.assertEqual(result.image.size, (1, 3))
        self.assertEqual(result.details["Channels"], "Red, Green, Blue")

    def test_channel_split_tool_rejects_unknown_action(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1))

        with self.assertRaises(ValueError):
            ChannelSplitTool().run_action(None, document, "Alpha")

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
            "horizontal": (1, 3, 1),
            "vertical": (3, 1, 1),
            "2d": (3, 3, 1),
        }
        for direction, shape in expected_shapes.items():
            with self.subTest(direction=direction):
                kernel = create_kernel(3, direction)
                self.assertEqual(kernel.shape, shape)
                self.assertAlmostEqual(float(kernel.sum()), 1.0)

    def test_convolution_kernel_size_sets_each_spatial_dimension(self) -> None:
        self.assertEqual(create_kernel(5, "horizontal").shape, (1, 5, 1))
        self.assertEqual(create_kernel(5, "vertical").shape, (5, 1, 1))
        self.assertEqual(create_kernel(5, "2d").shape, (5, 5, 1))

    def test_convolution_rejects_invalid_kernel_options(self) -> None:
        with self.assertRaises(ValueError):
            create_kernel(0, "horizontal")
        with self.assertRaises(ValueError):
            create_kernel(3, "diagonal")

    def test_convolution_preserves_constant_rgb_channels(self) -> None:
        image = Image.new("RGB", (5, 4), (73, 120, 201))

        output = apply_convolution(image, 3, "2d")

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)
        self.assertEqual(list(output.getdata()), [(73, 120, 201)] * 20)

    def test_convolution_filters_rgb_channels_independently(self) -> None:
        image = Image.new("RGB", (3, 1))
        image.putdata([(0, 30, 0), (90, 30, 0), (180, 30, 0)])

        output = apply_convolution(image, 3, "horizontal")

        self.assertEqual(
            list(output.getdata()),
            [(30, 30, 0), (90, 30, 0), (150, 30, 0)],
        )

    def test_convolution_tool_uses_selected_options(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (3, 3), (90, 120, 150))
        with patch(
            "forensics_app.tools.convolution.ask_convolution_options",
            return_value=(5, "vertical"),
        ) as ask_options:
            result = ConvolutionTool().run(None, document)

        ask_options.assert_called_once_with(None)
        self.assertEqual(result.image.mode, "RGB")
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

    def test_sharpen_image_preserves_uniform_pixels(self) -> None:
        image = Image.new("RGB", (8, 8), (200, 120, 40))

        output = sharpen_image(image)

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(list(output.getdata()), list(image.getdata()))

    def test_sharpen_image_enhances_edges_without_losing_flat_regions(self) -> None:
        pixels = np.zeros((9, 9), dtype=np.uint8)
        pixels[:, 4:] = 100
        image = Image.fromarray(pixels)

        output = sharpen_image(image, radius=1.0, amount=1.5)

        self.assertGreater(output.getpixel((4, 4)), image.getpixel((4, 4)))
        self.assertEqual(output.getpixel((8, 4)), image.getpixel((8, 4)))

    def test_sharpen_tool_uses_parameter_dialog(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (3, 3), "gray")

        with patch(
            "forensics_app.tools.sharpen.ask_parameters",
            return_value={"radius": 18.0, "amount": 1.2},
        ) as ask_parameters:
            result = SharpenTool().run(None, document)

        ask_parameters.assert_called_once_with(
            None,
            "Sharpen image",
            (
                ParameterField(
                    "radius",
                    "Gaussian blur radius (sigma)",
                    0.1,
                    10.0,
                    0.1,
                    2.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ("amount", "Sharpening amount", 0.0, 3.0, 0.1, 1.5),
            ),
        )
        self.assertEqual(result.image.size, document.current.size)
        self.assertEqual(result.details["Blur radius (sigma)"], 18.0)
        self.assertEqual(result.details["Sharpening amount"], 1.2)

    def test_sharpen_tool_returns_none_when_parameter_dialog_is_cancelled(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (1, 1))

        with patch("forensics_app.tools.sharpen.ask_parameters", return_value=None):
            result = SharpenTool().run(None, document)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
