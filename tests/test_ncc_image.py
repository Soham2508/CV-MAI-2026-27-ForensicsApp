import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.ncc_image import (
    NormalizedCrossCorrelationTool,
    compute_normalized_cross_correlation,
    normalized_cross_correlation,
    plot_template_matching,
    superimpose_matches,
    threshold_normalized_cross_correlation,
)
from forensics_app.utils.parameter_dialog import ParameterField


class NormalizedCrossCorrelationThresholdTests(unittest.TestCase):
    def test_compute_normalized_cross_correlation_returns_arrays(self) -> None:
        pixels = np.arange(100, dtype=np.uint8).reshape((10, 10))
        image = Image.fromarray(pixels)
        template = Image.fromarray(pixels[2:6, 3:7])

        img_arr, corr = compute_normalized_cross_correlation(image, template, angle=0.0)

        self.assertEqual(img_arr.shape, (10, 10))
        self.assertEqual(corr.shape, (10, 10))
        self.assertAlmostEqual(float(corr.max()), 1.0, places=2)

    def test_normalized_cross_correlation_with_threshold_superimposes_red(self) -> None:
        pixels = np.arange(144, dtype=np.uint8).reshape((12, 12))
        image = Image.fromarray(pixels)
        template = Image.fromarray(pixels[4:8, 5:9])

        output = normalized_cross_correlation(image, template, angle=0.0, threshold=0.8)

        self.assertEqual(output.size, image.size)
        self.assertEqual(output.mode, "RGB")

        # The peak match location should have pure red pixels (255, 0, 0)
        output_arr = np.asarray(output)
        red_pixels = (
            (output_arr[..., 0] == 255)
            & (output_arr[..., 1] == 0)
            & (output_arr[..., 2] == 0)
        )
        self.assertGreater(int(np.count_nonzero(red_pixels)), 0)

    def test_threshold_normalized_cross_correlation_helper(self) -> None:
        pixels = np.arange(144, dtype=np.uint8).reshape((12, 12))
        image = Image.fromarray(pixels)
        template = Image.fromarray(pixels[4:8, 5:9])

        output = threshold_normalized_cross_correlation(image, template, threshold=0.8)

        self.assertEqual(output.size, image.size)
        self.assertEqual(output.mode, "RGB")
        output_arr = np.asarray(output)
        red_pixels = (
            (output_arr[..., 0] == 255)
            & (output_arr[..., 1] == 0)
            & (output_arr[..., 2] == 0)
        )
        self.assertGreater(int(np.count_nonzero(red_pixels)), 0)

    def test_plot_template_matching_generates_side_by_side_figure(self) -> None:
        pixels = np.arange(144, dtype=np.uint8).reshape((12, 12))
        image = Image.fromarray(pixels)
        template = Image.fromarray(pixels[4:8, 5:9])

        fig_img = plot_template_matching(image, template, angle=0.0, threshold=0.8)

        self.assertEqual(fig_img.mode, "RGB")
        self.assertGreater(fig_img.width, image.width)
        self.assertGreater(fig_img.height, image.height)

    def test_normalized_cross_correlation_rejects_invalid_threshold(self) -> None:
        image = Image.new("L", (10, 10), 128)
        template = Image.new("L", (4, 4), 128)

        for invalid_threshold in (-1.5, 1.5, float("nan"), float("inf")):
            with (
                self.subTest(threshold=invalid_threshold),
                self.assertRaisesRegex(ValueError, "Correlation threshold"),
            ):
                normalized_cross_correlation(
                    image, template, threshold=invalid_threshold
                )

    def test_superimpose_matches_modes(self) -> None:
        image = Image.new("L", (10, 10), 100)
        correlation = np.zeros((10, 10), dtype=float)
        correlation[4:6, 4:6] = 1.0

        for mode in ("red", "transparent_red", "dimmed", "contours"):
            with self.subTest(mode=mode):
                output = superimpose_matches(image, correlation, threshold=0.8, mode=mode)
                self.assertEqual(output.size, (10, 10))
                self.assertEqual(output.mode, "RGB")

    def test_tool_menu_options_include_thresholded_image(self) -> None:
        tool = NormalizedCrossCorrelationTool()
        self.assertIn("Correlation map", tool.menu_options)
        self.assertIn("Thresholded image", tool.menu_options)

    def test_tool_runs_thresholded_action_with_configs(self) -> None:
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
                    return_value={"angle": 0.0, "threshold": 0.8},
                ) as ask_parameters,
            ):
                result = NormalizedCrossCorrelationTool().run_action(
                    None, document, "Thresholded image"
                )

        askopenfilename.assert_called_once()
        ask_parameters.assert_called_once_with(
            None,
            "Template matching configs",
            (
                ParameterField(
                    "angle",
                    "Rotation angle (degrees)",
                    -180.0,
                    180.0,
                    1.0,
                    0.0,
                ),
                ParameterField(
                    "threshold",
                    "Correlation threshold",
                    -1.0,
                    1.0,
                    0.01,
                    0.8,
                ),
            ),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.details["Rotation angle"], "0°")
        self.assertEqual(result.details["Threshold"], "0.80")
        self.assertGreater(result.details["Matches found"], 0)
        self.assertIn("matching pixels exceeding", result.message)

    def test_tool_returns_none_when_thresholded_parameters_cancelled(self) -> None:
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
                result = NormalizedCrossCorrelationTool().run_action(
                    None, document, "Thresholded image"
                )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
