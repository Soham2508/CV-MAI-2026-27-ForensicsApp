import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.corner_detection import CornerDetectionTool, detect_corners
from forensics_app.utils.parameter_dialog import ParameterField


class CornerDetectionTests(unittest.TestCase):
    def _create_test_image(self) -> Image.Image:
        """Create a test image with distinct features."""
        arr = np.zeros((64, 64), dtype=np.uint8)
        # Add isolated bright points for FAST and blocks for Harris
        arr[16, 16] = 255
        arr[16, 48] = 255
        arr[32:48, 32:48] = 255
        return Image.fromarray(arr, mode="L")

    def test_detect_corners_fast_returns_corners_and_rgb_image(self) -> None:
        image = self._create_test_image()

        corners, output = detect_corners(
            image,
            method="FAST",
            threshold_rel=0.01,
            fast_threshold=0.1,
            min_distance=2,
        )

        self.assertIsInstance(corners, np.ndarray)
        self.assertEqual(corners.ndim, 2)
        self.assertEqual(corners.shape[1], 2)
        self.assertGreater(len(corners), 0)
        self.assertIsInstance(output, Image.Image)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)

    def test_detect_corners_harris_returns_corners_and_rgb_image(self) -> None:
        image = self._create_test_image()

        corners, output = detect_corners(
            image,
            method="Harris",
            threshold_rel=0.01,
            harris_sigma=1.0,
            min_distance=2,
        )

        self.assertIsInstance(corners, np.ndarray)
        self.assertEqual(corners.ndim, 2)
        self.assertEqual(corners.shape[1], 2)
        self.assertGreater(len(corners), 0)
        self.assertIsInstance(output, Image.Image)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)

    def test_detect_corners_threshold_rel_filters_peaks(self) -> None:
        image = self._create_test_image()

        corners_lenient, _ = detect_corners(
            image, method="Harris", threshold_rel=0.01, min_distance=2
        )
        corners_strict, _ = detect_corners(
            image, method="Harris", threshold_rel=0.8, min_distance=2
        )

        self.assertGreaterEqual(len(corners_lenient), len(corners_strict))

    def test_detect_corners_rejects_invalid_inputs(self) -> None:
        image = Image.new("RGB", (32, 32), "white")

        with self.assertRaisesRegex(ValueError, "Choose FAST or Harris"):
            detect_corners(image, method="SIFT")

        for bad_thresh in (-0.1, 1.5, float("nan")):
            with self.assertRaisesRegex(ValueError, "Peak threshold"):
                detect_corners(image, threshold_rel=bad_thresh)

        for bad_distance in (0, -1, 2.5):  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "Minimum distance"):
                detect_corners(image, min_distance=bad_distance)

        for bad_fast in (-0.1, 1.1, float("nan")):
            with self.assertRaisesRegex(ValueError, "FAST intensity threshold"):
                detect_corners(image, fast_threshold=bad_fast)

        for bad_sigma in (0.0, -1.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "Harris sigma"):
                detect_corners(image, harris_sigma=bad_sigma)

        empty_image = Image.new("L", (0, 0))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            detect_corners(empty_image)

        small_image = Image.new("L", (6, 6))
        with self.assertRaisesRegex(ValueError, "at least 7 by 7"):
            detect_corners(small_image, method="FAST")

    def test_corner_detection_tool_run_action_fast(self) -> None:
        tool = CornerDetectionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        fake_params = {
            "threshold_rel": 0.1,
            "min_distance": 5.0,
            "fast_threshold": 0.15,
        }
        with patch(
            "forensics_app.tools.corner_detection.ask_parameters",
            return_value=fake_params,
        ) as mock_ask:
            result = tool.run_action(MagicMock(), document, "FAST")

        self.assertIsNotNone(result)
        mock_ask.assert_called_once()
        self.assertEqual(
            mock_ask.call_args[0][2],
            (
                ParameterField(
                    "threshold_rel", "Relative peak threshold", 0.0, 1.0, 0.01, 0.1
                ),
                ParameterField(
                    "min_distance",
                    "Minimum peak distance (px)",
                    1.0,
                    50.0,
                    1.0,
                    5.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ParameterField(
                    "fast_threshold",
                    "FAST intensity threshold",
                    0.0,
                    1.0,
                    0.01,
                    0.15,
                ),
            ),
        )
        assert result is not None
        self.assertIn("FAST: detected", result.message)
        self.assertEqual(result.details["Method"], "FAST")
        self.assertIn("Corners detected", result.details)
        self.assertEqual(result.details["FAST intensity threshold"], 0.15)
        self.assertIsInstance(result.image, Image.Image)

    def test_corner_detection_tool_run_action_harris(self) -> None:
        tool = CornerDetectionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        fake_params = {
            "threshold_rel": 0.05,
            "min_distance": 3.0,
            "harris_sigma": 1.2,
        }
        with patch(
            "forensics_app.tools.corner_detection.ask_parameters",
            return_value=fake_params,
        ) as mock_ask:
            result = tool.run_action(MagicMock(), document, "Harris")

        self.assertIsNotNone(result)
        mock_ask.assert_called_once()
        self.assertEqual(
            mock_ask.call_args[0][2],
            (
                ParameterField(
                    "threshold_rel", "Relative peak threshold", 0.0, 1.0, 0.01, 0.1
                ),
                ParameterField(
                    "min_distance",
                    "Minimum peak distance (px)",
                    1.0,
                    50.0,
                    1.0,
                    5.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ParameterField(
                    "harris_sigma",
                    "Harris sigma",
                    0.1,
                    5.0,
                    0.1,
                    1.0,
                    show_slider=False,
                    no_threshold=True,
                ),
            ),
        )
        assert result is not None
        self.assertIn("Harris: detected", result.message)
        self.assertEqual(result.details["Method"], "Harris")
        self.assertIn("Corners detected", result.details)
        self.assertEqual(result.details["Harris sigma"], 1.2)
        self.assertIsInstance(result.image, Image.Image)

    def test_corner_detection_tool_run_delegates_to_fast(self) -> None:
        tool = CornerDetectionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        with patch.object(
            tool, "run_action", return_value=MagicMock()
        ) as mock_run_action:
            tool.run(MagicMock(), document)
            mock_run_action.assert_called_once()
            self.assertEqual(mock_run_action.call_args[0][2], "FAST")

    def test_corner_detection_tool_cancelling_dialog_returns_none(self) -> None:
        tool = CornerDetectionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        with patch(
            "forensics_app.tools.corner_detection.ask_parameters",
            return_value=None,
        ):
            result = tool.run_action(MagicMock(), document, "FAST")

        self.assertIsNone(result)

    def test_corner_detection_tool_rejects_invalid_action(self) -> None:
        tool = CornerDetectionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        with self.assertRaisesRegex(ValueError, "Choose FAST or Harris"):
            tool.run_action(MagicMock(), document, "INVALID")


if __name__ == "__main__":
    unittest.main()
