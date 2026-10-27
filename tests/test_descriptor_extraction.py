import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.descriptor_extraction import (
    DescriptorExtractionTool,
    extract_descriptors,
    extract_features,
)
from forensics_app.utils.parameter_dialog import ParameterField


class DescriptorExtractionTests(unittest.TestCase):
    def _create_test_image(self) -> Image.Image:
        """Create a textured image with high-contrast dots and edges."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        for i in range(10, 90, 15):
            for j in range(10, 90, 15):
                arr[i : i + 6, j : j + 6] = 255
        return Image.fromarray(arr, mode="L")

    def test_extract_descriptors_orb_returns_keypoints_descriptors_and_image(
        self,
    ) -> None:
        image = self._create_test_image()

        keypoints, descriptors, output = extract_descriptors(
            image, method="ORB", n_keypoints=50, fast_threshold=0.05
        )

        self.assertIsInstance(keypoints, np.ndarray)
        self.assertEqual(keypoints.ndim, 2)
        self.assertEqual(keypoints.shape[1], 2)
        self.assertIsInstance(descriptors, np.ndarray)
        self.assertEqual(descriptors.shape[1], 256)
        self.assertEqual(descriptors.dtype, np.bool_)
        self.assertEqual(len(keypoints), len(descriptors))
        self.assertGreater(len(keypoints), 0)
        self.assertIsInstance(output, Image.Image)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)

    def test_extract_descriptors_sift_returns_keypoints_descriptors_and_image(
        self,
    ) -> None:
        image = self._create_test_image()

        keypoints, descriptors, output = extract_descriptors(
            image, method="SIFT", n_octaves=4, n_scales=3, sigma_min=1.6
        )

        self.assertIsInstance(keypoints, np.ndarray)
        self.assertEqual(keypoints.ndim, 2)
        self.assertEqual(keypoints.shape[1], 2)
        self.assertIsInstance(descriptors, np.ndarray)
        self.assertEqual(descriptors.shape[1], 128)
        self.assertEqual(descriptors.dtype, np.uint8)
        self.assertEqual(len(keypoints), len(descriptors))
        self.assertGreater(len(keypoints), 0)
        self.assertIsInstance(output, Image.Image)
        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, image.size)

    def test_extract_descriptors_textureless_image_returns_empty_arrays(
        self,
    ) -> None:
        flat_image = Image.new("L", (64, 64), 128)

        kp_orb, desc_orb, _ = extract_descriptors(flat_image, method="ORB")
        self.assertEqual(kp_orb.shape, (0, 2))
        self.assertEqual(desc_orb.shape, (0, 256))
        self.assertEqual(desc_orb.dtype, np.bool_)

        kp_sift, desc_sift, _ = extract_descriptors(flat_image, method="SIFT")
        self.assertEqual(kp_sift.shape, (0, 2))
        self.assertEqual(desc_sift.shape, (0, 128))
        self.assertEqual(desc_sift.dtype, np.uint8)

    def test_extract_descriptors_rejects_invalid_inputs(self) -> None:
        image = self._create_test_image()

        with self.assertRaisesRegex(ValueError, "Choose ORB or SIFT"):
            extract_descriptors(image, method="SURF")

        # ORB parameter validation
        for bad_count in (0, -10, 5.5):  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "keypoint count"):
                extract_descriptors(image, method="ORB", n_keypoints=bad_count)

        for bad_thresh in (-0.1, 1.2, float("nan")):
            with self.assertRaisesRegex(ValueError, "FAST threshold"):
                extract_descriptors(image, method="ORB", fast_threshold=bad_thresh)

        # SIFT parameter validation
        for bad_octaves in (0, -2, 3.5):  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "octave count"):
                extract_descriptors(image, method="SIFT", n_octaves=bad_octaves)

        for bad_scales in (0, -1, 2.5):  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "scale count"):
                extract_descriptors(image, method="SIFT", n_scales=bad_scales)

        for bad_sigma in (0.5, -1.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "sigma_min"):
                extract_descriptors(image, method="SIFT", sigma_min=bad_sigma)

        # Image size validation
        small_image = Image.new("L", (31, 31))
        with self.assertRaisesRegex(ValueError, "at least 32 by 32"):
            extract_descriptors(small_image, method="ORB")

        tiny_image = Image.new("L", (11, 11))
        with self.assertRaisesRegex(ValueError, "at least 12 by 12"):
            extract_descriptors(tiny_image, method="SIFT")

    def test_extract_features_both_modes(self) -> None:
        image = self._create_test_image()

        features, output = extract_features(image, mode="Both")

        self.assertIn("ORB", features)
        self.assertIn("SIFT", features)
        self.assertIsInstance(features["ORB"]["keypoints"], np.ndarray)
        self.assertIsInstance(features["SIFT"]["descriptors"], np.ndarray)
        self.assertEqual(output.size, image.size)
        self.assertEqual(output.mode, "RGB")

    def test_extract_features_rejects_invalid_mode(self) -> None:
        image = self._create_test_image()

        with self.assertRaisesRegex(ValueError, "Choose ORB, SIFT, or Both"):
            extract_features(image, mode="INVALID")

    def test_descriptor_extraction_tool_run_action_orb(self) -> None:
        tool = DescriptorExtractionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        fake_params = {
            "orb_n_keypoints": 100.0,
            "orb_fast_threshold": 0.08,
        }
        with patch(
            "forensics_app.tools.descriptor_extraction.ask_parameters",
            return_value=fake_params,
        ) as mock_ask:
            result = tool.run_action(MagicMock(), document, "ORB")

        self.assertIsNotNone(result)
        mock_ask.assert_called_once()
        self.assertEqual(
            mock_ask.call_args[0][2],
            (
                ParameterField(
                    "orb_n_keypoints",
                    "ORB: maximum keypoints",
                    1.0,
                    5000.0,
                    1.0,
                    200.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ParameterField(
                    "orb_fast_threshold",
                    "ORB: FAST threshold",
                    0.0,
                    1.0,
                    0.01,
                    0.08,
                ),
            ),
        )
        assert result is not None
        self.assertIn("ORB:", result.message)
        self.assertEqual(result.details["Mode"], "ORB")
        self.assertIn("ORB keypoints", result.details)
        self.assertIn("ORB descriptor matrix", result.details)
        self.assertIsInstance(result.image, Image.Image)

    def test_descriptor_extraction_tool_run_action_sift(self) -> None:
        tool = DescriptorExtractionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        fake_params = {
            "sift_n_octaves": 4.0,
            "sift_n_scales": 3.0,
            "sift_sigma_min": 1.6,
        }
        with patch(
            "forensics_app.tools.descriptor_extraction.ask_parameters",
            return_value=fake_params,
        ) as mock_ask:
            result = tool.run_action(MagicMock(), document, "SIFT")

        self.assertIsNotNone(result)
        mock_ask.assert_called_once()
        self.assertEqual(
            mock_ask.call_args[0][2],
            (
                ParameterField(
                    "sift_n_octaves",
                    "SIFT: maximum octaves",
                    1.0,
                    8.0,
                    1.0,
                    8.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ParameterField(
                    "sift_n_scales",
                    "SIFT: scales per octave",
                    1.0,
                    8.0,
                    1.0,
                    3.0,
                    show_slider=False,
                    no_threshold=True,
                ),
                ParameterField(
                    "sift_sigma_min",
                    "SIFT: initial sigma",
                    1.0,
                    3.0,
                    0.1,
                    1.6,
                    show_slider=False,
                    no_threshold=True,
                ),
            ),
        )
        assert result is not None
        self.assertIn("SIFT:", result.message)
        self.assertEqual(result.details["Mode"], "SIFT")
        self.assertIn("SIFT keypoints", result.details)
        self.assertIn("SIFT descriptor matrix", result.details)
        self.assertIsInstance(result.image, Image.Image)

    def test_descriptor_extraction_tool_run_action_both(self) -> None:
        tool = DescriptorExtractionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        fake_params = {
            "orb_n_keypoints": 50.0,
            "orb_fast_threshold": 0.08,
            "sift_n_octaves": 3.0,
            "sift_n_scales": 2.0,
            "sift_sigma_min": 1.6,
        }
        with patch(
            "forensics_app.tools.descriptor_extraction.ask_parameters",
            return_value=fake_params,
        ):
            result = tool.run_action(MagicMock(), document, "Both")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("ORB:", result.message)
        self.assertIn("SIFT:", result.message)
        self.assertEqual(result.details["Mode"], "Both")
        self.assertIn("ORB keypoints", result.details)
        self.assertIn("SIFT keypoints", result.details)

    def test_descriptor_extraction_tool_empty_message_hint(self) -> None:
        tool = DescriptorExtractionTool()
        document = ImageDocument()
        document.current = Image.new("L", (64, 64), 128)

        fake_params = {
            "orb_n_keypoints": 50.0,
            "orb_fast_threshold": 0.08,
        }
        with patch(
            "forensics_app.tools.descriptor_extraction.ask_parameters",
            return_value=fake_params,
        ):
            result = tool.run_action(MagicMock(), document, "ORB")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("No keypoints found. Try a more textured image.", result.message)

    def test_descriptor_extraction_tool_run_delegates_to_orb(self) -> None:
        tool = DescriptorExtractionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        with patch.object(
            tool, "run_action", return_value=MagicMock()
        ) as mock_run_action:
            tool.run(MagicMock(), document)
            mock_run_action.assert_called_once()
            self.assertEqual(mock_run_action.call_args[0][2], "ORB")

    def test_descriptor_extraction_tool_cancelling_returns_none(self) -> None:
        tool = DescriptorExtractionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        with patch(
            "forensics_app.tools.descriptor_extraction.ask_parameters",
            return_value=None,
        ):
            result = tool.run_action(MagicMock(), document, "ORB")

        self.assertIsNone(result)

    def test_descriptor_extraction_tool_rejects_invalid_action(self) -> None:
        tool = DescriptorExtractionTool()
        document = ImageDocument()
        document.current = self._create_test_image()

        with self.assertRaisesRegex(ValueError, "Choose ORB, SIFT, or Both"):
            tool.run_action(MagicMock(), document, "INVALID")


if __name__ == "__main__":
    unittest.main()
