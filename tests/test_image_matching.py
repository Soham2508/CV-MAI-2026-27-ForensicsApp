import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.image_matching import (
    ImageMatchingTool,
    extract_descriptors,
    match_image_descriptors,
    plot_matches,
)


class ImageMatchingTests(unittest.TestCase):
    def _create_textured_image(
        self, size: tuple[int, int] = (120, 120), color: tuple[int, int, int] = (255, 0, 0)
    ) -> Image.Image:
        """Create a textured full-color RGB image with high-contrast checker-like features."""
        arr = np.zeros((*size, 3), dtype=np.uint8)
        for i in range(10, size[0] - 10, 15):
            for j in range(10, size[1] - 10, 15):
                arr[i : i + 8, j : j + 8] = color
        return Image.fromarray(arr, mode="RGB")

    def test_tool_registered_in_tool_registry(self) -> None:
        registry = build_tool_registry()
        matching_tools = [
            tool for tool in registry.all() if isinstance(tool, ImageMatchingTool)
        ]
        self.assertEqual(len(matching_tools), 1)
        tool = matching_tools[0]
        self.assertEqual(tool.tool_id, "image_matching")
        self.assertEqual(tool.category, "Matching & Descriptors")
        # Ensure it renders as a regular button, not a dropdown
        self.assertFalse(hasattr(tool, "menu_options"))
        self.assertTrue(tool.requires_image)

    def test_extract_descriptors_orb_success(self) -> None:
        image = self._create_textured_image()
        keypoints, descriptors = extract_descriptors(image, method="ORB", n_keypoints=100)

        self.assertIsInstance(keypoints, np.ndarray)
        self.assertEqual(keypoints.ndim, 2)
        self.assertEqual(keypoints.shape[1], 2)
        self.assertIsInstance(descriptors, np.ndarray)
        self.assertEqual(descriptors.shape[1], 256)
        self.assertEqual(descriptors.dtype, np.bool_)
        self.assertEqual(len(keypoints), len(descriptors))
        self.assertGreater(len(keypoints), 0)

    def test_extract_descriptors_sift_success(self) -> None:
        image = self._create_textured_image()
        keypoints, descriptors = extract_descriptors(
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

    def test_extract_descriptors_textureless_returns_empty(self) -> None:
        blank = Image.new("RGB", (64, 64), (128, 128, 128))

        kp_orb, desc_orb = extract_descriptors(blank, method="ORB")
        self.assertEqual(kp_orb.shape, (0, 2))
        self.assertEqual(desc_orb.shape, (0, 256))

        kp_sift, desc_sift = extract_descriptors(blank, method="SIFT")
        self.assertEqual(kp_sift.shape, (0, 2))
        self.assertEqual(desc_sift.shape, (0, 128))

    def test_extract_descriptors_invalid_arguments(self) -> None:
        image = self._create_textured_image()

        with self.assertRaisesRegex(ValueError, "Choose ORB or SIFT"):
            extract_descriptors(image, method="SURF")

        with self.assertRaisesRegex(ValueError, "keypoint count"):
            extract_descriptors(image, method="ORB", n_keypoints=0)

        with self.assertRaisesRegex(ValueError, "keypoint count"):
            extract_descriptors(image, method="ORB", n_keypoints=-5)

        with self.assertRaisesRegex(ValueError, "octave count"):
            extract_descriptors(image, method="SIFT", n_octaves=0)

        with self.assertRaisesRegex(ValueError, "scale count"):
            extract_descriptors(image, method="SIFT", n_scales=0)

        with self.assertRaisesRegex(ValueError, "sigma_min"):
            extract_descriptors(image, method="SIFT", sigma_min=0.5)

        tiny_image = Image.new("RGB", (8, 8), (255, 255, 255))
        with self.assertRaisesRegex(ValueError, "at least 32"):
            extract_descriptors(tiny_image, method="ORB")

        with self.assertRaisesRegex(ValueError, "at least 12"):
            extract_descriptors(tiny_image, method="SIFT")

    def test_match_image_descriptors_orb_and_sift(self) -> None:
        image1 = self._create_textured_image((120, 120), (255, 0, 0))
        image2 = self._create_textured_image((120, 120), (0, 255, 0))

        kp1, desc1 = extract_descriptors(image1, "ORB", n_keypoints=100)
        kp2, desc2 = extract_descriptors(image2, "ORB", n_keypoints=100)
        matches_orb = match_image_descriptors(
            desc1, desc2, method="ORB", cross_check=True, max_ratio=0.8
        )
        self.assertIsInstance(matches_orb, np.ndarray)
        self.assertEqual(matches_orb.ndim, 2)
        self.assertEqual(matches_orb.shape[1], 2)
        self.assertGreater(len(matches_orb), 0)

        kp1_s, desc1_s = extract_descriptors(image1, "SIFT", n_octaves=4, n_scales=3, sigma_min=1.6)
        kp2_s, desc2_s = extract_descriptors(image2, "SIFT", n_octaves=4, n_scales=3, sigma_min=1.6)
        matches_sift = match_image_descriptors(
            desc1_s, desc2_s, method="SIFT", cross_check=True, max_ratio=0.8
        )
        self.assertIsInstance(matches_sift, np.ndarray)
        self.assertEqual(matches_sift.ndim, 2)
        self.assertEqual(matches_sift.shape[1], 2)
        self.assertGreater(len(matches_sift), 0)

    def test_match_image_descriptors_empty_returns_empty(self) -> None:
        empty_desc = np.empty((0, 256), dtype=bool)
        valid_desc = np.ones((10, 256), dtype=bool)

        matches1 = match_image_descriptors(empty_desc, valid_desc, method="ORB")
        self.assertEqual(matches1.shape, (0, 2))

        matches2 = match_image_descriptors(valid_desc, empty_desc, method="ORB")
        self.assertEqual(matches2.shape, (0, 2))

    def test_match_image_descriptors_mismatched_dimensions_raises(self) -> None:
        d1 = np.ones((5, 256), dtype=bool)
        d2 = np.ones((5, 128), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Descriptor length mismatch"):
            match_image_descriptors(d1, d2, method="ORB")

    def test_plot_matches_renders_full_color_image(self) -> None:
        image1 = Image.new("RGB", (100, 100), (255, 0, 0))
        image2 = Image.new("RGB", (120, 100), (0, 0, 255))
        k1 = np.array([[20, 20], [50, 50]])
        k2 = np.array([[20, 20], [50, 50]])
        matches = np.array([[0, 0], [1, 1]])

        output = plot_matches(image1, image2, k1, k2, matches, show=False)
        self.assertIsInstance(output, Image.Image)
        self.assertEqual(output.mode, "RGB")
        self.assertGreater(output.width, 0)
        self.assertGreater(output.height, 0)

    def test_tool_uses_opened_image_and_prompts_second(self) -> None:
        tool = ImageMatchingTool()
        document = ImageDocument()

        first_img = self._create_textured_image((120, 120), (255, 0, 0))
        second_img = self._create_textured_image((120, 120), (0, 255, 0))

        document.current = first_img.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path2 = str(Path(tmpdir) / "second.png")
            second_img.save(path2)

            params = {
                "method": "ORB",
                "left_settings": {"n_keypoints": 80},
                "right_settings": {"n_keypoints": 120},
                "cross_check": True,
                "max_ratio": 0.8,
            }

            parent = MagicMock()
            result = tool.run(
                parent,
                document,
                parameters=params,
                second_image_path=path2,
                show_plot=False,
            )

            self.assertIsNotNone(result)
            self.assertIsInstance(result.image, Image.Image)
            self.assertEqual(result.image.mode, "RGB")
            self.assertIn("Found", result.message)
            self.assertEqual(result.details["Method"], "ORB")
            self.assertEqual(result.details["First max keypoints"], 80)
            self.assertEqual(result.details["Second max keypoints"], 120)
            self.assertEqual(result.details["Metric"], "Hamming")
            self.assertGreater(result.details["Matches found"], 0)

    def test_tool_run_cancelled_second_image_selection(self) -> None:
        tool = ImageMatchingTool()
        document = ImageDocument()
        document.current = self._create_textured_image()
        parent = MagicMock()

        with patch("forensics_app.tools.image_matching.filedialog.askopenfilename", return_value=""):
            result = tool.run(parent, document)
            self.assertIsNone(result)

    def test_tool_run_cancelled_parameters(self) -> None:
        tool = ImageMatchingTool()
        document = ImageDocument()
        document.current = self._create_textured_image()
        img2 = self._create_textured_image()

        with tempfile.TemporaryDirectory() as tmpdir:
            path2 = str(Path(tmpdir) / "second.png")
            img2.save(path2)

            with patch(
                "forensics_app.tools.image_matching.ask_matching_parameters",
                return_value=None,
            ):
                result = tool.run(
                    MagicMock(),
                    document,
                    second_image_path=path2,
                    show_plot=False,
                )
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
