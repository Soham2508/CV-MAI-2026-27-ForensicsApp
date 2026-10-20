import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.hog_person_detection import (
    BOX_COLORS,
    COSINE_BOX_COLOR,
    EUCLIDEAN_BOX_COLOR,
    HOGPersonDetectionTool,
    _window_starts,
    detect_with_hog,
    draw_detection_boxes,
)


class HOGPersonDetectionTests(unittest.TestCase):
    def test_window_starts_includes_final_position(self) -> None:
        self.assertEqual(_window_starts(11, 4, 3), [0, 3, 6, 7])

    def test_window_starts_returns_single_position_for_equal_lengths(self) -> None:
        self.assertEqual(_window_starts(4, 4, 3), [0])

    def test_detect_with_hog_selects_best_euclidean_and_cosine_matches(self) -> None:
        scene = Image.new("L", (3, 1))
        template = Image.new("L", (1, 1))
        features = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.9, 1.0]),
            np.array([5.0, 0.0]),
        ]

        with patch(
            "forensics_app.tools.hog_person_detection.compute_hog_features",
            side_effect=features,
        ):
            result = detect_with_hog(scene, template, 9, 8, 2, 1)

        self.assertEqual(result["template_size"], (1, 1))
        self.assertEqual(result["euclidean_location"], (1, 0))
        self.assertAlmostEqual(result["euclidean_distance"], np.sqrt(1.01))
        self.assertEqual(result["cosine_location"], (2, 0))
        self.assertEqual(result["cosine_similarity"], 1.0)
        self.assertEqual(result["windows_checked"], 3)

    def test_detect_with_hog_rejects_template_larger_than_scene(self) -> None:
        with self.assertRaisesRegex(ValueError, "must fit inside"):
            detect_with_hog(Image.new("L", (2, 2)), Image.new("L", (3, 2)), 9, 8, 2, 1)

    def test_draw_detection_boxes_uses_independent_configured_colors(self) -> None:
        image = Image.new("RGB", (8, 8), "white")
        result = {
            "template_size": (3, 3),
            "euclidean_location": (1, 1),
            "cosine_location": (4, 4),
        }

        output = draw_detection_boxes(image, result, "Green", "Yellow")

        self.assertEqual(output.getpixel((1, 1)), (0, 128, 0))
        self.assertEqual(output.getpixel((4, 4)), (255, 255, 0))
        self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(image.getpixel((1, 1)), (255, 255, 255))

    def test_draw_detection_boxes_defaults_to_red_and_blue(self) -> None:
        output = draw_detection_boxes(
            Image.new("RGB", (4, 4), "white"),
            {
                "template_size": (2, 2),
                "euclidean_location": (0, 0),
                "cosine_location": (2, 2),
            },
        )

        self.assertEqual(output.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(output.getpixel((2, 2)), (0, 0, 255))

    def test_tool_uses_selected_box_colors_and_reports_them(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 4), "white")
        detection = {
            "template_size": (2, 2),
            "euclidean_location": (0, 0),
            "euclidean_distance": 0.25,
            "cosine_location": (2, 2),
            "cosine_similarity": 0.75,
            "windows_checked": 2,
        }
        with (
            patch(
                "forensics_app.tools.hog_person_detection.filedialog.askopenfilename",
                return_value="template.png",
            ),
            patch("forensics_app.tools.hog_person_detection.Image.open") as open_image,
            patch(
                "forensics_app.tools.hog_person_detection.ask_parameters",
                return_value={
                    "orientations": 9,
                    "pixels_per_cell": 8,
                    "cells_per_block": 2,
                    "step_size": 1,
                },
            ),
            patch(
                "forensics_app.tools.hog_person_detection.ask_box_colors",
                return_value=("Green", "Yellow"),
            ) as ask_box_colors,
            patch(
                "forensics_app.tools.hog_person_detection.detect_with_hog",
                return_value=detection,
            ),
            patch(
                "forensics_app.tools.hog_person_detection.draw_detection_boxes",
                return_value=Image.new("RGB", (4, 4)),
            ) as draw_boxes,
        ):
            open_image.return_value.__enter__.return_value.copy.return_value = (
                Image.new("RGB", (2, 2))
            )
            result = HOGPersonDetectionTool().run(None, document)

        ask_box_colors.assert_called_once_with(None)
        draw_boxes.assert_called_once_with(
            document.current, detection, "Green", "Yellow"
        )
        self.assertEqual(result.details["Euclidean box color"], "Green")
        self.assertEqual(result.details["Cosine box color"], "Yellow")

    def test_box_color_defaults_are_available_in_the_palette(self) -> None:
        self.assertIn(EUCLIDEAN_BOX_COLOR, BOX_COLORS)
        self.assertIn(COSINE_BOX_COLOR, BOX_COLORS)


if __name__ == "__main__":
    unittest.main()
