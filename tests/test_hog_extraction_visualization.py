import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.hog_extraction_visualization import (
    HOGVisualizationTool,
    compute_hog_features,
    hog_visualization,
)
from forensics_app.utils.parameter_dialog import ParameterField


class HOGVisualizationTests(unittest.TestCase):
    def test_compute_hog_features_returns_one_dimensional_descriptor(self) -> None:
        gray = np.zeros((32, 32), dtype=np.float32)
        gray[:, 16:] = 1.0

        descriptor = compute_hog_features(
            gray,
            orientations=9,
            pixels_per_cell=8,
            cells_per_block=2,
        )

        self.assertEqual(descriptor.ndim, 1)
        self.assertEqual(descriptor.size, 324)
        self.assertGreater(float(descriptor.max()), 0.0)

    def test_hog_visualization_returns_descriptor_and_normalized_grayscale_image(
        self,
    ) -> None:
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[:, 16:, :] = 255
        image = Image.fromarray(pixels, mode="RGB")

        descriptor, output = hog_visualization(
            image,
            orientations=9,
            pixels_per_cell=8,
            cells_per_block=2,
        )

        self.assertEqual(descriptor.ndim, 1)
        self.assertEqual(output.size, image.size)
        self.assertEqual(output.mode, "L")
        self.assertEqual(output.getextrema(), (0, 255))
        self.assertGreater(float(descriptor.max()), 0.0)

    def test_hog_visualization_rejects_nonpositive_settings(self) -> None:
        image = Image.new("L", (32, 32))

        for settings in ((0, 8, 2), (9, 0, 2), (9, 8, 0)):
            with (
                self.subTest(settings=settings),
                self.assertRaisesRegex(
                    ValueError, "All HOG settings must be positive integers"
                ),
            ):
                hog_visualization(image, *settings)

    def test_hog_visualization_rejects_images_smaller_than_selected_cells(self) -> None:
        image = Image.new("L", (16, 15))

        with self.assertRaisesRegex(ValueError, "at least 16"):
            hog_visualization(
                image,
                orientations=9,
                pixels_per_cell=8,
                cells_per_block=2,
            )

    def test_hog_visualization_tool_uses_parameters_and_reports_descriptor(
        self,
    ) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (32, 32))
        output = Image.new("L", (32, 32))
        descriptor = np.zeros(72)

        with (
            patch(
                "forensics_app.tools.hog_extraction_visualization.ask_parameters",
                return_value={
                    "orientations": 6,
                    "pixels_per_cell": 4,
                    "cells_per_block": 2,
                },
            ) as ask_parameters,
            patch(
                "forensics_app.tools.hog_extraction_visualization.hog_visualization",
                return_value=(descriptor, output),
            ) as visualize,
        ):
            result = HOGVisualizationTool().run(None, document)

        ask_parameters.assert_called_once_with(
            None,
            "HOG visualization",
            (
                ParameterField("orientations", "Orientations", 2, 18, 1, 9),
                ParameterField("pixels_per_cell", "Pixels per cell", 4, 32, 1, 8),
                ParameterField("cells_per_block", "Cells per block", 1, 4, 1, 2),
            ),
        )
        visualize.assert_called_once_with(document.current, 6, 4, 2)
        self.assertIs(result.image, output)
        self.assertEqual(result.message, "Extracted HOG descriptor and visualization.")
        self.assertEqual(
            result.details,
            {
                "Operation": "Histogram of Oriented Gradients",
                "Orientations": 6,
                "Pixels per cell": "4 × 4",
                "Cells per block": "2 × 2",
                "Descriptor length": 72,
            },
        )

    def test_hog_visualization_tool_returns_none_when_dialog_is_cancelled(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (32, 32))

        with (
            patch(
                "forensics_app.tools.hog_extraction_visualization.ask_parameters",
                return_value=None,
            ),
            patch(
                "forensics_app.tools.hog_extraction_visualization.hog_visualization"
            ) as visualize,
        ):
            result = HOGVisualizationTool().run(None, document)

        self.assertIsNone(result)
        visualize.assert_not_called()

    def test_hog_visualization_tool_is_registered_by_default(self) -> None:
        tool_ids = {tool.tool_id for tool in build_tool_registry().all()}

        self.assertIn(HOGVisualizationTool.tool_id, tool_ids)


if __name__ == "__main__":
    unittest.main()
