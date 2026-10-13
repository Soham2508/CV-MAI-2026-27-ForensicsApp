import numpy as np
from PIL import Image
from skimage.feature import canny

from forensics_app.core import ImageDocument
from forensics_app.utils.parameter_dialog import ParameterField, ask_parameters

from .base import ForensicsTool, ToolResult


class CannyTool(ForensicsTool):
    tool_id = "canny"
    title = "Canny edge detection"
    category = "Filtering & Edges"
    description = "Detect edges with the Canny method."

    def run(self, parent, document: ImageDocument) -> ToolResult | None:
        parameters = ask_parameters(
            parent,
            self.title,
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
            lambda values: (
                "The low threshold must not exceed the high threshold."
                if values["low"] > values["high"]
                else None
            ),
        )
        if parameters is None:
            return None

        assert document.current is not None

        # Canny expects a single-channel image.
        gray = document.current.convert("L")
        pixels = np.asarray(gray, dtype=np.float32) / 255.0

        # The result is a Boolean array: True where an edge was found.
        edges = canny(
            pixels,
            sigma=parameters["sigma"],
            low_threshold=parameters["low"],
            high_threshold=parameters["high"],
        )

        # Convert True/False to white/black pixels for the app display.
        output = Image.fromarray(edges.astype(np.uint8) * 255)

        return ToolResult(
            image=output,
            message=(
                f"Canny edge detection completed (sigma={parameters['sigma']:g}, "
                f"low={parameters['low']:g}, high={parameters['high']:g})."
            ),
            details={
                "Operation": "Canny edge detection",
                "Sigma": parameters["sigma"],
                "Low threshold": parameters["low"],
                "High threshold": parameters["high"],
                "Edge pixels": int(edges.sum()),
            },
        )
