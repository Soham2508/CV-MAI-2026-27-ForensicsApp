"""Find a template in the current image using normalized cross-correlation and thresholding."""

from __future__ import annotations

import io
import math
import tkinter as tk
from tkinter import filedialog
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import match_template
from skimage.transform import rotate

from forensics_app.core import ImageDocument
from forensics_app.utils import ask_parameters
from forensics_app.utils.parameter_dialog import ParameterField

from .base import ForensicsTool, ToolResult

IMAGE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def compute_normalized_cross_correlation(
    image: Image.Image,
    template: Image.Image,
    angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grayscale image array and normalized cross-correlation map.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image in which the template will be searched.
    template : PIL.Image.Image
        Template image to locate.
    angle : float
        Rotation angle in degrees to apply to the template (-180 to 180).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (image_array, correlation_map) where correlation_map has values in [-1.0, 1.0].
    """
    if not math.isfinite(angle) or not -180 <= angle <= 180:
        raise ValueError("Rotation angle must be between -180 and 180 degrees.")

    image_array = np.asarray(image.convert("L"))
    template_array = rotate(
        np.asarray(template.convert("L")),
        angle=angle,
        preserve_range=True,
    )

    if (
        template_array.shape[0] > image_array.shape[0]
        or template_array.shape[1] > image_array.shape[1]
    ):
        raise ValueError("The template must not be larger than the original image.")

    correlation = match_template(image_array, template_array, pad_input=True)
    return image_array, correlation


def superimpose_matches(
    image: Image.Image,
    correlation: np.ndarray,
    threshold: float,
    *,
    mode: str = "red",
    color: tuple[int, int, int] = (255, 0, 0),
    point_radius: int = 5,
) -> Image.Image:
    """Superimpose matching regions (correlation >= threshold) on the image.

    Modes supported (referencing Set 3 superimposition techniques):
    1. 'red' / 'red_on_gray': Red pixels on image (default).
    2. 'transparent_red' / 'overlay': 50% semi-transparent red overlay.
    3. 'dimmed' / 'darken': Darken background by 0.4 and highlight matches in white.
    4. 'contours' / 'contour': Red boundary contour around matching regions.

    Parameters
    ----------
    image : PIL.Image.Image
        Base image to superimpose onto.
    correlation : np.ndarray
        Normalized cross-correlation map with same spatial dimensions as image.
    threshold : float
        Correlation threshold in [-1.0, 1.0].
    mode : str
        Superimposition style.
    color : tuple[int, int, int]
        RGB color tuple for overlay (default bright red (255, 0, 0)).
    point_radius : int
        Visual size in pixels to expand single-pixel detections for clarity.

    Returns
    -------
    PIL.Image.Image
        RGB image with matches superimposed.
    """
    mask = correlation >= threshold
    rgb = np.asarray(image.convert("RGB")).copy()

    if point_radius > 1 and np.any(mask):
        from scipy.ndimage import maximum_filter

        visible_mask = maximum_filter(mask, size=point_radius, mode="constant", cval=0)
    else:
        visible_mask = mask

    mode_lower = mode.lower()
    if mode_lower in ("red", "red_on_gray", "default"):
        rgb[visible_mask] = color
    elif mode_lower in ("transparent_red", "overlay", "transparent"):
        overlay_color = np.array(color, dtype=np.float32)
        blended = rgb.astype(np.float32)
        blended[visible_mask] = 0.5 * blended[visible_mask] + 0.5 * overlay_color
        rgb = np.clip(blended, 0, 255).astype(np.uint8)
    elif mode_lower in ("dimmed", "darken"):
        dimmed = np.clip(rgb.astype(np.float32) * 0.4, 0, 255).astype(np.uint8)
        dimmed[visible_mask] = (255, 255, 255)
        rgb = dimmed
    elif mode_lower in ("contours", "contour"):
        from scipy.ndimage import binary_dilation

        dilated = binary_dilation(visible_mask, iterations=1)
        contour = dilated & (~visible_mask)
        rgb[contour if contour.any() else visible_mask] = color
    else:
        rgb[visible_mask] = color

    return Image.fromarray(rgb, mode="RGB")


def plot_template_matching(
    image: Image.Image,
    template: Image.Image,
    angle: float = 0.0,
    threshold: float = 0.8,
    *,
    superimpose_mode: str = "red",
    point_radius: int = 5,
) -> Image.Image:
    """Plot rotated template and thresholded image side-by-side matching Set 4.2.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.
    template : PIL.Image.Image
        Template image to search for.
    angle : float
        Rotation angle in degrees.
    threshold : float
        Correlation threshold in [-1.0, 1.0].
    superimpose_mode : str
        Superimposition style (default 'red').
    point_radius : int
        Visual size in pixels of matching points (default 5).

    Returns
    -------
    PIL.Image.Image
        Rendered figure showing 'Template' and 'Thresholded image'.
    """
    image_array, correlation = compute_normalized_cross_correlation(
        image, template, angle=angle
    )

    template_array = rotate(
        np.asarray(template.convert("L")),
        angle=angle,
        preserve_range=True,
    )

    thresholded = superimpose_matches(
        image,
        correlation,
        threshold=threshold,
        mode=superimpose_mode,
        point_radius=point_radius,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.0), dpi=100)

    # Subplot 1: Template
    axes[0].imshow(template_array, cmap="gray")
    axes[0].set_title("Template")
    axes[0].axis("off")

    # Subplot 2: Thresholded image
    axes[1].imshow(thresholded)
    axes[1].set_title("Thresholded image")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    output = Image.open(buf).convert("RGB")
    plt.close(fig)

    return output


def normalized_cross_correlation(
    image: Image.Image,
    template: Image.Image,
    angle: float = 0.0,
    threshold: float | None = None,
    superimpose_mode: str = "red",
    show_side_by_side: bool = False,
    point_radius: int = 5,
) -> Image.Image:
    """Return a visualization of the normalized cross-correlation map or thresholded matches.

    If threshold is None, returns a grayscale visualization of the correlation map (Set 4.1).
    If threshold is specified (-1.0 <= threshold <= 1.0), returns an image with
    matches superimposed in red (Set 4.2). If show_side_by_side is True, renders both
    Template and Thresholded image side by side.
    """
    image_array, correlation = compute_normalized_cross_correlation(
        image, template, angle=angle
    )

    if threshold is None:
        visualization = np.rint(
            np.clip((correlation + 1) * 127.5, 0, 255)
        ).astype(np.uint8)
        return Image.fromarray(visualization)

    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("Correlation threshold must be a finite number between -1.0 and 1.0.")

    if show_side_by_side:
        return plot_template_matching(
            image,
            template,
            angle=angle,
            threshold=threshold,
            superimpose_mode=superimpose_mode,
            point_radius=point_radius,
        )

    return superimpose_matches(
        image,
        correlation,
        threshold=threshold,
        mode=superimpose_mode,
        point_radius=point_radius,
    )


def threshold_normalized_cross_correlation(
    image: Image.Image,
    template: Image.Image,
    threshold: float = 0.8,
    angle: float = 0.0,
    superimpose_mode: str = "red",
    show_side_by_side: bool = False,
    point_radius: int = 5,
) -> Image.Image:
    """Return image with template matches exceeding threshold superimposed."""
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("Correlation threshold must be a finite number between -1.0 and 1.0.")

    return normalized_cross_correlation(
        image=image,
        template=template,
        angle=angle,
        threshold=threshold,
        superimpose_mode=superimpose_mode,
        show_side_by_side=show_side_by_side,
        point_radius=point_radius,
    )


class NormalizedCrossCorrelationTool(ForensicsTool):
    tool_id = "normalized_cross_correlation"
    title = "Normalized cross-correlation"
    category = "Template Matching"
    description = "Rotate a template and find it in the currently selected image."
    menu_options = ("Correlation map", "Thresholded image")

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        *,
        action: str | None = None,
        template_path: str | None = None,
        parameters: dict[str, float] | None = None,
    ) -> ToolResult | None:
        default_action = (
            "Thresholded image"
            if (action is not None and "thresh" in action.lower())
            or (parameters is not None and "threshold" in parameters)
            else "Correlation map"
        )
        chosen_action = action or default_action
        return self.run_action(
            parent,
            document,
            chosen_action,
            template_path=template_path,
            parameters=parameters,
        )

    def run_action(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        action: str,
        *,
        template_path: str | None = None,
        parameters: dict[str, float] | None = None,
    ) -> ToolResult | None:
        assert document.current is not None

        if template_path is None:
            template_path = filedialog.askopenfilename(
                parent=parent,
                title="Select template image",
                filetypes=IMAGE_TYPES,
            )
            if not template_path:
                return None

        with Image.open(template_path) as image:
            template = image.copy()

        is_thresholded = (
            "thresh" in action.lower()
            or (parameters is not None and "threshold" in parameters)
        )

        if parameters is None:
            if is_thresholded:
                parameters = ask_parameters(
                    parent,
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
            else:
                parameters = ask_parameters(
                    parent,
                    "Template rotation",
                    (("angle", "Rotation angle (degrees)", -180, 180, 1, 0),),
                )
            if parameters is None:
                return None

        angle = float(parameters["angle"])
        threshold = (
            float(parameters["threshold"])
            if is_thresholded and "threshold" in parameters
            else None
        )

        original = document.current

        details: dict[str, Any] = {
            "Original image size": f"{original.width} × {original.height}",
            "Template size": f"{template.width} × {template.height}",
            "Rotation angle": f"{angle:g}°",
        }

        if is_thresholded and threshold is not None:
            output = plot_template_matching(
                original,
                template,
                angle=angle,
                threshold=threshold,
            )
            _, correlation = compute_normalized_cross_correlation(
                original, template, angle=angle
            )
            matches_count = int(np.count_nonzero(correlation >= threshold))
            details["Threshold"] = f"{threshold:.2f}"
            details["Matches found"] = matches_count
            details["Output size"] = f"{output.width} × {output.height}"

            message = (
                f"Found {matches_count} matching pixels exceeding "
                f"correlation threshold {threshold:.2f} (rotation {angle:g}°)."
            )
            if matches_count == 0:
                message += " Try lowering the threshold or adjusting rotation."
        else:
            output = normalized_cross_correlation(
                original, template, angle=angle, threshold=None
            )
            details["Output size"] = f"{output.width} × {output.height}"
            message = "Calculated the normalized cross-correlation image."

        return ToolResult(
            image=output,
            message=message,
            details=details,
        )
