"""Feature matching between two images using ORB or SIFT descriptors."""

from __future__ import annotations

import io
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import ORB, SIFT, match_descriptors, plot_matched_features

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

IMAGE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def extract_descriptors(
    image: Image.Image,
    method: str = "ORB",
    *,
    n_keypoints: int = 500,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract keypoints and descriptors from an image using ORB or SIFT.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    method : str
        Feature extraction method, either 'ORB' or 'SIFT'.
    n_keypoints : int
        Maximum number of keypoints for ORB.
    n_octaves : int
        Number of octaves for SIFT.
    n_scales : int
        Number of scales per octave for SIFT.
    sigma_min : float
        Initial Gaussian blur standard deviation for SIFT.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (keypoints, descriptors) where keypoints has shape (N, 2) with (row, col) coordinates,
        and descriptors has shape (N, 256) for ORB or (N, 128) for SIFT.
    """
    method_upper = method.upper()
    if method_upper not in ("ORB", "SIFT"):
        raise ValueError("Choose ORB or SIFT.")

    if method_upper == "ORB":
        if type(n_keypoints) is not int or n_keypoints < 1:
            raise ValueError("ORB keypoint count must be a positive integer.")
        extractor = ORB(n_keypoints=n_keypoints)
        descriptor_width, descriptor_dtype = 256, np.bool_
        minimum_size = 32
    else:
        if type(n_octaves) is not int or n_octaves < 1:
            raise ValueError("SIFT octave count must be a positive integer.")
        if type(n_scales) is not int or n_scales < 1:
            raise ValueError("SIFT scale count must be a positive integer.")
        if not math.isfinite(sigma_min) or sigma_min < 1.0:
            raise ValueError("SIFT sigma_min must be finite and at least 1.0.")
        extractor = SIFT(
            upsampling=2,
            n_octaves=n_octaves,
            n_scales=n_scales,
            sigma_min=sigma_min,
        )
        descriptor_width, descriptor_dtype = 128, np.uint8
        minimum_size = 12

    if min(image.size) < minimum_size:
        raise ValueError(
            f"{method_upper} needs an image at least "
            f"{minimum_size} by {minimum_size} pixels for this tool."
        )

    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0

    try:
        extractor.detect_and_extract(gray)
    except RuntimeError as error:
        if not str(error).startswith(f"{method_upper} found no features."):
            raise
        keypoints = np.empty((0, 2), dtype=np.float64)
        descriptors = np.empty((0, descriptor_width), dtype=descriptor_dtype)
    else:
        keypoints = extractor.keypoints.copy()
        descriptors = extractor.descriptors.copy()

    return keypoints, descriptors


def match_image_descriptors(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    method: str = "ORB",
    *,
    cross_check: bool = True,
    max_ratio: float = 0.8,
) -> np.ndarray:
    """Find matches between two sets of descriptors of the same type.

    Parameters
    ----------
    descriptors1 : np.ndarray
        Descriptor matrix from the first image.
    descriptors2 : np.ndarray
        Descriptor matrix from the second image.
    method : str
        Method used ('ORB' for Hamming distance, 'SIFT' for Euclidean distance).
    cross_check : bool
        Whether to enforce symmetric bidirectional matching.
    max_ratio : float
        Lowe's distance ratio threshold between 1st and 2nd nearest neighbors.

    Returns
    -------
    np.ndarray
        Array of matches with shape (M, 2), mapping descriptor indices in
        image 1 to descriptor indices in image 2.
    """
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return np.empty((0, 2), dtype=int)

    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError(
            f"Descriptor length mismatch ({descriptors1.shape[1]} vs {descriptors2.shape[1]}). "
            "Both images must use the same extraction method."
        )

    metric = "hamming" if method.upper() == "ORB" else "euclidean"
    matches = match_descriptors(
        descriptors1,
        descriptors2,
        metric=metric,
        cross_check=cross_check,
        max_ratio=max_ratio,
    )
    return matches


def plot_matches(
    image1: Image.Image,
    image2: Image.Image,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> Image.Image:
    """Plot matched features across two images on a clean plane in full colour.

    Parameters
    ----------
    image1 : PIL.Image.Image
        Left image (in full colour).
    image2 : PIL.Image.Image
        Right image (in full colour).
    keypoints1 : np.ndarray
        (N, 2) array of (row, col) coordinates for left image keypoints.
    keypoints2 : np.ndarray
        (M, 2) array of (row, col) coordinates for right image keypoints.
    matches : np.ndarray
        (Q, 2) array of correspondence indices.
    ax : plt.Axes | None
        Optional existing matplotlib axis.
    show : bool
        Whether to display via plt.show().

    Returns
    -------
    PIL.Image.Image
        Rendered visualization of side-by-side images with keypoint matches in full colour.
    """
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # Use full color RGB representations
    img1_rgb = np.asarray(image1.convert("RGB"))
    img2_rgb = np.asarray(image2.convert("RGB"))

    plot_matched_features(
        img1_rgb,
        img2_rgb,
        keypoints0=keypoints1,
        keypoints1=keypoints2,
        matches=matches,
        ax=ax,
        keypoints_color="k",
        matches_color=None,
    )

    # Clean plane presentation (no axes ticks or frame borders)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    output_image = Image.open(buf).convert("RGB")

    if show:
        plt.show()

    if owns_figure:
        plt.close(fig)

    return output_image


def ask_matching_parameters(
    parent: tk.Misc, initial_method: str = "ORB"
) -> dict[str, Any] | None:
    """Display dialog to configure descriptor extraction and matching settings.

    The extraction method (ORB or SIFT) is selected once for both images, while
    individual parameters for each image can be configured independently.
    """
    dialog = tk.Toplevel(parent)
    dialog.title("Image Matching Settings")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    method_var = tk.StringVar(
        value="SIFT" if initial_method.upper() == "SIFT" else "ORB"
    )

    # First image params
    first_orb_n_var = tk.StringVar(value="500")
    first_sift_oct_var = tk.StringVar(value="8")
    first_sift_scale_var = tk.StringVar(value="3")
    first_sift_sigma_var = tk.StringVar(value="1.6")

    # Second image params
    second_orb_n_var = tk.StringVar(value="500")
    second_sift_oct_var = tk.StringVar(value="8")
    second_sift_scale_var = tk.StringVar(value="3")
    second_sift_sigma_var = tk.StringVar(value="1.6")

    cross_check_var = tk.BooleanVar(value=True)
    max_ratio_var = tk.StringVar(value="0.8")

    first_widgets: dict[str, list[tk.Widget]] = {"ORB": [], "SIFT": []}
    second_widgets: dict[str, list[tk.Widget]] = {"ORB": [], "SIFT": []}

    main_container = ttk.Frame(dialog, padding=(12, 10))
    main_container.pack(fill="both", expand=True)

    # Top method selector (applies to both images)
    method_frame = ttk.LabelFrame(main_container, text="Descriptor Method", padding=(10, 6))
    method_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

    ttk.Label(method_frame, text="Extraction method for both images:").pack(
        side="left", padx=(0, 10)
    )
    combo = ttk.Combobox(
        method_frame,
        textvariable=method_var,
        values=("ORB", "SIFT"),
        state="readonly",
        width=10,
    )
    combo.pack(side="left")

    def _build_image_params_frame(
        master: tk.Widget,
        title: str,
        orb_n_var: tk.StringVar,
        sift_oct_var: tk.StringVar,
        sift_scale_var: tk.StringVar,
        sift_sigma_var: tk.StringVar,
        widgets_dict: dict[str, list[tk.Widget]],
    ) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(master, text=title, padding=(10, 8))

        # ORB field
        orb_lbl = ttk.Label(frame, text="Max descriptors / keypoints:")
        orb_lbl.grid(row=0, column=0, sticky="w", pady=4)
        orb_spin = ttk.Spinbox(
            frame,
            textvariable=orb_n_var,
            from_=1,
            to=5000,
            increment=50,
            width=8,
        )
        orb_spin.grid(row=0, column=1, sticky="w", padx=(8, 0), pady=4)
        widgets_dict["ORB"].extend([orb_lbl, orb_spin])

        # SIFT fields
        sift_oct_lbl = ttk.Label(frame, text="Max octaves:")
        sift_oct_lbl.grid(row=1, column=0, sticky="w", pady=4)
        sift_oct_spin = ttk.Spinbox(
            frame,
            textvariable=sift_oct_var,
            from_=1,
            to=8,
            increment=1,
            width=8,
        )
        sift_oct_spin.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=4)

        sift_scale_lbl = ttk.Label(frame, text="Scales per octave:")
        sift_scale_lbl.grid(row=2, column=0, sticky="w", pady=4)
        sift_scale_spin = ttk.Spinbox(
            frame,
            textvariable=sift_scale_var,
            from_=1,
            to=8,
            increment=1,
            width=8,
        )
        sift_scale_spin.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=4)

        sift_sigma_lbl = ttk.Label(frame, text="Initial sigma:")
        sift_sigma_lbl.grid(row=3, column=0, sticky="w", pady=4)
        sift_sigma_spin = ttk.Spinbox(
            frame,
            textvariable=sift_sigma_var,
            from_=1.0,
            to=3.0,
            increment=0.1,
            width=8,
        )
        sift_sigma_spin.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=4)

        widgets_dict["SIFT"].extend(
            [
                sift_oct_lbl,
                sift_oct_spin,
                sift_scale_lbl,
                sift_scale_spin,
                sift_sigma_lbl,
                sift_sigma_spin,
            ]
        )

        return frame

    def _update_visibility(method: str) -> None:
        for widgets_dict in (first_widgets, second_widgets):
            if method == "ORB":
                for widget in widgets_dict["ORB"]:
                    widget.configure(state="normal")
                for widget in widgets_dict["SIFT"]:
                    widget.configure(state="disabled")
            else:
                for widget in widgets_dict["ORB"]:
                    widget.configure(state="disabled")
                for widget in widgets_dict["SIFT"]:
                    widget.configure(state="normal")

    def _on_method_changed(*_args: object) -> None:
        _update_visibility(method_var.get())

    method_var.trace_add("write", _on_method_changed)

    first_frame = _build_image_params_frame(
        main_container,
        "First Image (Current)",
        first_orb_n_var,
        first_sift_oct_var,
        first_sift_scale_var,
        first_sift_sigma_var,
        first_widgets,
    )
    first_frame.grid(row=1, column=0, padx=(0, 6), sticky="nsew")

    second_frame = _build_image_params_frame(
        main_container,
        "Second Image",
        second_orb_n_var,
        second_sift_oct_var,
        second_sift_scale_var,
        second_sift_sigma_var,
        second_widgets,
    )
    second_frame.grid(row=1, column=1, padx=(6, 0), sticky="nsew")

    # Initial state
    _update_visibility(method_var.get())

    # Matching options
    match_frame = ttk.LabelFrame(main_container, text="Matching Settings", padding=(10, 8))
    match_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    ttk.Checkbutton(
        match_frame,
        text="Cross-check (bidirectional)",
        variable=cross_check_var,
    ).grid(row=0, column=0, sticky="w", pady=4)

    ttk.Label(match_frame, text="Max distance ratio (Lowe's test):").grid(
        row=0, column=1, sticky="w", padx=(20, 4), pady=4
    )
    ttk.Spinbox(
        match_frame,
        textvariable=max_ratio_var,
        from_=0.1,
        to=1.0,
        increment=0.05,
        width=6,
    ).grid(row=0, column=2, sticky="w", pady=4)

    result: dict[str, Any] | None = None

    def apply() -> None:
        nonlocal result
        method = method_var.get()

        try:
            if method == "ORB":
                first_orb_n = int(first_orb_n_var.get())
                second_orb_n = int(second_orb_n_var.get())
                if first_orb_n < 1 or second_orb_n < 1:
                    raise ValueError
                left_settings = {"n_keypoints": first_orb_n}
                right_settings = {"n_keypoints": second_orb_n}
            else:
                first_oct = int(first_sift_oct_var.get())
                first_scale = int(first_sift_scale_var.get())
                first_sigma = float(first_sift_sigma_var.get())
                second_oct = int(second_sift_oct_var.get())
                second_scale = int(second_sift_scale_var.get())
                second_sigma = float(second_sift_sigma_var.get())

                if (
                    first_oct < 1
                    or first_scale < 1
                    or first_sigma < 1.0
                    or second_oct < 1
                    or second_scale < 1
                    or second_sigma < 1.0
                ):
                    raise ValueError

                left_settings = {
                    "n_octaves": first_oct,
                    "n_scales": first_scale,
                    "sigma_min": first_sigma,
                }
                right_settings = {
                    "n_octaves": second_oct,
                    "n_scales": second_scale,
                    "sigma_min": second_sigma,
                }

            max_ratio = float(max_ratio_var.get())
            if not 0.0 < max_ratio <= 1.0:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror(
                "Invalid Settings",
                "Please ensure all numeric values are positive numbers within valid ranges "
                "(and SIFT initial sigma >= 1.0, max ratio between 0.0 and 1.0).",
                parent=dialog,
            )
            return

        result = {
            "method": method,
            "left_settings": left_settings,
            "right_settings": right_settings,
            "cross_check": cross_check_var.get(),
            "max_ratio": max_ratio,
        }
        dialog.destroy()

    button_bar = ttk.Frame(main_container)
    button_bar.grid(row=3, column=0, columnspan=2, pady=(12, 0), sticky="e")
    ttk.Button(button_bar, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(button_bar, text="Apply", command=apply).pack(side="right", padx=(0, 8))

    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class ImageMatchingTool(ForensicsTool):
    """Forensics tool for matching features across two images using ORB or SIFT."""

    tool_id = "image_matching"
    title = "Image matching"
    category = "Matching & Descriptors"
    description = "Match ORB or SIFT descriptors between current image and a second image."
    requires_image = True

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        *,
        parameters: dict[str, Any] | None = None,
        second_image_path: str | None = None,
        show_plot: bool = False,
    ) -> ToolResult | None:
        """Prompt user for second image and parameters, match descriptors, and display plot on plane."""
        assert document.current is not None
        image1 = document.current

        if second_image_path is None:
            initialdir = str(document.path.parent) if document.path else None
            second_image_path = filedialog.askopenfilename(
                parent=parent,
                title="Select second image to match",
                filetypes=IMAGE_TYPES,
                initialdir=initialdir,
            )
            if not second_image_path:
                return None

        with Image.open(second_image_path) as img:
            image2 = img.copy()

        if parameters is None:
            parameters = ask_matching_parameters(parent, initial_method="ORB")
            if parameters is None:
                return None

        method = parameters["method"]
        left_settings = parameters["left_settings"]
        right_settings = parameters["right_settings"]
        cross_check = parameters.get("cross_check", True)
        max_ratio = parameters.get("max_ratio", 0.8)

        # Extract descriptors using the same method for both images
        keypoints1, descriptors1 = extract_descriptors(
            image1, method, **left_settings
        )
        keypoints2, descriptors2 = extract_descriptors(
            image2, method, **right_settings
        )

        # Match descriptors
        matches = match_image_descriptors(
            descriptors1,
            descriptors2,
            method=method,
            cross_check=cross_check,
            max_ratio=max_ratio,
        )

        # Plot matches in full colour on a clean plane
        output_image = plot_matches(
            image1,
            image2,
            keypoints1,
            keypoints2,
            matches,
            show=show_plot,
        )

        metric = "Hamming" if method == "ORB" else "Euclidean"
        details: dict[str, Any] = {
            "Method": method,
            "First image size": f"{image1.width} × {image1.height}",
            "First keypoints": len(keypoints1),
            "Second image size": f"{image2.width} × {image2.height}",
            "Second keypoints": len(keypoints2),
            "Matches found": len(matches),
            "Metric": metric,
            "Cross-check": cross_check,
            "Max ratio": f"{max_ratio:.2f}",
        }

        if method == "ORB":
            details["First max keypoints"] = left_settings.get("n_keypoints")
            details["Second max keypoints"] = right_settings.get("n_keypoints")
        else:
            details["First octaves / scales / sigma"] = (
                f"{left_settings.get('n_octaves')} / "
                f"{left_settings.get('n_scales')} / "
                f"{left_settings.get('sigma_min')}"
            )
            details["Second octaves / scales / sigma"] = (
                f"{right_settings.get('n_octaves')} / "
                f"{right_settings.get('n_scales')} / "
                f"{right_settings.get('sigma_min')}"
            )

        message = (
            f"Found {len(matches)} matches using {method} descriptors "
            f"({len(keypoints1)} keypoints in first image, {len(keypoints2)} in second image)."
        )
        if len(matches) == 0:
            message += " No matches found. Try using higher keypoint limits or more textured images."

        return ToolResult(image=output_image, message=message, details=details)

    def run_action(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        action: str,
        **kwargs: Any,
    ) -> ToolResult | None:
        """Support callers invoking run_action directly."""
        return self.run(parent, document, **kwargs)
