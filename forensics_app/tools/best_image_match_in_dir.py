"""Find the closest image in a folder using feature matching and RANSAC."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image
from skimage import measure, transform
from skimage.feature import match_descriptors

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .image_matching import extract_descriptors, plot_matches

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def compare_features(
    query_keypoints: np.ndarray,
    query_descriptors: np.ndarray,
    candidate_keypoints: np.ndarray,
    candidate_descriptors: np.ndarray,
    method: str = "ORB",
    *,
    cross_check: bool = True,
    max_ratio: float = 0.8,
    residual_threshold: float = 2.0,
    max_trials: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any] | None:
    """Match descriptors between query and candidate, run RANSAC homography, and return metrics.

    Parameters
    ----------
    query_keypoints : np.ndarray
        (N, 2) array of query keypoints in (row, col) order.
    query_descriptors : np.ndarray
        Descriptor matrix for query image.
    candidate_keypoints : np.ndarray
        (M, 2) array of candidate keypoints in (row, col) order.
    candidate_descriptors : np.ndarray
        Descriptor matrix for candidate image.
    method : str
        'ORB' (Hamming metric) or 'SIFT' (Euclidean metric).
    cross_check : bool
        Whether to enforce bidirectional matching.
    max_ratio : float
        Lowe's nearest-neighbour distance ratio threshold.
    residual_threshold : float
        RANSAC inlier residual distance in pixels.
    max_trials : int
        Maximum RANSAC iterations.
    random_seed : int
        RNG seed for reproducible RANSAC results.

    Returns
    -------
    dict[str, Any] | None
        Dictionary with matching and RANSAC results, or None if insufficient matches/model failure.
    """
    if len(query_descriptors) == 0 or len(candidate_descriptors) == 0:
        return None

    if query_descriptors.shape[1] != candidate_descriptors.shape[1]:
        return None

    metric = "hamming" if method.upper() == "ORB" else "euclidean"

    matches = match_descriptors(
        query_descriptors,
        candidate_descriptors,
        metric=metric,
        cross_check=cross_check,
        max_ratio=max_ratio,
    )

    n_matches = len(matches)
    if n_matches < 4:
        return None

    # scikit-image keypoints are (row, col) = (y, x). Reversing gives (x, y) for transforms.
    src = query_keypoints[matches[:, 0]][:, ::-1]
    dst = candidate_keypoints[matches[:, 1]][:, ::-1]

    try:
        model, inliers = measure.ransac(
            (src, dst),
            transform.ProjectiveTransform,
            min_samples=4,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            rng=random_seed,
        )
    except (ValueError, RuntimeError):
        return None

    if model is None or inliers is None:
        return None

    n_inliers = int(np.count_nonzero(inliers))
    inlier_ratio = n_inliers / n_matches

    return {
        "matches": matches,
        "inliers": inliers,
        "model": model,
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
    }


def find_best_image_match(
    query_image: Image.Image,
    query_path: str | Path | None,
    folder_path: str | Path,
    method: str = "ORB",
    *,
    n_keypoints: int = 500,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
    cross_check: bool = True,
    max_ratio: float = 0.8,
    residual_threshold: float = 2.0,
    max_trials: int = 1000,
    min_inliers_for_confident_match: int = 8,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Search a directory of images for the closest match to the query image."""
    query_kp, query_desc = extract_descriptors(
        query_image,
        method=method,
        n_keypoints=n_keypoints,
        n_octaves=n_octaves,
        n_scales=n_scales,
        sigma_min=sigma_min,
    )

    if len(query_desc) == 0:
        return {
            "status": "no_match",
            "reason": "Query image does not have enough detectable features.",
        }

    query_resolved_path = Path(query_path).resolve() if query_path else None
    folder = Path(folder_path)

    results: list[dict[str, Any]] = []
    total_candidates_checked = 0

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Exclude the query image if found again in the search directory
        if query_resolved_path is not None and file_path.resolve() == query_resolved_path:
            continue

        total_candidates_checked += 1

        try:
            with Image.open(file_path) as img:
                cand_img = img.copy()
        except (OSError, ValueError):
            continue

        cand_kp, cand_desc = extract_descriptors(
            cand_img,
            method=method,
            n_keypoints=n_keypoints,
            n_octaves=n_octaves,
            n_scales=n_scales,
            sigma_min=sigma_min,
        )

        if len(cand_desc) == 0:
            continue

        comp = compare_features(
            query_kp,
            query_desc,
            cand_kp,
            cand_desc,
            method=method,
            cross_check=cross_check,
            max_ratio=max_ratio,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            random_seed=random_seed,
        )

        if comp is not None:
            comp["path"] = file_path
            comp["candidate_image"] = cand_img
            comp["candidate_keypoints"] = cand_kp
            results.append(comp)

    if not results:
        return {
            "status": "no_match",
            "reason": "No candidate in directory had enough geometrically consistent matches.",
            "total_candidates_checked": total_candidates_checked,
        }

    # Rank primarily by RANSAC inliers; break ties using inlier ratio and raw match count
    results.sort(
        key=lambda r: (r["n_inliers"], r["inlier_ratio"], r["n_matches"]),
        reverse=True,
    )

    best = results[0]
    best["status"] = (
        "matched"
        if best["n_inliers"] >= min_inliers_for_confident_match
        else "low_confidence"
    )
    best["query_keypoints"] = query_kp
    best["total_candidates_checked"] = total_candidates_checked
    best["total_matched_candidates"] = len(results)
    return best


def ask_directory_match_parameters(
    parent: tk.Misc, initial_method: str = "ORB"
) -> dict[str, Any] | None:
    """Display dialog to configure descriptor and RANSAC matching settings.

    The same descriptor extraction configuration is used for both the query and
    candidate images.
    """
    dialog = tk.Toplevel(parent)
    dialog.title("Directory Matching Settings")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    method_var = tk.StringVar(
        value="SIFT" if initial_method.upper() == "SIFT" else "ORB"
    )

    # Extraction parameters
    orb_n_var = tk.StringVar(value="500")
    sift_oct_var = tk.StringVar(value="8")
    sift_scale_var = tk.StringVar(value="3")
    sift_sigma_var = tk.StringVar(value="1.6")

    # Matching & RANSAC parameters
    cross_check_var = tk.BooleanVar(value=True)
    max_ratio_var = tk.StringVar(value="0.8")
    residual_thresh_var = tk.StringVar(value="2.0")
    max_trials_var = tk.StringVar(value="1000")
    min_inliers_var = tk.StringVar(value="8")

    main_container = ttk.Frame(dialog, padding=(12, 10))
    main_container.pack(fill="both", expand=True)

    # Method selector
    method_frame = ttk.LabelFrame(main_container, text="Descriptor Method", padding=(10, 6))
    method_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(method_frame, text="Extraction method:").pack(side="left", padx=(0, 10))
    combo = ttk.Combobox(
        method_frame,
        textvariable=method_var,
        values=("ORB", "SIFT"),
        state="readonly",
        width=10,
    )
    combo.pack(side="left")

    # Descriptor parameters frame
    desc_frame = ttk.LabelFrame(
        main_container,
        text="Descriptor Extraction Settings (Query & Candidates)",
        padding=(10, 8),
    )
    desc_frame.pack(fill="x", pady=(0, 8))

    # ORB controls
    orb_lbl = ttk.Label(desc_frame, text="ORB: Max descriptors / keypoints:")
    orb_lbl.grid(row=0, column=0, sticky="w", pady=4)
    orb_spin = ttk.Spinbox(
        desc_frame,
        textvariable=orb_n_var,
        from_=1,
        to=5000,
        increment=50,
        width=8,
    )
    orb_spin.grid(row=0, column=1, sticky="w", padx=(8, 0), pady=4)

    # SIFT controls
    sift_oct_lbl = ttk.Label(desc_frame, text="SIFT: Max octaves:")
    sift_oct_lbl.grid(row=1, column=0, sticky="w", pady=4)
    sift_oct_spin = ttk.Spinbox(
        desc_frame,
        textvariable=sift_oct_var,
        from_=1,
        to=8,
        increment=1,
        width=8,
    )
    sift_oct_spin.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=4)

    sift_scale_lbl = ttk.Label(desc_frame, text="SIFT: Scales per octave:")
    sift_scale_lbl.grid(row=2, column=0, sticky="w", pady=4)
    sift_scale_spin = ttk.Spinbox(
        desc_frame,
        textvariable=sift_scale_var,
        from_=1,
        to=8,
        increment=1,
        width=8,
    )
    sift_scale_spin.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=4)

    sift_sigma_lbl = ttk.Label(desc_frame, text="SIFT: Initial sigma:")
    sift_sigma_lbl.grid(row=3, column=0, sticky="w", pady=4)
    sift_sigma_spin = ttk.Spinbox(
        desc_frame,
        textvariable=sift_sigma_var,
        from_=1.0,
        to=3.0,
        increment=0.1,
        width=8,
    )
    sift_sigma_spin.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=4)

    orb_widgets = [orb_lbl, orb_spin]
    sift_widgets = [sift_oct_lbl, sift_oct_spin, sift_scale_lbl, sift_scale_spin, sift_sigma_lbl, sift_sigma_spin]

    def _update_visibility(method: str) -> None:
        if method == "ORB":
            for w in orb_widgets:
                w.configure(state="normal")
            for w in sift_widgets:
                w.configure(state="disabled")
        else:
            for w in orb_widgets:
                w.configure(state="disabled")
            for w in sift_widgets:
                w.configure(state="normal")

    method_var.trace_add("write", lambda *_args: _update_visibility(method_var.get()))
    _update_visibility(method_var.get())

    # Matching & RANSAC frame
    match_frame = ttk.LabelFrame(main_container, text="Matching & RANSAC Settings", padding=(10, 8))
    match_frame.pack(fill="x", pady=(0, 8))

    ttk.Checkbutton(
        match_frame,
        text="Cross-check matches",
        variable=cross_check_var,
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=3)

    ttk.Label(match_frame, text="Max distance ratio:").grid(row=1, column=0, sticky="w", pady=3)
    ttk.Spinbox(
        match_frame,
        textvariable=max_ratio_var,
        from_=0.1,
        to=1.0,
        increment=0.05,
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=3)

    ttk.Label(match_frame, text="RANSAC residual threshold (px):").grid(row=2, column=0, sticky="w", pady=3)
    ttk.Spinbox(
        match_frame,
        textvariable=residual_thresh_var,
        from_=0.5,
        to=20.0,
        increment=0.5,
        width=8,
    ).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=3)

    ttk.Label(match_frame, text="RANSAC max trials:").grid(row=3, column=0, sticky="w", pady=3)
    ttk.Spinbox(
        match_frame,
        textvariable=max_trials_var,
        from_=50,
        to=10000,
        increment=100,
        width=8,
    ).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=3)

    ttk.Label(match_frame, text="Min inliers for confident match:").grid(row=4, column=0, sticky="w", pady=3)
    ttk.Spinbox(
        match_frame,
        textvariable=min_inliers_var,
        from_=1,
        to=100,
        increment=1,
        width=8,
    ).grid(row=4, column=1, sticky="w", padx=(8, 0), pady=3)

    result: dict[str, Any] | None = None

    def apply() -> None:
        nonlocal result
        method = method_var.get()

        try:
            if method == "ORB":
                orb_n = int(orb_n_var.get())
                if orb_n < 1:
                    raise ValueError
                desc_settings = {"n_keypoints": orb_n}
            else:
                octaves = int(sift_oct_var.get())
                scales = int(sift_scale_var.get())
                sigma = float(sift_sigma_var.get())
                if octaves < 1 or scales < 1 or sigma < 1.0:
                    raise ValueError
                desc_settings = {
                    "n_octaves": octaves,
                    "n_scales": scales,
                    "sigma_min": sigma,
                }

            max_ratio = float(max_ratio_var.get())
            residual_thresh = float(residual_thresh_var.get())
            max_trials = int(max_trials_var.get())
            min_inliers = int(min_inliers_var.get())

            if not 0.0 < max_ratio <= 1.0 or residual_thresh <= 0 or max_trials < 1 or min_inliers < 1:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror(
                "Invalid Settings",
                "Please verify all numeric parameters are positive numbers within valid ranges "
                "(SIFT initial sigma >= 1.0, max ratio in (0, 1]).",
                parent=dialog,
            )
            return

        result = {
            "method": method,
            "desc_settings": desc_settings,
            "cross_check": cross_check_var.get(),
            "max_ratio": max_ratio,
            "residual_threshold": residual_thresh,
            "max_trials": max_trials,
            "min_inliers_for_confident_match": min_inliers,
        }
        dialog.destroy()

    button_bar = ttk.Frame(main_container)
    button_bar.pack(fill="x", pady=(8, 0))
    ttk.Button(button_bar, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(button_bar, text="Apply", command=apply).pack(side="right", padx=(0, 8))

    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class BestImageMatchInDirTool(ForensicsTool):
    """Forensics tool to search a folder and identify the best matching image using RANSAC."""

    tool_id = "best_image_match_in_dir"
    title = "Best image match in directory"
    category = "Matching & Descriptors"
    description = "Search an image directory and identify the best match using RANSAC inliers."
    requires_image = True

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        *,
        folder_path: str | Path | None = None,
        parameters: dict[str, Any] | None = None,
        show_plot: bool = False,
    ) -> ToolResult | None:
        """Search a directory for the best match to document.current."""
        assert document.current is not None
        query_image = document.current

        if folder_path is None:
            initialdir = str(document.path.parent) if document.path else None
            selected_dir = filedialog.askdirectory(
                parent=parent,
                title="Select image folder to search",
                initialdir=initialdir,
            )
            if not selected_dir:
                return None
            folder_path = Path(selected_dir)
        else:
            folder_path = Path(folder_path)

        if parameters is None:
            parameters = ask_directory_match_parameters(parent, initial_method="ORB")
            if parameters is None:
                return None

        method = parameters["method"]
        desc_settings = parameters["desc_settings"]
        cross_check = parameters.get("cross_check", True)
        max_ratio = parameters.get("max_ratio", 0.8)
        residual_threshold = parameters.get("residual_threshold", 2.0)
        max_trials = parameters.get("max_trials", 1000)
        min_inliers_for_confident_match = parameters.get(
            "min_inliers_for_confident_match", 8
        )

        match_kwargs = {
            "method": method,
            "cross_check": cross_check,
            "max_ratio": max_ratio,
            "residual_threshold": residual_threshold,
            "max_trials": max_trials,
            "min_inliers_for_confident_match": min_inliers_for_confident_match,
            **desc_settings,
        }

        best = find_best_image_match(
            query_image=query_image,
            query_path=document.path,
            folder_path=folder_path,
            **match_kwargs,
        )

        if best.get("status") == "no_match":
            return ToolResult(
                image=None,
                message=f"No match found: {best.get('reason', 'No candidate met match criteria.')}",
                details={
                    "Search directory": str(folder_path),
                    "Status": "no_match",
                    "Method": method,
                },
            )

        best_path = Path(best["path"])
        cand_img = best["candidate_image"]
        matches = best["matches"]
        inliers = best["inliers"]

        # Connect the inlier correspondences (or all matches if no inliers)
        plot_matches_array = matches[inliers] if np.any(inliers) else matches

        ref_name = Path(document.path).name if document.path else "Reference"
        best_name = best_path.name
        n_display_matches = len(plot_matches_array)
        plot_title = f"Reference: {ref_name} <---> Best: {best_name}  (matches={n_display_matches})"

        output_image = plot_matches(
            query_image,
            cand_img,
            best["query_keypoints"],
            best["candidate_keypoints"],
            plot_matches_array,
            show=show_plot,
            title=plot_title,
            keypoint_size=6.0,
            keypoint_color="red",
        )

        metric = "Hamming" if method == "ORB" else "Euclidean"
        status = best["status"]

        details: dict[str, Any] = {
            "Best match path": str(best_path),
            "Best match filename": best_path.name,
            "Status": status,
            "RANSAC inliers": best["n_inliers"],
            "Inlier ratio": f"{best['inlier_ratio']:.3f}",
            "Raw descriptor matches": best["n_matches"],
            "Confident threshold": min_inliers_for_confident_match,
            "Candidates searched": best["total_candidates_checked"],
            "Matched candidates": best["total_matched_candidates"],
            "Method": method,
            "Metric": metric,
            "Search directory": str(folder_path),
        }

        if method == "ORB":
            details["Max descriptors"] = desc_settings.get("n_keypoints")
        else:
            details["SIFT octaves / scales / sigma"] = (
                f"{desc_settings.get('n_octaves')} / "
                f"{desc_settings.get('n_scales')} / "
                f"{desc_settings.get('sigma_min')}"
            )

        if status == "matched":
            message = (
                f"Best match found: {best_path.name} "
                f"({best['n_inliers']} inliers, {best['inlier_ratio']:.1%} inlier ratio)."
            )
        else:
            message = (
                f"Low confidence match: {best_path.name} "
                f"({best['n_inliers']} inliers, below confident threshold of {min_inliers_for_confident_match})."
            )

        return ToolResult(image=output_image, message=message, details=details)
