"""Tool 5 — Face Recognition (PCA Projection + K-NN & Nearest Training Faces).

Exclusively performs inference using a previously saved Eigenface / PCA model
and K-NN classifier: preprocesses the test image, projects it onto the PCA basis
``w_test = U_K^T (x_test - μ)``, predicts the identity with K-NN (including
neighbour vote fraction), retrieves the closest training faces by Euclidean
distance ``d_i = ||w_test - w_i||_2``, and renders the recognition layout.
"""

from __future__ import annotations

import io
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.paths import resolve_path

from .base import ForensicsTool, ToolResult
from .eigenface_trainer import (
    DEFAULT_EIGENFACE_MODEL_DIR,
    load_eigenface_model,
    load_knn_classifier,
    project_faces_with_saved_pca,
)
from .face_detection import create_face_detector
from .hog_person_detection import IMAGE_TYPES


def preprocess_test_face(
    image: Image.Image,
    target_shape: tuple[int, int],
    *,
    value_scale: str = "0_1",
    detect_and_crop: bool = False,
    detector_type: str = "YuNet",
    status: dict[str, str] | None = None,
) -> tuple[np.ndarray, Image.Image]:
    """Convert test image to grayscale, optionally detect/crop, and resize to (h, w).

    Returns ``(flattened_vector, resized_pil_gray_image)``. When ``status`` is
    given, ``status["crop"]`` records what happened to the optional face crop,
    so a failed detection is reported instead of silently ignored.
    """
    working = image.copy()
    crop_note = "Not requested"
    if detect_and_crop:
        try:
            det = create_face_detector(detector_type)
            faces = det.detect(working)
            crop_note = "No face found; used the whole image"
            if faces:
                x, y, w_box, h_box = faces[0]["bbox"]
                x0 = max(0, int(x))
                y0 = max(0, int(y))
                x1 = min(working.width, int(x + w_box))
                y1 = min(working.height, int(y + h_box))
                if x1 > x0 and y1 > y0:
                    working = working.crop((x0, y0, x1, y1))
                    crop_note = (
                        f"Cropped to first face ({x0}, {y0}, {x1 - x0}, {y1 - y0})"
                    )
        except Exception as exc:
            crop_note = f"Face detection failed ({exc}); used the whole image"
    if status is not None:
        status["crop"] = crop_note

    h, w = int(target_shape[0]), int(target_shape[1])
    gray = working.convert("L").resize((w, h), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32).reshape(-1)
    if value_scale == "0_1":
        arr = arr / 255.0
    return arr, gray


def _infer_ground_truth_name(
    test_image_path: str | Path | None,
    pca_model: dict[str, Any],
) -> str | None:
    """Infer ground-truth identity if the test image belongs to the saved test set."""
    if not test_image_path:
        return None
    p = Path(test_image_path)
    fname = p.name
    for rec in pca_model.get("test_records", []):
        if rec.get("filename") == fname:
            return str(rec.get("person_name"))
        if rec.get("path") and Path(rec["path"]).name == fname:
            return str(rec.get("person_name"))

    # Check if filename contains one of target_names (with spaces or underscores)
    stem_norm = p.stem.lower().replace("_", " ")
    for name in pca_model.get("target_names", []):
        if str(name).lower() in stem_norm:
            return str(name)
    return None


def _load_training_match_image(
    record: dict[str, Any],
    pca_model: dict[str, Any],
) -> Image.Image:
    """Load the original training face image for a given training record."""
    h, w = pca_model["image_shape"]
    model_dir = pca_model.get("model_dir")
    candidates: list[Path] = []
    if record.get("path"):
        candidates.append(Path(record["path"]))
    if model_dir and record.get("relative_path"):
        candidates.append(Path(model_dir) / record["relative_path"])
    if model_dir and record.get("filename"):
        candidates.append(Path(model_dir) / "train_images" / record["filename"])

    for cand in candidates:
        if cand.exists() and cand.is_file():
            try:
                with Image.open(cand) as img:
                    return img.convert("L").copy()
            except OSError:
                pass

    # Fallback: reconstruct from PCA coefficient vector w_i via U_K w_i + μ
    idx = int(record.get("index", 0))
    w_i = pca_model["X_train_pca"][idx]
    if pca_model.get("whiten", True):
        scale = np.sqrt(np.maximum(pca_model["explained_variance"], 1e-12))
        w_unwhitened = w_i * scale
    else:
        w_unwhitened = w_i
    recon = w_unwhitened @ pca_model["components"] + pca_model["mean"]
    recon_2d = recon.reshape((h, w))
    vmin, vmax = float(recon_2d.min()), float(recon_2d.max())
    if vmax > vmin:
        u8 = np.clip((recon_2d - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    else:
        u8 = np.zeros((h, w), dtype=np.uint8)
    return Image.fromarray(u8, mode="L")


def recognise_face(
    image: Image.Image,
    pca_model: dict[str, Any],
    knn_bundle: dict[str, Any] | None = None,
    *,
    n_nearest: int = 3,
    test_image_path: str | Path | None = None,
    ground_truth_name: str | None = None,
    detect_and_crop: bool = False,
) -> dict[str, Any]:
    """Perform PCA projection, K-NN classification, and nearest-face retrieval."""
    t0 = time.perf_counter()
    h, w = pca_model["image_shape"]
    value_scale = pca_model.get("value_scale", "0_1")

    crop_status: dict[str, str] = {}
    x_test, preprocessed_pil = preprocess_test_face(
        image,
        (h, w),
        value_scale=value_scale,
        detect_and_crop=detect_and_crop,
        status=crop_status,
    )

    # Calculate test image's PCA coefficients: w_test = U_K^T (x_test - μ)
    w_test = project_faces_with_saved_pca(
        x_test,
        mean=pca_model["mean"],
        components=pca_model["components"],
        explained_variance=pca_model["explained_variance"],
        whiten=bool(pca_model.get("whiten", True)),
    )[0]

    X_train_pca = np.asarray(pca_model["X_train_pca"], dtype=np.float32)
    y_train = np.asarray(pca_model["y_train"], dtype=np.int64)
    target_names = [str(n) for n in pca_model["target_names"]]
    train_records = pca_model.get("train_records", [])

    # Euclidean distances in PCA space: d_i = ||w_test - w_i||_2
    diffs = X_train_pca - w_test[None, :]
    distances = np.linalg.norm(diffs, axis=1)
    sorted_indices = np.argsort(distances)
    top_k = max(1, min(int(n_nearest), len(sorted_indices)))

    nearest_matches: list[dict[str, Any]] = []
    for rank, train_idx in enumerate(sorted_indices[:top_k], start=1):
        idx_int = int(train_idx)
        lbl = int(y_train[idx_int])
        person_name = target_names[lbl]
        rec = (
            train_records[idx_int]
            if idx_int < len(train_records)
            else {"index": idx_int, "filename": f"train_{idx_int:04d}.png"}
        )
        match_img = _load_training_match_image(rec, pca_model)
        nearest_matches.append(
            {
                "rank": rank,
                "train_index": idx_int,
                "person_label": lbl,
                "person_name": person_name,
                "distance": float(distances[idx_int]),
                "filename": rec.get("filename", f"train_{idx_int:04d}.png"),
                "image": match_img,
            }
        )

    # Predict identity with K-NN if available, else nearest neighbor (i* = argmin d_i)
    probability: float | None = None
    if knn_bundle is not None:
        if knn_bundle.get("pca_model_id") != pca_model.get("model_id"):
            raise ValueError(
                "K-NN classifier model_id does not match the loaded PCA model."
            )
        clf = knn_bundle["classifier"]
        pred_label = int(clf.predict(w_test[None, :])[0])
        predicted_name = target_names[pred_label]
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(w_test[None, :])[0]
            classes = list(clf.classes_)
            if pred_label in classes:
                probability = float(probs[classes.index(pred_label)])
    else:
        pred_label = nearest_matches[0]["person_label"]
        predicted_name = nearest_matches[0]["person_name"]

    gt_name = ground_truth_name or _infer_ground_truth_name(test_image_path, pca_model)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result_data = {
        "test_image": preprocessed_pil,
        "w_test": w_test,
        "predicted_label": pred_label,
        "predicted_name": predicted_name,
        "probability": probability,
        "ground_truth_name": gt_name,
        "nearest_matches": nearest_matches,
        "image_shape": (h, w),
        "n_components": int(pca_model["n_components"]),
        "processing_time_ms": elapsed_ms,
        "crop_status": crop_status.get("crop", "Not requested"),
    }
    result_data["visualization_image"] = render_recognition_result(result_data)
    return result_data


def render_recognition_result(recognition_output: dict[str, Any]) -> Image.Image:
    """Render the proposed Face Recognition layout (Plan Section 6.C)."""
    test_img = recognition_output["test_image"]
    pred_name = recognition_output["predicted_name"]
    gt_name = recognition_output.get("ground_truth_name")
    prob = recognition_output.get("probability")
    matches = recognition_output["nearest_matches"]
    h, w = recognition_output["image_shape"]

    n_matches = max(1, len(matches))
    aspect = w / max(1, h)
    col_w = max(2.6, 3.2 * aspect)
    fig_w = max(9.5, col_w * n_matches)
    fig_h = 7.2

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    gs = fig.add_gridspec(
        2,
        n_matches,
        height_ratios=[1.0, 1.45],
        hspace=0.35,
        wspace=0.22,
    )

    # Top row: Test image + Predicted Identity summary
    ax_test = fig.add_subplot(gs[0, 0])
    ax_test.imshow(np.asarray(test_img), cmap="gray")
    ax_test.set_title("Test Image", fontsize=11, fontweight="bold")
    ax_test.set_xticks(())
    ax_test.set_yticks(())

    ax_meta = fig.add_subplot(gs[0, 1:])
    ax_meta.axis("off")

    prob_line = (
        f"K-NN neighbour vote fraction: {prob * 100:.1f}% ({prob:.2f})"
        if prob is not None
        else "Classifier: Nearest Eigenface Neighbour (1-NN)"
    )
    gt_line = (
        f"Ground-truth identity: {gt_name}"
        if gt_name
        else "Ground-truth identity: Unlabelled test image"
    )
    badge_color = (
        "#dcfce7"
        if (gt_name is None or gt_name.lower() == pred_name.lower())
        else "#fee2e2"
    )
    border_color = (
        "#22c55e"
        if (gt_name is None or gt_name.lower() == pred_name.lower())
        else "#ef4444"
    )

    summary_text = (
        f"Predicted Identity:  {pred_name}\n\n"
        f"{gt_line}\n"
        f"{prob_line}\n"
        f"Closest Euclidean distance d₁: {matches[0]['distance']:.3f}"
    )
    ax_meta.text(
        0.02,
        0.85,
        summary_text,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": badge_color,
            "edgecolor": border_color,
            "linewidth": 1.5,
        },
    )

    # Bottom row: Top K nearest training images preserving original aspect ratio
    for idx, match in enumerate(matches):
        ax_m = fig.add_subplot(gs[1, idx])
        ax_m.imshow(np.asarray(match["image"]), cmap="gray")
        ax_m.set_title(
            f"Match {match['rank']}: {match['person_name']}\n"
            f"Distance: {match['distance']:.3f}",
            fontsize=10,
            fontweight="bold" if idx == 0 else "normal",
        )
        ax_m.set_xticks(())
        ax_m.set_yticks(())

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def ask_face_recognition_parameters(
    parent: tk.Misc, document: ImageDocument
) -> dict[str, Any] | None:
    """Display configuration dialog for Tool 5 — Face Recognition."""
    dialog = tk.Toplevel(parent)
    dialog.title("Face Recognition (PCA + K-NN)")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    pca_dir_var = tk.StringVar(value=str(DEFAULT_EIGENFACE_MODEL_DIR))
    knn_file_var = tk.StringVar(
        value=str(DEFAULT_EIGENFACE_MODEL_DIR / "knn_classifier.joblib")
    )
    use_current_img_var = tk.BooleanVar(value=document.is_loaded)
    test_img_path_var = tk.StringVar(value=str(document.path) if document.path else "")
    n_nearest_var = tk.IntVar(value=3)
    detect_crop_var = tk.BooleanVar(value=False)

    main_frame = ttk.Frame(dialog, padding=12)
    main_frame.pack(fill="both", expand=True)

    inputs_box = ttk.LabelFrame(
        main_frame, text="A. Models & Test Face Inputs", padding=10
    )
    inputs_box.pack(fill="x", pady=(0, 8))

    def _browse_pca() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog,
            title="Select saved PCA model directory",
            initialdir=str(DEFAULT_EIGENFACE_MODEL_DIR.parent),
        )
        if chosen:
            pca_dir_var.set(chosen)
            cand_knn = Path(chosen) / "knn_classifier.joblib"
            if cand_knn.exists():
                knn_file_var.set(str(cand_knn))

    def _browse_knn() -> None:
        chosen = filedialog.askopenfilename(
            parent=dialog,
            title="Select trained K-NN classifier (.joblib)",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
        )
        if chosen:
            knn_file_var.set(chosen)

    def _browse_test_img() -> None:
        init_dir = Path(pca_dir_var.get()) / "test_images"
        if not init_dir.exists():
            init_dir = Path.cwd()
        chosen = filedialog.askopenfilename(
            parent=dialog,
            title="Select test face image to recognise",
            initialdir=str(init_dir),
            filetypes=IMAGE_TYPES,
        )
        if chosen:
            test_img_path_var.set(chosen)
            use_current_img_var.set(False)

    ttk.Label(inputs_box, text="PCA model directory:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Entry(inputs_box, textvariable=pca_dir_var, width=38).grid(
        row=0, column=1, padx=6, pady=3
    )
    ttk.Button(inputs_box, text="Browse…", command=_browse_pca).grid(
        row=0, column=2, pady=3
    )

    ttk.Label(inputs_box, text="K-NN classifier (.joblib):").grid(
        row=1, column=0, sticky="w", pady=3
    )
    ttk.Entry(inputs_box, textvariable=knn_file_var, width=38).grid(
        row=1, column=1, padx=6, pady=3
    )
    ttk.Button(inputs_box, text="Browse…", command=_browse_knn).grid(
        row=1, column=2, pady=3
    )

    if document.is_loaded:
        ttk.Checkbutton(
            inputs_box,
            text="Use currently open canvas image as test face",
            variable=use_current_img_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 2))

    ttk.Label(inputs_box, text="Or select test image file:").grid(
        row=3, column=0, sticky="w", pady=3
    )
    ttk.Entry(inputs_box, textvariable=test_img_path_var, width=38).grid(
        row=3, column=1, padx=6, pady=3
    )
    ttk.Button(inputs_box, text="Browse…", command=_browse_test_img).grid(
        row=3, column=2, pady=3
    )

    opts_box = ttk.LabelFrame(main_frame, text="B. Recognition Options", padding=10)
    opts_box.pack(fill="x", pady=(0, 8))

    ttk.Label(opts_box, text="Number of nearest training images:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(opts_box, textvariable=n_nearest_var, from_=1, to=6, width=8).grid(
        row=0, column=1, sticky="w", padx=6, pady=3
    )

    ttk.Checkbutton(
        opts_box,
        text="Detect and crop face before PCA projection (for full photos)",
        variable=detect_crop_var,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 2))

    result: dict[str, Any] | None = None

    def apply() -> None:
        nonlocal result
        pca_dir = pca_dir_var.get().strip()
        if not pca_dir or not resolve_path(pca_dir).exists():
            messagebox.showerror(
                "Missing PCA Model",
                "Please select a valid trained Eigenface / PCA model directory.",
                parent=dialog,
            )
            return

        use_cur = bool(use_current_img_var.get()) and document.is_loaded
        tpath = test_img_path_var.get().strip()
        if not use_cur and (not tpath or not resolve_path(tpath).exists()):
            messagebox.showerror(
                "Missing Test Image",
                "Please open an image or select a test face image file.",
                parent=dialog,
            )
            return

        result = {
            "pca_model_path": pca_dir,
            "knn_model_path": knn_file_var.get().strip() or None,
            "use_current_image": use_cur,
            "test_image_path": tpath
            if not use_cur
            else (str(document.path) if document.path else None),
            "n_nearest": max(1, int(n_nearest_var.get())),
            "detect_and_crop": bool(detect_crop_var.get()),
        }
        dialog.destroy()

    btn_bar = ttk.Frame(main_frame)
    btn_bar.pack(fill="x", pady=(6, 0))
    ttk.Button(btn_bar, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(btn_bar, text="Recognise Face", command=apply).pack(
        side="right", padx=(0, 8)
    )

    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class FaceRecognitionTool(ForensicsTool):
    """Tool 5 — Load saved PCA and K-NN models and recognise a test face."""

    tool_id = "face_recognition"
    title = "Face Recognition"
    category = "Face Analysis"
    description = (
        "Load the existing PCA and K-NN models, recognise a test face and show "
        "nearest training images."
    )
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> ToolResult | None:
        if parameters is None:
            parameters = ask_face_recognition_parameters(parent, document)
            if parameters is None:
                return None

        pca_model = load_eigenface_model(parameters["pca_model_path"])

        knn_bundle: dict[str, Any] | None = None
        knn_path = parameters.get("knn_model_path")
        if knn_path and resolve_path(knn_path).exists():
            knn_bundle = load_knn_classifier(
                knn_path, expected_pca_model_id=pca_model["model_id"]
            )
        else:
            default_knn = Path(pca_model["model_dir"]) / "knn_classifier.joblib"
            if default_knn.exists():
                knn_bundle = load_knn_classifier(
                    default_knn, expected_pca_model_id=pca_model["model_id"]
                )

        if parameters.get("test_image") is not None:
            test_img = parameters["test_image"]
            test_path = parameters.get("test_image_path")
        elif parameters.get("use_current_image") and document.current is not None:
            test_img = document.current
            test_path = document.path
        else:
            test_path = resolve_path(parameters["test_image_path"])
            with Image.open(test_path) as img:
                test_img = img.copy()

        rec_out = recognise_face(
            test_img,
            pca_model,
            knn_bundle,
            n_nearest=parameters.get("n_nearest", 3),
            test_image_path=test_path,
            ground_truth_name=parameters.get("ground_truth_name"),
            detect_and_crop=parameters.get("detect_and_crop", False),
        )

        pred_name = rec_out["predicted_name"]
        prob = rec_out.get("probability")
        gt_name = rec_out.get("ground_truth_name")
        matches = rec_out["nearest_matches"]

        details: dict[str, Any] = {
            "Predicted identity": pred_name,
            "Ground-truth identity": gt_name if gt_name else "Unknown",
            "Vote fraction / prob": (
                f"{prob:.2f} ({prob * 100:.1f}%)" if prob is not None else "1-NN"
            ),
            "PCA components": rec_out["n_components"],
            "Face crop": rec_out["crop_status"],
            "Processing time": f"{rec_out['processing_time_ms']:.1f} ms",
        }
        for match in matches:
            details[f"Match {match['rank']}"] = (
                f"{match['person_name']} (d={match['distance']:.3f})"
            )

        return ToolResult(
            image=rec_out["visualization_image"],
            message=(
                f"Recognised face as {pred_name} "
                f"(closest training match: {matches[0]['person_name']}, "
                f"d={matches[0]['distance']:.3f})."
            ),
            details=details,
        )
