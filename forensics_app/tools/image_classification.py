"""Set 7.3 — Image Classification (Bag of Visual Words Stage 3).

Provides both:
  A. Classifier Training on visual-word histograms (Linear SVM, RBF SVM, K-NN,
     Random Forest) with probability estimation/calibration, evaluation metrics,
     and persistence (``classifier.joblib``, ``classifier_config.json``).
  B. Interactive Classification / Inference on a new test image: extracts local
     descriptors, builds and visualises the K-dimensional visual-word histogram
     (from 7.2), and upon clicking "Classify" runs the trained classifier and
     displays the top probability ``p(class) = ...`` along with the full class
     probability distribution.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .codebook_construction import (
    DEFAULT_VISUAL_WORDS_MODEL_DIR,
    ensure_document_has_image,
    find_latest_codebook_dir,
    load_codebook,
)
from .histogram_construction import (
    _draw_histogram_panel,
    build_and_save_histogram_datasets,
    build_histogram,
    load_histogram_dataset,
)
from .hog_person_detection import IMAGE_TYPES


class _SoftmaxDecisionWrapper:
    """Fallback probability wrapper around an estimator with ``decision_function``
    when sample counts per class are too small for cross-validated calibration.
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.classes_ = getattr(estimator, "classes_", np.array([]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.asarray(self.estimator.decision_function(X), dtype=np.float64)
        if scores.ndim == 1:
            # Binary case: shape (N,) -> (N, 2)
            probs_pos = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
            return np.column_stack([1.0 - probs_pos, probs_pos])
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(np.clip(shifted, -30.0, 30.0))
        return exp_s / np.maximum(1e-12, np.sum(exp_s, axis=1, keepdims=True))


def _normalize_algorithm_name(algorithm: str) -> str:
    norm = str(algorithm).strip().lower().replace("-", "_").replace(" ", "_")
    if norm in ("linear_svm", "svm", "svm_linear", "linearsvm"):
        return "linear_svm"
    if norm in ("rbf_svm", "svm_rbf", "rbfsvm"):
        return "rbf_svm"
    if norm in ("knn", "k_nn", "k_nearest_neighbors"):
        return "knn"
    if norm in ("random_forest", "rf", "randomforest"):
        return "random_forest"
    raise ValueError(
        f"Unsupported classifier algorithm: {algorithm!r}. "
        "Choose Linear SVM, RBF SVM, K-NN, or Random Forest."
    )


def _display_algorithm_name(alg_key: str) -> str:
    mapping = {
        "linear_svm": "Linear SVM",
        "rbf_svm": "RBF SVM",
        "knn": "K-NN",
        "random_forest": "Random Forest",
    }
    return mapping.get(alg_key, alg_key)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    class_names: list[str],
    codebook_id: str = "",
    normalization: str = "l1",
    algorithm: str = "linear_svm",
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight: str | None = None,
    n_neighbors: int = 5,
    knn_weights: str = "distance",
    knn_metric: str = "euclidean",
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train a classifier on visual-word histograms with probability outputs."""
    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.int64).reshape(-1)
    if X_tr.ndim != 2 or len(X_tr) == 0:
        raise ValueError("X_train must be a non-empty 2D histogram matrix.")
    if len(X_tr) != len(y_tr):
        raise ValueError("X_train and y_train must have the same number of samples.")

    alg_key = _normalize_algorithm_name(algorithm)
    cw = None if (class_weight in (None, "", "None", "none")) else str(class_weight)

    t0 = time.perf_counter()
    if alg_key in ("linear_svm", "rbf_svm"):
        kernel = "linear" if alg_key == "linear_svm" else "rbf"
        gamma_val: str | float = gamma
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            try:
                gamma_val = float(gamma)
            except ValueError:
                gamma_val = "scale"
        base_svc = SVC(
            kernel=kernel,
            C=max(1e-4, float(C)),
            gamma=gamma_val if kernel == "rbf" else "scale",
            class_weight=cw,
            random_state=int(random_state),
        )
        unique_classes, class_counts = np.unique(y_tr, return_counts=True)
        min_class_count = int(np.min(class_counts)) if len(class_counts) > 0 else 0
        if len(unique_classes) >= 2 and min_class_count >= 2:
            cv_folds = min(3, min_class_count)
            estimator: Any = CalibratedClassifierCV(base_svc, cv=cv_folds)
            estimator.fit(X_tr, y_tr)
        else:
            base_svc.fit(X_tr, y_tr)
            estimator = _SoftmaxDecisionWrapper(base_svc)
    elif alg_key == "knn":
        eff_k = min(max(1, int(n_neighbors)), len(X_tr))
        estimator = KNeighborsClassifier(
            n_neighbors=eff_k,
            weights=str(knn_weights),
            metric=str(knn_metric),
        )
        estimator.fit(X_tr, y_tr)
    else:
        eff_depth = (
            int(max_depth) if (max_depth is not None and int(max_depth) > 0) else None
        )
        estimator = RandomForestClassifier(
            n_estimators=max(10, int(n_estimators)),
            max_depth=eff_depth,
            min_samples_leaf=max(1, int(min_samples_leaf)),
            class_weight=cw,
            random_state=int(random_state),
        )
        estimator.fit(X_tr, y_tr)

    elapsed_s = time.perf_counter() - t0
    train_preds = estimator.predict(X_tr)
    train_acc = float(accuracy_score(y_tr, train_preds))

    test_acc: float | None = None
    conf_mat: list[list[int]] | None = None
    report_str: str = ""
    if (
        X_test is not None
        and y_test is not None
        and len(X_test) > 0
        and len(y_test) > 0
    ):
        X_te = np.asarray(X_test, dtype=np.float32)
        y_te = np.asarray(y_test, dtype=np.int64).reshape(-1)
        test_preds = estimator.predict(X_te)
        test_acc = float(accuracy_score(y_te, test_preds))
        labels_idx = list(range(len(class_names)))
        conf_mat = confusion_matrix(y_te, test_preds, labels=labels_idx).tolist()
        report_str = str(
            classification_report(
                y_te,
                test_preds,
                labels=labels_idx,
                target_names=class_names,
                zero_division=0,
            )
        )

    config: dict[str, Any] = {
        "codebook_id": str(codebook_id),
        "n_clusters": int(X_tr.shape[1]),
        "class_names": list(class_names),
        "normalization": str(normalization).strip().lower(),
        "algorithm_key": alg_key,
        "algorithm": _display_algorithm_name(alg_key),
        "C": float(C),
        "gamma": str(gamma),
        "class_weight": cw,
        "n_neighbors": int(n_neighbors),
        "knn_weights": str(knn_weights),
        "knn_metric": str(knn_metric),
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth)
        if (max_depth is not None and int(max_depth) > 0)
        else None,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "num_train_samples": int(len(X_tr)),
        "num_test_samples": int(len(X_test)) if X_test is not None else 0,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "confusion_matrix": conf_mat,
        "classification_report": report_str,
        "training_time_s": elapsed_s,
    }

    return {
        "classifier": estimator,
        "config": config,
    }


def save_classifier(
    classifier_bundle: dict[str, Any],
    output_dir: str | Path = DEFAULT_VISUAL_WORDS_MODEL_DIR,
) -> dict[str, Path]:
    """Save ``classifier.joblib`` and ``classifier_config.json`` in ``output_dir``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf_path = out_dir / "classifier.joblib"
    cfg_path = out_dir / "classifier_config.json"

    joblib.dump(classifier_bundle["classifier"], clf_path)
    cfg = dict(classifier_bundle["config"])
    cfg["model_dir"] = str(out_dir)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return {
        "model_dir": out_dir,
        "classifier_path": clf_path,
        "config_path": cfg_path,
    }


def load_classifier(
    model_path_or_dir: str | Path = DEFAULT_VISUAL_WORDS_MODEL_DIR,
    *,
    expected_codebook_id: str | None = None,
) -> dict[str, Any]:
    """Load a persisted classifier and verify compatibility with the codebook."""
    target = Path(model_path_or_dir)
    if target.is_file():
        if target.suffix == ".joblib":
            clf_path = target
            model_dir = target.parent
        else:
            model_dir = target.parent
            clf_path = model_dir / "classifier.joblib"
    else:
        model_dir = target
        clf_path = model_dir / "classifier.joblib"

    cfg_path = model_dir / "classifier_config.json"
    if not clf_path.exists():
        raise ValueError(f"Trained classifier file not found: {clf_path}")
    if not cfg_path.exists():
        raise ValueError(f"Classifier config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if (
        expected_codebook_id
        and config.get("codebook_id")
        and config.get("codebook_id") != expected_codebook_id
    ):
        raise ValueError(
            f"Classifier codebook_id ({config.get('codebook_id')}) does not match "
            f"selected codebook ({expected_codebook_id})."
        )

    estimator = joblib.load(clf_path)
    return {
        "classifier": estimator,
        "config": config,
        "model_dir": str(model_dir),
    }


def classify_image(
    image: Image.Image,
    codebook_bundle: dict[str, Any],
    classifier_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Classify a single image using the codebook and trained classifier (Set 7.3).

    Extracts descriptors with the codebook's locked extractor, builds the visual-word
    histogram with the classifier's normalisation, runs ``predict_proba``, and returns
    the top predicted class, top probability ``p(class)``, full class probability list,
    and the rendered classification figure.
    """
    t0 = time.perf_counter()
    cfg = classifier_bundle["config"]
    cb_meta = codebook_bundle["metadata"]

    if (
        cfg.get("codebook_id")
        and codebook_bundle.get("codebook_id")
        and cfg["codebook_id"] != codebook_bundle["codebook_id"]
    ):
        raise ValueError(
            f"Model mismatch: classifier was trained on codebook {cfg['codebook_id']}, "
            f"but current codebook is {codebook_bundle['codebook_id']}."
        )

    norm_mode = str(cfg.get("normalization", "l1"))
    hist_res = build_histogram(image, codebook_bundle, normalization=norm_mode)
    hist_vec = hist_res["histogram"].reshape(1, -1)

    estimator = classifier_bundle["classifier"]
    class_names: list[str] = list(
        cfg.get("class_names", cb_meta.get("class_names", []))
    )

    if hasattr(estimator, "predict_proba"):
        probs_raw = np.asarray(estimator.predict_proba(hist_vec)[0], dtype=np.float64)
        est_classes = getattr(estimator, "classes_", np.arange(len(probs_raw)))
        full_probs = np.zeros(len(class_names), dtype=np.float64)
        for idx_pos, cls_id in enumerate(est_classes):
            if 0 <= int(cls_id) < len(full_probs):
                full_probs[int(cls_id)] = probs_raw[idx_pos]
    else:
        pred_only = int(estimator.predict(hist_vec)[0])
        full_probs = np.zeros(len(class_names), dtype=np.float64)
        if 0 <= pred_only < len(full_probs):
            full_probs[pred_only] = 1.0

    top_idx = int(np.argmax(full_probs))
    predicted_class = (
        class_names[top_idx] if top_idx < len(class_names) else str(top_idx)
    )
    top_probability = float(full_probs[top_idx])

    ranked_indices = np.argsort(full_probs)[::-1]
    class_probabilities = [
        {
            "class_id": int(i),
            "class_name": class_names[int(i)] if int(i) < len(class_names) else str(i),
            "probability": float(full_probs[int(i)]),
        }
        for i in ranked_indices
    ]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    probability_label = f"p({predicted_class}) = {top_probability:.2f}"

    result_data: dict[str, Any] = {
        "predicted_class_id": top_idx,
        "predicted_class": predicted_class,
        "top_probability": top_probability,
        "probability_label": probability_label,
        "class_probabilities": class_probabilities,
        "histogram_result": hist_res,
        "classifier_algorithm": cfg.get("algorithm", "Linear SVM"),
        "processing_time_ms": elapsed_ms,
        "input_image": image.convert("RGB"),
    }
    result_data["visualization_image"] = render_classification_visualization(
        result_data,
        visual_word_patches=codebook_bundle.get("visual_word_patches"),
        classifier_config=cfg,
    )
    return result_data


def render_classification_visualization(
    classification_result: dict[str, Any],
    *,
    visual_word_patches: np.ndarray | None = None,
    classifier_config: dict[str, Any] | None = None,
) -> Image.Image:
    """Render Set 7.3 figure: Query Image with ``p(class) = ...``, Histogram,
    and Class Probability Distribution.
    """
    rgb_img = classification_result["input_image"]
    prob_label = classification_result["probability_label"]
    hist_res = classification_result["histogram_result"]
    class_probs = classification_result["class_probabilities"]
    alg_name = classification_result.get("classifier_algorithm", "Linear SVM")

    fig = plt.figure(figsize=(13.5, 7.6), dpi=100)
    fig.suptitle(
        f"Set 7.3 — Bag of Visual Words Classification ({alg_name})   |   {prob_label}",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.05, 1.35],
        height_ratios=[1.15, 0.95],
        left=0.05,
        right=0.97,
        top=0.89,
        bottom=0.08,
        wspace=0.22,
        hspace=0.36,
    )

    # Top-left: Input image with prominent Set 7.3 title `p(kangaroo) = 0.64`
    ax_query = fig.add_subplot(gs[0, 0])
    ax_query.imshow(rgb_img)
    ax_query.set_title(
        prob_label, fontsize=14, fontweight="bold", color="#1a202c", pad=8
    )
    ax_query.set_xticks(())
    ax_query.set_yticks(())
    for spine in ax_query.spines.values():
        spine.set_edgecolor("#744210")
        spine.set_linewidth(2.0)

    # Top-right: Keypoints + Histogram of visual words (from 7.2)
    top_right_gs = gs[0, 1].subgridspec(1, 2, width_ratios=[0.85, 1.35], wspace=0.22)
    ax_kp = fig.add_subplot(top_right_gs[0, 0])
    ax_hist = fig.add_subplot(top_right_gs[0, 1])
    _draw_histogram_panel(
        ax_kp,
        ax_hist,
        hist_res,
        title_label=f"Keypoints ({len(hist_res['keypoints'])})",
        visual_word_patches=visual_word_patches,
        show_assignments=True,
    )

    # Bottom-left: Class Probability Distribution horizontal bar chart
    ax_probs = fig.add_subplot(gs[1, 0])
    c_names = [item["class_name"] for item in class_probs][::-1]
    c_vals = [item["probability"] * 100.0 for item in class_probs][::-1]
    bar_colors = [
        "#2f855a" if i == len(c_names) - 1 else "#4a5568" for i in range(len(c_names))
    ]
    bars = ax_probs.barh(
        c_names, c_vals, color=bar_colors, height=0.55, edgecolor="#1a202c"
    )
    ax_probs.set_xlim(0.0, 108.0)
    ax_probs.set_xlabel("Predicted Probability (%)", fontsize=9)
    ax_probs.set_title(
        "Class Probability Distribution", fontsize=10.5, fontweight="bold"
    )
    ax_probs.grid(axis="x", linestyle="--", alpha=0.35)
    for bar, val in zip(bars, c_vals):
        ax_probs.text(
            min(102.0, val + 1.8),
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Bottom-right: Classifier Summary & Confusion Matrix (if available)
    ax_info = fig.add_subplot(gs[1, 1])
    conf_mat = classifier_config.get("confusion_matrix") if classifier_config else None
    all_classes = classifier_config.get("class_names", []) if classifier_config else []
    if conf_mat is not None and len(conf_mat) > 0:
        cm_arr = np.asarray(conf_mat, dtype=np.int64)
        im = ax_info.imshow(cm_arr, cmap="Blues")
        plt.colorbar(im, ax=ax_info, fraction=0.046, pad=0.04)
        ax_info.set_xticks(np.arange(len(all_classes)))
        ax_info.set_yticks(np.arange(len(all_classes)))
        ax_info.set_xticklabels(all_classes, fontsize=8.5, rotation=20)
        ax_info.set_yticklabels(all_classes, fontsize=8.5)
        ax_info.set_xlabel("Predicted class", fontsize=8.5)
        ax_info.set_ylabel("True class", fontsize=8.5)
        tr_acc = classifier_config.get("train_accuracy", 0.0) * 100.0
        te_acc = (classifier_config.get("test_accuracy") or 0.0) * 100.0
        ax_info.set_title(
            f"Held-out Confusion Matrix (Train: {tr_acc:.1f}%, Test: {te_acc:.1f}%)",
            fontsize=10,
            fontweight="bold",
        )
        thresh = cm_arr.max() / 2.0 if cm_arr.size > 0 else 0
        for r in range(cm_arr.shape[0]):
            for c in range(cm_arr.shape[1]):
                ax_info.text(
                    c,
                    r,
                    str(cm_arr[r, c]),
                    ha="center",
                    va="center",
                    color="white" if cm_arr[r, c] > thresh else "black",
                    fontsize=9,
                    fontweight="bold",
                )
    else:
        ax_info.set_axis_off()
        lines = [
            f"Predicted Class: {classification_result['predicted_class']}",
            f"Top Probability: {classification_result['top_probability']:.2f} "
            f"({classification_result['top_probability'] * 100:.1f}%)",
            f"Classifier: {alg_name}",
            f"Visual Words (K): {hist_res['n_clusters']}",
            f"Extracted Keypoints: {len(hist_res['keypoints'])}",
        ]
        ax_info.text(0.05, 0.5, "\n".join(lines), va="center", fontsize=10.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _render_preview_strip(
    image: Image.Image,
    hist_res: dict[str, Any],
    *,
    prediction_text: str | None = None,
) -> Image.Image:
    """Render a compact [Input Image + Visual-Word Histogram] preview for dialog."""
    fig = plt.figure(figsize=(8.2, 3.0), dpi=90)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.85, 1.45],
        left=0.04,
        right=0.98,
        top=0.86,
        bottom=0.18,
        wspace=0.24,
    )
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image.convert("RGB"))
    title_str = prediction_text if prediction_text else "Selected Test Image"
    ax_img.set_title(
        title_str,
        fontsize=11 if prediction_text else 9.5,
        fontweight="bold",
        color="#2f855a" if prediction_text else "#1a202c",
    )
    ax_img.set_xticks(())
    ax_img.set_yticks(())

    ax_hist = fig.add_subplot(gs[0, 1])
    k_words = hist_res["n_clusters"]
    bins = np.arange(k_words)
    hist = hist_res["histogram"]
    ax_hist.bar(bins, hist, width=0.8, color="#3182ce", edgecolor="#1a365d")
    ax_hist.set_title(
        f"Histogram of visual words "
        f"(K={k_words}, keypoints={len(hist_res['keypoints'])})",
        fontsize=9.5,
    )
    ax_hist.set_xlabel("Visual word index", fontsize=8)
    ax_hist.set_ylabel("Normalized frequency", fontsize=8)
    ax_hist.set_xlim(-0.8, k_words - 0.2)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def ask_image_classification_parameters(
    parent: tk.Misc,
    document: ImageDocument,
) -> dict[str, Any] | None:
    """Interactive Set 7.3 dialog for Classifier Training and Classification."""
    dialog = tk.Toplevel(parent)
    dialog.title("Image Classification (Set 7.3 — Bag of Visual Words)")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    default_cb = str(find_latest_codebook_dir())
    codebook_dir_var = tk.StringVar(value=default_cb)

    # Section A: Classifier Training variables
    retrain_var = tk.BooleanVar(value=False)
    alg_var = tk.StringVar(value="Linear SVM")
    svm_c_var = tk.DoubleVar(value=1.0)
    svm_gamma_var = tk.StringVar(value="scale")
    class_weight_var = tk.StringVar(value="None")
    knn_k_var = tk.IntVar(value=5)
    rf_trees_var = tk.IntVar(value=100)
    norm_var = tk.StringVar(value="l1")
    seed_var = tk.IntVar(value=42)

    # Section B: Classification / Inference variables
    source_mode_var = tk.StringVar(
        value="current_image" if document.current is not None else "test_split_image"
    )
    custom_img_var = tk.StringVar(value="")

    cb_status_var = tk.StringVar(value="")
    clf_status_var = tk.StringVar(value="")
    pred_headline_var = tk.StringVar(
        value="Click 'Classify' to run the trained classifier."
    )
    pred_dist_var = tk.StringVar(value="")

    test_choices: list[dict[str, Any]] = []
    preview_photo_ref: list[ImageTk.PhotoImage] = []
    cached_bundle: dict[str, Any] | None = None
    cached_clf_bundle: dict[str, Any] | None = None

    main = ttk.Frame(dialog, padding=10)
    main.pack(fill="both", expand=True)

    # 1. Codebook & Model Path
    top_frame = ttk.LabelFrame(
        main, text="1. Codebook & Classifier Model Folder", padding=8
    )
    top_frame.pack(fill="x", pady=(0, 6))

    ttk.Label(top_frame, text="Codebook directory:").grid(
        row=0, column=0, sticky="w", pady=2
    )
    ttk.Entry(top_frame, textvariable=codebook_dir_var, width=46).grid(
        row=0, column=1, padx=6, pady=2, sticky="ew"
    )

    test_combo_ref: list[ttk.Combobox] = []

    def _refresh_model_state(*_args: object) -> None:
        nonlocal test_choices, cached_bundle, cached_clf_bundle
        cb_dir = codebook_dir_var.get().strip()
        try:
            cached_bundle = load_codebook(cb_dir)
            meta = cached_bundle["metadata"]
            cb_status_var.set(
                f"Codebook: K={meta['n_clusters']} words | "
                f"Classes: {', '.join(meta['class_names'])} | "
                f"ID: {cached_bundle['codebook_id']}"
            )
            test_choices = list(meta.get("test_records", [])) or list(
                meta.get("train_records", [])
            )
            if test_combo_ref:
                combo = test_combo_ref[0]
                if test_choices:
                    combo.configure(
                        values=[
                            f"[{r['class_name']}] {r['filename']}" for r in test_choices
                        ]
                    )
                    combo.current(0)
                else:
                    combo.configure(values=("(No dataset records)",))
                    combo.current(0)

            try:
                cached_clf_bundle = load_classifier(
                    cb_dir, expected_codebook_id=cached_bundle["codebook_id"]
                )
                c_cfg = cached_clf_bundle["config"]
                tr_a = c_cfg.get("train_accuracy", 0.0) * 100.0
                te_a = c_cfg.get("test_accuracy")
                te_str = f", Test acc={te_a * 100:.1f}%" if te_a is not None else ""
                clf_status_var.set(
                    f"Trained Classifier: {c_cfg.get('algorithm')} "
                    f"(Train acc={tr_a:.1f}%{te_str})"
                )
                retrain_var.set(False)
            except Exception:
                cached_clf_bundle = None
                clf_status_var.set(
                    "No trained classifier found for this codebook — "
                    "will train automatically."
                )
                retrain_var.set(True)
        except Exception as exc:
            cached_bundle = None
            cached_clf_bundle = None
            cb_status_var.set(f"Codebook error: {exc}")
            clf_status_var.set("Classifier: N/A")

    def _browse_cb() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select Codebook Directory"
        )
        if chosen:
            codebook_dir_var.set(chosen)
            _refresh_model_state()
            _update_histogram_preview()

    ttk.Button(top_frame, text="Browse…", command=_browse_cb).grid(
        row=0, column=2, pady=2
    )
    ttk.Label(top_frame, textvariable=cb_status_var, foreground="#2b6cb0").grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(2, 1)
    )
    ttk.Label(top_frame, textvariable=clf_status_var, foreground="#2f855a").grid(
        row=2, column=0, columnspan=3, sticky="w", pady=(0, 2)
    )

    # 2. Section A: Classifier Training
    train_frame = ttk.LabelFrame(
        main, text="2. Section A — Classifier Training Configuration", padding=8
    )
    train_frame.pack(fill="x", pady=(0, 6))

    ttk.Label(train_frame, text="Classifier:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Combobox(
        train_frame,
        textvariable=alg_var,
        values=("Linear SVM", "RBF SVM", "K-NN", "Random Forest"),
        state="readonly",
        width=15,
    ).grid(row=0, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(train_frame, text="SVM C:").grid(
        row=0, column=2, sticky="w", padx=(8, 0), pady=2
    )
    ttk.Spinbox(
        train_frame, textvariable=svm_c_var, from_=0.1, to=100.0, increment=0.5, width=7
    ).grid(row=0, column=3, sticky="w", padx=6, pady=2)

    ttk.Label(train_frame, text="SVM gamma:").grid(
        row=0, column=4, sticky="w", padx=(8, 0), pady=2
    )
    ttk.Combobox(
        train_frame,
        textvariable=svm_gamma_var,
        values=("scale", "auto", "0.1", "1.0", "10.0"),
        width=8,
    ).grid(row=0, column=5, sticky="w", padx=6, pady=2)

    ttk.Label(train_frame, text="K-NN neighbours:").grid(
        row=1, column=0, sticky="w", pady=2
    )
    ttk.Spinbox(
        train_frame, textvariable=knn_k_var, from_=1, to=25, increment=2, width=7
    ).grid(row=1, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(train_frame, text="RF trees:").grid(
        row=1, column=2, sticky="w", padx=(8, 0), pady=2
    )
    ttk.Spinbox(
        train_frame, textvariable=rf_trees_var, from_=10, to=500, increment=25, width=7
    ).grid(row=1, column=3, sticky="w", padx=6, pady=2)

    ttk.Label(train_frame, text="Class weight:").grid(
        row=1, column=4, sticky="w", padx=(8, 0), pady=2
    )
    ttk.Combobox(
        train_frame,
        textvariable=class_weight_var,
        values=("None", "balanced"),
        state="readonly",
        width=8,
    ).grid(row=1, column=5, sticky="w", padx=6, pady=2)

    def _run_training_now() -> None:
        nonlocal cached_clf_bundle
        try:
            if cached_bundle is None:
                _refresh_model_state()
            if cached_bundle is None:
                raise ValueError("Please select a valid codebook directory first.")

            cb_dir = Path(codebook_dir_var.get().strip())
            norm_m = norm_var.get()
            try:
                hist_ds = load_histogram_dataset(
                    cb_dir, expected_codebook_id=cached_bundle["codebook_id"]
                )
            except Exception:
                hist_ds = build_and_save_histogram_datasets(
                    cached_bundle, normalization=norm_m
                )

            trained = train_classifier(
                hist_ds["X_train"],
                hist_ds["y_train"],
                class_names=cached_bundle["metadata"]["class_names"],
                codebook_id=cached_bundle["codebook_id"],
                normalization=hist_ds["metadata"].get("normalization", norm_m),
                algorithm=alg_var.get(),
                C=float(svm_c_var.get()),
                gamma=svm_gamma_var.get(),
                class_weight=None
                if class_weight_var.get() == "None"
                else class_weight_var.get(),
                n_neighbors=int(knn_k_var.get()),
                n_estimators=int(rf_trees_var.get()),
                random_state=int(seed_var.get()),
                X_test=hist_ds["X_test"],
                y_test=hist_ds["y_test"],
            )
            save_classifier(trained, output_dir=cb_dir)
            cached_clf_bundle = trained
            _refresh_model_state()
        except Exception as exc:
            messagebox.showerror("Classifier Training Failed", str(exc), parent=dialog)

    ttk.Checkbutton(
        train_frame,
        text="Retrain classifier before classification",
        variable=retrain_var,
    ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 2))
    ttk.Button(
        train_frame, text="Train / Update Classifier Now", command=_run_training_now
    ).grid(row=2, column=4, columnspan=2, sticky="e", pady=(4, 2))

    # 3. Section B: Interactive Classification / Inference (Set 7.3)
    inf_frame = ttk.LabelFrame(
        main,
        text=(
            "3. Section B — New Image Selection, Histogram Visualisation "
            "& Classify (Set 7.3)"
        ),
        padding=8,
    )
    inf_frame.pack(fill="both", expand=True, pady=(0, 8))

    test_combo = ttk.Combobox(inf_frame, state="readonly", width=40)
    test_combo_ref.append(test_combo)

    def _resolve_selected_image() -> Image.Image | None:
        mode = source_mode_var.get()
        if mode == "current_image" and document.current is not None:
            return document.current.copy()
        if mode == "test_split_image" and test_choices:
            idx = max(0, test_combo.current())
            p = Path(test_choices[idx]["path"])
            if p.exists():
                with Image.open(p) as im:
                    return im.copy()
        if mode == "custom_file":
            cp = Path(custom_img_var.get().strip())
            if cp.exists() and cp.is_file():
                with Image.open(cp) as im:
                    return im.copy()
        return None

    preview_label = ttk.Label(inf_frame)

    def _update_histogram_preview(*_args: object) -> None:
        if cached_bundle is None:
            return
        img_obj = _resolve_selected_image()
        if img_obj is None:
            return
        try:
            h_res = build_histogram(
                img_obj, cached_bundle, normalization=norm_var.get()
            )
            strip = _render_preview_strip(img_obj, h_res)
            photo = ImageTk.PhotoImage(strip)
            preview_photo_ref.clear()
            preview_photo_ref.append(photo)
            preview_label.configure(image=photo)
            pred_headline_var.set(
                "Histogram visualised (7.2). Click 'Classify' to predict class."
            )
            pred_dist_var.set("")
        except Exception:
            pass

    ttk.Radiobutton(
        inf_frame,
        text="Held-out test image:",
        variable=source_mode_var,
        value="test_split_image",
        command=_update_histogram_preview,
    ).grid(row=0, column=0, sticky="w", pady=2)
    test_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
    test_combo.bind("<<ComboboxSelected>>", _update_histogram_preview)

    rb_canvas = ttk.Radiobutton(
        inf_frame,
        text="Currently open canvas image",
        variable=source_mode_var,
        value="current_image",
        command=_update_histogram_preview,
    )
    rb_canvas.grid(row=0, column=2, sticky="w", padx=6, pady=2)
    if document.current is None:
        rb_canvas.configure(state="disabled")

    ttk.Radiobutton(
        inf_frame,
        text="Custom image file:",
        variable=source_mode_var,
        value="custom_file",
        command=_update_histogram_preview,
    ).grid(row=1, column=0, sticky="w", pady=2)
    ttk.Entry(inf_frame, textvariable=custom_img_var, width=40).grid(
        row=1, column=1, sticky="ew", padx=6, pady=2
    )

    def _browse_img() -> None:
        chosen = filedialog.askopenfilename(
            parent=dialog, title="Select Query Image to Classify", filetypes=IMAGE_TYPES
        )
        if chosen:
            custom_img_var.set(chosen)
            source_mode_var.set("custom_file")
            _update_histogram_preview()

    ttk.Button(inf_frame, text="Browse Image…", command=_browse_img).grid(
        row=1, column=2, sticky="w", padx=6, pady=2
    )

    preview_label.grid(row=2, column=0, columnspan=3, pady=6)

    # Interactive "Classify" button and probability readout (Slide 4)
    classify_bar = ttk.Frame(inf_frame)
    classify_bar.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(2, 2))

    def _on_interactive_classify() -> None:
        if cached_bundle is None:
            _refresh_model_state()
        if cached_bundle is None:
            messagebox.showerror("Error", "No valid codebook loaded.", parent=dialog)
            return
        if cached_clf_bundle is None or retrain_var.get():
            _run_training_now()
        if cached_clf_bundle is None:
            return
        img_obj = _resolve_selected_image()
        if img_obj is None:
            messagebox.showerror(
                "Error", "Please select a valid test image.", parent=dialog
            )
            return
        try:
            cls_res = classify_image(img_obj, cached_bundle, cached_clf_bundle)
            strip = _render_preview_strip(
                img_obj,
                cls_res["histogram_result"],
                prediction_text=cls_res["probability_label"],
            )
            photo = ImageTk.PhotoImage(strip)
            preview_photo_ref.clear()
            preview_photo_ref.append(photo)
            preview_label.configure(image=photo)
            pred_headline_var.set(
                f"Prediction: {cls_res['probability_label']}  "
                f"({cls_res['top_probability'] * 100:.1f}%)"
            )
            dist_parts = [
                f"{item['class_name']}: {item['probability'] * 100:.1f}%"
                for item in cls_res["class_probabilities"]
            ]
            pred_dist_var.set("Class probabilities:  " + "   |   ".join(dist_parts))
        except Exception as exc:
            messagebox.showerror("Classification Failed", str(exc), parent=dialog)

    ttk.Button(
        classify_bar,
        text="Classify (Preview Top Probability)",
        command=_on_interactive_classify,
    ).pack(side="left", padx=(0, 12))

    ttk.Label(
        classify_bar,
        textvariable=pred_headline_var,
        font=("Segoe UI", 10, "bold"),
        foreground="#2f855a",
    ).pack(side="left")

    ttk.Label(
        inf_frame,
        textvariable=pred_dist_var,
        font=("Segoe UI", 9),
        foreground="#2d3748",
    ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

    _refresh_model_state()
    _update_histogram_preview()

    result: dict[str, Any] | None = None

    def _on_apply() -> None:
        nonlocal result
        try:
            cb_dir = codebook_dir_var.get().strip()
            load_codebook(cb_dir)
            mode = source_mode_var.get()
            test_img_path: str | None = None
            use_current = False
            if mode == "test_split_image":
                if not test_choices:
                    raise ValueError("No dataset test records available.")
                idx = max(0, test_combo.current())
                test_img_path = str(test_choices[idx]["path"])
            elif mode == "current_image":
                use_current = True
            elif mode == "custom_file":
                cpath = custom_img_var.get().strip()
                if not cpath or not Path(cpath).exists():
                    raise ValueError("Please select a valid query image file.")
                test_img_path = cpath

            result = {
                "codebook_dir": cb_dir,
                "train_classifier": bool(retrain_var.get())
                or (cached_clf_bundle is None),
                "algorithm": alg_var.get(),
                "C": float(svm_c_var.get()),
                "gamma": svm_gamma_var.get(),
                "class_weight": None
                if class_weight_var.get() == "None"
                else class_weight_var.get(),
                "n_neighbors": int(knn_k_var.get()),
                "n_estimators": int(rf_trees_var.get()),
                "normalization": norm_var.get(),
                "random_state": int(seed_var.get()),
                "test_image_path": test_img_path,
                "use_current_image": use_current,
            }
            dialog.destroy()
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc), parent=dialog)

    btn_row = ttk.Frame(main)
    btn_row.pack(fill="x")
    ttk.Button(btn_row, text="Classify & Render Result", command=_on_apply).pack(
        side="right", padx=(6, 0)
    )
    ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="right")

    dialog.wait_window()
    return result


class ImageClassificationTool(ForensicsTool):
    """Tool 03 — Image Classification (Set 7.3 Bag of Visual Words)."""

    tool_id = "image_classification"
    title = "Image Classification"
    category = "Visual Vocabulary"
    description = (
        "Train a classifier on visual-word histograms, visualise the histogram of a "
        "new image, and classify it showing the top probability p(class)."
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
            parameters = ask_image_classification_parameters(parent, document)
            if parameters is None:
                return None

        cb_dir = Path(parameters.get("codebook_dir", find_latest_codebook_dir()))
        codebook_bundle = load_codebook(cb_dir)
        norm_mode = str(parameters.get("normalization", "l1"))

        should_train = bool(parameters.get("train_classifier", False))
        clf_bundle: dict[str, Any] | None = None
        if not should_train:
            try:
                clf_bundle = load_classifier(
                    cb_dir, expected_codebook_id=codebook_bundle["codebook_id"]
                )
            except Exception:
                should_train = True

        if should_train or clf_bundle is None:
            try:
                hist_ds = load_histogram_dataset(
                    cb_dir, expected_codebook_id=codebook_bundle["codebook_id"]
                )
            except Exception:
                hist_ds = build_and_save_histogram_datasets(
                    codebook_bundle, normalization=norm_mode
                )

            clf_bundle = train_classifier(
                hist_ds["X_train"],
                hist_ds["y_train"],
                class_names=codebook_bundle["metadata"]["class_names"],
                codebook_id=codebook_bundle["codebook_id"],
                normalization=hist_ds["metadata"].get("normalization", norm_mode),
                algorithm=str(parameters.get("algorithm", "linear_svm")),
                C=float(parameters.get("C", 1.0)),
                gamma=parameters.get("gamma", "scale"),
                class_weight=parameters.get("class_weight"),
                n_neighbors=int(parameters.get("n_neighbors", 5)),
                knn_weights=str(parameters.get("knn_weights", "distance")),
                knn_metric=str(parameters.get("knn_metric", "euclidean")),
                n_estimators=int(parameters.get("n_estimators", 100)),
                max_depth=parameters.get("max_depth"),
                min_samples_leaf=int(parameters.get("min_samples_leaf", 1)),
                random_state=int(parameters.get("random_state", 42)),
                X_test=hist_ds["X_test"],
                y_test=hist_ds["y_test"],
            )
            save_classifier(clf_bundle, output_dir=cb_dir)

        # Resolve test image for classification
        if parameters.get("test_image") is not None:
            query_img = parameters["test_image"]
        elif parameters.get("use_current_image") and document.current is not None:
            query_img = document.current
        elif parameters.get("test_image_path"):
            with Image.open(parameters["test_image_path"]) as im:
                query_img = im.copy()
        elif document.current is not None:
            query_img = document.current
        else:
            test_recs = codebook_bundle["metadata"].get(
                "test_records"
            ) or codebook_bundle["metadata"].get("train_records", [])
            if not test_recs:
                raise ValueError("No query image provided for classification.")
            with Image.open(test_recs[0]["path"]) as im:
                query_img = im.copy()

        cls_out = classify_image(query_img, codebook_bundle, clf_bundle)
        cfg = clf_bundle["config"]

        details: dict[str, Any] = {
            "Predicted class": cls_out["predicted_class"],
            "Top probability": cls_out["probability_label"],
            "Probability (%)": f"{cls_out['top_probability'] * 100:.1f}%",
            "Classifier": cfg["algorithm"],
            "Codebook ID": codebook_bundle["codebook_id"],
            "Visual words (K)": codebook_bundle["metadata"]["n_clusters"],
            "Train accuracy": f"{cfg['train_accuracy'] * 100:.1f}%",
        }
        if cfg.get("test_accuracy") is not None:
            details["Test accuracy"] = f"{cfg['test_accuracy'] * 100:.1f}%"
        for item in cls_out["class_probabilities"]:
            details[f"P({item['class_name']})"] = (
                f"{item['probability']:.2f} ({item['probability'] * 100:.1f}%)"
            )
        details["Processing time"] = f"{cls_out['processing_time_ms']:.1f} ms"

        top_pct = cls_out["top_probability"] * 100.0
        vis_image = cls_out["visualization_image"]
        ensure_document_has_image(document, vis_image)
        return ToolResult(
            image=vis_image,
            message=(
                f"Classified image as '{cls_out['predicted_class']}': "
                f"{cls_out['probability_label']} ({top_pct:.1f}%) "
                f"using {cfg['algorithm']}."
            ),
            details=details,
        )
