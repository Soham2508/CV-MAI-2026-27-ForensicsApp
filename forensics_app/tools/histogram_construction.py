"""Set 7.2 — Histogram Construction (Bag of Visual Words Stage 2).

Loads a saved visual vocabulary codebook, locks descriptor extraction to the
codebook's configuration, maps each local feature descriptor to its nearest
centroid by Euclidean distance, builds K-dimensional visual-word frequency
histograms (raw or L1/L2 normalised), visualises histograms for test images,
and batch-exports ``X_train.npy``, ``y_train.npy``, ``X_test.npy``, ``y_test.npy``
for downstream classifier training.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .codebook_construction import (
    DEFAULT_VISUAL_WORDS_MODEL_DIR,
    ensure_document_has_image,
    extract_local_descriptors_from_image,
    find_latest_codebook_dir,
    load_codebook,
)
from .hog_person_detection import IMAGE_TYPES


def assign_descriptors_to_words(
    descriptors: np.ndarray,
    codebook: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each descriptor ``d_i`` to its closest visual-word centroid ``c_j``.

    Computes ``j* = argmin_j ||d_i - c_j||_2`` using Euclidean distance.
    Returns ``(word_indices, min_distances)``.
    """
    centroids = np.asarray(codebook, dtype=np.float32)
    if centroids.ndim != 2 or len(centroids) == 0:
        raise ValueError("codebook must be a non-empty 2D array of shape (K, D).")

    descs = np.asarray(descriptors, dtype=np.float32)
    if len(descs) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    if descs.ndim != 2 or descs.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"Descriptor shape {descs.shape} does not match codebook {centroids.shape}."
        )

    # Compute pairwise squared Euclidean distances: ||d||^2 + ||c||^2 - 2 d . c^T
    d_sq = np.sum(descs * descs, axis=1, keepdims=True)
    c_sq = np.sum(centroids * centroids, axis=1, keepdims=True).T
    sq_dists = np.maximum(0.0, d_sq + c_sq - 2.0 * (descs @ centroids.T))
    word_indices = np.argmin(sq_dists, axis=1).astype(np.int64)
    min_dists = np.sqrt(sq_dists[np.arange(len(descs)), word_indices]).astype(
        np.float32
    )
    return word_indices, min_dists


def normalize_histogram(
    counts: np.ndarray,
    normalization: str = "l1",
) -> np.ndarray:
    """Normalise a 1D visual-word count histogram using 'l1', 'l2', or 'none'."""
    hist = np.asarray(counts, dtype=np.float32).reshape(-1)
    norm_mode = str(normalization).strip().lower()
    if norm_mode in ("l1", "freq", "frequency", "normalized"):
        total = float(np.sum(np.abs(hist)))
        return hist / total if total > 0.0 else hist
    if norm_mode == "l2":
        norm_val = float(np.linalg.norm(hist))
        return hist / norm_val if norm_val > 0.0 else hist
    if norm_mode in ("none", "raw", "counts", ""):
        return hist
    raise ValueError(f"Unsupported histogram normalization: {normalization!r}")


def build_histogram(
    image: Image.Image,
    codebook_bundle: dict[str, Any],
    *,
    normalization: str = "l1",
) -> dict[str, Any]:
    """Extract descriptors from ``image``, assign words, and build a K-bin histogram."""
    centroids = np.asarray(codebook_bundle["codebook"], dtype=np.float32)
    desc_cfg = codebook_bundle["descriptor_config"]
    k_words = len(centroids)

    keypoints, descriptors, gray_pil = extract_local_descriptors_from_image(
        image, desc_cfg
    )
    word_indices, distances = assign_descriptors_to_words(descriptors, centroids)
    raw_counts = np.bincount(word_indices, minlength=k_words).astype(np.int64)
    histogram = normalize_histogram(raw_counts, normalization=normalization)

    return {
        "histogram": histogram,
        "raw_counts": raw_counts,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "word_indices": word_indices,
        "distances": distances,
        "gray_image": gray_pil,
        "n_clusters": k_words,
        "normalization": str(normalization).strip().lower(),
    }


def generate_histogram_dataset(
    records: list[dict[str, Any]],
    codebook_bundle: dict[str, Any],
    *,
    normalization: str = "l1",
    progress_callback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build matrix ``X`` of shape ``(N, K)`` and label vector ``y`` for records."""
    centroids = np.asarray(codebook_bundle["codebook"], dtype=np.float32)
    k_words = len(centroids)

    x_rows: list[np.ndarray] = []
    y_labels: list[int] = []
    kept_records: list[dict[str, Any]] = []
    total_recs = len(records)

    for idx, rec in enumerate(records):
        if progress_callback is not None and (idx % 5 == 0 or idx == total_recs - 1):
            progress_callback(idx + 1, total_recs)
        img_path = Path(rec["path"])
        if not img_path.exists():
            continue
        with Image.open(img_path) as img:
            res = build_histogram(img, codebook_bundle, normalization=normalization)
        x_rows.append(res["histogram"])
        y_labels.append(int(rec["class_id"]))
        kept_records.append(dict(rec))

    if not x_rows:
        return (
            np.empty((0, k_words), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    return (
        np.vstack(x_rows).astype(np.float32),
        np.asarray(y_labels, dtype=np.int64),
        kept_records,
    )


def build_and_save_histogram_datasets(
    codebook_bundle: dict[str, Any],
    *,
    normalization: str = "l1",
    output_dir: str | Path | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Generate and save train and test visual-word histogram matrices."""
    meta = codebook_bundle["metadata"]
    train_records = meta.get("train_records", [])
    test_records = meta.get("test_records", [])
    if not train_records:
        raise ValueError("Codebook metadata contains no training image records.")

    model_dir = Path(codebook_bundle.get("model_dir", DEFAULT_VISUAL_WORDS_MODEL_DIR))
    hist_dir = Path(output_dir) if output_dir else (model_dir / "histograms")
    hist_dir.mkdir(parents=True, exist_ok=True)

    k_words = int(meta["n_clusters"])
    train_raw = codebook_bundle.get("train_raw_counts")
    if (
        train_raw is not None
        and isinstance(train_raw, np.ndarray)
        and train_raw.shape == (len(train_records), k_words)
    ):
        X_train = np.vstack(
            [normalize_histogram(row, normalization) for row in train_raw]
        ).astype(np.float32)
        y_train = np.asarray(
            [int(r["class_id"]) for r in train_records], dtype=np.int64
        )
        kept_train = [dict(r) for r in train_records]
    else:
        X_train, y_train, kept_train = generate_histogram_dataset(
            train_records,
            codebook_bundle,
            normalization=normalization,
            progress_callback=progress_callback,
        )

    X_test, y_test, kept_test = generate_histogram_dataset(
        test_records,
        codebook_bundle,
        normalization=normalization,
        progress_callback=progress_callback,
    )

    np.save(hist_dir / "X_train.npy", X_train)
    np.save(hist_dir / "y_train.npy", y_train)
    np.save(hist_dir / "X_test.npy", X_test)
    np.save(hist_dir / "y_test.npy", y_test)

    hist_meta: dict[str, Any] = {
        "codebook_id": codebook_bundle["codebook_id"],
        "n_clusters": int(meta["n_clusters"]),
        "class_names": list(meta["class_names"]),
        "normalization": str(normalization).strip().lower(),
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
        "train_records": kept_train,
        "test_records": kept_test,
    }
    with open(hist_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(hist_meta, f, indent=2)

    return {
        "histograms_dir": hist_dir,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": hist_meta,
    }


def load_histogram_dataset(
    histograms_or_codebook_dir: str | Path,
    *,
    expected_codebook_id: str | None = None,
) -> dict[str, Any]:
    """Load saved train/test histogram matrices and verify codebook compatibility."""
    base = Path(histograms_or_codebook_dir)
    if base.is_file():
        base = base.parent
    hist_dir = base / "histograms" if (base / "histograms").is_dir() else base

    x_tr_path = hist_dir / "X_train.npy"
    y_tr_path = hist_dir / "y_train.npy"
    x_te_path = hist_dir / "X_test.npy"
    y_te_path = hist_dir / "y_test.npy"
    meta_path = hist_dir / "metadata.json"

    if not x_tr_path.exists() or not y_tr_path.exists() or not meta_path.exists():
        raise ValueError(f"Saved histogram dataset not found in {hist_dir}.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if expected_codebook_id and meta.get("codebook_id") != expected_codebook_id:
        raise ValueError(
            f"Histogram dataset codebook_id ({meta.get('codebook_id')}) does not match "
            f"selected codebook ({expected_codebook_id})."
        )

    X_train = np.asarray(np.load(x_tr_path), dtype=np.float32)
    y_train = np.asarray(np.load(y_tr_path), dtype=np.int64)
    X_test = (
        np.asarray(np.load(x_te_path), dtype=np.float32)
        if x_te_path.exists()
        else np.empty((0, X_train.shape[1]), dtype=np.float32)
    )
    y_test = (
        np.asarray(np.load(y_te_path), dtype=np.int64)
        if y_te_path.exists()
        else np.empty((0,), dtype=np.int64)
    )

    return {
        "histograms_dir": str(hist_dir),
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "metadata": meta,
    }


def _draw_histogram_panel(
    ax_img: plt.Axes,
    ax_hist: plt.Axes,
    hist_result: dict[str, Any],
    *,
    title_label: str,
    visual_word_patches: np.ndarray | None = None,
    show_assignments: bool = True,
) -> None:
    """Draw one row of [Test Image with Keypoints] + [Visual-Word Histogram]."""
    gray_pil = hist_result["gray_image"]
    kps = hist_result["keypoints"]
    words = hist_result["word_indices"]
    hist = hist_result["histogram"]
    k_words = hist_result["n_clusters"]
    norm_mode = hist_result["normalization"]

    ax_img.imshow(gray_pil, cmap="gray")
    if show_assignments and len(kps) > 0:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(int(w) % 20) for w in words]
        ax_img.scatter(
            kps[:, 1],
            kps[:, 0],
            c=colors,
            s=14,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3,
        )
    ax_img.set_title(title_label, fontsize=10, fontweight="bold")
    ax_img.set_xticks(())
    ax_img.set_yticks(())

    bins = np.arange(k_words)
    ax_hist.bar(
        bins,
        hist,
        width=0.82,
        color="#cbd5e0",
        edgecolor="#2d3748",
        linewidth=0.8,
    )
    ax_hist.set_title("Histogram of visual words", fontsize=10)
    ax_hist.set_xlabel("Visual word index", fontsize=8.5)
    ylabel = "Normalized frequency" if norm_mode != "none" else "Descriptor count"
    ax_hist.set_ylabel(ylabel, fontsize=8.5)
    ax_hist.set_xlim(-0.8, k_words - 0.2)
    ymax = max(float(np.max(hist)) * 1.18, 0.05)
    ax_hist.set_ylim(0.0, ymax)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.3)

    # Overlay small visual-word patch thumbnails at top-5 active bins (Set 7.2 slide)
    if (
        visual_word_patches is not None
        and len(visual_word_patches) == k_words
        and float(np.max(hist)) > 0
    ):
        top_bins = np.argsort(hist)[::-1][: min(6, k_words)]
        for b_idx in top_bins:
            if hist[b_idx] <= 0:
                continue
            patch_im = OffsetImage(
                visual_word_patches[int(b_idx)], cmap="gray", zoom=0.45
            )
            ab = AnnotationBbox(
                patch_im,
                (float(b_idx), float(hist[b_idx]) * 0.12),
                frameon=True,
                pad=0.1,
                bboxprops={"edgecolor": "#4a5568", "linewidth": 0.5},
            )
            ax_hist.add_artist(ab)


def render_test_histograms_visualization(
    sample_items: list[tuple[str, dict[str, Any]]],
    *,
    visual_word_patches: np.ndarray | None = None,
    show_assignments: bool = True,
) -> Image.Image:
    """Render Set 7.2 test image visual-word histograms (1 or multiple rows)."""
    n_rows = max(1, len(sample_items))
    fig_h = max(3.6, 2.85 * n_rows)
    fig = plt.figure(figsize=(12.0, fig_h), dpi=100)
    fig.subplots_adjust(
        left=0.04, right=0.98, top=0.92, bottom=0.09, wspace=0.22, hspace=0.42
    )

    for r_idx, (label, h_res) in enumerate(sample_items):
        ax_img = fig.add_subplot(n_rows, 2, 2 * r_idx + 1)
        ax_hist = fig.add_subplot(n_rows, 2, 2 * r_idx + 2)
        _draw_histogram_panel(
            ax_img,
            ax_hist,
            h_res,
            title_label=label,
            visual_word_patches=visual_word_patches,
            show_assignments=show_assignments,
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def select_representative_test_records(
    codebook_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select one test image record per class from the codebook's test split."""
    meta = codebook_bundle["metadata"]
    test_records: list[dict[str, Any]] = list(meta.get("test_records", []))
    train_records: list[dict[str, Any]] = list(meta.get("train_records", []))
    source = test_records if test_records else train_records

    selected: list[dict[str, Any]] = []
    seen_classes: set[str] = set()
    for rec in source:
        cname = str(rec.get("class_name", ""))
        if cname not in seen_classes and Path(rec["path"]).exists():
            seen_classes.add(cname)
            selected.append(rec)
    return selected


def ask_histogram_construction_parameters(
    parent: tk.Misc,
    document: ImageDocument,
) -> dict[str, Any] | None:
    """Display the Set 7.2 Histogram Construction configuration dialog."""
    dialog = tk.Toplevel(parent)
    dialog.title("Histogram Construction (Set 7.2 — Bag of Visual Words)")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    default_cb = str(find_latest_codebook_dir())
    codebook_dir_var = tk.StringVar(value=default_cb)
    norm_var = tk.StringVar(value="l1")
    batch_var = tk.BooleanVar(value=True)
    show_assign_var = tk.BooleanVar(value=True)

    # Source mode: "class_overview", "test_split_image", "current_image", "custom_file"
    source_mode_var = tk.StringVar(value="class_overview")
    custom_path_var = tk.StringVar(value="")

    cb_summary_var = tk.StringVar(value="")
    locked_extractor_var = tk.StringVar(value="")
    test_choices: list[dict[str, Any]] = []

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    # Section A: Codebook Selection & Locked Extractor
    cb_frame = ttk.LabelFrame(
        main, text="A. Codebook & Locked Descriptor Extractor", padding=10
    )
    cb_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(cb_frame, text="Codebook directory:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Entry(cb_frame, textvariable=codebook_dir_var, width=42).grid(
        row=0, column=1, padx=6, pady=3, sticky="ew"
    )

    test_combo_ref: list[ttk.Combobox] = []

    def _refresh_cb_info(*_args: object) -> None:
        nonlocal test_choices
        try:
            bundle = load_codebook(codebook_dir_var.get().strip())
            meta = bundle["metadata"]
            cfg = bundle["descriptor_config"]
            cb_summary_var.set(
                f"K = {meta['n_clusters']} words | "
                f"Classes ({meta['num_classes']}): {', '.join(meta['class_names'])} | "
                f"ID: {bundle['codebook_id']}"
            )
            locked_extractor_var.set(
                f"Locked extractor: {cfg['method']} ({meta['descriptor_dim']}-D, "
                f"max_keypoints={cfg['max_keypoints']}, sigma={cfg['sigma']})"
            )
            test_choices = list(meta.get("test_records", []))
            if test_combo_ref:
                combo = test_combo_ref[0]
                if test_choices:
                    labels = [
                        f"[{r['class_name']}] {r['filename']}" for r in test_choices
                    ]
                    combo.configure(values=labels)
                    combo.current(0)
                else:
                    combo.configure(values=("(No test records in codebook)",))
                    combo.current(0)
        except Exception as exc:
            cb_summary_var.set(f"Codebook status: {exc}")
            locked_extractor_var.set("Locked extractor: N/A")
            test_choices = []

    def _browse_cb() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select Saved Codebook Directory"
        )
        if chosen:
            codebook_dir_var.set(chosen)
            _refresh_cb_info()

    ttk.Button(cb_frame, text="Browse…", command=_browse_cb).grid(
        row=0, column=2, pady=3
    )
    ttk.Label(cb_frame, textvariable=cb_summary_var, foreground="#2b6cb0").grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(2, 2)
    )
    ttk.Label(cb_frame, textvariable=locked_extractor_var, foreground="#4a5568").grid(
        row=2, column=0, columnspan=3, sticky="w", pady=(0, 2)
    )

    # Section B: Histogram Assignment & Normalisation Parameters
    param_frame = ttk.LabelFrame(
        main, text="B. Histogram Construction Parameters", padding=10
    )
    param_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(param_frame, text="Assignment strategy:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Label(
        param_frame, text="Hard assignment (Nearest Centroid, Euclidean L2)"
    ).grid(row=0, column=1, columnspan=2, sticky="w", padx=6, pady=3)

    ttk.Label(param_frame, text="Histogram normalisation:").grid(
        row=1, column=0, sticky="w", pady=3
    )
    ttk.Combobox(
        param_frame,
        textvariable=norm_var,
        values=("l1", "l2", "none"),
        state="readonly",
        width=14,
    ).grid(row=1, column=1, sticky="w", padx=6, pady=3)

    ttk.Checkbutton(
        param_frame,
        text=(
            "Batch-generate & save train/test histogram matrices "
            "(X_train, y_train, X_test, y_test)"
        ),
        variable=batch_var,
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 2))

    ttk.Checkbutton(
        param_frame,
        text="Show keypoint visual-word assignments on test image",
        variable=show_assign_var,
    ).grid(row=3, column=0, columnspan=3, sticky="w", pady=2)

    # Section C: Test Image Visualisation Selection (Set 7.2)
    vis_frame = ttk.LabelFrame(
        main, text="C. Test Image Histogram Visualisation (Set 7.2)", padding=10
    )
    vis_frame.pack(fill="x", pady=(0, 10))

    test_combo = ttk.Combobox(vis_frame, state="disabled", width=42)
    test_combo_ref.append(test_combo)

    def _update_source_widgets(*_args: object) -> None:
        mode = source_mode_var.get()
        test_combo.configure(
            state="readonly" if mode == "test_split_image" else "disabled"
        )

    ttk.Radiobutton(
        vis_frame,
        text="One test image per class (Set 7.2 multi-class overview)",
        variable=source_mode_var,
        value="class_overview",
        command=_update_source_widgets,
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

    ttk.Radiobutton(
        vis_frame,
        text="Select image from held-out test split:",
        variable=source_mode_var,
        value="test_split_image",
        command=_update_source_widgets,
    ).grid(row=1, column=0, sticky="w", pady=2)
    test_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=2)

    has_current = document.current is not None
    rb_cur = ttk.Radiobutton(
        vis_frame,
        text="Use currently open canvas image",
        variable=source_mode_var,
        value="current_image",
        command=_update_source_widgets,
    )
    rb_cur.grid(row=2, column=0, columnspan=3, sticky="w", pady=2)
    if not has_current:
        rb_cur.configure(state="disabled")

    ttk.Radiobutton(
        vis_frame,
        text="Browse custom test image file:",
        variable=source_mode_var,
        value="custom_file",
        command=_update_source_widgets,
    ).grid(row=3, column=0, sticky="w", pady=2)
    ttk.Entry(vis_frame, textvariable=custom_path_var, width=32).grid(
        row=3, column=1, sticky="ew", padx=6, pady=2
    )

    def _browse_custom_img() -> None:
        chosen = filedialog.askopenfilename(
            parent=dialog, title="Select Test Image", filetypes=IMAGE_TYPES
        )
        if chosen:
            custom_path_var.set(chosen)
            source_mode_var.set("custom_file")
            _update_source_widgets()

    ttk.Button(vis_frame, text="Browse…", command=_browse_custom_img).grid(
        row=3, column=2, pady=2
    )

    _refresh_cb_info()
    _update_source_widgets()

    result: dict[str, Any] | None = None

    def _on_run() -> None:
        nonlocal result
        try:
            cb_dir = codebook_dir_var.get().strip()
            load_codebook(cb_dir)
            mode = source_mode_var.get()
            test_img_path: str | None = None
            use_current = False

            if mode == "test_split_image":
                idx = max(0, test_combo.current())
                if not test_choices:
                    raise ValueError(
                        "No held-out test records available in this codebook."
                    )
                test_img_path = str(test_choices[idx]["path"])
            elif mode == "current_image":
                use_current = True
            elif mode == "custom_file":
                cpath = custom_path_var.get().strip()
                if not cpath or not Path(cpath).exists():
                    raise ValueError("Please select a valid test image file.")
                test_img_path = cpath

            result = {
                "codebook_dir": cb_dir,
                "normalization": norm_var.get(),
                "batch_process": bool(batch_var.get()),
                "show_assignments": bool(show_assign_var.get()),
                "source_mode": mode,
                "test_image_path": test_img_path,
                "use_current_image": use_current,
            }
            dialog.destroy()
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc), parent=dialog)

    btn_row = ttk.Frame(main)
    btn_row.pack(fill="x")
    ttk.Button(btn_row, text="Construct Histograms", command=_on_run).pack(
        side="right", padx=(6, 0)
    )
    ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="right")

    dialog.wait_window()
    return result


class HistogramConstructionTool(ForensicsTool):
    """Tool 02 — Histogram Construction (Set 7.2 Bag of Visual Words)."""

    tool_id = "histogram_construction"
    title = "Histogram Construction"
    category = "Visual Vocabulary"
    description = (
        "Load the codebook, assign local descriptors to their closest visual words "
        "and construct K-dimensional frequency histograms."
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
            parameters = ask_histogram_construction_parameters(parent, document)
            if parameters is None:
                return None

        t0 = time.perf_counter()
        cb_dir = parameters.get("codebook_dir", find_latest_codebook_dir())
        bundle = load_codebook(cb_dir)
        norm_mode = str(parameters.get("normalization", "l1"))
        batch_process = bool(parameters.get("batch_process", True))
        show_assignments = bool(parameters.get("show_assignments", True))

        def _pump_ui(_cur: int, _tot: int) -> None:
            if parent is not None and hasattr(parent, "update_idletasks"):
                try:
                    parent.update_idletasks()
                except Exception:
                    pass

        batch_out: dict[str, Any] | None = None
        if batch_process and bundle["metadata"].get("train_records"):
            batch_out = build_and_save_histogram_datasets(
                bundle,
                normalization=norm_mode,
                output_dir=parameters.get("output_dir"),
                progress_callback=_pump_ui,
            )

        # Build test image visualisations
        sample_items: list[tuple[str, dict[str, Any]]] = []
        if parameters.get("test_image") is not None:
            img_obj = parameters["test_image"]
            h_res = build_histogram(img_obj, bundle, normalization=norm_mode)
            label = str(parameters.get("test_label", "Test image"))
            sample_items.append((label, h_res))
        elif parameters.get("use_current_image") and document.current is not None:
            h_res = build_histogram(document.current, bundle, normalization=norm_mode)
            label = document.path.name if document.path else "Current image"
            sample_items.append((label, h_res))
        elif parameters.get("test_image_path"):
            tpath = Path(parameters["test_image_path"])
            with Image.open(tpath) as img_obj:
                h_res = build_histogram(img_obj, bundle, normalization=norm_mode)
            sample_items.append((tpath.name, h_res))
        else:
            rep_records = select_representative_test_records(bundle)
            for rec in rep_records:
                rpath = Path(rec["path"])
                with Image.open(rpath) as img_obj:
                    h_res = build_histogram(img_obj, bundle, normalization=norm_mode)
                sample_items.append((str(rec["class_name"]), h_res))

        if not sample_items and document.current is not None:
            h_res = build_histogram(document.current, bundle, normalization=norm_mode)
            sample_items.append(("Test image", h_res))

        if not sample_items:
            raise ValueError("No test image available to visualise histogram.")

        vis_image = render_test_histograms_visualization(
            sample_items,
            visual_word_patches=bundle.get("visual_word_patches"),
            show_assignments=show_assignments,
        )

        elapsed_s = time.perf_counter() - t0
        first_label, first_res = sample_items[0]
        top_bin = int(np.argmax(first_res["histogram"]))
        top_freq = float(first_res["histogram"][top_bin])

        details: dict[str, Any] = {
            "Codebook ID": bundle["codebook_id"],
            "Visual word bins (K)": bundle["metadata"]["n_clusters"],
            "Locked descriptor": bundle["descriptor_config"]["method"],
            "Assignment strategy": "Hard assignment (Euclidean L2)",
            "Normalisation": norm_mode.upper(),
            "Visualised test images": len(sample_items),
            f"{first_label} keypoints": len(first_res["keypoints"]),
            f"{first_label} dominant word": f"vw {top_bin} ({top_freq:.3f})",
        }
        if batch_out is not None:
            details["X_train shape"] = (
                f"{batch_out['X_train'].shape[0]} × {batch_out['X_train'].shape[1]}"
            )
            details["X_test shape"] = (
                f"{batch_out['X_test'].shape[0]} × {batch_out['X_test'].shape[1]}"
            )
            details["Saved histograms"] = str(batch_out["histograms_dir"])
        details["Processing time"] = f"{elapsed_s:.2f} s"

        msg = (
            f"Constructed K={bundle['metadata']['n_clusters']}-bin "
            f"visual-word histogram(s) ({norm_mode.upper()} normalisation)."
        )
        if batch_out is not None:
            msg += (
                f" Saved X_train ({len(batch_out['X_train'])}) and "
                f"X_test ({len(batch_out['X_test'])}) to {batch_out['histograms_dir']}."
            )

        ensure_document_has_image(document, vis_image)
        return ToolResult(image=vis_image, message=msg, details=details)
