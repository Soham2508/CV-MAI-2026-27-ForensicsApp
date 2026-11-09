"""Tool 4 — Eigenface / PCA Trainer and K-NN Classifier Trainer.

Implements dataset preparation (LFW via ``fetch_lfw_people`` or local identity
folders), train/test splitting with image-to-name correspondence, PCA/eigenface
construction, mean face and eigenface gallery visualization, explained variance
and PCA coefficient inspection, model persistence, and explicit K-NN training
with optional ``GridSearchCV`` and quantitative/qualitative evaluation.
"""

from __future__ import annotations

import hashlib
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
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm"}
DEFAULT_EIGENFACE_MODEL_DIR = Path("models/eigenface_model")
DEFAULT_LFW_EXPORT_DIR = Path("datasets/lfw_faces")


def plot_gallery(
    images: np.ndarray,
    titles: list[str] | tuple[str, ...],
    h: int,
    w: int,
    n_row: int = 3,
    n_col: int = 4,
    *,
    show: bool = False,
) -> Image.Image:
    """Plot a gallery of portraits preserving the original (h, w) aspect ratio.

    Matches the reference notebook's ``plot_gallery`` function and returns a
    rendered ``PIL.Image.Image``.
    """
    aspect = w / max(1, h)
    cell_h = 2.4
    cell_w = max(1.5, cell_h * aspect)
    fig = plt.figure(figsize=(cell_w * n_col, cell_h * n_row), dpi=100)
    fig.subplots_adjust(
        bottom=0.02, left=0.02, right=0.98, top=0.90, hspace=0.38, wspace=0.18
    )

    count = min(len(images), len(titles), n_row * n_col)
    for i in range(count):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        ax.set_title(titles[i], size=10)
        ax.set_xticks(())
        ax.set_yticks(())

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.08)
    if show:
        plt.show()
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def title(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str] | np.ndarray,
    i: int,
) -> str:
    """Format predicted vs true identity title for test gallery."""
    pred_name = str(target_names[int(y_pred[i])]).rsplit(" ", 1)[-1]
    true_name = str(target_names[int(y_test[i])]).rsplit(" ", 1)[-1]
    return "predicted: %s\ntrue:      %s" % (pred_name, true_name)


def _uint8_from_face_array(arr_2d: np.ndarray) -> np.ndarray:
    """Convert a 2D float or uint8 face image array to uint8 [0, 255]."""
    arr = np.asarray(arr_2d, dtype=np.float64)
    if arr.max() <= 1.0 and arr.min() >= 0.0:
        return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    if arr.min() >= 0.0 and arr.max() <= 255.0:
        return np.clip(np.round(arr), 0, 255).astype(np.uint8)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        norm = (arr - vmin) / (vmax - vmin)
        return np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def save_split_dataset_to_folder(
    dataset: dict[str, Any], export_dir: str | Path
) -> Path:
    """Save train and test face images to disk and record image <-> name correspondence.

    Implements Set 6.2 slide 1:
    1. Save the images to a folder (train and test)
    2. Preserve labels and correspondence (image <-> name pairs)
    """
    out_root = Path(export_dir)
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    h, w = dataset["image_shape"]
    target_names = [str(n) for n in dataset["target_names"]]

    train_records: list[dict[str, Any]] = []
    for idx, (vec, lbl) in enumerate(zip(dataset["X_train"], dataset["y_train"])):
        person_id = int(lbl)
        person_name = target_names[person_id]
        safe_name = person_name.replace(" ", "_")
        fname = f"train_{idx:04d}_{safe_name}.png"
        fpath = train_dir / fname
        img_u8 = _uint8_from_face_array(vec.reshape((h, w)))
        Image.fromarray(img_u8, mode="L").save(fpath)
        train_records.append(
            {
                "index": idx,
                "filename": fname,
                "path": str(fpath),
                "person_label": person_id,
                "person_name": person_name,
                "split": "train",
            }
        )

    test_records: list[dict[str, Any]] = []
    for idx, (vec, lbl) in enumerate(zip(dataset["X_test"], dataset["y_test"])):
        person_id = int(lbl)
        person_name = target_names[person_id]
        safe_name = person_name.replace(" ", "_")
        fname = f"test_{idx:04d}_{safe_name}.png"
        fpath = test_dir / fname
        img_u8 = _uint8_from_face_array(vec.reshape((h, w)))
        Image.fromarray(img_u8, mode="L").save(fpath)
        test_records.append(
            {
                "index": idx,
                "filename": fname,
                "path": str(fpath),
                "person_label": person_id,
                "person_name": person_name,
                "split": "test",
            }
        )

    correspondence = {
        "image_shape": [int(h), int(w)],
        "target_names": target_names,
        "value_scale": dataset.get("value_scale", "0_1"),
        "train": train_records,
        "test": test_records,
    }
    corr_path = out_root / "correspondence.json"
    corr_path.write_text(json.dumps(correspondence, indent=2), encoding="utf-8")

    dataset["train_records"] = train_records
    dataset["test_records"] = test_records
    dataset["export_dir"] = str(out_root)
    return out_root


def load_face_dataset(
    *,
    source: str = "LFW",
    folder_path: str | Path | None = None,
    min_faces_per_person: int = 70,
    resize: float = 0.4,
    test_size: float = 0.25,
    random_state: int = 42,
    image_shape: tuple[int, int] | None = None,
    export_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load and split a face dataset from LFW or a local folder.

    Preserves image arrays, person labels, person names, image paths/filenames,
    and train/test assignment as specified in Plan Section 5.A.
    """
    src_norm = source.strip().upper()

    if src_norm == "LFW":
        lfw_people = fetch_lfw_people(
            min_faces_per_person=int(min_faces_per_person),
            resize=float(resize),
        )
        n_samples, h, w = lfw_people.images.shape
        X = np.asarray(lfw_people.data, dtype=np.float32)
        y = np.asarray(lfw_people.target, dtype=np.int64)
        target_names = [str(name) for name in lfw_people.target_names]

        # Standardize pixel scale info
        value_scale = "0_1" if float(X.max()) <= 1.0 else "0_255"

        indices = np.arange(n_samples)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            indices,
            test_size=float(test_size),
            random_state=int(random_state),
        )

        train_records = [
            {
                "index": int(i),
                "dataset_index": int(orig_i),
                "filename": (
                    f"lfw_train_{i:04d}_"
                    f"{target_names[int(lbl)].replace(' ', '_')}.png"
                ),
                "path": None,
                "person_label": int(lbl),
                "person_name": target_names[int(lbl)],
                "split": "train",
            }
            for i, (orig_i, lbl) in enumerate(zip(idx_train, y_train))
        ]
        test_records = [
            {
                "index": int(i),
                "dataset_index": int(orig_i),
                "filename": (
                    f"lfw_test_{i:04d}_"
                    f"{target_names[int(lbl)].replace(' ', '_')}.png"
                ),
                "path": None,
                "person_label": int(lbl),
                "person_name": target_names[int(lbl)],
                "split": "test",
            }
            for i, (orig_i, lbl) in enumerate(zip(idx_test, y_test))
        ]

        dataset: dict[str, Any] = {
            "source": "LFW",
            "X": X,
            "y": y,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "target_names": target_names,
            "n_classes": len(target_names),
            "image_shape": (int(h), int(w)),
            "value_scale": value_scale,
            "train_records": train_records,
            "test_records": test_records,
            "test_size": float(test_size),
            "random_state": int(random_state),
            "resize": float(resize),
            "min_faces_per_person": int(min_faces_per_person),
        }

        if export_dir:
            save_split_dataset_to_folder(dataset, export_dir)

        return dataset

    # Local dataset folder
    if not folder_path:
        raise ValueError("folder_path is required when loading a local dataset.")
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Local dataset folder does not exist: {root}")

    # Case 1: Previously exported folder containing correspondence.json
    corr_file = root / "correspondence.json"
    if corr_file.exists():
        corr = json.loads(corr_file.read_text(encoding="utf-8"))
        h, w = int(corr["image_shape"][0]), int(corr["image_shape"][1])
        target_names = [str(n) for n in corr["target_names"]]
        value_scale = corr.get("value_scale", "0_1")
        scale_div = 255.0 if value_scale == "0_1" else 1.0

        X_train_list: list[np.ndarray] = []
        y_train_list: list[int] = []
        train_records = []
        for rec in corr.get("train", []):
            img_path = Path(rec["path"])
            if not img_path.exists():
                img_path = root / "train" / rec["filename"]
            with Image.open(img_path) as img:
                gray = img.convert("L").resize((w, h), Image.Resampling.BILINEAR)
                arr = np.asarray(gray, dtype=np.float32).reshape(-1) / scale_div
            X_train_list.append(arr)
            y_train_list.append(int(rec["person_label"]))
            rec_copy = dict(rec)
            rec_copy["path"] = str(img_path)
            train_records.append(rec_copy)

        X_test_list: list[np.ndarray] = []
        y_test_list: list[int] = []
        test_records = []
        for rec in corr.get("test", []):
            img_path = Path(rec["path"])
            if not img_path.exists():
                img_path = root / "test" / rec["filename"]
            with Image.open(img_path) as img:
                gray = img.convert("L").resize((w, h), Image.Resampling.BILINEAR)
                arr = np.asarray(gray, dtype=np.float32).reshape(-1) / scale_div
            X_test_list.append(arr)
            y_test_list.append(int(rec["person_label"]))
            rec_copy = dict(rec)
            rec_copy["path"] = str(img_path)
            test_records.append(rec_copy)

        X_train_arr = np.stack(X_train_list, axis=0)
        y_train_arr = np.asarray(y_train_list, dtype=np.int64)
        X_test_arr = (
            np.stack(X_test_list, axis=0)
            if X_test_list
            else np.empty((0, h * w), dtype=np.float32)
        )
        y_test_arr = np.asarray(y_test_list, dtype=np.int64)

        return {
            "source": "Local",
            "folder_path": str(root),
            "X": np.vstack([X_train_arr, X_test_arr])
            if len(X_test_arr)
            else X_train_arr,
            "y": np.concatenate([y_train_arr, y_test_arr])
            if len(y_test_arr)
            else y_train_arr,
            "X_train": X_train_arr,
            "X_test": X_test_arr,
            "y_train": y_train_arr,
            "y_test": y_test_arr,
            "target_names": target_names,
            "n_classes": len(target_names),
            "image_shape": (h, w),
            "value_scale": value_scale,
            "train_records": train_records,
            "test_records": test_records,
            "test_size": float(test_size),
            "random_state": int(random_state),
        }

    # Case 2: Folder organized by identity subfolders (root/Person_Name/img.png)
    subdirs = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name not in ("train", "test")
    )
    target_names = []
    all_vectors: list[np.ndarray] = []
    all_labels: list[int] = []
    all_paths: list[Path] = []
    inferred_shape: tuple[int, int] | None = image_shape

    if subdirs:
        for person_dir in subdirs:
            files = sorted(
                f
                for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not files:
                continue
            person_id = len(target_names)
            person_name = person_dir.name.replace("_", " ")
            target_names.append(person_name)
            for fpath in files:
                with Image.open(fpath) as img:
                    gray = img.convert("L")
                    if inferred_shape is None:
                        inferred_shape = (gray.height, gray.width)
                    else:
                        gray = gray.resize(
                            (inferred_shape[1], inferred_shape[0]),
                            Image.Resampling.BILINEAR,
                        )
                    vec = np.asarray(gray, dtype=np.float32).reshape(-1) / 255.0
                all_vectors.append(vec)
                all_labels.append(person_id)
                all_paths.append(fpath)
    else:
        # Flat directory with filename convention: personName_001.png
        files = sorted(
            f
            for f in root.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        name_to_id: dict[str, int] = {}
        for fpath in files:
            stem_parts = fpath.stem.rsplit("_", 1)
            person_name = stem_parts[0].replace("_", " ")
            if person_name not in name_to_id:
                name_to_id[person_name] = len(target_names)
                target_names.append(person_name)
            person_id = name_to_id[person_name]
            with Image.open(fpath) as img:
                gray = img.convert("L")
                if inferred_shape is None:
                    inferred_shape = (gray.height, gray.width)
                else:
                    gray = gray.resize(
                        (inferred_shape[1], inferred_shape[0]),
                        Image.Resampling.BILINEAR,
                    )
                vec = np.asarray(gray, dtype=np.float32).reshape(-1) / 255.0
            all_vectors.append(vec)
            all_labels.append(person_id)
            all_paths.append(fpath)

    if not all_vectors or inferred_shape is None:
        raise ValueError(f"No valid face images found in {root}.")

    X = np.stack(all_vectors, axis=0)
    y = np.asarray(all_labels, dtype=np.int64)
    indices = np.arange(len(all_vectors))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=float(test_size),
        random_state=int(random_state),
    )

    train_records = [
        {
            "index": int(i),
            "dataset_index": int(orig_i),
            "filename": all_paths[int(orig_i)].name,
            "path": str(all_paths[int(orig_i)]),
            "person_label": int(lbl),
            "person_name": target_names[int(lbl)],
            "split": "train",
        }
        for i, (orig_i, lbl) in enumerate(zip(idx_train, y_train))
    ]
    test_records = [
        {
            "index": int(i),
            "dataset_index": int(orig_i),
            "filename": all_paths[int(orig_i)].name,
            "path": str(all_paths[int(orig_i)]),
            "person_label": int(lbl),
            "person_name": target_names[int(lbl)],
            "split": "test",
        }
        for i, (orig_i, lbl) in enumerate(zip(idx_test, y_test))
    ]

    dataset = {
        "source": "Local",
        "folder_path": str(root),
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_names": target_names,
        "n_classes": len(target_names),
        "image_shape": inferred_shape,
        "value_scale": "0_1",
        "train_records": train_records,
        "test_records": test_records,
        "test_size": float(test_size),
        "random_state": int(random_state),
    }
    if export_dir:
        save_split_dataset_to_folder(dataset, export_dir)
    return dataset


def train_eigenface_pca(
    dataset: dict[str, Any],
    *,
    n_components: int = 100,
    svd_solver: str = "randomized",
    whiten: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train PCA on training faces and project train and test sets onto eigenfaces.

    Matches Exercise #1 of the reference notebook:
    - Mean face: ``pca.mean_``
    - Eigenfaces: ``pca.components_.reshape((n_components, h, w))``
    - Training weights / coefficients: ``X_train_pca = pca.transform(X_train)``
    """
    X_train = np.asarray(dataset["X_train"], dtype=np.float32)
    X_test = np.asarray(dataset["X_test"], dtype=np.float32)
    h, w = dataset["image_shape"]

    max_valid_components = min(X_train.shape[0], X_train.shape[1])
    eff_components = min(int(n_components), max_valid_components)
    if eff_components < 1:
        raise ValueError("Number of PCA components must be at least 1.")

    t0 = time.perf_counter()
    pca = PCA(
        n_components=eff_components,
        svd_solver=svd_solver,
        whiten=bool(whiten),
        random_state=int(random_state)
        if svd_solver in ("randomized", "arpack", "auto")
        else None,
    ).fit(X_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    X_train_pca = pca.transform(X_train)
    X_test_pca = (
        pca.transform(X_test)
        if len(X_test) > 0
        else np.empty((0, eff_components), dtype=np.float32)
    )
    project_time = time.perf_counter() - t1

    eigenfaces = pca.components_.reshape((eff_components, h, w))
    mean_face = pca.mean_.reshape((h, w))
    total_variance = float(np.sum(pca.explained_variance_ratio_))

    # Compute a deterministic model_id hash to pair K-NN classifiers safely
    hash_digest = hashlib.sha256(
        pca.components_.tobytes() + pca.mean_.tobytes()
    ).hexdigest()[:16]

    return {
        "pca": pca,
        "model_id": hash_digest,
        "mean_face": mean_face,
        "components": pca.components_,
        "eigenfaces": eigenfaces,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "singular_values": pca.singular_values_,
        "total_explained_variance": total_variance,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "y_train": dataset["y_train"],
        "y_test": dataset["y_test"],
        "X_train": X_train,
        "X_test": X_test,
        "target_names": dataset["target_names"],
        "image_shape": (h, w),
        "value_scale": dataset.get("value_scale", "0_1"),
        "n_components": eff_components,
        "svd_solver": svd_solver,
        "whiten": bool(whiten),
        "fit_time_sec": fit_time,
        "project_time_sec": project_time,
        "train_records": dataset.get("train_records", []),
        "test_records": dataset.get("test_records", []),
    }


def render_pca_summary_image(
    pca_result: dict[str, Any],
    *,
    n_row: int = 3,
    n_col: int = 4,
) -> Image.Image:
    """Render composite PCA output: Mean Face + Eigenface Gallery + Variance."""
    h, w = pca_result["image_shape"]
    eigenfaces = pca_result["eigenfaces"]
    mean_face = pca_result["mean_face"]
    var_ratios = pca_result["explained_variance_ratio"]
    total_var = pca_result["total_explained_variance"]

    n_show = min(len(eigenfaces), n_row * n_col)
    total_cols = n_col + 1  # Column 0 holds Mean Face, Columns 1..n_col hold Eigenfaces
    aspect = w / max(1, h)
    fig_w = max(10.0, total_cols * 2.1 * max(0.75, aspect))
    fig_h = max(7.5, (n_row + 1.2) * 2.2)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    gs = fig.add_gridspec(
        n_row + 1,
        total_cols,
        height_ratios=[1.0] * n_row + [1.15],
        hspace=0.42,
        wspace=0.22,
    )

    # Mean face in top-left panel
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_mean.imshow(mean_face, cmap=plt.cm.gray)
    ax_mean.set_title("Mean Face (μ)", fontsize=10, fontweight="bold")
    ax_mean.set_xticks(())
    ax_mean.set_yticks(())

    # Summary text under mean face
    ax_info = fig.add_subplot(gs[1:n_row, 0])
    ax_info.axis("off")
    info_lines = [
        f"Components: {pca_result['n_components']}",
        f"Solver: {pca_result['svd_solver']}",
        f"Whitening: {pca_result['whiten']}",
        f"Image: {w} × {h} px",
        f"Train faces: {len(pca_result['X_train'])}",
        f"Test faces: {len(pca_result['X_test'])}",
        f"Identities: {len(pca_result['target_names'])}",
        f"Explained Var:\n  {total_var * 100:.2f}%",
    ]
    ax_info.text(
        0.05,
        0.95,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f0f4f8",
            "edgecolor": "#cbd5e1",
        },
    )

    # Eigenface gallery preserving aspect ratio
    for i in range(n_show):
        r = i // n_col
        c = (i % n_col) + 1
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
        pct = var_ratios[i] * 100.0 if i < len(var_ratios) else 0.0
        ax.set_title(f"eigenface {i}\n({pct:.1f}%)", fontsize=9)
        ax.set_xticks(())
        ax.set_yticks(())

    # Bottom row: Explained variance bar chart
    ax_var = fig.add_subplot(gs[n_row, :])
    top_n_bars = min(25, len(var_ratios))
    xs = np.arange(1, top_n_bars + 1)
    bars_pct = var_ratios[:top_n_bars] * 100.0
    ax_var.bar(
        xs, bars_pct, color="#2563eb", alpha=0.85, label="Per-component variance (%)"
    )
    ax_var.set_xticks(xs)
    ax_var.set_xticklabels([f"PC{k}" for k in xs], rotation=45, fontsize=8)
    ax_var.set_ylabel("Variance (%)")
    ax_var.set_title(
        f"Explained Variance Ratio (Top {top_n_bars} shown — Total across "
        f"{pca_result['n_components']} components: {total_var * 100:.2f}%)",
        fontsize=10,
    )
    ax_var.grid(axis="y", linestyle="--", alpha=0.35)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_eigenface_model(
    pca_result: dict[str, Any],
    model_dir: str | Path = DEFAULT_EIGENFACE_MODEL_DIR,
    *,
    gallery_rows: int = 3,
    gallery_cols: int = 4,
) -> dict[str, Path]:
    """Persist trained PCA / Eigenface model and training representations (Sec. 5.D).

    Directory structure created:
    ``eigenface_model/``
      - ``pca_model.npz``
      - ``metadata.json``
      - ``training_embeddings.npy`` (N × K matrix of PCA weights w_i)
      - ``training_labels.npy``
      - ``training_image_paths.json``
      - ``eigenfaces_gallery.png``
      - ``train_images/`` and ``test_images/`` (for retrieving training & test faces)
    """
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_imgs_dir = out_dir / "train_images"
    test_imgs_dir = out_dir / "test_images"
    train_imgs_dir.mkdir(parents=True, exist_ok=True)
    test_imgs_dir.mkdir(parents=True, exist_ok=True)

    h, w = pca_result["image_shape"]
    target_names = [str(n) for n in pca_result["target_names"]]

    # Save training and test face images so Tool 5 can load closest training images
    saved_train_records: list[dict[str, Any]] = []
    for idx, (vec, lbl) in enumerate(zip(pca_result["X_train"], pca_result["y_train"])):
        person_id = int(lbl)
        person_name = target_names[person_id]
        safe_name = person_name.replace(" ", "_")
        fname = f"train_{idx:04d}_{safe_name}.png"
        fpath = train_imgs_dir / fname
        img_u8 = _uint8_from_face_array(vec.reshape((h, w)))
        Image.fromarray(img_u8, mode="L").save(fpath)
        saved_train_records.append(
            {
                "index": idx,
                "filename": fname,
                "path": str(fpath),
                "relative_path": f"train_images/{fname}",
                "person_label": person_id,
                "person_name": person_name,
                "split": "train",
            }
        )

    saved_test_records: list[dict[str, Any]] = []
    for idx, (vec, lbl) in enumerate(zip(pca_result["X_test"], pca_result["y_test"])):
        person_id = int(lbl)
        person_name = target_names[person_id]
        safe_name = person_name.replace(" ", "_")
        fname = f"test_{idx:04d}_{safe_name}.png"
        fpath = test_imgs_dir / fname
        img_u8 = _uint8_from_face_array(vec.reshape((h, w)))
        Image.fromarray(img_u8, mode="L").save(fpath)
        saved_test_records.append(
            {
                "index": idx,
                "filename": fname,
                "path": str(fpath),
                "relative_path": f"test_images/{fname}",
                "person_label": person_id,
                "person_name": person_name,
                "split": "test",
            }
        )

    npz_path = out_dir / "pca_model.npz"
    np.savez_compressed(
        npz_path,
        mean=pca_result["pca"].mean_,
        mean_face=pca_result["mean_face"],
        components=pca_result["components"],
        eigenfaces=pca_result["eigenfaces"],
        explained_variance=pca_result["explained_variance"],
        explained_variance_ratio=pca_result["explained_variance_ratio"],
        singular_values=pca_result["singular_values"],
        image_shape=np.asarray([h, w], dtype=np.int64),
        whiten=np.asarray(bool(pca_result["whiten"])),
    )

    embeddings_path = out_dir / "training_embeddings.npy"
    np.save(embeddings_path, np.asarray(pca_result["X_train_pca"], dtype=np.float32))

    labels_path = out_dir / "training_labels.npy"
    np.save(labels_path, np.asarray(pca_result["y_train"], dtype=np.int64))

    test_embeddings_path = out_dir / "test_embeddings.npy"
    np.save(
        test_embeddings_path, np.asarray(pca_result["X_test_pca"], dtype=np.float32)
    )
    test_labels_path = out_dir / "test_labels.npy"
    np.save(test_labels_path, np.asarray(pca_result["y_test"], dtype=np.int64))

    paths_json = out_dir / "training_image_paths.json"
    paths_json.write_text(
        json.dumps(
            {
                "train": saved_train_records,
                "test": saved_test_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "model_id": pca_result["model_id"],
        "n_components": int(pca_result["n_components"]),
        "svd_solver": pca_result["svd_solver"],
        "whiten": bool(pca_result["whiten"]),
        "image_shape": [int(h), int(w)],
        "value_scale": pca_result.get("value_scale", "0_1"),
        "target_names": target_names,
        "n_classes": len(target_names),
        "n_train_samples": len(saved_train_records),
        "n_test_samples": len(saved_test_records),
        "total_explained_variance": float(pca_result["total_explained_variance"]),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Render and save eigenfaces gallery PNG
    eigenface_titles = [
        f"eigenface {i}" for i in range(pca_result["eigenfaces"].shape[0])
    ]
    gallery_img = plot_gallery(
        pca_result["eigenfaces"],
        eigenface_titles,
        h,
        w,
        n_row=gallery_rows,
        n_col=gallery_cols,
    )
    gallery_path = out_dir / "eigenfaces_gallery.png"
    gallery_img.save(gallery_path)

    return {
        "model_dir": out_dir,
        "pca_model_npz": npz_path,
        "metadata_json": metadata_path,
        "training_embeddings_npy": embeddings_path,
        "training_labels_npy": labels_path,
        "training_image_paths_json": paths_json,
        "eigenfaces_gallery_png": gallery_path,
    }


def project_faces_with_saved_pca(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
    whiten: bool,
) -> np.ndarray:
    """Compute PCA projection coefficients w = U_K^T (x - μ) (with optional whitening).

    Matches both ``sklearn.decomposition.PCA.transform`` and the formula
    ``w_i = U_K^T (x_i - μ)`` in the Set 6 slides.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]
    centered = X_arr - np.asarray(mean, dtype=np.float32)
    weights = centered @ np.asarray(components, dtype=np.float32).T
    if whiten:
        scale = np.sqrt(
            np.maximum(np.asarray(explained_variance, dtype=np.float32), 1e-12)
        )
        weights = weights / scale
    return weights


def load_eigenface_model(model_path: str | Path) -> dict[str, Any]:
    """Load a persisted Eigenface / PCA model directory or ``pca_model.npz`` file."""
    p = Path(model_path)
    model_dir = p if p.is_dir() else p.parent
    npz_path = model_dir / "pca_model.npz" if p.is_dir() else p
    if not npz_path.exists():
        raise FileNotFoundError(f"PCA model file not found: {npz_path}")

    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"PCA metadata.json not found in {model_dir}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path, allow_pickle=False)

    mean = np.asarray(npz["mean"], dtype=np.float32)
    components = np.asarray(npz["components"], dtype=np.float32)
    eigenfaces = np.asarray(npz["eigenfaces"], dtype=np.float32)
    explained_variance = np.asarray(npz["explained_variance"], dtype=np.float32)
    explained_variance_ratio = np.asarray(
        npz["explained_variance_ratio"], dtype=np.float32
    )
    singular_values = np.asarray(npz["singular_values"], dtype=np.float32)
    h, w = int(metadata["image_shape"][0]), int(metadata["image_shape"][1])
    whiten = bool(metadata.get("whiten", True))

    emb_path = model_dir / "training_embeddings.npy"
    lbl_path = model_dir / "training_labels.npy"
    if not emb_path.exists() or not lbl_path.exists():
        raise FileNotFoundError(
            f"Missing training_embeddings.npy or training_labels.npy in {model_dir}"
        )
    X_train_pca = np.load(emb_path, allow_pickle=False)
    y_train = np.load(lbl_path, allow_pickle=False)

    test_emb_path = model_dir / "test_embeddings.npy"
    test_lbl_path = model_dir / "test_labels.npy"
    X_test_pca = (
        np.load(test_emb_path, allow_pickle=False)
        if test_emb_path.exists()
        else np.empty((0, components.shape[0]), dtype=np.float32)
    )
    y_test = (
        np.load(test_lbl_path, allow_pickle=False)
        if test_lbl_path.exists()
        else np.empty((0,), dtype=np.int64)
    )

    paths_file = model_dir / "training_image_paths.json"
    image_records = (
        json.loads(paths_file.read_text(encoding="utf-8"))
        if paths_file.exists()
        else {"train": [], "test": []}
    )

    # Reconstruct a fitted sklearn PCA instance for seamless transform compatibility
    pca_obj = PCA(
        n_components=int(metadata["n_components"]),
        svd_solver=metadata.get("svd_solver", "randomized"),
        whiten=whiten,
    )
    pca_obj.mean_ = mean
    pca_obj.components_ = components
    pca_obj.explained_variance_ = explained_variance
    pca_obj.explained_variance_ratio_ = explained_variance_ratio
    pca_obj.singular_values_ = singular_values
    pca_obj.n_components_ = int(metadata["n_components"])
    pca_obj.n_features_in_ = mean.shape[0]

    return {
        "model_dir": model_dir,
        "pca": pca_obj,
        "model_id": metadata["model_id"],
        "mean": mean,
        "mean_face": mean.reshape((h, w)),
        "components": components,
        "eigenfaces": eigenfaces,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "singular_values": singular_values,
        "total_explained_variance": float(np.sum(explained_variance_ratio)),
        "X_train_pca": X_train_pca,
        "y_train": y_train,
        "X_test_pca": X_test_pca,
        "y_test": y_test,
        "target_names": [str(n) for n in metadata["target_names"]],
        "image_shape": (h, w),
        "value_scale": metadata.get("value_scale", "0_1"),
        "n_components": int(metadata["n_components"]),
        "svd_solver": metadata.get("svd_solver", "randomized"),
        "whiten": whiten,
        "train_records": image_records.get("train", []),
        "test_records": image_records.get("test", []),
    }


def train_knn_classifier(
    pca_data: dict[str, Any],
    *,
    n_neighbors: int = 5,
    metric: str = "minkowski",
    p: int = 2,
    weights: str = "uniform",
    algorithm: str = "auto",
    use_grid_search: bool = False,
    k_candidates: tuple[int, ...] | list[int] = (3, 5, 7, 9, 11),
    cv_folds: int = 5,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train a K-NN classifier on PCA training coefficients and evaluate on test set.

    Implements Exercises #2–#5 of the reference notebook and Plan Section 5.E.
    """
    X_train_pca = np.asarray(pca_data["X_train_pca"], dtype=np.float32)
    y_train = np.asarray(pca_data["y_train"], dtype=np.int64)
    X_test_pca = np.asarray(pca_data.get("X_test_pca", []), dtype=np.float32)
    y_test = np.asarray(pca_data.get("y_test", []), dtype=np.int64)
    target_names = [str(n) for n in pca_data["target_names"]]
    n_classes = len(target_names)

    if len(X_train_pca) == 0:
        raise ValueError("PCA training embeddings are empty.")

    # scikit-learn's ball_tree and kd_tree searches do not support the cosine
    # metric and raise an error; brute force gives the same neighbours.
    if metric.lower() == "cosine" and algorithm.lower() in ("ball_tree", "kd_tree"):
        algorithm = "brute"

    t0 = time.perf_counter()
    base_knn = KNeighborsClassifier(
        n_neighbors=min(int(n_neighbors), len(X_train_pca)),
        metric=metric.lower(),
        p=int(p),
        weights=weights.lower(),
        algorithm=algorithm.lower(),
    )

    best_k = int(n_neighbors)
    cv_results_summary: dict[str, float] = {}

    if use_grid_search:
        _, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = int(np.min(class_counts)) if len(class_counts) else 2
        eff_cv = max(2, min(int(cv_folds), min_class_count, len(X_train_pca) // 2))
        max_train_in_fold = max(1, len(X_train_pca) - (len(X_train_pca) // eff_cv))
        valid_ks = [int(k) for k in k_candidates if 1 <= int(k) <= max_train_in_fold]
        if not valid_ks:
            valid_ks = [min(int(n_neighbors), max_train_in_fold)]
        param_grid = {"n_neighbors": valid_ks}
        grid = GridSearchCV(base_knn, param_grid, cv=eff_cv)
        grid.fit(X_train_pca, y_train)
        classifier: KNeighborsClassifier = grid.best_estimator_
        best_k = int(grid.best_params_["n_neighbors"])
        for k_val, mean_score in zip(
            grid.cv_results_["param_n_neighbors"],
            grid.cv_results_["mean_test_score"],
        ):
            cv_results_summary[f"k={int(k_val)}"] = float(mean_score)
    else:
        classifier = base_knn.fit(X_train_pca, y_train)
        best_k = int(classifier.n_neighbors)

    fit_time = time.perf_counter() - t0

    accuracy: float | None = None
    report_str = ""
    conf_mat: np.ndarray | None = None
    y_pred: np.ndarray | None = None

    if len(X_test_pca) > 0 and len(y_test) == len(X_test_pca):
        y_pred = classifier.predict(X_test_pca)
        accuracy = float(accuracy_score(y_test, y_pred))
        present_labels = list(range(n_classes))
        report_str = str(
            classification_report(
                y_test,
                y_pred,
                labels=present_labels,
                target_names=target_names,
                zero_division=0,
            )
        )
        conf_mat = confusion_matrix(y_test, y_pred, labels=present_labels)

    knn_bundle = {
        "classifier": classifier,
        "pca_model_id": pca_data["model_id"],
        "n_components": int(pca_data["n_components"]),
        "image_shape": tuple(pca_data["image_shape"]),
        "target_names": target_names,
        "best_k": best_k,
        "metric": metric.lower(),
        "p": int(p),
        "weights": weights.lower(),
        "algorithm": algorithm.lower(),
        "use_grid_search": bool(use_grid_search),
        "cv_results": cv_results_summary,
        "accuracy": accuracy,
        "classification_report": report_str,
        "confusion_matrix": conf_mat.tolist() if conf_mat is not None else None,
        "fit_time_sec": fit_time,
    }

    saved_joblib_path: Path | None = None
    if save_path is not None:
        saved_joblib_path = save_knn_classifier(knn_bundle, save_path)
    elif "model_dir" in pca_data and pca_data["model_dir"] is not None:
        saved_joblib_path = save_knn_classifier(
            knn_bundle, Path(pca_data["model_dir"]) / "knn_classifier.joblib"
        )

    eval_image = render_knn_evaluation_image(pca_data, knn_bundle, y_pred)

    return {
        **knn_bundle,
        "y_pred": y_pred,
        "joblib_path": saved_joblib_path,
        "evaluation_image": eval_image,
    }


def save_knn_classifier(knn_bundle: dict[str, Any], path: str | Path) -> Path:
    """Persist the trained K-NN classifier bundle with its paired PCA model_id."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": knn_bundle["classifier"],
        "pca_model_id": knn_bundle["pca_model_id"],
        "n_components": knn_bundle["n_components"],
        "image_shape": knn_bundle["image_shape"],
        "target_names": knn_bundle["target_names"],
        "best_k": knn_bundle["best_k"],
        "metric": knn_bundle["metric"],
        "p": knn_bundle["p"],
        "weights": knn_bundle["weights"],
        "algorithm": knn_bundle["algorithm"],
        "use_grid_search": knn_bundle["use_grid_search"],
        "accuracy": knn_bundle.get("accuracy"),
    }
    joblib.dump(payload, out_path)
    return out_path


def load_knn_classifier(
    path: str | Path, expected_pca_model_id: str | None = None
) -> dict[str, Any]:
    """Load a persisted K-NN classifier and verify compatibility with the PCA model."""
    p = Path(path)
    if p.is_dir():
        p = p / "knn_classifier.joblib"
    if not p.exists():
        raise FileNotFoundError(f"K-NN classifier file not found: {p}")

    bundle = joblib.load(p)
    if not isinstance(bundle, dict) or "classifier" not in bundle:
        raise ValueError(f"Invalid K-NN classifier bundle in {p}")

    if (
        expected_pca_model_id is not None
        and bundle.get("pca_model_id") != expected_pca_model_id
    ):
        raise ValueError(
            "K-NN classifier was trained against a different PCA model representation "
            f"(expected model_id={expected_pca_model_id}, "
            f"got {bundle.get('pca_model_id')})."
        )
    return bundle


def _get_test_images_array(pca_data: dict[str, Any]) -> np.ndarray | None:
    """Retrieve test images array (N_test, h*w) from pca_data or saved test_images."""
    h, w = pca_data["image_shape"]
    if "X_test" in pca_data and len(pca_data["X_test"]) > 0:
        return np.asarray(pca_data["X_test"], dtype=np.float32)

    test_records = pca_data.get("test_records", [])
    model_dir = pca_data.get("model_dir")
    loaded: list[np.ndarray] = []
    for rec in test_records:
        cand = Path(rec["path"]) if rec.get("path") else None
        if (
            (cand is None or not cand.exists())
            and model_dir
            and rec.get("relative_path")
        ):
            cand = Path(model_dir) / rec["relative_path"]
        if cand is not None and cand.exists():
            with Image.open(cand) as img:
                gray = img.convert("L").resize((w, h), Image.Resampling.BILINEAR)
                loaded.append(np.asarray(gray, dtype=np.float32).reshape(-1))
    if loaded:
        return np.stack(loaded, axis=0)
    return None


def render_knn_evaluation_image(
    pca_data: dict[str, Any],
    knn_bundle: dict[str, Any],
    y_pred: np.ndarray | None,
) -> Image.Image:
    """Render a 4×4 qualitative test prediction gallery alongside confusion matrix."""
    h, w = pca_data["image_shape"]
    target_names = [str(n) for n in pca_data["target_names"]]
    y_test = np.asarray(pca_data.get("y_test", []), dtype=np.int64)
    X_test = _get_test_images_array(pca_data)
    conf_mat = knn_bundle.get("confusion_matrix")
    acc = knn_bundle.get("accuracy")

    fig = plt.figure(figsize=(13.5, 7.5), dpi=100)
    gs = fig.add_gridspec(4, 6, wspace=0.28, hspace=0.48)

    # Left 4x4 grid: Qualitative predictions (Exercise #5 of reference notebook)
    if X_test is not None and y_pred is not None and len(X_test) > 0:
        n_show = min(16, len(X_test), len(y_pred), len(y_test))
        for i in range(n_show):
            r = i // 4
            c = i % 4
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(X_test[i].reshape((h, w)), cmap=plt.cm.gray)
            lbl_title = title(y_pred, y_test, target_names, i)
            is_correct = int(y_pred[i]) == int(y_test[i])
            ax.set_title(
                lbl_title,
                fontsize=8,
                color="#15803d" if is_correct else "#dc2626",
            )
            ax.set_xticks(())
            ax.set_yticks(())

    # Right 2 columns: Confusion Matrix & Quantitative Metrics
    ax_cm = fig.add_subplot(gs[:3, 4:])
    if conf_mat is not None:
        cm_arr = np.asarray(conf_mat, dtype=np.int64)
        im = ax_cm.imshow(cm_arr, cmap="Blues")
        short_names = [n.rsplit(" ", 1)[-1] for n in target_names]
        ax_cm.set_xticks(range(len(short_names)))
        ax_cm.set_yticks(range(len(short_names)))
        ax_cm.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax_cm.set_yticklabels(short_names, fontsize=8)
        ax_cm.set_xlabel("Predicted label", fontsize=9)
        ax_cm.set_ylabel("True label", fontsize=9)
        ax_cm.set_title("Confusion Matrix", fontsize=10, fontweight="bold")
        thresh = cm_arr.max() / 2.0 if cm_arr.size else 0
        for r_idx in range(cm_arr.shape[0]):
            for c_idx in range(cm_arr.shape[1]):
                ax_cm.text(
                    c_idx,
                    r_idx,
                    str(cm_arr[r_idx, c_idx]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm_arr[r_idx, c_idx] > thresh else "black",
                )
        fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    else:
        ax_cm.axis("off")

    ax_txt = fig.add_subplot(gs[3, 4:])
    ax_txt.axis("off")
    acc_str = f"{acc * 100:.2f}%" if acc is not None else "N/A"
    summary_lines = [
        f"K-NN Test Accuracy: {acc_str}",
        f"Best k (neighbours): {knn_bundle['best_k']}",
        f"Distance metric: {knn_bundle['metric']} (p={knn_bundle['p']})",
        f"Weighting: {knn_bundle['weights']}",
        f"GridSearchCV: {'Enabled' if knn_bundle['use_grid_search'] else 'Disabled'}",
        f"PCA components: {knn_bundle['n_components']}",
    ]
    ax_txt.text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f0fdf4",
            "edgecolor": "#86efac",
        },
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class EigenfaceTrainerDialog:
    """Interactive 3-stage dialog for PCA training, weights, and K-NN training."""

    def __init__(self, parent: tk.Misc, document: ImageDocument) -> None:
        self.parent = parent
        self.document = document

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Eigenface / PCA & K-NN Trainer")
        self.dialog.transient(parent)
        self.dialog.geometry("980x690")
        self.dialog.minsize(860, 600)
        self.dialog.grab_set()

        self.pca_result: dict[str, Any] | None = None
        self.knn_result: dict[str, Any] | None = None
        self.tool_result: ToolResult | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None

        self._init_vars()
        self._build_ui()
        self.dialog.protocol("WM_DELETE_WINDOW", self.dialog.destroy)

    def _init_vars(self) -> None:
        # Section 1: Dataset & PCA
        self.source_var = tk.StringVar(value="LFW")
        self.local_folder_var = tk.StringVar(value="")
        self.export_images_var = tk.BooleanVar(value=True)
        self.export_dir_var = tk.StringVar(value=str(DEFAULT_LFW_EXPORT_DIR))
        self.min_faces_var = tk.IntVar(value=70)
        self.resize_var = tk.DoubleVar(value=0.4)
        self.test_split_var = tk.DoubleVar(value=0.25)
        self.seed_var = tk.IntVar(value=42)

        self.n_components_var = tk.IntVar(value=100)
        self.solver_var = tk.StringVar(value="randomized")
        self.whiten_var = tk.BooleanVar(value=True)
        self.gallery_rows_var = tk.IntVar(value=3)
        self.gallery_cols_var = tk.IntVar(value=4)
        self.model_dir_var = tk.StringVar(value=str(DEFAULT_EIGENFACE_MODEL_DIR))

        # Section 3: K-NN
        self.k_neighbors_var = tk.IntVar(value=5)
        self.metric_var = tk.StringVar(value="minkowski")
        self.p_var = tk.IntVar(value=2)
        self.weights_var = tk.StringVar(value="uniform")
        self.algo_var = tk.StringVar(value="auto")
        self.use_grid_var = tk.BooleanVar(value=False)
        self.k_candidates_var = tk.StringVar(value="3, 5, 7, 9, 11")
        self.cv_folds_var = tk.IntVar(value=5)

        self.status_var = tk.StringVar(
            value=(
                "Configure dataset and PCA parameters, then click "
                "'1. Train PCA & Save Model'."
            )
        )

    def _build_ui(self) -> None:
        main = ttk.Frame(self.dialog, padding=10)
        main.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Dataset & PCA Configuration
        tab_pca = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_pca, text="1. Dataset & PCA Training")
        self._build_pca_tab(tab_pca)

        # Tab 2: PCA Coefficients & Eigenfaces
        tab_coeffs = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_coeffs, text="2. PCA Coefficients (Weights)")
        self._build_coeffs_tab(tab_coeffs)

        # Tab 3: Train K-NN & Evaluation
        tab_knn = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab_knn, text="3. Train K-NN & Evaluation")
        self._build_knn_tab(tab_knn)

        bottom = ttk.Frame(main, padding=(0, 8, 0, 0))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left")
        ttk.Button(bottom, text="Close", command=self.dialog.destroy).pack(side="right")
        ttk.Button(
            bottom,
            text="Apply Result to Main Canvas",
            command=self._apply_and_close,
        ).pack(side="right", padx=(0, 8))

    def _build_pca_tab(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent)
        left.pack(side="left", fill="y", padx=(0, 12))

        ds_box = ttk.LabelFrame(left, text="A. Dataset Preparation", padding=10)
        ds_box.pack(fill="x", pady=(0, 8))

        ttk.Label(ds_box, text="Dataset source:").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Combobox(
            ds_box,
            textvariable=self.source_var,
            values=("LFW", "Local"),
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(ds_box, text="Local dataset folder:").grid(
            row=1, column=0, sticky="w", pady=3
        )
        loc_row = ttk.Frame(ds_box)
        loc_row.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=3)
        ttk.Entry(loc_row, textvariable=self.local_folder_var, width=24).pack(
            side="left"
        )
        ttk.Button(
            loc_row,
            text="…",
            width=3,
            command=lambda: self._browse_dir(self.local_folder_var),
        ).pack(side="left", padx=(4, 0))

        ttk.Label(ds_box, text="LFW min faces / resize:").grid(
            row=2, column=0, sticky="w", pady=3
        )
        lfw_sub = ttk.Frame(ds_box)
        lfw_sub.grid(row=2, column=1, sticky="w", padx=6, pady=3)
        ttk.Spinbox(
            lfw_sub, textvariable=self.min_faces_var, from_=10, to=200, width=6
        ).pack(side="left")
        ttk.Spinbox(
            lfw_sub,
            textvariable=self.resize_var,
            from_=0.2,
            to=1.0,
            increment=0.1,
            width=6,
        ).pack(side="left", padx=(6, 0))

        ttk.Checkbutton(
            ds_box,
            text="Save train/test images & correspondence to folder:",
            variable=self.export_images_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 2))
        exp_row = ttk.Frame(ds_box)
        exp_row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Entry(exp_row, textvariable=self.export_dir_var, width=34).pack(side="left")
        ttk.Button(
            exp_row,
            text="…",
            width=3,
            command=lambda: self._browse_dir(self.export_dir_var),
        ).pack(side="left", padx=(4, 0))

        pca_box = ttk.LabelFrame(left, text="B. PCA Configuration", padding=10)
        pca_box.pack(fill="x", pady=(0, 8))

        ttk.Label(pca_box, text="Number of components:").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Spinbox(
            pca_box, textvariable=self.n_components_var, from_=2, to=500, width=10
        ).grid(row=0, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(pca_box, text="PCA solver:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(
            pca_box,
            textvariable=self.solver_var,
            values=("randomized", "full", "arpack", "auto"),
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=3)

        ttk.Checkbutton(
            pca_box, text="Whitening enabled", variable=self.whiten_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=3)

        ttk.Label(pca_box, text="Test split / Seed:").grid(
            row=3, column=0, sticky="w", pady=3
        )
        split_sub = ttk.Frame(pca_box)
        split_sub.grid(row=3, column=1, sticky="w", padx=6, pady=3)
        ttk.Spinbox(
            split_sub,
            textvariable=self.test_split_var,
            from_=0.1,
            to=0.5,
            increment=0.05,
            width=6,
        ).pack(side="left")
        ttk.Spinbox(
            split_sub, textvariable=self.seed_var, from_=0, to=9999, width=6
        ).pack(side="left", padx=(6, 0))

        ttk.Label(pca_box, text="Eigenface grid (rows × cols):").grid(
            row=4, column=0, sticky="w", pady=3
        )
        grid_sub = ttk.Frame(pca_box)
        grid_sub.grid(row=4, column=1, sticky="w", padx=6, pady=3)
        ttk.Spinbox(
            grid_sub, textvariable=self.gallery_rows_var, from_=1, to=8, width=5
        ).pack(side="left")
        ttk.Spinbox(
            grid_sub, textvariable=self.gallery_cols_var, from_=1, to=8, width=5
        ).pack(side="left", padx=(6, 0))

        ttk.Label(pca_box, text="Model output directory:").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 2)
        )
        mdir_row = ttk.Frame(pca_box)
        mdir_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Entry(mdir_row, textvariable=self.model_dir_var, width=32).pack(side="left")
        ttk.Button(
            mdir_row,
            text="…",
            width=3,
            command=lambda: self._browse_dir(self.model_dir_var),
        ).pack(side="left", padx=(4, 0))

        ttk.Button(
            left,
            text="1. Train PCA & Save Eigenface Model",
            command=self._run_pca_training,
        ).pack(fill="x", pady=(4, 4))

        ttk.Button(
            left,
            text="Load Existing Saved PCA Model…",
            command=self._load_existing_pca,
        ).pack(fill="x")

        # Right preview area
        right = ttk.LabelFrame(
            parent,
            text="C. Mean Face, Eigenface Gallery & Explained Variance",
            padding=8,
        )
        right.pack(side="left", fill="both", expand=True)
        self.pca_canvas = tk.Canvas(right, background="#181b20", highlightthickness=0)
        self.pca_canvas.pack(fill="both", expand=True)

    def _build_coeffs_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text=(
                "Learned Training Image Representations in PCA / Eigenface Space "
                "(w_i = U_K^T (x_i - μ)):"
            ),
        ).pack(anchor="w", pady=(0, 6))

        cols = ("person", "pc1", "pc2", "pc3", "pc4", "pc5", "pc6")
        self.coeffs_tree = ttk.Treeview(
            parent, columns=cols, show="tree headings", height=20
        )
        self.coeffs_tree.heading("#0", text="Image")
        self.coeffs_tree.column("#0", width=220)
        self.coeffs_tree.heading("person", text="Identity")
        self.coeffs_tree.column("person", width=150)
        for idx in range(1, 7):
            col_id = f"pc{idx}"
            self.coeffs_tree.heading(col_id, text=f"PC{idx}")
            self.coeffs_tree.column(col_id, width=85, anchor="e")
        self.coeffs_tree.pack(fill="both", expand=True)

    def _build_knn_tab(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent)
        left.pack(side="left", fill="y", padx=(0, 12))

        knn_box = ttk.LabelFrame(left, text="E. K-NN Parameters", padding=10)
        knn_box.pack(fill="x", pady=(0, 8))

        ttk.Label(knn_box, text="Number of neighbours (k):").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Spinbox(
            knn_box, textvariable=self.k_neighbors_var, from_=1, to=50, width=10
        ).grid(row=0, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(knn_box, text="Distance metric:").grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Combobox(
            knn_box,
            textvariable=self.metric_var,
            values=("minkowski", "euclidean", "manhattan", "cosine"),
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(knn_box, text="Minkowski p:").grid(
            row=2, column=0, sticky="w", pady=3
        )
        ttk.Spinbox(knn_box, textvariable=self.p_var, from_=1, to=5, width=10).grid(
            row=2, column=1, sticky="w", padx=6, pady=3
        )

        ttk.Label(knn_box, text="Weighting:").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Combobox(
            knn_box,
            textvariable=self.weights_var,
            values=("uniform", "distance"),
            state="readonly",
            width=12,
        ).grid(row=3, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(knn_box, text="Search algorithm:").grid(
            row=4, column=0, sticky="w", pady=3
        )
        ttk.Combobox(
            knn_box,
            textvariable=self.algo_var,
            values=("auto", "ball_tree", "kd_tree", "brute"),
            state="readonly",
            width=12,
        ).grid(row=4, column=1, sticky="w", padx=6, pady=3)

        gs_box = ttk.LabelFrame(
            left, text="Optional Hyperparameter Search (GridSearchCV)", padding=10
        )
        gs_box.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(
            gs_box, text="Use GridSearchCV", variable=self.use_grid_var
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(gs_box, text="k candidates:").grid(
            row=1, column=0, sticky="w", pady=3
        )
        ttk.Entry(gs_box, textvariable=self.k_candidates_var, width=16).grid(
            row=1, column=1, sticky="w", padx=6, pady=3
        )
        ttk.Label(gs_box, text="CV folds:").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Spinbox(
            gs_box, textvariable=self.cv_folds_var, from_=2, to=15, width=8
        ).grid(row=2, column=1, sticky="w", padx=6, pady=3)

        ttk.Button(
            left,
            text="2. Train K-NN Classifier & Evaluate",
            command=self._run_knn_training,
        ).pack(fill="x", pady=(4, 8))

        rep_box = ttk.LabelFrame(left, text="Classification Report", padding=6)
        rep_box.pack(fill="both", expand=True)
        self.report_text = tk.Text(rep_box, width=38, height=12, font=("Courier", 8))
        self.report_text.pack(fill="both", expand=True)

        right = ttk.LabelFrame(
            parent,
            text="Qualitative Predictions (4×4 Gallery) & Confusion Matrix",
            padding=8,
        )
        right.pack(side="left", fill="both", expand=True)
        self.knn_canvas = tk.Canvas(right, background="#181b20", highlightthickness=0)
        self.knn_canvas.pack(fill="both", expand=True)

    def _browse_dir(self, var: tk.StringVar) -> None:
        chosen = filedialog.askdirectory(parent=self.dialog, title="Select folder")
        if chosen:
            var.set(chosen)

    def _render_on_canvas(self, canvas: tk.Canvas, img: Image.Image) -> None:
        canvas.update_idletasks()
        cw = max(320, canvas.winfo_width())
        ch = max(240, canvas.winfo_height())
        preview = img.copy()
        preview.thumbnail((cw, ch), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(preview)
        canvas.delete("all")
        canvas.create_image(
            cw // 2, ch // 2, image=self._preview_photo, anchor="center"
        )

    def _populate_coeffs_table(self, pca_res: dict[str, Any]) -> None:
        for item in self.coeffs_tree.get_children():
            self.coeffs_tree.delete(item)
        X_train_pca = pca_res["X_train_pca"]
        y_train = pca_res["y_train"]
        target_names = pca_res["target_names"]
        records = pca_res.get("train_records", [])

        for idx in range(min(200, len(X_train_pca))):
            row_weights = X_train_pca[idx]
            fname = (
                records[idx]["filename"]
                if idx < len(records)
                else f"face_{idx + 1:03d}"
            )
            person_name = target_names[int(y_train[idx])]
            vals = [person_name] + [
                f"{float(row_weights[c]):.2f}" if c < len(row_weights) else "—"
                for c in range(6)
            ]
            self.coeffs_tree.insert("", "end", text=fname, values=tuple(vals))

    def _run_pca_training(self) -> None:
        try:
            self.status_var.set("Loading dataset and computing PCA eigenfaces…")
            self.dialog.update_idletasks()

            export_dir = (
                self.export_dir_var.get().strip()
                if self.export_images_var.get()
                else None
            )
            dataset = load_face_dataset(
                source=self.source_var.get(),
                folder_path=self.local_folder_var.get().strip() or None,
                min_faces_per_person=int(self.min_faces_var.get()),
                resize=float(self.resize_var.get()),
                test_size=float(self.test_split_var.get()),
                random_state=int(self.seed_var.get()),
                export_dir=export_dir,
            )
            pca_res = train_eigenface_pca(
                dataset,
                n_components=int(self.n_components_var.get()),
                svd_solver=self.solver_var.get(),
                whiten=bool(self.whiten_var.get()),
                random_state=int(self.seed_var.get()),
            )
            model_dir = Path(
                self.model_dir_var.get().strip() or str(DEFAULT_EIGENFACE_MODEL_DIR)
            )
            saved_paths = save_eigenface_model(
                pca_res,
                model_dir=model_dir,
                gallery_rows=int(self.gallery_rows_var.get()),
                gallery_cols=int(self.gallery_cols_var.get()),
            )
            pca_res["model_dir"] = saved_paths["model_dir"]
            summary_img = render_pca_summary_image(
                pca_res,
                n_row=int(self.gallery_rows_var.get()),
                n_col=int(self.gallery_cols_var.get()),
            )
            pca_res["summary_image"] = summary_img
            self.pca_result = pca_res

            self._render_on_canvas(self.pca_canvas, summary_img)
            self._populate_coeffs_table(pca_res)

            h, w = pca_res["image_shape"]
            exp_pct = pca_res["total_explained_variance"] * 100
            w_shape = pca_res["X_train_pca"].shape
            self.tool_result = ToolResult(
                image=summary_img,
                message=(
                    f"Trained Eigenface PCA ({pca_res['n_components']} components, "
                    f"{exp_pct:.1f}% explained variance) -> {model_dir.name}."
                ),
                details={
                    "Operation": "Eigenface / PCA Training",
                    "Dataset source": dataset["source"],
                    "Train / Test faces": (
                        f"{len(pca_res['X_train'])} / {len(pca_res['X_test'])}"
                    ),
                    "Identities": len(pca_res["target_names"]),
                    "Image dimensions": f"{w} × {h}",
                    "PCA components": pca_res["n_components"],
                    "Solver / Whitening": (
                        f"{pca_res['svd_solver']} / {pca_res['whiten']}"
                    ),
                    "Explained variance": f"{exp_pct:.2f}%",
                    "PCA weights matrix": f"{w_shape[0]} × {w_shape[1]}",
                    "Model directory": str(model_dir),
                },
            )
            self.status_var.set(
                f"PCA trained and saved to {model_dir}. "
                "You can now inspect coefficients or train K-NN."
            )
        except Exception as exc:
            messagebox.showerror("PCA Training Failed", str(exc), parent=self.dialog)
            self.status_var.set("PCA training failed.")

    def _load_existing_pca(self) -> None:
        chosen = filedialog.askdirectory(
            parent=self.dialog,
            title="Select saved Eigenface model directory",
            initialdir=str(DEFAULT_EIGENFACE_MODEL_DIR.parent),
        )
        if not chosen:
            return
        try:
            pca_res = load_eigenface_model(chosen)
            summary_img = render_pca_summary_image(
                pca_res,
                n_row=int(self.gallery_rows_var.get()),
                n_col=int(self.gallery_cols_var.get()),
            )
            pca_res["summary_image"] = summary_img
            self.pca_result = pca_res
            self.model_dir_var.set(str(pca_res["model_dir"]))
            self._render_on_canvas(self.pca_canvas, summary_img)
            self._populate_coeffs_table(pca_res)
            self.status_var.set(f"Loaded PCA model from {chosen}.")
        except Exception as exc:
            messagebox.showerror("Load PCA Failed", str(exc), parent=self.dialog)

    def _run_knn_training(self) -> None:
        if self.pca_result is None:
            model_dir = Path(self.model_dir_var.get().strip())
            if (model_dir / "pca_model.npz").exists():
                self.pca_result = load_eigenface_model(model_dir)
            else:
                messagebox.showinfo(
                    "Train or Load PCA First",
                    "Please train or load a PCA model in Tab 1 before training K-NN.",
                    parent=self.dialog,
                )
                return

        try:
            k_cands = [
                int(x.strip())
                for x in self.k_candidates_var.get().split(",")
                if x.strip()
            ]
            knn_out = train_knn_classifier(
                self.pca_result,
                n_neighbors=int(self.k_neighbors_var.get()),
                metric=self.metric_var.get(),
                p=int(self.p_var.get()),
                weights=self.weights_var.get(),
                algorithm=self.algo_var.get(),
                use_grid_search=bool(self.use_grid_var.get()),
                k_candidates=k_cands or [3, 5, 7, 9, 11],
                cv_folds=int(self.cv_folds_var.get()),
            )
            self.knn_result = knn_out

            self.report_text.delete("1.0", "end")
            self.report_text.insert("1.0", knn_out.get("classification_report", ""))

            eval_img = knn_out["evaluation_image"]
            self._render_on_canvas(self.knn_canvas, eval_img)

            acc = knn_out.get("accuracy")
            acc_str = f"{acc * 100:.2f}%" if acc is not None else "N/A"
            exp_pct = self.pca_result["total_explained_variance"] * 100
            self.tool_result = ToolResult(
                image=eval_img,
                message=(
                    f"Trained K-NN classifier (k={knn_out['best_k']}, "
                    f"accuracy={acc_str}) paired with PCA model "
                    f"{self.pca_result['model_id'][:8]}."
                ),
                details={
                    "Operation": "PCA + K-NN Training",
                    "PCA components": self.pca_result["n_components"],
                    "Explained variance": f"{exp_pct:.2f}%",
                    "K-NN best k": knn_out["best_k"],
                    "Metric / Weighting": (
                        f"{knn_out['metric']} / {knn_out['weights']}"
                    ),
                    "GridSearchCV": "Yes" if knn_out["use_grid_search"] else "No",
                    "Test accuracy": acc_str,
                    "Saved classifier": (
                        Path(knn_out["joblib_path"]).name
                        if knn_out.get("joblib_path")
                        else "—"
                    ),
                    "Model directory": str(self.pca_result.get("model_dir", "")),
                },
            )
            self.status_var.set(
                f"K-NN trained (k={knn_out['best_k']}, Test Accuracy={acc_str})."
            )
        except Exception as exc:
            messagebox.showerror("K-NN Training Failed", str(exc), parent=self.dialog)

    def _apply_and_close(self) -> None:
        self.dialog.destroy()

    def show_and_wait(self) -> ToolResult | None:
        self.parent.wait_window(self.dialog)
        return self.tool_result


class EigenfaceTrainerTool(ForensicsTool):
    """Tool 4 — Eigenface / PCA Trainer and K-NN Classifier Trainer."""

    tool_id = "eigenface_trainer"
    title = "Eigenface / PCA Trainer"
    category = "Face Analysis"
    description = (
        "Prepare the face dataset, train PCA, visualise eigenfaces and projection "
        "coefficients, train K-NN and persist the models."
    )
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> ToolResult | None:
        # Support headless programmatic execution for automated testing
        if parameters is not None:
            dataset = load_face_dataset(
                source=parameters.get("source", "LFW"),
                folder_path=parameters.get("folder_path"),
                min_faces_per_person=parameters.get("min_faces_per_person", 70),
                resize=parameters.get("resize", 0.4),
                test_size=parameters.get("test_size", 0.25),
                random_state=parameters.get("random_state", 42),
                export_dir=parameters.get("export_dir"),
            )
            pca_res = train_eigenface_pca(
                dataset,
                n_components=parameters.get("n_components", 100),
                svd_solver=parameters.get("svd_solver", "randomized"),
                whiten=parameters.get("whiten", True),
                random_state=parameters.get("random_state", 42),
            )
            model_dir = Path(parameters.get("model_dir", DEFAULT_EIGENFACE_MODEL_DIR))
            save_eigenface_model(
                pca_res,
                model_dir=model_dir,
                gallery_rows=parameters.get("gallery_rows", 3),
                gallery_cols=parameters.get("gallery_cols", 4),
            )
            pca_res["model_dir"] = model_dir
            summary_img = render_pca_summary_image(
                pca_res,
                n_row=parameters.get("gallery_rows", 3),
                n_col=parameters.get("gallery_cols", 4),
            )

            exp_pct = pca_res["total_explained_variance"] * 100
            w_shape = pca_res["X_train_pca"].shape
            details: dict[str, Any] = {
                "Operation": "Eigenface / PCA Training",
                "Dataset source": dataset["source"],
                "Train / Test faces": (
                    f"{len(pca_res['X_train'])} / {len(pca_res['X_test'])}"
                ),
                "Identities": len(pca_res["target_names"]),
                "PCA components": pca_res["n_components"],
                "Explained variance": f"{exp_pct:.2f}%",
                "PCA weights matrix": f"{w_shape[0]} × {w_shape[1]}",
                "Model directory": str(model_dir),
            }

            if parameters.get("train_knn", True):
                knn_out = train_knn_classifier(
                    pca_res,
                    n_neighbors=parameters.get("n_neighbors", 5),
                    metric=parameters.get("metric", "minkowski"),
                    p=parameters.get("p", 2),
                    weights=parameters.get("weights", "uniform"),
                    algorithm=parameters.get("algorithm", "auto"),
                    use_grid_search=parameters.get("use_grid_search", False),
                    k_candidates=parameters.get("k_candidates", (3, 5, 7, 9, 11)),
                    cv_folds=parameters.get("cv_folds", 5),
                )
                acc = knn_out.get("accuracy")
                details["K-NN best k"] = knn_out["best_k"]
                details["Test accuracy"] = (
                    f"{acc * 100:.2f}%" if acc is not None else "N/A"
                )
                if parameters.get("return_knn_image", False):
                    summary_img = knn_out["evaluation_image"]

            return ToolResult(
                image=summary_img,
                message=(
                    f"Trained Eigenface PCA ({pca_res['n_components']} components, "
                    f"{pca_res['total_explained_variance'] * 100:.1f}% variance)."
                ),
                details=details,
            )

        dialog = EigenfaceTrainerDialog(parent, document)
        return dialog.show_and_wait()
