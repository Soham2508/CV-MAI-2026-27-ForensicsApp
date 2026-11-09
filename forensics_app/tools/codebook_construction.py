"""Set 7.1 — Codebook Construction (Bag of Visual Words Stage 1).

Extracts local descriptors (SIFT/ORB, reusing ``descriptor_extraction.py``) from
training images only, clusters them with ``MiniBatchKMeans`` / ``KMeans`` using
``K = 10 * num_classes``, extracts representative visual-word image patches
nearest to each centroid, visualises the visual vocabulary gallery and cluster
assignment statistics, and persists the codebook and metadata for downstream
histogram construction and classification.
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

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .descriptor_extraction import extract_descriptors

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm"}
DEFAULT_DATASET_DIR = Path("datasets/Caltech_101_subset")
DEFAULT_VISUAL_WORDS_ROOT = Path("models/visual_words")
DEFAULT_VISUAL_WORDS_MODEL_DIR = DEFAULT_VISUAL_WORDS_ROOT / "codebook_001"


def default_descriptor_config(
    *,
    method: str = "SIFT",
    max_keypoints: int = 500,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
    n_octaves: int = 4,
    n_scales: int = 3,
    upsampling: int = 1,
    fast_threshold: float = 0.08,
    grayscale: bool = True,
) -> dict[str, Any]:
    """Return a canonical descriptor extraction configuration dictionary."""
    method_norm = str(method).strip().upper()
    if method_norm not in ("SIFT", "ORB"):
        raise ValueError("Descriptor method must be 'SIFT' or 'ORB'.")
    return {
        "method": method_norm,
        "max_keypoints": max(1, int(max_keypoints)),
        "contrast_threshold": float(contrast_threshold),
        "edge_threshold": float(edge_threshold),
        "sigma": float(sigma),
        "n_octaves": max(1, int(n_octaves)),
        "n_scales": max(1, int(n_scales)),
        "upsampling": max(1, int(upsampling)),
        "fast_threshold": float(fast_threshold),
        "grayscale": bool(grayscale),
    }


def _is_valid_image_file(path: Path) -> bool:
    """Return True if ``path`` is a non-hidden image file."""
    if not path.is_file():
        return False
    if path.name.startswith(".") or path.name.startswith("._"):
        return False
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _collect_class_folders(folder: Path) -> dict[str, list[Path]]:
    """Scan a directory of class subfolders and return sorted image paths per class."""
    class_map: dict[str, list[Path]] = {}
    if not folder.exists() or not folder.is_dir():
        return class_map
    for sub in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not sub.is_dir():
            continue
        if (
            sub.name.startswith(".")
            or sub.name.startswith("._")
            or sub.name == "__MACOSX"
        ):
            continue
        images = sorted(
            [f for f in sub.iterdir() if _is_valid_image_file(f)],
            key=lambda p: p.name.lower(),
        )
        if images:
            class_map[sub.name] = images
    return class_map


def inspect_dataset_directory(dataset_dir: str | Path) -> dict[str, Any]:
    """Inspect a dataset directory and return detected classes and image counts."""
    root = Path(dataset_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    train_dir = root / "train"
    test_dir = root / "test"
    if train_dir.is_dir() and test_dir.is_dir():
        train_map = _collect_class_folders(train_dir)
        test_map = _collect_class_folders(test_dir)
        class_names = sorted(set(train_map.keys()) | set(test_map.keys()))
        if not class_names:
            raise ValueError(
                f"No class image folders found inside {train_dir} / {test_dir}."
            )
        return {
            "dataset_dir": str(root),
            "split_mode": "pre_split",
            "class_names": class_names,
            "num_classes": len(class_names),
            "train_counts": {c: len(train_map.get(c, [])) for c in class_names},
            "test_counts": {c: len(test_map.get(c, [])) for c in class_names},
        }

    # Check if root has a single nested wrapper folder
    # (e.g. datasets/Caltech_101_subset/Caltech_101_subset)
    class_map = _collect_class_folders(root)
    if len(class_map) == 1:
        only_sub = root / next(iter(class_map.keys()))
        nested_map = _collect_class_folders(only_sub)
        if len(nested_map) >= 2:
            root = only_sub
            class_map = nested_map

    if not class_map:
        # Check one level deeper if root only has subdirectories without direct images
        for sub in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if sub.is_dir() and not sub.name.startswith(".") and sub.name != "__MACOSX":
                cand = _collect_class_folders(sub)
                if cand:
                    root = sub
                    class_map = cand
                    break

    if not class_map:
        raise ValueError(
            f"No class folders with images found in {root}. Expected either "
            "'train/<class>/' + 'test/<class>/' or '<class>/' folders."
        )

    class_names = sorted(class_map.keys())
    return {
        "dataset_dir": str(root),
        "split_mode": "stratified_split",
        "class_names": class_names,
        "num_classes": len(class_names),
        "total_counts": {c: len(class_map[c]) for c in class_names},
    }


def load_visual_words_dataset(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    *,
    test_size: float = 0.25,
    random_state: int = 0,
    max_images_per_class: int | None = None,
) -> dict[str, Any]:
    """Load a visual-words dataset and split into train/test records before K-Means.

    Supports both pre-split ``train/`` + ``test/`` directories and flat class-folder
    datasets (such as ``Caltech_101_subset/butterfly``, ``dalmatian``, ``kangaroo``).
    Only training records are used for codebook learning to prevent data leakage.
    """
    info = inspect_dataset_directory(dataset_dir)
    root = Path(info["dataset_dir"])
    class_names: list[str] = list(info["class_names"])
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    if info["split_mode"] == "pre_split":
        train_map = _collect_class_folders(root / "train")
        test_map = _collect_class_folders(root / "test")
        for cname in class_names:
            cid = class_to_id[cname]
            tr_files = train_map.get(cname, [])
            te_files = test_map.get(cname, [])
            if max_images_per_class is not None and max_images_per_class > 0:
                tr_files = tr_files[:max_images_per_class]
                te_files = te_files[:max_images_per_class]
            for p in tr_files:
                train_records.append(
                    {
                        "path": str(p),
                        "filename": p.name,
                        "class_name": cname,
                        "class_id": cid,
                    }
                )
            for p in te_files:
                test_records.append(
                    {
                        "path": str(p),
                        "filename": p.name,
                        "class_name": cname,
                        "class_id": cid,
                    }
                )
    else:
        class_map = _collect_class_folders(root)
        all_records: list[dict[str, Any]] = []
        all_labels: list[int] = []
        rng = np.random.default_rng(random_state)

        for cname in class_names:
            cid = class_to_id[cname]
            files = list(class_map[cname])
            if (
                max_images_per_class is not None
                and max_images_per_class > 0
                and len(files) > max_images_per_class
            ):
                indices = sorted(
                    rng.choice(len(files), size=max_images_per_class, replace=False)
                )
                files = [files[i] for i in indices]
            for p in files:
                all_records.append(
                    {
                        "path": str(p),
                        "filename": p.name,
                        "class_name": cname,
                        "class_id": cid,
                    }
                )
                all_labels.append(cid)

        if not all_records:
            raise ValueError(f"No valid images found in {root}.")

        # Check if stratified split is possible (at least 2 images per class)
        counts = np.bincount(all_labels, minlength=len(class_names))
        if float(test_size) <= 0.0:
            train_records = list(all_records)
            test_records = []
        elif np.all(counts >= 2) and len(all_records) >= 2 * len(class_names):
            tr_rec, te_rec = train_test_split(
                all_records,
                test_size=float(test_size),
                random_state=int(random_state),
                stratify=all_labels,
            )
            train_records = list(tr_rec)
            test_records = list(te_rec)
        else:
            # Per-class manual split fallback for tiny synthetic datasets
            for cname in class_names:
                c_recs = [r for r in all_records if r["class_name"] == cname]
                if len(c_recs) == 1:
                    train_records.extend(c_recs)
                    test_records.extend(c_recs)
                else:
                    n_te = max(1, int(round(len(c_recs) * float(test_size))))
                    n_tr = max(1, len(c_recs) - n_te)
                    train_records.extend(c_recs[:n_tr])
                    test_records.extend(c_recs[n_tr:])

    if not train_records:
        raise ValueError("Training split contains no images.")

    return {
        "dataset_dir": str(root),
        "split_mode": info["split_mode"],
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_records": train_records,
        "test_records": test_records,
        "test_size": float(test_size),
        "random_state": int(random_state),
    }


def extract_local_descriptors_from_image(
    image: Image.Image,
    descriptor_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """Extract local keypoints and float32 descriptors reusing ``extract_descriptors``.

    Returns ``(keypoints, descriptors_float32, gray_pil_image)`` where ``keypoints``
    has shape ``(N, 2)`` in ``(row, col)`` order and ``descriptors_float32`` has
    shape ``(N, D)``.
    """
    cfg = default_descriptor_config(**(descriptor_config or {}))
    method = cfg["method"]
    max_kp = int(cfg["max_keypoints"])
    gray_pil = image.convert("L") if cfg.get("grayscale", True) else image.copy()

    min_dim = 32 if method == "ORB" else 12
    if min(gray_pil.size) < min_dim:
        scale = min_dim / max(1, min(gray_pil.size))
        new_w = max(min_dim, int(round(gray_pil.width * scale)))
        new_h = max(min_dim, int(round(gray_pil.height * scale)))
        gray_pil = gray_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)

    if method == "SIFT":
        n_scales = int(cfg.get("n_scales", 3))
        c_dog = float(cfg.get("contrast_threshold", 0.04)) / max(1, n_scales)
        keypoints, descriptors, _ = extract_descriptors(
            gray_pil,
            method="SIFT",
            n_octaves=int(cfg.get("n_octaves", 4)),
            n_scales=n_scales,
            sigma_min=float(cfg.get("sigma", 1.6)),
            upsampling=int(cfg.get("upsampling", 1)),
            c_dog=c_dog,
            c_edge=float(cfg.get("edge_threshold", 10.0)),
            max_keypoints=max_kp,
        )
    else:
        keypoints, descriptors, _ = extract_descriptors(
            gray_pil,
            method="ORB",
            n_keypoints=max_kp,
            fast_threshold=float(cfg.get("fast_threshold", 0.08)),
            max_keypoints=max_kp,
        )

    return keypoints, np.asarray(descriptors, dtype=np.float32), gray_pil.convert("L")


def _crop_keypoint_patch(
    gray_arr: np.ndarray,
    row: float,
    col: float,
    patch_size: int = 28,
) -> np.ndarray:
    """Crop a square grayscale patch centered at ``(row, col)`` with reflect padding."""
    half = patch_size // 2
    r_int = int(round(float(row)))
    c_int = int(round(float(col)))
    padded = np.pad(gray_arr, pad_width=half + 2, mode="edge")
    pr = r_int + half + 2
    pc = c_int + half + 2
    patch = padded[
        pr - half : pr - half + patch_size, pc - half : pc - half + patch_size
    ]
    if patch.shape != (patch_size, patch_size):
        pil_p = Image.fromarray(patch.astype(np.uint8), mode="L").resize(
            (patch_size, patch_size), Image.Resampling.BILINEAR
        )
        return np.asarray(pil_p, dtype=np.uint8)
    return patch.astype(np.uint8)


def ensure_document_has_image(
    document: ImageDocument | None,
    image: Image.Image | None,
) -> None:
    """Seed ``document.current`` when no image is loaded so ``MainWindow.run_tool``
    displays the tool's output image instead of raising ``RuntimeError``.
    """
    if document is not None and document.current is None and image is not None:
        document.original = image.copy()
        document.current = image.copy()


def _compute_codebook_id(
    codebook: np.ndarray,
    class_names: list[str],
    descriptor_config: dict[str, Any],
) -> str:
    """Compute a deterministic ID for codebook-histogram-classifier verification."""
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(codebook, dtype=np.float32).tobytes())
    hasher.update(json.dumps(list(class_names), sort_keys=True).encode("utf-8"))
    hasher.update(json.dumps(descriptor_config, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()[:16]


def build_codebook(
    train_records: list[dict[str, Any]],
    *,
    class_names: list[str],
    descriptor_config: dict[str, Any] | None = None,
    clusters_per_class: int = 10,
    n_clusters: int | None = None,
    clustering_algorithm: str = "MiniBatchKMeans",
    random_state: int = 0,
    batch_size: int = 1024,
    max_iter: int = 100,
    n_init: int = 3,
    patch_size: int = 28,
    test_records: list[dict[str, Any]] | None = None,
    dataset_dir: str | Path | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Extract local descriptors from training images and build a K-Means codebook.

    Follows Set 7.1:
      - ``K = 10 * num_classes`` by default (``clusters_per_class * len(class_names)``)
      - Stacks training descriptors with ``np.vstack``
      - Fits ``MiniBatchKMeans(n_clusters=ncl, random_state=0).fit(descriptors)``
      - Finds the representative image patch closest to each centre ``vw 0 .. vw K-1``
    """
    if not train_records:
        raise ValueError("train_records must not be empty.")
    if not class_names:
        raise ValueError("class_names must not be empty.")

    t0 = time.perf_counter()
    cfg = default_descriptor_config(**(descriptor_config or {}))
    num_classes = len(class_names)
    k_clusters = (
        int(n_clusters)
        if (n_clusters is not None and int(n_clusters) > 0)
        else max(2, int(clusters_per_class) * num_classes)
    )

    feature_descriptors: list[np.ndarray] = []
    descriptor_meta: list[tuple[int, float, float]] = []  # (train_idx, row, col)
    gray_cache: dict[int, np.ndarray] = {}
    sample_keypoint_previews: list[dict[str, Any]] = []

    per_image_counts: list[int] = []
    image_descriptor_slices: list[tuple[int, int]] = []
    running_offset = 0
    total_train = len(train_records)

    for idx, rec in enumerate(train_records):
        if progress_callback is not None and (idx % 5 == 0 or idx == total_train - 1):
            progress_callback(idx + 1, total_train, "Extracting descriptors")
        img_path = Path(rec["path"])
        if not img_path.exists():
            per_image_counts.append(0)
            image_descriptor_slices.append((running_offset, running_offset))
            continue
        with Image.open(img_path) as img:
            kps, descs, gray_pil = extract_local_descriptors_from_image(img, cfg)

        gray_arr = np.asarray(gray_pil, dtype=np.uint8)
        n_desc = len(descs)
        per_image_counts.append(n_desc)
        image_descriptor_slices.append((running_offset, running_offset + n_desc))
        running_offset += n_desc

        if len(sample_keypoint_previews) < 10 and len(kps) > 0:
            sample_keypoint_previews.append(
                {
                    "gray_image": gray_arr,
                    "keypoints": kps[:120].copy(),
                    "class_name": rec.get("class_name", ""),
                    "filename": rec.get("filename", img_path.name),
                }
            )

        if n_desc > 0:
            feature_descriptors.append(descs)
            gray_cache[idx] = gray_arr
            for r_val, c_val in kps:
                descriptor_meta.append((idx, float(r_val), float(c_val)))

    if not feature_descriptors:
        raise ValueError(
            "No local descriptors could be extracted from the training images."
        )

    if progress_callback is not None:
        progress_callback(total_train, total_train, "Clustering visual words")

    # Stack descriptors matching Set 7.1 slide
    descriptors = feature_descriptors[0]
    for descriptor in feature_descriptors[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    descriptors = np.asarray(descriptors, dtype=np.float32)

    if len(descriptors) < k_clusters:
        k_clusters = max(1, len(descriptors))

    alg_norm = str(clustering_algorithm).strip()
    if alg_norm.lower() in ("minibatchkmeans", "mini_batch_kmeans", "minibatch"):
        alg_label = "MiniBatchKMeans"
        effective_batch = min(max(32, int(batch_size)), max(32, len(descriptors)))
        kmeans: MiniBatchKMeans | KMeans = MiniBatchKMeans(
            n_clusters=k_clusters,
            random_state=int(random_state),
            batch_size=effective_batch,
            max_iter=max(10, int(max_iter)),
            n_init=max(1, int(n_init)),
        ).fit(descriptors)
    elif alg_norm.lower() in ("kmeans", "k-means"):
        alg_label = "KMeans"
        kmeans = KMeans(
            n_clusters=k_clusters,
            random_state=int(random_state),
            max_iter=max(10, int(max_iter)),
            n_init=max(1, int(n_init)),
        ).fit(descriptors)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {clustering_algorithm!r}")

    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    labels = np.asarray(kmeans.labels_, dtype=np.int64)
    cluster_counts = np.bincount(labels, minlength=k_clusters).astype(np.int64)

    # Pre-compute per-training-image raw visual-word count histograms from labels
    train_raw_counts = np.zeros((len(train_records), k_clusters), dtype=np.int64)
    for idx, (s_off, e_off) in enumerate(image_descriptor_slices):
        if e_off > s_off:
            train_raw_counts[idx] = np.bincount(
                labels[s_off:e_off], minlength=k_clusters
            )

    # Extract representative visual-word patch closest to each centroid
    visual_word_patches = np.zeros((k_clusters, patch_size, patch_size), dtype=np.uint8)
    visual_word_sources: list[dict[str, Any]] = []

    for w_idx in range(k_clusters):
        member_indices = np.where(labels == w_idx)[0]
        if len(member_indices) > 0:
            member_descs = descriptors[member_indices]
            dists = np.linalg.norm(member_descs - centroids[w_idx], axis=1)
            best_local = int(member_indices[int(np.argmin(dists))])
            best_dist = float(np.min(dists))
        else:
            dists = np.linalg.norm(descriptors - centroids[w_idx], axis=1)
            best_local = int(np.argmin(dists))
            best_dist = float(np.min(dists))

        tr_idx, kp_r, kp_c = descriptor_meta[best_local]
        src_gray = gray_cache[tr_idx]
        visual_word_patches[w_idx] = _crop_keypoint_patch(
            src_gray, kp_r, kp_c, patch_size=patch_size
        )
        src_rec = train_records[tr_idx]
        visual_word_sources.append(
            {
                "word_index": w_idx,
                "train_record_index": tr_idx,
                "filename": src_rec.get("filename", ""),
                "class_name": src_rec.get("class_name", ""),
                "keypoint_row": kp_r,
                "keypoint_col": kp_c,
                "distance_to_centroid": best_dist,
            }
        )

    elapsed_s = time.perf_counter() - t0
    codebook_id = _compute_codebook_id(centroids, class_names, cfg)

    metadata: dict[str, Any] = {
        "codebook_id": codebook_id,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "n_clusters": k_clusters,
        "clusters_per_class": int(clusters_per_class),
        "descriptor_dim": int(centroids.shape[1]),
        "total_descriptors": int(descriptors.shape[0]),
        "num_train_images": len(train_records),
        "num_test_images": len(test_records) if test_records is not None else 0,
        "mean_descriptors_per_image": float(np.mean(per_image_counts))
        if per_image_counts
        else 0.0,
        "clustering_algorithm": alg_label,
        "random_state": int(random_state),
        "batch_size": int(batch_size),
        "max_iter": int(max_iter),
        "n_init": int(n_init),
        "patch_size": int(patch_size),
        "cluster_counts": [int(c) for c in cluster_counts.tolist()],
        "training_time_s": elapsed_s,
        "dataset_dir": str(dataset_dir) if dataset_dir else "",
        "descriptor_config": cfg,
        "visual_word_sources": visual_word_sources,
        "train_records": list(train_records),
        "test_records": list(test_records) if test_records is not None else [],
    }

    bundle: dict[str, Any] = {
        "codebook_id": codebook_id,
        "codebook": centroids,
        "visual_word_patches": visual_word_patches,
        "cluster_counts": cluster_counts,
        "train_raw_counts": train_raw_counts,
        "descriptor_config": cfg,
        "metadata": metadata,
        "sample_previews": sample_keypoint_previews,
    }
    bundle["visualization_image"] = render_codebook_visualization(bundle)
    return bundle


def render_codebook_visualization(codebook_bundle: dict[str, Any]) -> Image.Image:
    """Render Set 7.1 visual vocabulary gallery, keypoints, and cluster statistics."""
    patches = np.asarray(codebook_bundle["visual_word_patches"], dtype=np.uint8)
    counts = np.asarray(codebook_bundle["cluster_counts"], dtype=np.int64)
    meta = codebook_bundle["metadata"]
    previews: list[dict[str, Any]] = codebook_bundle.get("sample_previews", [])

    k_words = len(patches)
    fig = plt.figure(figsize=(14.5, 9.2), dpi=100)
    fig.suptitle(
        f"Set 7.1 — Codebook Construction ({meta['clustering_algorithm']}, "
        f"K = {k_words} visual words, {meta['total_descriptors']:,} descriptors)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.15, 1.25],
        height_ratios=[1.0, 1.15],
        left=0.04,
        right=0.98,
        top=0.91,
        bottom=0.07,
        wspace=0.18,
        hspace=0.28,
    )

    # Top-left: Sample training images with extracted keypoints (Slide 2 style)
    n_prev = min(len(previews), 10)
    if n_prev > 0:
        prev_cols = 5
        prev_rows = 2
        prev_gs = outer[0, 0].subgridspec(
            prev_rows, prev_cols, wspace=0.08, hspace=0.22
        )
        for i in range(prev_rows * prev_cols):
            ax = fig.add_subplot(prev_gs[i // prev_cols, i % prev_cols])
            if i < n_prev:
                item = previews[i]
                ax.imshow(item["gray_image"], cmap="gray")
                kps = item["keypoints"]
                if len(kps) > 0:
                    ax.scatter(
                        kps[:, 1],
                        kps[:, 0],
                        c="red",
                        s=8,
                        marker="+",
                        linewidths=0.8,
                    )
                ax.set_title(str(item.get("class_name", ""))[:11], fontsize=8)
            ax.set_xticks(())
            ax.set_yticks(())
    else:
        ax_empty = fig.add_subplot(outer[0, 0])
        ax_empty.text(
            0.5,
            0.5,
            f"Classes: {', '.join(meta['class_names'])}\n"
            f"Train images: {meta['num_train_images']}\n"
            f"Total descriptors: {meta['total_descriptors']:,}",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax_empty.set_axis_off()

    # Bottom-left: Cluster assignment frequency bar chart
    ax_bar = fig.add_subplot(outer[1, 0])
    indices = np.arange(k_words)
    ax_bar.bar(indices, counts, color="#2b6cb0", edgecolor="#1a365d", width=0.78)
    ax_bar.set_title(
        "Cluster Statistics (Training Descriptors per Visual Word)",
        fontsize=10,
        fontweight="bold",
    )
    ax_bar.set_xlabel("Visual word index (vw)", fontsize=9)
    ax_bar.set_ylabel("Assigned descriptors", fontsize=9)
    step_xtick = max(1, k_words // 10)
    ax_bar.set_xticks(indices[::step_xtick])
    ax_bar.set_xticklabels([f"vw {i}" for i in indices[::step_xtick]], fontsize=8)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.35)

    # Right column (spanning both rows): Visual Vocabulary Gallery (vw 0 .. vw K-1)
    max_display_words = min(k_words, 50)
    vw_cols = 5
    vw_rows = max(1, int(np.ceil(max_display_words / vw_cols)))
    vw_gs = outer[:, 1].subgridspec(vw_rows, vw_cols, wspace=0.14, hspace=0.36)

    for idx in range(vw_rows * vw_cols):
        ax_vw = fig.add_subplot(vw_gs[idx // vw_cols, idx % vw_cols])
        if idx < max_display_words:
            ax_vw.imshow(patches[idx], cmap="gray")
            ax_vw.set_title(f"vw {idx}", fontsize=8.5, pad=2)
        ax_vw.set_xticks(())
        ax_vw.set_yticks(())
        for spine in ax_vw.spines.values():
            spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_codebook(
    codebook_bundle: dict[str, Any],
    output_dir: str | Path = DEFAULT_VISUAL_WORDS_MODEL_DIR,
) -> dict[str, Path]:
    """Persist codebook centres, visual word patches, metadata, and visual_words.png."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook_path = out_dir / "codebook.npy"
    patches_path = out_dir / "visual_word_patches.npy"
    train_counts_path = out_dir / "train_raw_counts.npy"
    metadata_path = out_dir / "metadata.json"
    desc_cfg_path = out_dir / "descriptor_config.json"
    vis_path = out_dir / "visual_words.png"

    np.save(codebook_path, np.asarray(codebook_bundle["codebook"], dtype=np.float32))
    np.save(
        patches_path,
        np.asarray(codebook_bundle["visual_word_patches"], dtype=np.uint8),
    )
    if codebook_bundle.get("train_raw_counts") is not None:
        np.save(
            train_counts_path,
            np.asarray(codebook_bundle["train_raw_counts"], dtype=np.int64),
        )

    meta = dict(codebook_bundle["metadata"])
    meta["model_dir"] = str(out_dir)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(desc_cfg_path, "w", encoding="utf-8") as f:
        json.dump(codebook_bundle["descriptor_config"], f, indent=2)

    vis_img = codebook_bundle.get("visualization_image")
    if isinstance(vis_img, Image.Image):
        vis_img.save(vis_path)

    return {
        "model_dir": out_dir,
        "codebook_path": codebook_path,
        "patches_path": patches_path,
        "metadata_path": metadata_path,
        "descriptor_config_path": desc_cfg_path,
        "visualization_path": vis_path,
    }


def load_codebook(
    model_path_or_dir: str | Path = DEFAULT_VISUAL_WORDS_MODEL_DIR,
) -> dict[str, Any]:
    """Load a persisted visual vocabulary codebook and its locked descriptor config."""
    target = Path(model_path_or_dir)
    if target.is_file():
        model_dir = target.parent
    else:
        model_dir = target

    codebook_path = model_dir / "codebook.npy"
    metadata_path = model_dir / "metadata.json"
    desc_cfg_path = model_dir / "descriptor_config.json"
    patches_path = model_dir / "visual_word_patches.npy"

    if not codebook_path.exists():
        raise ValueError(f"Codebook file not found: {codebook_path}")
    if not metadata_path.exists():
        raise ValueError(f"Codebook metadata file not found: {metadata_path}")

    codebook = np.asarray(np.load(codebook_path), dtype=np.float32)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if desc_cfg_path.exists():
        with open(desc_cfg_path, "r", encoding="utf-8") as f:
            descriptor_config = json.load(f)
    else:
        descriptor_config = metadata.get(
            "descriptor_config", default_descriptor_config()
        )

    if patches_path.exists():
        patches = np.asarray(np.load(patches_path), dtype=np.uint8)
    else:
        p_sz = int(metadata.get("patch_size", 28))
        patches = np.zeros((len(codebook), p_sz, p_sz), dtype=np.uint8)

    cluster_counts = np.asarray(
        metadata.get("cluster_counts", [0] * len(codebook)), dtype=np.int64
    )
    train_counts_path = model_dir / "train_raw_counts.npy"
    train_raw_counts = (
        np.asarray(np.load(train_counts_path), dtype=np.int64)
        if train_counts_path.exists()
        else None
    )

    return {
        "codebook_id": str(metadata.get("codebook_id", "")),
        "model_dir": str(model_dir),
        "codebook": codebook,
        "visual_word_patches": patches,
        "cluster_counts": cluster_counts,
        "train_raw_counts": train_raw_counts,
        "descriptor_config": descriptor_config,
        "metadata": metadata,
    }


def find_latest_codebook_dir() -> Path:
    """Locate the most recently saved codebook directory, falling back to default."""
    if DEFAULT_VISUAL_WORDS_ROOT.exists():
        candidates = list(DEFAULT_VISUAL_WORDS_ROOT.glob("**/codebook.npy"))
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            return newest.parent
    return DEFAULT_VISUAL_WORDS_MODEL_DIR


def ask_codebook_construction_parameters(parent: tk.Misc) -> dict[str, Any] | None:
    """Display the Set 7.1 Codebook Construction configuration dialog."""
    dialog = tk.Toplevel(parent)
    dialog.title("Codebook Construction (Set 7.1 — Bag of Visual Words)")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    default_ds = (
        str(DEFAULT_DATASET_DIR)
        if DEFAULT_DATASET_DIR.exists()
        else "datasets/Caltech_101_subset"
    )
    dataset_var = tk.StringVar(value=default_ds)
    output_dir_var = tk.StringVar(value=str(DEFAULT_VISUAL_WORDS_MODEL_DIR))
    test_size_var = tk.DoubleVar(value=0.25)
    max_imgs_var = tk.IntVar(value=0)

    # Descriptor config
    method_var = tk.StringVar(value="SIFT")
    max_kp_var = tk.IntVar(value=500)
    contrast_var = tk.DoubleVar(value=0.04)
    edge_var = tk.DoubleVar(value=10.0)
    sigma_var = tk.DoubleVar(value=1.6)
    grayscale_var = tk.BooleanVar(value=True)

    # K-Means config
    alg_var = tk.StringVar(value="MiniBatchKMeans")
    clusters_per_class_var = tk.IntVar(value=10)
    computed_k_var = tk.StringVar(value="30 (10 × 3 classes)")
    random_state_var = tk.IntVar(value=0)
    batch_size_var = tk.IntVar(value=1024)
    max_iter_var = tk.IntVar(value=100)
    n_init_var = tk.IntVar(value=3)

    detected_info_var = tk.StringVar(value="")

    def _refresh_dataset_info(*_args: object) -> None:
        ds_path = dataset_var.get().strip()
        try:
            info = inspect_dataset_directory(ds_path)
            n_cls = info["num_classes"]
            c_names = ", ".join(info["class_names"])
            k_val = max(1, int(clusters_per_class_var.get())) * n_cls
            computed_k_var.set(
                f"{k_val} ({clusters_per_class_var.get()} × {n_cls} classes)"
            )
            detected_info_var.set(f"Detected {n_cls} classes: {c_names}")
        except Exception as exc:
            computed_k_var.set("Auto (10 × classes)")
            detected_info_var.set(f"Dataset status: {exc}")

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    # Section A: Dataset Configuration
    ds_frame = ttk.LabelFrame(
        main,
        text="A. Dataset Configuration (Train/Test Split before K-Means)",
        padding=10,
    )
    ds_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(ds_frame, text="Dataset directory:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Entry(ds_frame, textvariable=dataset_var, width=42).grid(
        row=0, column=1, columnspan=2, padx=6, pady=3, sticky="ew"
    )

    def _browse_ds() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select Visual Words Dataset Folder"
        )
        if chosen:
            dataset_var.set(chosen)
            _refresh_dataset_info()

    ttk.Button(ds_frame, text="Browse…", command=_browse_ds).grid(
        row=0, column=3, pady=3
    )

    ttk.Label(ds_frame, textvariable=detected_info_var, foreground="#2b6cb0").grid(
        row=1, column=0, columnspan=4, sticky="w", pady=(2, 4)
    )

    ttk.Label(ds_frame, text="Test split fraction:").grid(
        row=2, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(
        ds_frame,
        textvariable=test_size_var,
        from_=0.10,
        to=0.50,
        increment=0.05,
        width=8,
    ).grid(row=2, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(ds_frame, text="Max images/class (0=all):").grid(
        row=2, column=2, sticky="w", padx=(8, 0), pady=3
    )
    ttk.Spinbox(
        ds_frame, textvariable=max_imgs_var, from_=0, to=500, increment=10, width=8
    ).grid(row=2, column=3, sticky="w", padx=6, pady=3)

    # Section B: Local Descriptor Extraction
    desc_frame = ttk.LabelFrame(
        main, text="B. Local Descriptor Extraction Configuration", padding=10
    )
    desc_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(desc_frame, text="Descriptor algorithm:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Combobox(
        desc_frame,
        textvariable=method_var,
        values=("SIFT", "ORB"),
        state="readonly",
        width=12,
    ).grid(row=0, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(desc_frame, text="Max keypoints / image:").grid(
        row=0, column=2, sticky="w", padx=(10, 0), pady=3
    )
    ttk.Spinbox(
        desc_frame, textvariable=max_kp_var, from_=50, to=5000, increment=50, width=8
    ).grid(row=0, column=3, sticky="w", padx=6, pady=3)

    ttk.Label(desc_frame, text="Contrast threshold:").grid(
        row=1, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(
        desc_frame,
        textvariable=contrast_var,
        from_=0.01,
        to=0.20,
        increment=0.01,
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(desc_frame, text="Edge threshold:").grid(
        row=1, column=2, sticky="w", padx=(10, 0), pady=3
    )
    ttk.Spinbox(
        desc_frame, textvariable=edge_var, from_=2.0, to=30.0, increment=1.0, width=8
    ).grid(row=1, column=3, sticky="w", padx=6, pady=3)

    ttk.Label(desc_frame, text="Sigma (initial):").grid(
        row=2, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(
        desc_frame, textvariable=sigma_var, from_=1.0, to=3.0, increment=0.1, width=8
    ).grid(row=2, column=1, sticky="w", padx=6, pady=3)
    ttk.Checkbutton(
        desc_frame, text="Grayscale conversion", variable=grayscale_var
    ).grid(row=2, column=2, columnspan=2, sticky="w", padx=(10, 0), pady=3)

    # Section C: K-Means Clustering Configuration
    km_frame = ttk.LabelFrame(
        main, text="C. K-Means Codebook Configuration (K = 10 × N_classes)", padding=10
    )
    km_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(km_frame, text="Clustering algorithm:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Combobox(
        km_frame,
        textvariable=alg_var,
        values=("MiniBatchKMeans", "KMeans"),
        state="readonly",
        width=16,
    ).grid(row=0, column=1, sticky="w", padx=6, pady=3)

    ttk.Label(km_frame, text="Words per class:").grid(
        row=0, column=2, sticky="w", padx=(10, 0), pady=3
    )
    ttk.Spinbox(
        km_frame,
        textvariable=clusters_per_class_var,
        from_=2,
        to=100,
        increment=5,
        width=8,
        command=_refresh_dataset_info,
    ).grid(row=0, column=3, sticky="w", padx=6, pady=3)

    ttk.Label(km_frame, text="Visual words (K):").grid(
        row=1, column=0, sticky="w", pady=3
    )
    ttk.Label(km_frame, textvariable=computed_k_var, font=("Segoe UI", 9, "bold")).grid(
        row=1, column=1, columnspan=3, sticky="w", padx=6, pady=3
    )

    ttk.Label(km_frame, text="Random state:").grid(row=2, column=0, sticky="w", pady=3)
    ttk.Spinbox(
        km_frame, textvariable=random_state_var, from_=0, to=9999, increment=1, width=8
    ).grid(row=2, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(km_frame, text="Batch size:").grid(
        row=2, column=2, sticky="w", padx=(10, 0), pady=3
    )
    ttk.Spinbox(
        km_frame, textvariable=batch_size_var, from_=64, to=8192, increment=128, width=8
    ).grid(row=2, column=3, sticky="w", padx=6, pady=3)

    ttk.Label(km_frame, text="Max iterations:").grid(
        row=3, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(
        km_frame, textvariable=max_iter_var, from_=10, to=1000, increment=10, width=8
    ).grid(row=3, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(km_frame, text="Initialisations (n_init):").grid(
        row=3, column=2, sticky="w", padx=(10, 0), pady=3
    )
    ttk.Spinbox(
        km_frame, textvariable=n_init_var, from_=1, to=20, increment=1, width=8
    ).grid(row=3, column=3, sticky="w", padx=6, pady=3)

    # Section D: Model Persistence
    out_frame = ttk.LabelFrame(main, text="D. Codebook Persistence", padding=10)
    out_frame.pack(fill="x", pady=(0, 10))

    ttk.Label(out_frame, text="Output folder:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Entry(out_frame, textvariable=output_dir_var, width=42).grid(
        row=0, column=1, padx=6, pady=3, sticky="ew"
    )

    def _browse_out() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select Codebook Save Directory"
        )
        if chosen:
            output_dir_var.set(chosen)

    ttk.Button(out_frame, text="Browse…", command=_browse_out).grid(
        row=0, column=2, pady=3
    )

    _refresh_dataset_info()

    result: dict[str, Any] | None = None

    def _on_build() -> None:
        nonlocal result
        try:
            ds_dir = dataset_var.get().strip()
            inspect_dataset_directory(ds_dir)
            result = {
                "dataset_dir": ds_dir,
                "output_dir": output_dir_var.get().strip()
                or str(DEFAULT_VISUAL_WORDS_MODEL_DIR),
                "test_size": float(test_size_var.get()),
                "max_images_per_class": int(max_imgs_var.get())
                if int(max_imgs_var.get()) > 0
                else None,
                "descriptor_config": default_descriptor_config(
                    method=method_var.get(),
                    max_keypoints=int(max_kp_var.get()),
                    contrast_threshold=float(contrast_var.get()),
                    edge_threshold=float(edge_var.get()),
                    sigma=float(sigma_var.get()),
                    grayscale=bool(grayscale_var.get()),
                ),
                "clustering_algorithm": alg_var.get(),
                "clusters_per_class": max(1, int(clusters_per_class_var.get())),
                "random_state": int(random_state_var.get()),
                "batch_size": max(32, int(batch_size_var.get())),
                "max_iter": max(10, int(max_iter_var.get())),
                "n_init": max(1, int(n_init_var.get())),
            }
            dialog.destroy()
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc), parent=dialog)

    btn_row = ttk.Frame(main)
    btn_row.pack(fill="x")
    ttk.Button(btn_row, text="Build Codebook", command=_on_build).pack(
        side="right", padx=(6, 0)
    )
    ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="right")

    dialog.wait_window()
    return result


class CodebookConstructionTool(ForensicsTool):
    """Tool 01 — Codebook Construction (Set 7.1 Bag of Visual Words)."""

    tool_id = "codebook_construction"
    title = "Codebook Construction"
    category = "Visual Vocabulary"
    description = (
        "Extract local descriptors from training images, cluster them with K-Means "
        "(K = 10 × classes) and generate visual words."
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
            parameters = ask_codebook_construction_parameters(parent)
            if parameters is None:
                return None

        dataset_dir = parameters.get("dataset_dir", DEFAULT_DATASET_DIR)
        test_size = float(parameters.get("test_size", 0.25))
        random_state = int(parameters.get("random_state", 0))
        max_imgs = parameters.get("max_images_per_class")

        dataset = load_visual_words_dataset(
            dataset_dir,
            test_size=test_size,
            random_state=random_state,
            max_images_per_class=max_imgs,
        )

        def _pump_ui(_cur: int, _tot: int, _stage: str) -> None:
            if parent is not None and hasattr(parent, "update_idletasks"):
                try:
                    parent.update_idletasks()
                except Exception:
                    pass

        bundle = build_codebook(
            dataset["train_records"],
            class_names=dataset["class_names"],
            descriptor_config=parameters.get("descriptor_config"),
            clusters_per_class=int(parameters.get("clusters_per_class", 10)),
            n_clusters=parameters.get("n_clusters"),
            clustering_algorithm=str(
                parameters.get("clustering_algorithm", "MiniBatchKMeans")
            ),
            random_state=random_state,
            batch_size=int(parameters.get("batch_size", 1024)),
            max_iter=int(parameters.get("max_iter", 100)),
            n_init=int(parameters.get("n_init", 3)),
            test_records=dataset["test_records"],
            dataset_dir=dataset["dataset_dir"],
            progress_callback=_pump_ui,
        )

        output_dir = parameters.get("output_dir", DEFAULT_VISUAL_WORDS_MODEL_DIR)
        saved_paths = save_codebook(bundle, output_dir=output_dir)
        meta = bundle["metadata"]

        details: dict[str, Any] = {
            "Classes": f"{meta['num_classes']} ({', '.join(meta['class_names'])})",
            "Visual words": meta["n_clusters"],
            "Total descriptors": meta["total_descriptors"],
            "Descriptor dimensions": meta["descriptor_dim"],
            "Descriptor algorithm": bundle["descriptor_config"]["method"],
            "Clustering algorithm": meta["clustering_algorithm"],
            "Train / Test images": (
                f"{meta['num_train_images']} / {meta['num_test_images']}"
            ),
            "Codebook ID": bundle["codebook_id"],
            "Saved model folder": str(saved_paths["model_dir"]),
            "Training time": f"{meta['training_time_s']:.2f} s",
        }

        vis_img = bundle["visualization_image"]
        ensure_document_has_image(document, vis_img)
        method_name = bundle["descriptor_config"]["method"]
        return ToolResult(
            image=vis_img,
            message=(
                f"Constructed codebook with K={meta['n_clusters']} visual words from "
                f"{meta['total_descriptors']:,} {method_name} descriptors "
                f"across {meta['num_classes']} classes. "
                f"Saved to {saved_paths['model_dir']}."
            ),
            details=details,
        )
