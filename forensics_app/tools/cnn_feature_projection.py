"""Set 8.3 — CNN Feature Extraction & 2D Feature Projection (t-SNE & UMAP).

Extracts deep feature vectors from a trained CNN or ImageNet pretrained backbone
(e.g. ResNet50 ``avgpool``, InceptionV3, or VGG dense features) on the held-out
test set, saves them to ``features_test.npz`` + ``features_meta.json`` for easy
reloading, projects the high-dimensional vectors to 2D via t-SNE and/or UMAP,
and renders the resulting cluster visualization.
"""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import io
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib import colormaps
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import torch
from torch.utils.data import DataLoader, Dataset
import umap

from forensics_app.core import ImageDocument
from forensics_app.paths import MODELS_DIR, resolve_path
from forensics_app.utils.background import run_with_progress

from .base import ForensicsTool, ToolResult
from .cnn_common import (
    BatchPreprocessor,
    CIFAR10IndexedDataset,
    DEFAULT_CIFAR10_DIR,
    DEFAULT_CNN_ROOT,
    find_latest_cnn_model_dir,
    get_cifar10_test_cache,
    list_saved_cnn_models,
    load_cnn_bundle,
    load_imagenet_pretrained_bundle,
    load_test_record_image,
    prepare_cnn_dataset,
    preprocess_pil_image,
)
from .codebook_construction import (
    DEFAULT_DATASET_DIR,
    ensure_document_has_image,
    portable_path,
    resolve_record_path,
)

DEFAULT_FEATURES_DIR = MODELS_DIR / "cnn" / "extracted_features"


class _PassportRecordsDataset(Dataset):
    """Dataset reading held-out test records specified inside a model passport."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        passport: dict[str, Any],
    ) -> None:
        self.records = list(records)
        self.passport = passport
        self.preprocessing = passport["preprocessing"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rec = self.records[index]
        img = load_test_record_image(rec, self.passport)
        tensor = preprocess_pil_image(img, self.preprocessing)
        return tensor, int(rec["class_id"])


def _extract_batch_features(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """Extract penultimate/pooled features from custom or torchvision CNNs."""
    if hasattr(model, "extract_features"):
        if (
            layer_name == "avgpool"
            and model.__class__.__name__ == "InceptionV3Classifier"
        ):
            return model.extract_features(inputs, layer_name="avgpool")
        return model.extract_features(inputs)

    # Off-the-shelf torchvision ResNet50 / InceptionV3: capture avgpool output cleanly.
    captured: list[torch.Tensor] = []
    target_layer = getattr(model, "avgpool", None)
    if target_layer is None:
        raise ValueError(
            f"Model {model.__class__.__name__} does not expose "
            "'extract_features' or 'avgpool'."
        )
    handle = target_layer.register_forward_hook(
        lambda _mod, _inp, out: captured.append(torch.flatten(out, 1))
    )
    try:
        _ = model(inputs)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Forward hook failed to capture features from avgpool.")
    return captured[0]


def extract_and_save_test_features(
    *,
    model_source: str = "saved",
    model_dir: str | Path = DEFAULT_CNN_ROOT / "cnn_model_001",
    imagenet_arch: str = "resnet50",
    dataset_source: str = "image_folder",
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    max_samples_per_class: int = 0,
    layer_name: str = "auto",
    batch_size: int = 64,
    device_preference: str = "auto",
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Extract CNN feature vectors on the test set and persist them to disk."""
    t0 = time.perf_counter()
    if model_source == "imagenet":
        bundle = load_imagenet_pretrained_bundle(
            imagenet_arch, device_preference=device_preference
        )
        passport = bundle["passport"]
        prep = passport["preprocessing"]
        cap = int(max_samples_per_class) if int(max_samples_per_class) > 0 else None
        ds_bundle = prepare_cnn_dataset(
            source_type=dataset_source,
            dataset_dir=dataset_dir,
            test_size=0.25,
            val_split=0.0,
            random_state=seed,
            max_samples_per_class=cap,
            preprocessing=prep,
        )
        test_ds = ds_bundle["test_dataset"]
        class_names = list(ds_bundle["class_names"])
        records = list(ds_bundle["test_records"])
        dataset_name = ds_bundle["dataset_name"]
        save_root = resolve_path(output_dir or DEFAULT_FEATURES_DIR)
    else:
        bundle = load_cnn_bundle(model_dir, device_preference=device_preference)
        passport = bundle["passport"]
        records = list(passport.get("dataset", {}).get("test_records", []))
        if not records:
            raise ValueError(
                "Saved model passport has no held-out test records "
                "for feature extraction."
            )
        if max_samples_per_class and int(max_samples_per_class) > 0:
            cap = int(max_samples_per_class)
            per_cls: dict[int, int] = {}
            filtered: list[dict[str, Any]] = []
            for rec in records:
                cid = int(rec["class_id"])
                if per_cls.get(cid, 0) < cap:
                    filtered.append(rec)
                    per_cls[cid] = per_cls.get(cid, 0) + 1
            records = filtered
        if all("cifar_index" in rec for rec in records):
            ds_dir = passport.get("dataset", {}).get(
                "dataset_dir", DEFAULT_CIFAR10_DIR
            )
            data, targets = get_cifar10_test_cache(ds_dir)
            test_ds = CIFAR10IndexedDataset(
                data,
                targets,
                [int(r["cifar_index"]) for r in records],
                passport["preprocessing"],
            )
        else:
            test_ds = _PassportRecordsDataset(records, passport)
        class_names = list(passport["class_names"])
        dataset_name = str(passport.get("dataset", {}).get("dataset_name", "test"))
        save_root = resolve_path(output_dir or bundle["model_dir"])

    if len(test_ds) == 0:
        raise ValueError("Test set is empty; cannot extract feature vectors.")

    arch = passport.get("architecture", {})
    chosen_layer = (
        str(arch.get("feature_layer", "avgpool"))
        if layer_name in ("", "auto")
        else str(layer_name)
    )

    model = bundle["model"]
    device = bundle["device"]
    dev_info = bundle["device_info"]
    model.eval()

    use_cuda = device.type == "cuda"
    batch_prep: BatchPreprocessor | None = None
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if hasattr(test_ds, "enable_gpu_batch_preprocessing"):
            test_ds.enable_gpu_batch_preprocessing()
        batch_prep = BatchPreprocessor(passport["preprocessing"], device)

    input_size = int(passport.get("preprocessing", {}).get("input_size", [32, 32])[0])
    vram_mb = dev_info.get("vram_mb") or 0
    eff_batch = min(int(batch_size), len(test_ds))
    if use_cuda and input_size >= 224 and 0 < vram_mb <= 4096:
        eff_batch = min(eff_batch, 64)
    eff_batch = max(1, eff_batch)

    loader = DataLoader(
        test_ds,
        batch_size=eff_batch,
        shuffle=False,
        pin_memory=use_cuda,
    )

    feat_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    use_extract_method = hasattr(model, "extract_features")
    captured: list[torch.Tensor] = []
    hook_handle = None
    if not use_extract_method:
        target_layer = getattr(model, "avgpool", None)
        if target_layer is None:
            raise ValueError(
                f"Model {model.__class__.__name__} does not expose "
                "'extract_features' or 'avgpool'."
            )
        hook_handle = target_layer.register_forward_hook(
            lambda _mod, _inp, out: captured.append(torch.flatten(out, 1))
        )

    try:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_cuda):
            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=use_cuda)
                if batch_prep is not None:
                    inputs = batch_prep(inputs)
                if use_extract_method:
                    if (
                        chosen_layer == "avgpool"
                        and model.__class__.__name__ == "InceptionV3Classifier"
                    ):
                        feats = model.extract_features(inputs, layer_name="avgpool")
                    else:
                        feats = model.extract_features(inputs)
                else:
                    captured.clear()
                    _ = model(inputs)
                    if not captured:
                        raise RuntimeError(
                            "Forward hook failed to capture features from avgpool."
                        )
                    feats = captured[0]
                feat_chunks.append(
                    feats.detach().to(dtype=torch.float32).cpu().numpy()
                )
                label_chunks.append(targets.numpy().astype(np.int64))
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    features = np.concatenate(feat_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    filenames = np.array([str(r.get("filename", "")) for r in records])

    save_root.mkdir(parents=True, exist_ok=True)
    npz_path = save_root / "features_test.npz"
    meta_path = save_root / "features_meta.json"
    np.savez(
        npz_path,
        features=features,
        labels=labels,
        class_names=np.array(class_names),
        filenames=filenames,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    dev_label = dev_info.get("gpu_name") or dev_info.get("device")
    print(
        f"[CNNFeatureProjection] Extracted {features.shape[0]} x {features.shape[1]}-D "
        f"features ({chosen_layer}) in {elapsed_ms:.1f} ms on {dev_label} -> {npz_path}",
        flush=True,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": passport.get("model_id", ""),
        "model_name": passport.get("model_name", ""),
        "arch_key": arch.get("arch_key", ""),
        "layer_name": chosen_layer,
        "dataset_name": dataset_name,
        "split": "test",
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "class_names": class_names,
        "device_used": dev_info.get("gpu_name") or dev_info.get("device"),
        "extraction_time_ms": round(elapsed_ms, 2),
        "features_file": portable_path(npz_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "features": features,
        "labels": labels,
        "class_names": class_names,
        "filenames": filenames.tolist(),
        "metadata": metadata,
        "features_path": str(npz_path),
    }


def load_extracted_features(features_path: str | Path) -> dict[str, Any]:
    """Load previously saved test feature vectors from ``features_test.npz``."""
    p = resolve_record_path(features_path)
    if p.is_dir():
        p = p / "features_test.npz"
    if not p.is_file():
        raise FileNotFoundError(f"Extracted features file not found: {p}")

    with np.load(p, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        class_names = [str(x) for x in data["class_names"].tolist()]
        filenames = (
            [str(x) for x in data["filenames"].tolist()]
            if "filenames" in data
            else [f"sample_{i}" for i in range(len(labels))]
        )

    meta_path = p.parent / "features_meta.json"
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "model_name": p.parent.name,
            "layer_name": "features",
            "dataset_name": "test",
            "num_samples": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "class_names": class_names,
        }

    return {
        "features": features,
        "labels": labels,
        "class_names": class_names,
        "filenames": filenames,
        "metadata": metadata,
        "features_path": str(p),
    }


def project_features_2d(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "both",
    perplexity: float = 30.0,
    tsne_max_iter: int = 500,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    random_state: int = 42,
) -> dict[str, Any]:
    """Project high-dimensional feature vectors to 2D using t-SNE and/or UMAP."""
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("At least 3 feature vectors are required for 2D projection.")
    if len(X) != len(y):
        raise ValueError("features and labels must have the same number of rows.")

    n_samples = int(X.shape[0])
    mode = str(method).strip().lower()
    if mode not in ("tsne", "umap", "both"):
        raise ValueError(
            f"Unsupported projection method: {method!r}. Use 'tsne', 'umap', or 'both'."
        )

    t0 = time.perf_counter()
    results: dict[str, Any] = {}
    unique_labels = np.unique(y)
    can_silhouette = len(unique_labels) >= 2 and n_samples > len(unique_labels)
    sil_kwargs: dict[str, Any] = (
        {"sample_size": 2000, "random_state": int(random_state)}
        if n_samples > 2000
        else {}
    )

    if X.shape[1] > 50 and n_samples > 50:
        pca_dim = min(50, n_samples - 1)
        X_proj = np.asarray(
            PCA(n_components=pca_dim, random_state=int(random_state)).fit_transform(X),
            dtype=np.float32,
        )
    else:
        X_proj = X

    if mode in ("tsne", "both"):
        eff_perp = min(float(perplexity), max(2.0, float(n_samples - 1) / 3.0))
        eff_iter = max(250, int(tsne_max_iter))
        init_mode = "pca" if X_proj.shape[1] >= 2 and n_samples >= 4 else "random"
        tsne_kwargs: dict[str, Any] = {
            "n_components": 2,
            "perplexity": eff_perp,
            "max_iter": eff_iter,
            "init": init_mode,
            "learning_rate": "auto",
            "random_state": int(random_state),
        }
        if "n_jobs" in inspect.signature(TSNE.__init__).parameters:
            tsne_kwargs["n_jobs"] = -1
        tsne = TSNE(**tsne_kwargs)
        coords_tsne = tsne.fit_transform(X_proj)
        sil_tsne = (
            float(silhouette_score(coords_tsne, y, **sil_kwargs))
            if can_silhouette
            else None
        )
        results["tsne"] = {
            "coords": coords_tsne,
            "perplexity_used": round(eff_perp, 2),
            "max_iter": eff_iter,
            "silhouette": sil_tsne,
        }

    if mode in ("umap", "both"):
        eff_neighbors = min(max(2, int(umap_n_neighbors)), n_samples - 1)
        umap_kwargs: dict[str, Any] = {
            "n_components": 2,
            "n_neighbors": eff_neighbors,
            "min_dist": float(umap_min_dist),
            "metric": str(umap_metric),
            "random_state": int(random_state),
            "n_jobs": 1,
        }
        if "compatibility_layout" in inspect.signature(umap.UMAP.__init__).parameters:
            umap_kwargs["compatibility_layout"] = False
        reducer = umap.UMAP(**umap_kwargs)
        coords_umap = np.asarray(reducer.fit_transform(X_proj), dtype=np.float32)
        sil_umap = (
            float(silhouette_score(coords_umap, y, **sil_kwargs))
            if can_silhouette
            else None
        )
        results["umap"] = {
            "coords": coords_umap,
            "n_neighbors_used": eff_neighbors,
            "min_dist": float(umap_min_dist),
            "metric": str(umap_metric),
            "silhouette": sil_umap,
        }

    proj_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[CNNFeatureProjection] 2D projection ({mode}) on {n_samples} samples "
        f"({X.shape[1]}-D -> {X_proj.shape[1]}-D) finished in {proj_ms:.1f} ms",
        flush=True,
    )
    return results


def render_projection_figure(
    projections: dict[str, Any],
    labels: np.ndarray,
    class_names: list[str],
    metadata: dict[str, Any],
) -> Image.Image:
    """Render 2D t-SNE and/or UMAP scatter plots colored by class."""
    panels: list[tuple[str, str, dict[str, Any]]] = []
    if "tsne" in projections:
        panels.append(("tsne", "t-SNE", projections["tsne"]))
    if "umap" in projections:
        panels.append(("umap", "UMAP", projections["umap"]))

    n_panels = len(panels)
    fig = Figure(figsize=(6.4 * n_panels, 5.4), dpi=120)
    cmap = colormaps.get_cmap("tab10" if len(class_names) <= 10 else "tab20")

    for idx, (key, label_title, proj) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, n_panels, idx)
        coords = proj["coords"]
        for cid, cname in enumerate(class_names):
            mask = labels == cid
            if not np.any(mask):
                continue
            color = cmap(cid % cmap.N)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=24,
                alpha=0.82,
                color=color,
                label=cname,
                edgecolors="white",
                linewidths=0.3,
            )
        if key == "tsne":
            sub = f"perplexity={proj['perplexity_used']}, iter={proj['max_iter']}"
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
        else:
            sub = f"n_neighbors={proj['n_neighbors_used']}, min_dist={proj['min_dist']}"
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
        sil = proj.get("silhouette")
        sil_str = f" | sil={sil:.3f}" if sil is not None else ""
        ax.set_title(f"{label_title} ({sub}{sil_str})", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.2)
        if len(class_names) <= 15:
            ax.legend(fontsize=8, loc="best", framealpha=0.85)

    layer_txt = metadata.get("layer_name", "features")
    feat_dim = metadata.get("feature_dim", "?")
    fig.suptitle(
        f"CNN Feature Projection — {metadata.get('model_name', 'CNN')} "
        f"[{layer_txt}: {feat_dim}-D, "
        f"N={metadata.get('num_samples', len(labels))}, "
        f"Dataset={metadata.get('dataset_name', 'test')}]",
        fontsize=10.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def ask_feature_projection_parameters(parent: tk.Misc) -> dict[str, Any] | None:
    """Open the Set 8.3 Feature Extraction & 2D Projection configuration dialog."""
    dialog = tk.Toplevel(parent)
    dialog.title("Set 8.3 — CNN Feature Extraction & Projection (t-SNE / UMAP)")
    dialog.transient(parent.winfo_toplevel())
    dialog.grab_set()
    dialog.resizable(False, False)

    saved_dirs = list_saved_cnn_models()
    default_model = (
        portable_path(find_latest_cnn_model_dir())
        if saved_dirs
        else portable_path(DEFAULT_CNN_ROOT / "cnn_model_001")
    )
    default_npz = Path(default_model) / "features_test.npz"

    feature_source_var = tk.StringVar(
        value="load_saved"
        if resolve_record_path(default_npz).is_file()
        else ("extract_saved" if saved_dirs else "extract_imagenet")
    )
    model_dir_var = tk.StringVar(value=default_model)
    imagenet_arch_var = tk.StringVar(value="resnet50")
    dataset_source_var = tk.StringVar(value="image_folder")
    dataset_dir_var = tk.StringVar(value=portable_path(DEFAULT_DATASET_DIR))
    max_per_class_var = tk.IntVar(value=50)
    layer_var = tk.StringVar(value="auto")
    device_var = tk.StringVar(value="auto")
    features_file_var = tk.StringVar(value=portable_path(default_npz))

    method_var = tk.StringVar(value="both")
    perplexity_var = tk.DoubleVar(value=30.0)
    tsne_iter_var = tk.IntVar(value=500)
    umap_neighbors_var = tk.IntVar(value=15)
    umap_min_dist_var = tk.DoubleVar(value=0.1)
    umap_metric_var = tk.StringVar(value="euclidean")
    seed_var = tk.IntVar(value=42)
    status_var = tk.StringVar(value="")

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    # 1. Feature Vector Source (Extract & Save OR Load Saved)
    src_frame = ttk.LabelFrame(
        main, text="1. Test-Set Feature Vectors (Extract & Save or Load)", padding=8
    )
    src_frame.pack(fill="x", pady=(0, 8))

    def _update_status(*_args: object) -> None:
        mode = feature_source_var.get()
        if mode == "load_saved":
            fp = resolve_record_path(features_file_var.get().strip())
            if fp.is_dir():
                fp = fp / "features_test.npz"
            if fp.is_file():
                try:
                    loaded = load_extracted_features(fp)
                    m = loaded["metadata"]
                    status_var.set(
                        f"Ready to load {m['num_samples']} x {m['feature_dim']}-D "
                        f"features ({m.get('layer_name', 'features')}) from {fp.name}"
                    )
                except Exception as exc:
                    status_var.set(f"Invalid features file: {exc}")
            else:
                status_var.set("No saved features_test.npz at that path yet.")
        elif mode == "extract_saved":
            mdir = resolve_record_path(model_dir_var.get().strip())
            has_feats = (mdir / "features_test.npz").is_file()
            hint = (
                " (features_test.npz already exists — will refresh)"
                if has_feats
                else ""
            )
            status_var.set(f"Will extract test-set features from {mdir.name}{hint}.")
        else:
            status_var.set(
                f"Will extract {imagenet_arch_var.get()} avgpool features on "
                f"{dataset_source_var.get()} test set."
            )

    ttk.Radiobutton(
        src_frame,
        text="Load saved feature vectors (.npz)",
        variable=feature_source_var,
        value="load_saved",
        command=_update_status,
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

    ttk.Label(src_frame, text="Features (.npz):").grid(
        row=1, column=0, sticky="w", pady=2
    )
    ttk.Entry(src_frame, textvariable=features_file_var, width=44).grid(
        row=1, column=1, sticky="ew", padx=6, pady=2
    )

    def _browse_npz() -> None:
        chosen = filedialog.askopenfilename(
            parent=dialog,
            title="Select Saved Feature Vectors (.npz)",
            filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")],
        )
        if chosen:
            features_file_var.set(chosen)
            feature_source_var.set("load_saved")
            _update_status()

    ttk.Button(src_frame, text="Browse…", command=_browse_npz).grid(
        row=1, column=2, sticky="w", pady=2
    )

    ttk.Separator(src_frame, orient="horizontal").grid(
        row=2, column=0, columnspan=3, sticky="ew", pady=6
    )

    ttk.Radiobutton(
        src_frame,
        text="Extract from saved trained CNN & save features_test.npz",
        variable=feature_source_var,
        value="extract_saved",
        command=_update_status,
    ).grid(row=3, column=0, columnspan=3, sticky="w", pady=2)

    ttk.Label(src_frame, text="Saved CNN folder:").grid(
        row=4, column=0, sticky="w", pady=2
    )
    model_combo = ttk.Combobox(
        src_frame,
        textvariable=model_dir_var,
        values=[portable_path(p) for p in saved_dirs] or [default_model],
        width=42,
    )
    model_combo.grid(row=4, column=1, sticky="ew", padx=6, pady=2)

    def _browse_model() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select Saved CNN Model Folder"
        )
        if chosen:
            model_dir_var.set(chosen)
            features_file_var.set(portable_path(Path(chosen) / "features_test.npz"))
            feature_source_var.set("extract_saved")
            _update_status()

    ttk.Button(src_frame, text="Browse…", command=_browse_model).grid(
        row=4, column=2, sticky="w", pady=2
    )

    ttk.Separator(src_frame, orient="horizontal").grid(
        row=5, column=0, columnspan=3, sticky="ew", pady=6
    )

    ttk.Radiobutton(
        src_frame,
        text="Extract from ImageNet pretrained backbone (ResNet50 / InceptionV3)",
        variable=feature_source_var,
        value="extract_imagenet",
        command=_update_status,
    ).grid(row=6, column=0, columnspan=3, sticky="w", pady=2)

    img_row = ttk.Frame(src_frame)
    img_row.grid(row=7, column=0, columnspan=3, sticky="ew", pady=2)
    ttk.Label(img_row, text="Backbone:").pack(side="left")
    ttk.Combobox(
        img_row,
        textvariable=imagenet_arch_var,
        values=("resnet50", "inception_v3"),
        state="readonly",
        width=13,
    ).pack(side="left", padx=(4, 10))
    ttk.Label(img_row, text="Dataset:").pack(side="left")
    ttk.Combobox(
        img_row,
        textvariable=dataset_source_var,
        values=("image_folder", "cifar10"),
        state="readonly",
        width=13,
    ).pack(side="left", padx=(4, 10))
    ttk.Label(img_row, text="Max/class (0=all):").pack(side="left")
    ttk.Spinbox(
        img_row, textvariable=max_per_class_var, from_=0, to=1000, increment=10, width=6
    ).pack(side="left", padx=4)

    ttk.Label(src_frame, text="Dataset folder:").grid(
        row=8, column=0, sticky="w", pady=2
    )
    ttk.Entry(src_frame, textvariable=dataset_dir_var, width=44).grid(
        row=8, column=1, sticky="ew", padx=6, pady=2
    )

    def _browse_ds() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select Dataset Directory"
        )
        if chosen:
            dataset_dir_var.set(chosen)
            _update_status()

    ttk.Button(src_frame, text="Browse…", command=_browse_ds).grid(
        row=8, column=2, sticky="w", pady=2
    )

    extra_row = ttk.Frame(src_frame)
    extra_row.grid(row=9, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Label(extra_row, text="Feature layer:").pack(side="left")
    ttk.Combobox(
        extra_row,
        textvariable=layer_var,
        values=("auto", "avgpool", "custom_dense", "fc1"),
        state="readonly",
        width=13,
    ).pack(side="left", padx=(4, 12))
    ttk.Label(extra_row, text="Device:").pack(side="left")
    ttk.Combobox(
        extra_row,
        textvariable=device_var,
        values=("auto", "cuda", "cpu"),
        state="readonly",
        width=8,
    ).pack(side="left", padx=4)

    def _extract_now() -> None:
        try:
            mode = feature_source_var.get()
            src_kind = "imagenet" if mode == "extract_imagenet" else "saved"
            out = run_with_progress(
                dialog,
                "Extracting Features",
                "Running CNN forward pass on test images and saving features…",
                extract_and_save_test_features,
                model_source=src_kind,
                model_dir=model_dir_var.get().strip(),
                imagenet_arch=imagenet_arch_var.get(),
                dataset_source=dataset_source_var.get(),
                dataset_dir=dataset_dir_var.get().strip(),
                max_samples_per_class=int(max_per_class_var.get()),
                layer_name=layer_var.get(),
                device_preference=device_var.get(),
                seed=int(seed_var.get()),
            )
            features_file_var.set(portable_path(out["features_path"]))
            feature_source_var.set("load_saved")
            _update_status()
            m = out["metadata"]
            messagebox.showinfo(
                "Features Saved",
                f"Saved {m['num_samples']} x {m['feature_dim']}-D vectors to:\n"
                f"{out['features_path']}",
                parent=dialog,
            )
        except Exception as exc:
            messagebox.showerror("Feature Extraction Failed", str(exc), parent=dialog)

    ttk.Button(
        extra_row, text="Extract & Save Features Now", command=_extract_now
    ).pack(side="left", padx=(14, 0))

    ttk.Label(src_frame, textvariable=status_var, foreground="#2b6cb0").grid(
        row=10, column=0, columnspan=3, sticky="w", pady=(4, 0)
    )

    # 2. 2D Projection Parameters (t-SNE & UMAP)
    proj_frame = ttk.LabelFrame(
        main, text="2. 2D Projection Settings (t-SNE / UMAP)", padding=8
    )
    proj_frame.pack(fill="x", pady=(0, 10))

    ttk.Label(proj_frame, text="Method:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Combobox(
        proj_frame,
        textvariable=method_var,
        values=("both", "tsne", "umap"),
        state="readonly",
        width=10,
    ).grid(row=0, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(proj_frame, text="Random seed:").grid(row=0, column=2, sticky="e", pady=2)
    ttk.Spinbox(
        proj_frame, textvariable=seed_var, from_=0, to=9999, increment=1, width=7
    ).grid(row=0, column=3, sticky="w", padx=6, pady=2)

    ttk.Label(proj_frame, text="t-SNE perplexity:").grid(
        row=1, column=0, sticky="w", pady=2
    )
    ttk.Spinbox(
        proj_frame,
        textvariable=perplexity_var,
        from_=2.0,
        to=100.0,
        increment=5.0,
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(proj_frame, text="t-SNE max_iter:").grid(
        row=1, column=2, sticky="e", pady=2
    )
    ttk.Spinbox(
        proj_frame,
        textvariable=tsne_iter_var,
        from_=250,
        to=3000,
        increment=100,
        width=8,
    ).grid(row=1, column=3, sticky="w", padx=6, pady=2)

    ttk.Label(proj_frame, text="UMAP n_neighbors:").grid(
        row=2, column=0, sticky="w", pady=2
    )
    ttk.Spinbox(
        proj_frame,
        textvariable=umap_neighbors_var,
        from_=2,
        to=100,
        increment=5,
        width=8,
    ).grid(row=2, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(proj_frame, text="UMAP min_dist / metric:").grid(
        row=2, column=2, sticky="e", pady=2
    )
    umap_box = ttk.Frame(proj_frame)
    umap_box.grid(row=2, column=3, sticky="w", padx=6, pady=2)
    ttk.Spinbox(
        umap_box,
        textvariable=umap_min_dist_var,
        from_=0.0,
        to=0.99,
        increment=0.05,
        width=6,
    ).pack(side="left")
    ttk.Combobox(
        umap_box,
        textvariable=umap_metric_var,
        values=("euclidean", "cosine"),
        state="readonly",
        width=9,
    ).pack(side="left", padx=(4, 0))

    _update_status()

    result: dict[str, Any] | None = None

    def _on_apply() -> None:
        nonlocal result
        result = {
            "feature_source": feature_source_var.get(),
            "features_path": features_file_var.get().strip(),
            "model_dir": model_dir_var.get().strip(),
            "imagenet_arch": imagenet_arch_var.get(),
            "dataset_source": dataset_source_var.get(),
            "dataset_dir": dataset_dir_var.get().strip(),
            "max_samples_per_class": int(max_per_class_var.get()),
            "layer_name": layer_var.get(),
            "device": device_var.get(),
            "method": method_var.get(),
            "perplexity": float(perplexity_var.get()),
            "tsne_max_iter": int(tsne_iter_var.get()),
            "umap_n_neighbors": int(umap_neighbors_var.get()),
            "umap_min_dist": float(umap_min_dist_var.get()),
            "umap_metric": umap_metric_var.get(),
            "random_state": int(seed_var.get()),
        }
        dialog.destroy()

    btns = ttk.Frame(main)
    btns.pack(fill="x")
    ttk.Button(btns, text="Project & Visualize", command=_on_apply).pack(
        side="right", padx=(6, 0)
    )
    ttk.Button(btns, text="Cancel", command=dialog.destroy).pack(side="right")

    dialog.wait_window()
    return result


def run_feature_projection(parameters: dict[str, Any]) -> dict[str, Any]:
    """Load or extract feature vectors, project onto 2D (t-SNE / UMAP), and plot."""
    src = str(parameters.get("feature_source", "extract_saved"))
    if src == "load_saved":
        feat_bundle = load_extracted_features(parameters["features_path"])
    else:
        feat_bundle = extract_and_save_test_features(
            model_source="imagenet" if src == "extract_imagenet" else "saved",
            model_dir=parameters.get("model_dir", find_latest_cnn_model_dir()),
            imagenet_arch=str(parameters.get("imagenet_arch", "resnet50")),
            dataset_source=str(parameters.get("dataset_source", "image_folder")),
            dataset_dir=parameters.get("dataset_dir", DEFAULT_DATASET_DIR),
            max_samples_per_class=int(parameters.get("max_samples_per_class", 0)),
            layer_name=str(parameters.get("layer_name", "auto")),
            device_preference=str(parameters.get("device", "auto")),
            output_dir=parameters.get("output_dir"),
            seed=int(parameters.get("random_state", 42)),
        )

    projections = project_features_2d(
        feat_bundle["features"],
        feat_bundle["labels"],
        method=str(parameters.get("method", "both")),
        perplexity=float(parameters.get("perplexity", 30.0)),
        tsne_max_iter=int(parameters.get("tsne_max_iter", 500)),
        umap_n_neighbors=int(parameters.get("umap_n_neighbors", 15)),
        umap_min_dist=float(parameters.get("umap_min_dist", 0.1)),
        umap_metric=str(parameters.get("umap_metric", "euclidean")),
        random_state=int(parameters.get("random_state", 42)),
    )
    vis_img = render_projection_figure(
        projections,
        feat_bundle["labels"],
        feat_bundle["class_names"],
        feat_bundle["metadata"],
    )

    feat_path = Path(feat_bundle["features_path"])
    plot_path = feat_path.parent / "projection_2d.png"
    vis_img.save(plot_path)

    return {
        "feature_bundle": feat_bundle,
        "projections": projections,
        "visualization_image": vis_img,
        "plot_path": str(plot_path),
    }


class CNNFeatureProjectionTool(ForensicsTool):
    """Set 8.3 — Extract/load CNN test features and project via t-SNE & UMAP."""

    tool_id = "cnn_feature_projection"
    title = "CNN Feature Projection (t-SNE / UMAP)"
    category = "CNN Classification"
    description = (
        "Extract and save CNN feature vectors on the test set, load saved features, "
        "and visualize 2D projections using t-SNE and UMAP."
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
            parameters = ask_feature_projection_parameters(parent)
            if parameters is None:
                return None

        out = run_with_progress(
            parent,
            "CNN Feature Projection",
            "Loading/extracting test feature vectors and computing 2D projection…",
            run_feature_projection,
            parameters,
        )
        fb = out["feature_bundle"]
        meta = fb["metadata"]
        projs = out["projections"]
        vis_img = out["visualization_image"]
        ensure_document_has_image(document, vis_img)

        method_labels = ["t-SNE" if k == "tsne" else k.upper() for k in projs]
        details: dict[str, Any] = {
            "Model": meta.get("model_name", "—"),
            "Feature layer": meta.get("layer_name", "—"),
            "Feature shape": f"{meta.get('num_samples')} × {meta.get('feature_dim')}",
            "Dataset": meta.get("dataset_name", "—"),
            "Saved features": portable_path(fb["features_path"]),
            "Saved plot": portable_path(out["plot_path"]),
            "Methods": ", ".join(method_labels),
        }
        if "tsne" in projs:
            details["t-SNE perplexity"] = projs["tsne"]["perplexity_used"]
            if projs["tsne"].get("silhouette") is not None:
                details["t-SNE silhouette"] = f"{projs['tsne']['silhouette']:.3f}"
        if "umap" in projs:
            details["UMAP n_neighbors"] = projs["umap"]["n_neighbors_used"]
            if projs["umap"].get("silhouette") is not None:
                details["UMAP silhouette"] = f"{projs['umap']['silhouette']:.3f}"

        return ToolResult(
            image=vis_img,
            message=(
                f"Projected {meta.get('num_samples')} {meta.get('feature_dim')}-D "
                f"test feature vectors via {' & '.join(method_labels)}."
            ),
            details=details,
        )
