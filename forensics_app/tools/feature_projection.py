"""Set 8.3 — CNN Feature Projection (t-SNE and UMAP).

Loads the test-set feature vectors exported with each model by
``notebooks/train_cnn.ipynb`` (``features_test.npz``: the activations of the
``features`` layer for the 10,000 CIFAR-10 test images), optionally reduces
them with PCA, projects them to 2-D with t-SNE and UMAP, and measures how well
the classes stay separated (k-NN accuracy and silhouette score) before and
after the projection. scikit-learn and umap-learn are imported lazily.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import inspect
from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.paths import PROJECT_ROOT, resolve_path
from forensics_app.utils.background import run_with_progress

from .base import ForensicsTool, ToolResult
from .cnn_classification import (
    AXIS,
    CNN_MODELS_DIR,
    GRID,
    INK,
    INK_SECONDARY,
    SURFACE,
    CnnModelInfo,
    _figure_to_image,
    list_cnn_models,
    load_passport,
)

FEATURES_FILE = "features_test.npz"
REQUIRED_FEATURE_KEYS = ("features", "labels", "predictions")
METHODS = ("t-SNE", "UMAP")
KNN_K = 5
SILHOUETTE_MAX_SAMPLES = 5000


# ---------------------------------------------------------------------------
# Feature files
# ---------------------------------------------------------------------------


def features_path(info: CnnModelInfo) -> Path:
    """Return the path of a model's exported feature file."""
    files = info.passport.get("files", {})
    return info.folder / files.get("features", FEATURES_FILE)


def list_feature_models(root: str | Path = CNN_MODELS_DIR) -> list[CnnModelInfo]:
    """Return the exported models whose folder contains a feature file."""
    return [info for info in list_cnn_models(root) if features_path(info).is_file()]


@dataclass(frozen=True)
class FeatureSet:
    """Feature vectors of one model with their true and predicted labels.

    ``indices`` are the row numbers in the original split (CIFAR-10 test
    image numbers), kept through subsampling.
    """

    model_name: str
    display_name: str
    layer_name: str
    split: str
    class_names: list[str]
    features: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    indices: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.features.shape[1])

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.labels == self.predictions))

    def subset(self, indices: np.ndarray) -> FeatureSet:
        """Return the rows at positions ``indices``."""
        return FeatureSet(
            model_name=self.model_name,
            display_name=self.display_name,
            layer_name=self.layer_name,
            split=self.split,
            class_names=self.class_names,
            features=self.features[indices],
            labels=self.labels[indices],
            predictions=self.predictions[indices],
            indices=self.indices[indices],
        )


def load_feature_set(info: CnnModelInfo) -> FeatureSet:
    """Read and validate a model's ``features_test.npz``."""
    path = features_path(info)
    if not path.is_file():
        raise FileNotFoundError(f"No feature file for {info.display_name}: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_FEATURE_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path} is not a feature file (missing {missing})")
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        predictions = np.asarray(data["predictions"], dtype=np.int64)
        class_names = (
            [str(c) for c in data["class_names"]]
            if "class_names" in data.files
            else info.class_names
        )
        layer = str(data["layer_name"]) if "layer_name" in data.files else "?"
        split = str(data["split"]) if "split" in data.files else "test"

    if features.ndim != 2:
        raise ValueError(f"{path}: features must be 2-D, got {features.shape}")
    n = features.shape[0]
    if labels.shape != (n,) or predictions.shape != (n,):
        raise ValueError(f"{path}: labels/predictions do not match {n} rows")
    num_classes = len(class_names)
    if n and max(labels.max(), predictions.max()) >= num_classes:
        raise ValueError(f"{path}: label outside the {num_classes} classes")

    return FeatureSet(
        model_name=info.name,
        display_name=info.display_name,
        layer_name=layer,
        split=split,
        class_names=class_names,
        features=features,
        labels=labels,
        predictions=predictions,
        indices=np.arange(n, dtype=np.int64),
    )


def subsample_per_class(
    labels: np.ndarray, per_class: int | None, seed: int
) -> np.ndarray:
    """Return sorted row positions with at most ``per_class`` rows per class.

    ``None`` or a non-positive value keeps every row.
    """
    if not per_class or per_class <= 0:
        return np.arange(labels.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if rows.size > per_class:
            rows = rng.choice(rows, size=per_class, replace=False)
        chosen.append(rows)
    return np.sort(np.concatenate(chosen)).astype(np.int64)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------


def umap_available() -> bool:
    """Return whether umap-learn is installed."""
    return importlib.util.find_spec("umap") is not None


def reduce_with_pca(
    x: np.ndarray, n_components: int, seed: int
) -> tuple[np.ndarray, float]:
    """Return the PCA projection of ``x`` and its explained-variance fraction.

    ``x`` is returned unchanged (fraction 1.0) when it already has no more
    than ``n_components`` dimensions.
    """
    n_components = min(n_components, x.shape[0], x.shape[1])
    if n_components >= x.shape[1]:
        return x, 1.0
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(x).astype(np.float32)
    return reduced, float(pca.explained_variance_ratio_.sum())


def project_tsne(x: np.ndarray, *, perplexity: float, seed: int) -> np.ndarray:
    """Return the 2-D t-SNE embedding of ``x``.

    Perplexity is capped at ``(n - 1) / 3`` so small samples still run.
    """
    from sklearn.manifold import TSNE

    perplexity = max(1.0, min(float(perplexity), (x.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(x).astype(np.float32)


def _import_umap() -> Any:
    import umap

    return umap


def project_umap(
    x: np.ndarray, *, n_neighbors: int, min_dist: float, seed: int
) -> np.ndarray:
    """Return the 2-D UMAP embedding of ``x``.

    ``n_neighbors`` is capped at ``n - 1``.
    """
    umap = _import_umap()
    n_neighbors = max(2, min(int(n_neighbors), x.shape[0] - 1))
    options: dict[str, Any] = {
        "n_components": 2,
        "n_neighbors": n_neighbors,
        "min_dist": float(min_dist),
        "random_state": seed,
        "n_jobs": 1,
    }
    if "compatibility_layout" in inspect.signature(umap.UMAP).parameters:
        options["compatibility_layout"] = True  # umap-learn 0.6: keep 0.5 layout
    reducer = umap.UMAP(**options)
    return np.asarray(reducer.fit_transform(x), dtype=np.float32)


# ---------------------------------------------------------------------------
# Cluster quality
# ---------------------------------------------------------------------------


def knn_correct(x: np.ndarray, labels: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Return, per point, whether its ``k`` nearest other points vote its label.

    Majority vote with Euclidean distance; ties go to the lowest class index.
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(k, x.shape[0] - 1)
    if k < 1:
        return np.zeros(x.shape[0], dtype=bool)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)
    neighbour_labels = labels[idx[:, 1:]]
    num_classes = int(labels.max()) + 1
    votes = np.zeros((x.shape[0], num_classes), dtype=np.int64)
    np.add.at(votes, (np.arange(x.shape[0])[:, None], neighbour_labels), 1)
    return votes.argmax(axis=1) == labels


def knn_accuracy(x: np.ndarray, labels: np.ndarray, k: int = KNN_K) -> float:
    """Return the leave-one-out k-NN accuracy of ``labels`` in space ``x``."""
    if x.shape[0] < 2:
        return float("nan")
    return float(np.mean(knn_correct(x, labels, k)))


def silhouette(x: np.ndarray, labels: np.ndarray, seed: int) -> float:
    """Return the silhouette score of ``labels`` in space ``x`` (-1 to 1).

    Computed on at most ``SILHOUETTE_MAX_SAMPLES`` rows.
    """
    from sklearn.metrics import silhouette_score

    if np.unique(labels).size < 2:
        return float("nan")
    sample = SILHOUETTE_MAX_SAMPLES if x.shape[0] > SILHOUETTE_MAX_SAMPLES else None
    return float(silhouette_score(x, labels, sample_size=sample, random_state=seed))


def cluster_quality(x: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    """Return k-NN accuracy and silhouette score of ``labels`` in ``x``."""
    return {
        "knn_accuracy": knn_accuracy(x, labels),
        "silhouette": silhouette(x, labels, seed),
    }


# ---------------------------------------------------------------------------
# Projection pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionSettings:
    """User-selectable options of a projection run."""

    samples_per_class: int | None = 300
    pca_components: int | None = 50
    perplexity: float = 30.0
    n_neighbors: int = 15
    min_dist: float = 0.1
    seed: int = 42
    methods: tuple[str, ...] = METHODS


@dataclass
class ProjectionResult:
    """Embeddings and measurements of one projection run."""

    feature_set: FeatureSet
    settings: ProjectionSettings
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    quality: dict[str, dict[str, float]] = field(default_factory=dict)
    pca_variance: float | None = None
    skipped: dict[str, str] = field(default_factory=dict)


def compute_projections(
    full: FeatureSet, settings: ProjectionSettings
) -> ProjectionResult:
    """Subsample, optionally reduce with PCA, and project with each method.

    Methods whose library is missing are reported in ``skipped`` instead of
    raising. Quality is measured on the true labels in the original feature
    space, after PCA, and in each 2-D embedding.
    """
    unknown = [m for m in settings.methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown projection method(s): {unknown}")

    rows = subsample_per_class(full.labels, settings.samples_per_class, settings.seed)
    data = full.subset(rows)
    result = ProjectionResult(feature_set=data, settings=settings)
    seed = settings.seed

    start = time.perf_counter()
    result.quality["Original"] = cluster_quality(data.features, data.labels, seed)
    result.timings["Quality (original)"] = time.perf_counter() - start

    x = data.features
    if settings.pca_components:
        start = time.perf_counter()
        x, result.pca_variance = reduce_with_pca(x, settings.pca_components, seed)
        result.timings["PCA"] = time.perf_counter() - start
        result.quality["PCA"] = cluster_quality(x, data.labels, seed)

    for method in settings.methods:
        if method == "UMAP" and not umap_available():
            result.skipped[method] = "umap-learn is not installed"
            continue
        start = time.perf_counter()
        if method == "t-SNE":
            embedding = project_tsne(x, perplexity=settings.perplexity, seed=seed)
        else:
            embedding = project_umap(
                x,
                n_neighbors=settings.n_neighbors,
                min_dist=settings.min_dist,
                seed=seed,
            )
        result.timings[method] = time.perf_counter() - start
        result.embeddings[method] = embedding
        result.quality[method] = cluster_quality(embedding, data.labels, seed)
    return result


def projection_summary(result: ProjectionResult) -> dict[str, Any]:
    """Return a flat, printable summary of a projection run."""
    data, s = result.feature_set, result.settings
    summary: dict[str, Any] = {
        "Model": data.display_name,
        "Layer": f"{data.layer_name} ({data.dimension}-D)",
        "Points": f"{data.num_samples} ({data.split} split)",
        "Classifier accuracy on points": f"{data.accuracy:.3f}",
    }
    if result.pca_variance is not None:
        summary["PCA"] = (
            f"{s.pca_components} components, {result.pca_variance:.1%} variance kept"
        )
    for space, q in result.quality.items():
        summary[f"{space} — {KNN_K}-NN acc / silhouette"] = (
            f"{q['knn_accuracy']:.3f} / {q['silhouette']:.3f}"
        )
    for name, seconds in result.timings.items():
        summary[f"Time {name}"] = f"{seconds:.2f} s"
    for method, reason in result.skipped.items():
        summary[f"{method} skipped"] = reason
    return summary


# ---------------------------------------------------------------------------
# Figures (matplotlib Figure, not pyplot: safe in a worker thread)
# ---------------------------------------------------------------------------

COLOR_MODES = {
    "true": "true class",
    "predicted": "predicted class",
    "errors": "classifier errors",
}

# CIFAR-10 colours: the 8 validated categorical hues plus two documented
# extras, assigned so frequently confused classes get well-separated colours.
CLASS_COLORS = {
    "airplane": "#eb6834",
    "automobile": "#e87ba4",
    "bird": "#6250d6",
    "cat": "#008300",
    "deer": "#86b6ef",
    "dog": "#e34948",
    "frog": "#eda100",
    "horse": "#52514e",
    "ship": "#2a78d6",
    "truck": "#1baf7a",
}
FALLBACK_COLORS = list(CLASS_COLORS.values())
CORRECT_COLOR = AXIS
ERROR_COLOR = "#eb6834"
HIGHLIGHT_COLOR = "#2a78d6"
CONTEXT_COLOR = GRID


def class_colors(class_names: list[str]) -> list[str]:
    """Return one colour per class, CIFAR-10 names first, then by index."""
    colors = []
    for i, name in enumerate(class_names):
        if name in CLASS_COLORS:
            colors.append(CLASS_COLORS[name])
        elif i < len(FALLBACK_COLORS):
            colors.append(FALLBACK_COLORS[i])
        else:
            colors.append(INK_SECONDARY)
    return colors


def _style_embedding_axes(ax: Any, title: str, *, fontsize: int = 11) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_title(title, loc="left", fontsize=fontsize, color=INK)


def _robust_limits(ax: Any, embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Set axis limits to the central 99% of the points plus a margin."""
    lo = np.percentile(embedding, 0.5, axis=0)
    hi = np.percentile(embedding, 99.5, axis=0)
    pad = 0.06 * np.maximum(hi - lo, 1e-6)
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    return lo, hi


def _spread_labels(
    anchors: np.ndarray, widths: np.ndarray, height: float, steps: int = 200
) -> np.ndarray:
    """Return label positions (axes fractions) moved apart until boxes do not overlap.

    Labels are pushed mostly vertically and kept inside the axes.
    """
    pos = anchors.astype(float).copy()
    for _ in range(steps):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[j] - pos[i]
                min_dx = (widths[i] + widths[j]) / 2
                if abs(d[0]) >= min_dx or abs(d[1]) >= height:
                    continue
                push = (height - abs(d[1])) / 2 + 1e-3
                sign = 1.0 if d[1] > 0 or (d[1] == 0 and j > i) else -1.0
                pos[i, 1] -= sign * push
                pos[j, 1] += sign * push
                moved = True
        pos = np.clip(pos, [0.06, 0.04], [0.94, 0.96])
        if not moved:
            break
    return pos


def _label_clusters(
    ax: Any,
    embedding: np.ndarray,
    keys: np.ndarray,
    names: list[str],
    limits: tuple[np.ndarray, np.ndarray],
) -> None:
    from matplotlib import patheffects

    lo, hi = limits
    span = hi - lo
    present = [c for c in range(len(names)) if np.any(keys == c)]
    if not present:
        return
    anchors = np.array(
        [(np.median(embedding[keys == c], axis=0) - lo) / span for c in present]
    )
    widths = np.array([0.018 * len(names[c]) + 0.02 for c in present])
    placed = _spread_labels(anchors, widths, height=0.055)

    halo = [patheffects.withStroke(linewidth=3, foreground=SURFACE)]
    for c, anchor, pos in zip(present, anchors, placed):
        if np.hypot(*(pos - anchor)) > 0.03:
            ax.plot(
                *zip(anchor, pos),
                color=INK_SECONDARY,
                linewidth=0.8,
                transform=ax.transAxes,
            )
        ax.text(
            *pos,
            names[c],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=INK,
            ha="center",
            va="center",
            path_effects=halo,
        )


def _legend_handle(color: str, label: str) -> Any:
    from matplotlib.lines import Line2D

    return Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=7,
        markerfacecolor=color,
        markeredgecolor=color,
        label=label,
    )


def _check_color_mode(color_by: str) -> None:
    if color_by not in COLOR_MODES:
        raise ValueError(f"Unknown colour mode {color_by!r}")


def _draw_projection(
    ax: Any, result: ProjectionResult, method: str, color_by: str, title: str
) -> None:
    data = result.feature_set
    emb = result.embeddings[method]
    if color_by == "errors":
        wrong = data.labels != data.predictions
        ax.scatter(*emb[~wrong].T, s=5, c=CORRECT_COLOR, linewidths=0)
        ax.scatter(
            *emb[wrong].T, s=14, c=ERROR_COLOR, edgecolors=SURFACE, linewidths=0.6
        )
    else:
        colors = np.array(class_colors(data.class_names))
        keys = data.labels if color_by == "true" else data.predictions
        order = np.random.default_rng(0).permutation(data.num_samples)
        ax.scatter(*emb[order].T, s=6, c=colors[keys[order]], linewidths=0, alpha=0.85)
    _style_embedding_axes(ax, title)
    _label_clusters(ax, emb, data.labels, data.class_names, _robust_limits(ax, emb))


def _knn_title(result: ProjectionResult, method: str) -> str:
    knn = result.quality.get(method, {}).get("knn_accuracy", float("nan"))
    return f"{method}   ·   {KNN_K}-NN accuracy {knn:.3f}"


def _add_legend(fig: Any, handles: list[Any]) -> None:
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 5),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )


def _class_handles(class_names: list[str]) -> list[Any]:
    return [
        _legend_handle(c, n) for c, n in zip(class_colors(class_names), class_names)
    ]


def render_projection_figure(
    result: ProjectionResult, color_by: str = "true"
) -> Image.Image:
    """Render the 2-D embeddings side by side, coloured by ``color_by``.

    ``color_by`` is ``"true"``, ``"predicted"`` or ``"errors"`` (misclassified
    points highlighted). Class names are placed at each true-class median.
    """
    _check_color_mode(color_by)
    methods = list(result.embeddings)
    if not methods:
        raise ValueError("No projection was computed")
    from matplotlib.figure import Figure

    data = result.feature_set
    fig = Figure(figsize=(6.2 * len(methods), 6.8))
    axes = fig.subplots(1, len(methods), squeeze=False)[0]
    for ax, method in zip(axes, methods):
        _draw_projection(ax, result, method, color_by, _knn_title(result, method))

    if color_by == "errors":
        wrong = data.labels != data.predictions
        handles = [
            _legend_handle(CORRECT_COLOR, f"correct ({int((~wrong).sum())})"),
            _legend_handle(ERROR_COLOR, f"misclassified ({int(wrong.sum())})"),
        ]
    else:
        handles = _class_handles(data.class_names)
    _add_legend(fig, handles)
    fig.suptitle(
        f"{data.display_name}: {data.layer_name} layer ({data.dimension}-D), "
        f"{data.num_samples} {data.split} images, coloured by "
        f"{COLOR_MODES[color_by]}",
        x=0.02,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.12, wspace=0.06)
    return _figure_to_image(fig)


def _short_name(result: ProjectionResult) -> str:
    return result.feature_set.display_name.split(" (")[0]


def render_projection_comparison(
    results: list[ProjectionResult], color_by: str = "true"
) -> Image.Image:
    """Render one row per model and one column per method, with one legend.

    Every model must have the same methods. Row labels give the model, its
    feature dimension and classifier accuracy on the projected points.
    """
    _check_color_mode(color_by)
    if not results:
        raise ValueError("No projection was computed")
    methods = list(results[0].embeddings)
    if not methods or any(list(r.embeddings) != methods for r in results):
        raise ValueError("Every model needs the same projections to compare")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.2 * len(methods), 6.0 * len(results) + 1.0))
    axes = fig.subplots(len(results), len(methods), squeeze=False)
    for r, result in enumerate(results):
        data = result.feature_set
        for c, method in enumerate(methods):
            _draw_projection(
                axes[r, c], result, method, color_by, _knn_title(result, method)
            )
        wrong = int(np.sum(data.labels != data.predictions))
        row_label = (
            f"{_short_name(result)}\n{data.dimension}-D features · "
            f"classifier accuracy {data.accuracy:.1%}"
        )
        if color_by == "errors":
            row_label += f" · {wrong} misclassified"
        axes[r, 0].set_ylabel(row_label, fontsize=11, color=INK, labelpad=8)

    if color_by == "errors":
        handles = [
            _legend_handle(CORRECT_COLOR, "correct"),
            _legend_handle(ERROR_COLOR, "misclassified"),
        ]
    else:
        handles = _class_handles(results[0].feature_set.class_names)
    _add_legend(fig, handles)
    n = results[0].feature_set.num_samples
    fig.suptitle(
        f"{' vs '.join(_short_name(r) for r in results)}: "
        f"{' and '.join(methods)} of the {results[0].feature_set.layer_name} layer, "
        f"{n} {results[0].feature_set.split} images, coloured by "
        f"{COLOR_MODES[color_by]}",
        x=0.02,
        ha="left",
        fontsize=13,
        color=INK,
    )
    height = fig.get_figheight()
    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=1 - 0.75 / height,
        bottom=0.75 / height,
        wspace=0.06,
        hspace=0.12,
    )
    return _figure_to_image(fig)


def _class_accuracies(
    result: ProjectionResult, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-class k-NN accuracy in 2-D and in the original space."""
    data = result.feature_set
    correct_2d = knn_correct(result.embeddings[method], data.labels)
    correct_nd = knn_correct(data.features, data.labels)
    acc_2d, acc_nd = [], []
    for c in range(len(data.class_names)):
        mask = data.labels == c
        acc_2d.append(correct_2d[mask].mean() if mask.any() else np.nan)
        acc_nd.append(correct_nd[mask].mean() if mask.any() else np.nan)
    return np.array(acc_2d), np.array(acc_nd)


def _draw_class_panel(
    ax: Any, embedding: np.ndarray, mask: np.ndarray, title: str, fontsize: int
) -> None:
    ax.scatter(*embedding[~mask].T, s=2, c=CONTEXT_COLOR, linewidths=0)
    ax.scatter(*embedding[mask].T, s=5, c=HIGHLIGHT_COLOR, linewidths=0)
    _style_embedding_axes(ax, title, fontsize=fontsize)
    _robust_limits(ax, embedding)


def _check_method(results: list[ProjectionResult], method: str) -> None:
    for result in results:
        if method not in result.embeddings:
            raise ValueError(f"No {method} projection was computed")


def render_class_panels(result: ProjectionResult, method: str) -> Image.Image:
    """Render one small panel per class with that class highlighted.

    Each panel title gives the class's k-NN accuracy in 2-D and in the
    original feature space.
    """
    _check_method([result], method)
    from matplotlib.figure import Figure

    data = result.feature_set
    emb = result.embeddings[method]
    acc_2d, acc_nd = _class_accuracies(result, method)
    names = data.class_names
    cols = min(5, len(names))
    rows = -(-len(names) // cols)

    fig = Figure(figsize=(2.7 * cols, 2.9 * rows + 0.5))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    for c, ax in enumerate(axes):
        if c >= len(names):
            ax.set_visible(False)
            continue
        _draw_class_panel(
            ax,
            emb,
            data.labels == c,
            f"{names[c]}\n2-D {acc_2d[c]:.2f}  ·  {data.dimension}-D {acc_nd[c]:.2f}",
            10,
        )
    fig.suptitle(
        f"{method} of {data.display_name}: one class per panel; "
        f"{KNN_K}-NN accuracy of the class in 2-D and in the original space",
        x=0.02,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.86 if rows > 1 else 0.78,
        bottom=0.02,
        hspace=0.35,
        wspace=0.08,
    )
    return _figure_to_image(fig)


def render_class_panel_rows(
    results: list[ProjectionResult], method: str
) -> Image.Image:
    """Render one row per model and one column per class, class highlighted.

    Panel titles give the class's k-NN accuracy in 2-D and in the original
    feature space.
    """
    _check_method(results, method)
    from matplotlib.figure import Figure

    names = results[0].feature_set.class_names
    cols = len(names)
    fig = Figure(figsize=(1.24 * cols, 1.75 * len(results) + 0.45))
    axes = fig.subplots(len(results), cols, squeeze=False)
    for r, result in enumerate(results):
        data = result.feature_set
        emb = result.embeddings[method]
        acc_2d, acc_nd = _class_accuracies(result, method)
        for c in range(cols):
            _draw_class_panel(
                axes[r, c],
                emb,
                data.labels == c,
                f"{names[c]}\n{acc_2d[c]:.2f} · {acc_nd[c]:.2f}",
                8,
            )
        axes[r, 0].set_ylabel(
            data.display_name.split(" (")[0], fontsize=8, color=INK_SECONDARY
        )
    fig.suptitle(
        f"{method}, one class per panel; {KNN_K}-NN accuracy of the class "
        "in 2-D · in the original space",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.subplots_adjust(
        left=0.03, right=0.99, top=0.84, bottom=0.01, hspace=0.45, wspace=0.06
    )
    return _figure_to_image(fig)


def stack_images(images: list[Image.Image]) -> Image.Image:
    """Stack images vertically on the figure surface, centred."""
    if len(images) == 1:
        return images[0]
    width = max(img.width for img in images)
    canvas = Image.new("RGB", (width, sum(img.height for img in images)), SURFACE)
    y = 0
    for img in images:
        canvas.paste(img, ((width - img.width) // 2, y))
        y += img.height
    return canvas


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def feature_file_summary(info: CnnModelInfo) -> str:
    """Return a one-line description of a model's exported features."""
    fe = info.passport.get("feature_extraction", {})
    acc = info.passport.get("evaluation", {}).get("test_accuracy")
    text = (
        f"{fe.get('layer', '?')} layer, {fe.get('dimension', '?')}-D, "
        f"{fe.get('num_samples', '?')} {fe.get('split', 'test')} images"
    )
    if acc is not None:
        text += f"  ·  test accuracy {acc:.1%}"
    return text


def ask_feature_projection_parameters(parent: tk.Misc) -> dict[str, Any] | None:
    """Display the feature file, projection and display options dialog."""
    models = list_feature_models()
    if not models:
        messagebox.showinfo(
            "No feature files",
            f"No exported models with {FEATURES_FILE} were found in\n"
            f"{CNN_MODELS_DIR}\n\nTrain the CNN first.",
            parent=parent,
        )
        return None
    has_umap = umap_available()
    defaults = ProjectionSettings()

    dialog = tk.Toplevel(parent)
    dialog.title("CNN Feature Projection")
    dialog.transient(parent)
    dialog.grab_set()

    model_var = tk.StringVar(value=models[0].display_name)
    info_var = tk.StringVar()
    compare_var = tk.BooleanVar(value=False)
    per_class_var = tk.IntVar(value=defaults.samples_per_class or 0)
    seed_var = tk.IntVar(value=defaults.seed)
    pca_var = tk.BooleanVar(value=True)
    pca_dims_var = tk.IntVar(value=defaults.pca_components or 50)
    tsne_var = tk.BooleanVar(value=True)
    perplexity_var = tk.DoubleVar(value=defaults.perplexity)
    umap_var = tk.BooleanVar(value=has_umap)
    neighbors_var = tk.IntVar(value=defaults.n_neighbors)
    min_dist_var = tk.DoubleVar(value=defaults.min_dist)
    color_var = tk.StringVar(value="true")
    panels_var = tk.StringVar(value="none")
    save_var = tk.BooleanVar(value=False)

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    def spin_row(box: ttk.Frame, row: int, label: str, var: tk.Variable, **kw) -> None:
        ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(box, textvariable=var, width=8, **kw).grid(
            row=row, column=1, sticky="w", padx=6, pady=2
        )

    # A. Feature file -----------------------------------------------------------
    model_box = ttk.LabelFrame(main, text="A. Feature file", padding=10)
    model_box.pack(fill="x", pady=(0, 8))
    row = ttk.Frame(model_box)
    row.pack(fill="x")
    ttk.Label(row, text="Model:").pack(side="left")
    combo = ttk.Combobox(
        row,
        textvariable=model_var,
        values=[m.display_name for m in models],
        state="readonly",
        width=52,
    )
    combo.pack(side="left", padx=6)
    ttk.Label(model_box, textvariable=info_var, foreground=INK_SECONDARY).pack(
        anchor="w", pady=(6, 0)
    )
    if len(models) > 1:
        ttk.Checkbutton(
            model_box,
            text=f"Compare all {len(models)} models (same settings, one figure each)",
            variable=compare_var,
        ).pack(anchor="w", pady=(4, 0))

    # B. Sampling and PCA -------------------------------------------------------
    sample_box = ttk.LabelFrame(main, text="B. Sampling and PCA", padding=10)
    sample_box.pack(fill="x", pady=(0, 8))
    spin_row(
        sample_box,
        0,
        "Images per class (0 = all):",
        per_class_var,
        from_=0,
        to=1000,
        increment=50,
    )
    spin_row(sample_box, 1, "Random seed:", seed_var, from_=0, to=9999)
    ttk.Checkbutton(
        sample_box, text="Reduce with PCA first, components:", variable=pca_var
    ).grid(row=2, column=0, sticky="w", pady=2)
    ttk.Spinbox(sample_box, textvariable=pca_dims_var, from_=2, to=512, width=8).grid(
        row=2, column=1, sticky="w", padx=6, pady=2
    )

    # C. t-SNE ------------------------------------------------------------------
    tsne_box = ttk.LabelFrame(main, text="C. t-SNE", padding=10)
    tsne_box.pack(fill="x", pady=(0, 8))
    ttk.Checkbutton(tsne_box, text="Compute t-SNE", variable=tsne_var).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    spin_row(tsne_box, 1, "Perplexity:", perplexity_var, from_=5, to=100, increment=5)

    # D. UMAP -------------------------------------------------------------------
    umap_box = ttk.LabelFrame(main, text="D. UMAP", padding=10)
    umap_box.pack(fill="x", pady=(0, 8))
    umap_check = ttk.Checkbutton(umap_box, text="Compute UMAP", variable=umap_var)
    umap_check.grid(row=0, column=0, columnspan=2, sticky="w")
    if not has_umap:
        umap_check.state(["disabled"])
        ttk.Label(
            umap_box, text="umap-learn is not installed", foreground="#b42318"
        ).grid(row=0, column=2, sticky="w", padx=6)
    spin_row(umap_box, 1, "Neighbours:", neighbors_var, from_=2, to=200)
    spin_row(
        umap_box,
        2,
        "Minimum distance:",
        min_dist_var,
        from_=0.0,
        to=0.99,
        increment=0.05,
    )

    # E. Display ----------------------------------------------------------------
    display_box = ttk.LabelFrame(main, text="E. Display", padding=10)
    display_box.pack(fill="x", pady=(0, 8))
    ttk.Label(display_box, text="Colour points by:").grid(row=0, column=0, sticky="w")
    for col, (value, label) in enumerate(COLOR_MODES.items(), 1):
        ttk.Radiobutton(display_box, text=label, value=value, variable=color_var).grid(
            row=0, column=col, sticky="w", padx=(6, 0)
        )
    ttk.Label(display_box, text="Per-class panels:").grid(
        row=1, column=0, sticky="w", pady=(6, 0)
    )
    ttk.Combobox(
        display_box,
        textvariable=panels_var,
        values=["none", *METHODS],
        state="readonly",
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
    ttk.Checkbutton(
        display_box,
        text="Also save every figure (full size) and the 2-D coordinates to "
        "outputs/set8_3_projection",
        variable=save_var,
    ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

    # Buttons -------------------------------------------------------------------
    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(4, 0))
    result: dict[str, Any] | None = None

    def refresh(_event: object = None) -> None:
        info_var.set(feature_file_summary(models[max(0, combo.current())]))

    def finish() -> None:
        nonlocal result
        methods = [
            m
            for m, var in (("t-SNE", tsne_var), ("UMAP", umap_var))
            if var.get() and (m != "UMAP" or has_umap)
        ]
        if not methods:
            messagebox.showerror(
                "No method", "Select t-SNE, UMAP or both.", parent=dialog
            )
            return
        try:
            result = {
                "model_dirs": [
                    str(m.folder)
                    for m in (
                        models
                        if compare_var.get()
                        else [models[max(0, combo.current())]]
                    )
                ],
                "samples_per_class": int(per_class_var.get()),
                "seed": int(seed_var.get()),
                "pca_components": int(pca_dims_var.get()) if pca_var.get() else None,
                "methods": methods,
                "perplexity": float(perplexity_var.get()),
                "n_neighbors": int(neighbors_var.get()),
                "min_dist": float(min_dist_var.get()),
                "color_by": color_var.get(),
                "class_panels": None
                if panels_var.get() == "none"
                else panels_var.get(),
                "save_dir": str(EXPORT_DIR) if save_var.get() else None,
            }
        except (tk.TclError, ValueError):
            messagebox.showerror(
                "Invalid value", "Enter numbers in every field.", parent=dialog
            )
            return
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Project", command=finish).pack(side="right", padx=(0, 8))

    combo.bind("<<ComboboxSelected>>", refresh)
    refresh()
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def settings_from_parameters(parameters: dict[str, Any]) -> ProjectionSettings:
    """Build and validate projection settings from dialog parameters."""
    methods = tuple(parameters.get("methods", METHODS))
    if not methods:
        raise ValueError("Select t-SNE, UMAP or both.")
    per_class = int(parameters.get("samples_per_class", 300))
    pca = parameters.get("pca_components", 50)
    perplexity = float(parameters.get("perplexity", 30.0))
    n_neighbors = int(parameters.get("n_neighbors", 15))
    min_dist = float(parameters.get("min_dist", 0.1))
    if per_class < 0:
        raise ValueError("Images per class must be 0 (all) or positive.")
    if pca is not None and int(pca) < 2:
        raise ValueError("PCA needs at least 2 components.")
    if perplexity <= 0:
        raise ValueError("Perplexity must be positive.")
    if n_neighbors < 2:
        raise ValueError("UMAP needs at least 2 neighbours.")
    if not 0.0 <= min_dist < 1.0:
        raise ValueError("UMAP minimum distance must be in [0, 1).")
    return ProjectionSettings(
        samples_per_class=per_class or None,
        pca_components=int(pca) if pca is not None else None,
        perplexity=perplexity,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        seed=int(parameters.get("seed", 42)),
        methods=methods,
    )


def _projection_job(
    infos: list[CnnModelInfo],
    settings: ProjectionSettings,
    color_by: str,
    class_panels: str | None,
    save_dir: str | Path | None = None,
) -> tuple[list[ProjectionResult], Image.Image]:
    """Load, project, render and optionally export; runs in a worker thread."""
    results: list[ProjectionResult] = []
    figures: list[Image.Image] = []
    for info in infos:
        result = compute_projections(load_feature_set(info), settings)
        if not result.embeddings:
            raise RuntimeError(
                "No projection was computed: "
                + "; ".join(f"{m} {why}" for m, why in result.skipped.items())
                + "."
            )
        results.append(result)
        if save_dir:
            export_projection_outputs(result, save_dir)
    if len(results) == 1:
        figures.append(render_projection_figure(results[0], color_by))
    else:
        figures.append(render_projection_comparison(results, color_by))
        if save_dir:
            export_comparison_outputs(results, save_dir)
    if class_panels and all(class_panels in r.embeddings for r in results):
        if len(results) == 1:
            figures.append(render_class_panels(results[0], class_panels))
        else:
            figures.append(render_class_panel_rows(results, class_panels))
    return results, stack_images(figures)


def comparison_summary(results: list[ProjectionResult]) -> dict[str, Any]:
    """Merge projection summaries into ``|``-separated rows."""
    summaries = [projection_summary(r) for r in results]
    if len(summaries) == 1:
        return summaries[0]
    keys: list[str] = []
    for summary in summaries:
        keys.extend(k for k in summary if k not in keys)
    return {k: " | ".join(str(s.get(k, "—")) for s in summaries) for k in keys}


def _result_message(result: ProjectionResult) -> str:
    scores = ", ".join(
        f"{method} {KNN_K}-NN {result.quality[method]['knn_accuracy']:.3f}"
        for method in result.embeddings
    )
    data = result.feature_set
    text = f"{data.display_name}: {data.num_samples} points ({scores})"
    if result.skipped:
        text += "; " + ", ".join(f"{m} skipped" for m in result.skipped)
    return text


class FeatureProjectionTool(ForensicsTool):
    """Project exported CNN feature vectors to 2-D with t-SNE and UMAP."""

    tool_id = "feature_projection"
    title = "CNN Feature Projection"
    category = "Deep Learning"
    description = (
        "Project the test-set feature vectors exported with each CNN to 2-D "
        "with t-SNE and UMAP and measure how well the classes separate."
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

        settings = settings_from_parameters(parameters)
        color_by = parameters.get("color_by", "true")
        if color_by not in COLOR_MODES:
            raise ValueError(f"Unknown colour mode {color_by!r}")
        infos = [load_passport(resolve_path(d)) for d in parameters["model_dirs"]]
        if not infos:
            raise ValueError("Select a model.")

        results, figure = run_with_progress(
            parent,
            "CNN feature projection",
            f"Projecting the features of {len(infos)} model(s) with "
            f"{' and '.join(settings.methods)}…",
            _projection_job,
            infos,
            settings,
            color_by,
            parameters.get("class_panels"),
            parameters.get("save_dir"),
        )
        details = comparison_summary(results)
        message = (
            "Feature projection — "
            + "; ".join(_result_message(r) for r in results)
            + "."
        )
        if parameters.get("save_dir"):
            details["Saved to"] = str(parameters["save_dir"])
            message += f" Figures and coordinates saved to {parameters['save_dir']}."
        return ToolResult(message=message, image=figure, details=details)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

EXPORT_DIR = PROJECT_ROOT / "outputs" / "set8_3_projection"
COMPARISON_FOLDER = "comparison"


def save_projection(result: ProjectionResult, path: str | Path) -> Path:
    """Save a projection's 2-D coordinates, labels and settings to ``.npz``.

    Keys: ``indices`` (test image numbers), ``labels``, ``predictions``,
    ``class_names``, one ``(N, 2)`` array per method (``tsne``, ``umap``),
    and ``settings`` / ``quality`` as JSON strings.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = result.feature_set
    arrays = {
        "indices": data.indices,
        "labels": data.labels,
        "predictions": data.predictions,
        "class_names": np.array(data.class_names),
        "model_name": np.array(data.model_name),
        "layer_name": np.array(data.layer_name),
        "settings": np.array(json.dumps(asdict(result.settings))),
        "quality": np.array(json.dumps(result.quality)),
    }
    for method, embedding in result.embeddings.items():
        arrays[_method_key(method)] = embedding
    np.savez_compressed(path, **arrays)
    return path


def load_projection(path: str | Path) -> dict[str, Any]:
    """Load a file written by :func:`save_projection`."""
    import json

    with np.load(path, allow_pickle=False) as data:
        out: dict[str, Any] = {k: data[k] for k in data.files}
    out["class_names"] = [str(c) for c in out["class_names"]]
    for key in ("model_name", "layer_name"):
        out[key] = str(out[key])
    for key in ("settings", "quality"):
        out[key] = json.loads(str(out[key]))
    return out


def _method_key(method: str) -> str:
    return method.lower().replace("-", "")


def export_projection_outputs(
    result: ProjectionResult, out_dir: str | Path = EXPORT_DIR
) -> list[Path]:
    """Write every figure of a projection at full size, plus its ``.npz``.

    One figure per colour mode, per-class panels per method, and the
    coordinates file, in ``<out_dir>/<model_name>/`` and prefixed with the
    model name.
    """
    name = result.feature_set.model_name
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [save_projection(result, out_dir / f"{name}_projection.npz")]
    for mode in COLOR_MODES:
        path = out_dir / f"{name}_{mode}.png"
        render_projection_figure(result, mode).save(path)
        paths.append(path)
    for method in result.embeddings:
        path = out_dir / f"{name}_panels_{_method_key(method)}.png"
        render_class_panels(result, method).save(path)
        paths.append(path)
    return paths


def export_comparison_outputs(
    results: list[ProjectionResult], out_dir: str | Path = EXPORT_DIR
) -> list[Path]:
    """Write model-comparison figures to ``<out_dir>/comparison/``."""
    out_dir = Path(out_dir) / COMPARISON_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode in COLOR_MODES:
        path = out_dir / f"comparison_{mode}.png"
        render_projection_comparison(results, mode).save(path)
        paths.append(path)
    for method in results[0].embeddings:
        path = out_dir / f"comparison_panels_{_method_key(method)}.png"
        render_class_panel_rows(results, method).save(path)
        paths.append(path)
    return paths


def export_all(
    settings: ProjectionSettings | None = None,
    out_dir: str | Path = EXPORT_DIR,
    root: str | Path = CNN_MODELS_DIR,
) -> list[Path]:
    """Project every exported model and write all figures and coordinates."""
    settings = settings or ProjectionSettings()
    paths: list[Path] = []
    results: list[ProjectionResult] = []
    for info in list_feature_models(root):
        result = compute_projections(load_feature_set(info), settings)
        results.append(result)
        written = export_projection_outputs(result, out_dir)
        paths.extend(written)
        print(f"{info.display_name}: {len(written)} files")
    if len(results) > 1:
        written = export_comparison_outputs(results, out_dir)
        paths.extend(written)
        print(f"Model comparison: {len(written)} files")
    print(f"Saved in {Path(out_dir)}")
    return paths
