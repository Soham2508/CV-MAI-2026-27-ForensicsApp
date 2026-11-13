"""Set 8.2 — CNN Image Classification (pretrained Keras models).

Lists the models exported by ``notebooks/train_cnn.ipynb`` under
``models/cnn/<model_name>/``, displays their model passport (dataset, sample
sizes, class names, preprocessing, training history, test evaluation), compares
models side by side, and classifies an image using the preprocessing recorded
in the passport. TensorFlow is imported only when a model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.paths import MODELS_DIR, resolve_path
from forensics_app.utils.background import run_with_progress

from .base import ForensicsTool, ToolResult
from .hog_person_detection import IMAGE_TYPES

CNN_MODELS_DIR = MODELS_DIR / "cnn"
SAMPLE_IMAGES_DIR = CNN_MODELS_DIR / "sample_images"
PASSPORT_FILE = "passport.json"
REQUIRED_PASSPORT_KEYS = ("model_name", "class_names", "preprocessing")

# Passport "scaling" values written by the training notebook.
SCALINGS = {
    "0_1": lambda x: x / 255.0,  # [0, 255] -> [0, 1]
    "-1_1": lambda x: x / 127.5 - 1.0,  # [0, 255] -> [-1, 1]
}


# ---------------------------------------------------------------------------
# Passport discovery and parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CnnModelInfo:
    """Exported model folder and its parsed passport."""

    folder: Path
    passport: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.passport["model_name"])

    @property
    def display_name(self) -> str:
        return str(self.passport.get("display_name") or self.name)

    @property
    def class_names(self) -> list[str]:
        return [str(c) for c in self.passport["class_names"]]

    @property
    def model_path(self) -> Path:
        files = self.passport.get("files", {})
        return self.folder / files.get("model", "model.keras")

    @property
    def has_weights(self) -> bool:
        return self.model_path.is_file()


def load_passport(folder: str | Path) -> CnnModelInfo:
    """Read and validate ``<folder>/passport.json``."""
    folder = Path(folder)
    path = folder / PASSPORT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"No {PASSPORT_FILE} in {folder}")
    with path.open(encoding="utf-8") as fh:
        passport = json.load(fh)

    missing = [key for key in REQUIRED_PASSPORT_KEYS if key not in passport]
    if missing:
        raise ValueError(f"{path} is not a model passport (missing {missing})")
    scaling = passport["preprocessing"].get("scaling")
    if scaling not in SCALINGS:
        raise ValueError(f"{path}: unknown pixel scaling {scaling!r}")
    return CnnModelInfo(folder=folder, passport=passport)


def list_cnn_models(root: str | Path = CNN_MODELS_DIR) -> list[CnnModelInfo]:
    """Return every sub-folder of ``root`` with a valid passport.

    Folders without a passport (e.g. ``sample_images``) and invalid passports
    are skipped.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    models: list[CnnModelInfo] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            models.append(load_passport(folder))
        except (FileNotFoundError, ValueError):
            continue
    return models


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def resize_bilinear(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Float bilinear resize of an (H, W, C) array.

    Same formula as ``tf.image.resize(method="bilinear")``: half-pixel centres,
    clamped edges, no antialiasing and no rounding.
    """
    in_h, in_w = x.shape[:2]

    def coords(out_size: int, in_size: int):
        src = (np.arange(out_size, dtype=np.float32) + 0.5) * (in_size / out_size)
        src = np.clip(src - 0.5, 0, in_size - 1)
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, in_size - 1)
        return lo, hi, (src - lo).astype(np.float32)

    y0, y1, wy = coords(out_h, in_h)
    x0, x1, wx = coords(out_w, in_w)
    wx = wx[None, :, None]
    top = x[y0][:, x0] * (1 - wx) + x[y0][:, x1] * wx
    bot = x[y1][:, x0] * (1 - wx) + x[y1][:, x1] * wx
    wy = wy[:, None, None]
    return top * (1 - wy) + bot * wy


def preprocess_image(
    image: Image.Image,
    passport: dict[str, Any],
    *,
    match_training_resolution: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Return the ``(1, h, w, 3)`` float32 model input and the steps applied.

    RGB conversion, bilinear resize to the passport input size and pixel
    scaling. With ``match_training_resolution``, images larger than the
    dataset's original size are first downscaled to it with a box filter.
    """
    prep = passport["preprocessing"]
    in_h, in_w = (int(v) for v in prep["input_size"])
    rgb = image.convert("RGB")
    steps = [f"Loaded {image.width}x{image.height} image, converted to RGB"]

    native = passport.get("dataset", {}).get("original_image_size")
    if match_training_resolution and native:
        nat_h, nat_w = (int(v) for v in native)
        if rgb.width > nat_w or rgb.height > nat_h:
            rgb = rgb.resize((nat_w, nat_h), Image.Resampling.BOX)
            steps.append(
                f"Anti-aliased downscale (box filter) to the dataset's "
                f"{nat_w}x{nat_h} resolution"
            )

    x = np.asarray(rgb, dtype=np.float32)
    if x.shape[:2] != (in_h, in_w):
        x = resize_bilinear(x, in_h, in_w)
        steps.append(f"Float bilinear resize to the model input {in_w}x{in_h}")

    x = SCALINGS[prep["scaling"]](x).astype(np.float32)
    steps.append(f"Pixel scaling: {prep.get('formula', prep['scaling'])}")
    steps.append(f"Batch shape: (1, {in_h}, {in_w}, 3)")
    return x[None, ...], steps


def batch_to_image(batch: np.ndarray, scaling: str) -> Image.Image:
    """Convert a preprocessed batch back to an RGB image."""
    x = batch[0]
    x = x * 255.0 if scaling == "0_1" else (x + 1.0) * 127.5
    return Image.fromarray(np.clip(np.rint(x), 0, 255).astype(np.uint8), "RGB")


def true_label_from_filename(
    path: str | Path | None, class_names: list[str]
) -> str | None:
    """Return the class encoded in a ``<class>_<index>.png`` file name, if any."""
    if not path:
        return None
    prefix = Path(path).stem.rsplit("_", 1)[0].lower()
    return prefix if prefix in class_names else None


# ---------------------------------------------------------------------------
# Model loading and prediction
# ---------------------------------------------------------------------------


class MissingWeightsError(FileNotFoundError):
    """Raised when a passport has no ``model.keras`` next to it."""


def missing_weights_message(info: CnnModelInfo) -> str:
    return (
        f"The weights file for '{info.display_name}' was not found:\n"
        f"{info.model_path}\n\n"
        "Train the CNN first. The passport can still be viewed."
    )


# Loaded model, keyed by (path, modification time).
_MODEL_CACHE: dict[tuple[str, int], Any] = {}


def load_keras_model(info: CnnModelInfo) -> Any:
    """Load ``model.keras`` (cached until the file changes)."""
    if not info.has_weights:
        raise MissingWeightsError(missing_weights_message(info))
    path = info.model_path.resolve()
    key = (str(path), path.stat().st_mtime_ns)
    model = _MODEL_CACHE.get(key)
    if model is None:
        try:
            import keras
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow/Keras is not installed. Install it with:\n"
                "    pip install tensorflow"
            ) from exc
        model = keras.models.load_model(path, compile=False)
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = model
    return model


def predict_probabilities(model: Any, batch: np.ndarray) -> np.ndarray:
    """Return softmax probabilities ``(N, C)`` in inference mode."""
    return np.asarray(model(batch, training=False), dtype=np.float32)


def classify_image(
    info: CnnModelInfo,
    image: Image.Image,
    *,
    model: Any | None = None,
    top_k: int = 5,
    match_training_resolution: bool = True,
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    """Preprocess an image, predict and rank the classes.

    ``model`` overrides the loaded Keras model (used by tests).
    """
    started = time.perf_counter()
    if model is None:
        model = load_keras_model(info)
    batch, steps = preprocess_image(
        image, info.passport, match_training_resolution=match_training_resolution
    )
    probs = predict_probabilities(model, batch)[0]

    class_names = info.class_names
    if probs.shape != (len(class_names),):
        raise ValueError(
            f"Model output has shape {probs.shape}, but the passport lists "
            f"{len(class_names)} classes."
        )

    order = np.argsort(probs)[::-1]
    k = max(1, min(int(top_k), len(class_names)))
    top = [(class_names[i], float(probs[i])) for i in order[:k]]
    return {
        "model_name": info.name,
        "display_name": info.display_name,
        "predicted_index": int(order[0]),
        "predicted_name": top[0][0],
        "confidence": top[0][1],
        "top_k": top,
        "probabilities": probs,
        "true_name": true_label_from_filename(image_path, class_names),
        "steps": steps,
        "input_image": image.convert("RGB"),
        "model_view": batch_to_image(batch, info.passport["preprocessing"]["scaling"]),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


# ---------------------------------------------------------------------------
# Passport text
# ---------------------------------------------------------------------------


def _fmt_int(value: Any) -> str:
    return f"{int(value):,}" if isinstance(value, (int, float)) else str(value)


def passport_summary_text(info: CnnModelInfo) -> str:
    """Return the passport as readable plain text."""
    p = info.passport
    arch = p.get("architecture", {})
    data = p.get("dataset", {})
    prep = p.get("preprocessing", {})
    train = p.get("training", {})
    ev = p.get("evaluation", {})
    params = arch.get("parameters", {})
    sizes = data.get("sample_sizes", {})

    lines = [info.display_name, "=" * len(info.display_name)]
    if p.get("description"):
        lines += [str(p["description"])]
    lines += ["", "MODEL"]
    lines += [
        f"  {key}: {arch[key]}" for key in arch if key not in ("parameters", "summary")
    ]
    lines += [
        f"  parameters: {_fmt_int(params.get('total', '?'))} total, "
        f"{_fmt_int(params.get('trainable', '?'))} trainable, "
        f"{_fmt_int(params.get('non_trainable', '?'))} non-trainable",
        f"  weights file: {info.model_path.name} "
        f"({'found' if info.has_weights else 'MISSING'})",
        "",
        "DATASET",
        f"  name: {data.get('name', '?')}",
        f"  source: {data.get('source', '?')}",
        f"  original image size: {data.get('original_image_size', '?')}",
        "  sample sizes: "
        + ", ".join(f"{split} {_fmt_int(n)}" for split, n in sizes.items()),
        f"  validation: {data.get('validation_split', '?')}",
        f"  classes ({len(info.class_names)}): {', '.join(info.class_names)}",
        "",
        "PREPROCESSING (applied to every image before prediction)",
    ]
    lines += [f"  {i}. {step}" for i, step in enumerate(prep.get("steps", []), 1)]
    lines += ["", "TRAINING"]
    for key in ("optimizer", "learning_rate", "loss", "batch_size", "max_epochs"):
        if key in train:
            lines.append(f"  {key}: {train[key]}")
    lines.append(f"  epochs run: {train.get('epochs_run', '?')}")
    if "best_epoch" in train:
        lines.append(f"  best epoch (weights kept): {train['best_epoch']}")
    for phase in train.get("phases", []):
        lines.append(
            f"  phase '{phase.get('name')}': {phase.get('epochs')} epochs, "
            f"lr {phase.get('learning_rate')}, "
            f"{_fmt_int(phase.get('parameters', {}).get('trainable', '?'))} "
            "trainable params"
        )
    lines += [f"  augmentation: {a}" for a in train.get("augmentation", [])]
    lines += [f"  callback: {c}" for c in train.get("callbacks", [])]
    lines += [
        "",
        f"EVALUATION ({ev.get('split', 'test')} set, "
        f"{_fmt_int(ev.get('num_samples', '?'))} images)",
    ]
    if "test_accuracy" in ev:
        lines.append(f"  accuracy: {ev['test_accuracy']:.2%}")
    if "test_loss" in ev:
        lines.append(f"  loss: {ev['test_loss']:.4f}")
    for name, acc in ev.get("per_class_accuracy", {}).items():
        lines.append(f"  {name:<12} {acc:.1%}")
    env = p.get("environment", {})
    if env:
        lines += [
            "",
            "ENVIRONMENT",
            "  " + ", ".join(f"{k} {v}" for k, v in env.items()),
        ]
    if p.get("created_utc"):
        lines.append(f"  created (UTC): {p['created_utc']}")
    if arch.get("summary"):
        lines += ["", "LAYER SUMMARY", str(arch["summary"])]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures (matplotlib Figure, not pyplot: safe in a worker thread)
# ---------------------------------------------------------------------------

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
BAR_COLOR = "#2a78d6"
SEQUENTIAL_BLUES = [
    SURFACE,
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]


def _style_axes(ax: Any, *, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figure_to_image(fig: Any) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=SURFACE, bbox_inches="tight")
    buf.seek(0)
    with Image.open(buf) as img:
        return img.convert("RGB")


def _plot_history(ax: Any, train: dict[str, Any], metric: str, title: str) -> None:
    history = train.get("history", {})
    values = history.get(metric, [])
    epochs = np.arange(1, len(values) + 1)
    if len(values):
        ax.plot(epochs, values, color=TRAIN_COLOR, linewidth=2, label="train")
    val = history.get(f"val_{metric}", [])
    if len(val):
        ax.plot(
            np.arange(1, len(val) + 1),
            val,
            color=VAL_COLOR,
            linewidth=2,
            label="validation",
        )
    for boundary in train.get("phase_boundaries", []):
        ax.axvline(boundary + 0.5, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.text(
            boundary + 0.7,
            0.97,
            "fine-tuning",
            transform=ax.get_xaxis_transform(),
            color=INK_MUTED,
            fontsize=8,
            va="top",
        )
    if metric == "accuracy" and len(val):
        best = int(train.get("best_epoch") or (int(np.argmax(val)) + 1))
        ax.plot(
            [best],
            [val[best - 1]],
            "o",
            markersize=8,
            color=VAL_COLOR,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.text(
            0.98,
            0.04,
            f"\u25cf weights kept: epoch {best} (val {val[best - 1]:.1%})",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color=INK_SECONDARY,
        )
    _style_axes(ax)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("epoch", fontsize=9, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)


def render_passport_figure(info: CnnModelInfo) -> Image.Image:
    """Render training curves, confusion matrix and per-class accuracy."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    p = info.passport
    train = p.get("training", {})
    ev = p.get("evaluation", {})
    names = info.class_names

    fig = Figure(figsize=(13, 10), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.35, wspace=0.28)
    acc = ev.get("test_accuracy")
    fig.suptitle(
        info.display_name + (f" — test accuracy {acc:.1%}" if acc is not None else ""),
        x=0.06,
        ha="left",
        fontsize=14,
        color=INK,
    )

    _plot_history(fig.add_subplot(grid[0, 0]), train, "loss", "Loss")
    _plot_history(fig.add_subplot(grid[0, 1]), train, "accuracy", "Accuracy")

    # Confusion matrix coloured by row share (diagonal = per-class recall).
    ax_cm = fig.add_subplot(grid[1, 0])
    cm = np.asarray(ev.get("confusion_matrix", []), dtype=np.float64)
    if cm.shape == (len(names), len(names)):
        rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cmap = LinearSegmentedColormap.from_list("blues", SEQUENTIAL_BLUES)
        ax_cm.imshow(rates, cmap=cmap, vmin=0, vmax=1)
        for r in range(len(names)):
            for c in range(len(names)):
                if r == c or rates[r, c] >= 0.05:
                    ax_cm.text(
                        c,
                        r,
                        f"{int(cm[r, c])}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="#ffffff" if rates[r, c] > 0.45 else INK,
                    )
        ax_cm.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax_cm.set_yticks(range(len(names)), names)
        ax_cm.set_xlabel("predicted class", fontsize=9, color=INK_SECONDARY)
        ax_cm.set_ylabel("true class", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax_cm, grid_axis="")
        for spine in ax_cm.spines.values():
            spine.set_visible(False)
    else:
        ax_cm.axis("off")
    ax_cm.set_title(
        "Confusion matrix (test counts; colour = row share)",
        loc="left",
        fontsize=11,
        color=INK,
    )

    # Per-class accuracy in class order, aligned with the confusion matrix rows.
    ax_pc = fig.add_subplot(grid[1, 1])
    per_class = ev.get("per_class_accuracy", {})
    values = [float(per_class.get(n, 0.0)) for n in names]
    ypos = np.arange(len(names))
    ax_pc.barh(ypos, values, height=0.6, color=BAR_COLOR)
    for y, v in zip(ypos, values):
        ax_pc.text(
            v + 0.01, y, f"{v:.1%}", va="center", fontsize=8.5, color=INK_SECONDARY
        )
    if acc is not None:
        ax_pc.axvline(acc, color=INK_MUTED, linestyle="--", linewidth=1)
        ax_pc.text(
            acc, -0.9, f"overall {acc:.1%}", ha="center", fontsize=8, color=INK_MUTED
        )
    ax_pc.set_yticks(ypos, names)
    ax_pc.invert_yaxis()
    ax_pc.set_xlim(0, 1.1)
    ax_pc.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    _style_axes(ax_pc, grid_axis="x")
    ax_pc.set_title("Per-class test accuracy", loc="left", fontsize=11, color=INK)
    return _figure_to_image(fig)


def render_model_comparison(models: list[CnnModelInfo]) -> Image.Image:
    """Render the passport figures of several models side by side."""
    figures = [render_passport_figure(info) for info in models]
    gap = 24
    width = sum(f.width for f in figures) + gap * (len(figures) - 1)
    height = max(f.height for f in figures)
    canvas = Image.new("RGB", (width, height), SURFACE)
    x = 0
    for figure in figures:
        canvas.paste(figure, (x, 0))
        x += figure.width + gap
    return canvas


def comparison_details(models: list[CnnModelInfo]) -> dict[str, Any]:
    """Return comparison rows with one value per model, separated by ' | '."""

    def row(get: Any) -> str:
        return " | ".join(str(get(m)) for m in models)

    def ev(m: CnnModelInfo) -> dict[str, Any]:
        return m.passport.get("evaluation", {})

    details: dict[str, Any] = {
        "Models": row(lambda m: m.name),
        "Test accuracy": row(lambda m: f"{ev(m).get('test_accuracy', 0):.2%}"),
        "Test loss": row(lambda m: f"{ev(m).get('test_loss', 0):.3f}"),
        "Parameters": row(
            lambda m: _fmt_int(
                m.passport.get("architecture", {})
                .get("parameters", {})
                .get("total", "?")
            )
        ),
        "Input size": row(lambda m: m.passport["preprocessing"].get("input_size")),
        "Pixel scaling": row(lambda m: m.passport["preprocessing"].get("scaling")),
        "Train images": row(
            lambda m: _fmt_int(
                m.passport.get("dataset", {}).get("sample_sizes", {}).get("train", "?")
            )
        ),
        "Epochs run": row(
            lambda m: m.passport.get("training", {}).get("epochs_run", "?")
        ),
        "Weights": row(lambda m: "found" if m.has_weights else "missing"),
    }
    for name in models[0].class_names:
        details[f"  {name}"] = row(
            lambda m, n=name: f"{ev(m).get('per_class_accuracy', {}).get(n, 0):.1%}"
        )
    return details


def render_classification_figure(result: dict[str, Any]) -> Image.Image:
    """Render the input, the model input and the top-k probabilities."""
    from matplotlib.figure import Figure

    top = result["top_k"]
    true_name = result.get("true_name")
    fig = Figure(figsize=(13, 4.6), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.25)

    ax_in = fig.add_subplot(grid[0, 0])
    img = result["input_image"]
    ax_in.imshow(np.asarray(img))
    ax_in.set_title(f"Input ({img.width}x{img.height})", fontsize=10, color=INK)
    ax_view = fig.add_subplot(grid[0, 1])
    view = result["model_view"]
    ax_view.imshow(np.asarray(view), interpolation="nearest")
    ax_view.set_title(
        f"What the CNN sees ({view.width}x{view.height})", fontsize=10, color=INK
    )
    for ax in (ax_in, ax_view):
        ax.set_xticks(())
        ax.set_yticks(())
        for spine in ax.spines.values():
            spine.set_color(AXIS)

    ax_bar = fig.add_subplot(grid[0, 2])
    labels = [name + ("  (true)" if name == true_name else "") for name, _ in top]
    probs = [p for _, p in top]
    ypos = np.arange(len(top))
    ax_bar.barh(ypos, probs, height=0.6, color=BAR_COLOR)
    for y, v in zip(ypos, probs):
        ax_bar.text(
            v + 0.01, y, f"{v:.1%}", va="center", fontsize=9, color=INK_SECONDARY
        )
    ax_bar.set_yticks(ypos, labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    _style_axes(ax_bar, grid_axis="x")
    ax_bar.set_title(
        f"Top-{len(top)} class probabilities (softmax)",
        loc="left",
        fontsize=10,
        color=INK,
    )

    title = (
        f"{result['display_name']}: {result['predicted_name']} "
        f"({result['confidence']:.1%})"
    )
    if true_name:
        verdict = "correct" if true_name == result["predicted_name"] else "wrong"
        title += f"   |   true class: {true_name} ({verdict})"
    fig.suptitle(title, x=0.06, y=1.0, ha="left", fontsize=13, color=INK)
    return _figure_to_image(fig)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def ask_cnn_classification_parameters(
    parent: tk.Misc, document: ImageDocument
) -> dict[str, Any] | None:
    """Display the model selection, passport and classification dialog."""
    models = list_cnn_models()
    if not models:
        messagebox.showinfo(
            "No CNN models",
            f"No exported models (folders with {PASSPORT_FILE}) were found in\n"
            f"{CNN_MODELS_DIR}\n\nTrain and export one with "
            "notebooks/train_cnn.ipynb.",
            parent=parent,
        )
        return None

    dialog = tk.Toplevel(parent)
    dialog.title("CNN Image Classification")
    dialog.transient(parent)
    dialog.grab_set()

    model_var = tk.StringVar(value=models[0].display_name)
    status_var = tk.StringVar()
    use_current_var = tk.BooleanVar(value=document.is_loaded)
    path_var = tk.StringVar(value=str(document.path) if document.path else "")
    top_k_var = tk.IntVar(value=5)
    match_res_var = tk.BooleanVar(value=True)

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    # A. Model + passport ------------------------------------------------------
    model_box = ttk.LabelFrame(main, text="A. Model and passport", padding=10)
    model_box.pack(fill="both", expand=True, pady=(0, 8))
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
    status_label = ttk.Label(row, textvariable=status_var)
    status_label.pack(side="left", padx=6)

    text_frame = ttk.Frame(model_box)
    text_frame.pack(fill="both", expand=True, pady=(8, 0))
    text = tk.Text(text_frame, width=96, height=22, wrap="none", font="TkFixedFont")
    yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
    xscroll = ttk.Scrollbar(text_frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    text.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    text_frame.rowconfigure(0, weight=1)
    text_frame.columnconfigure(0, weight=1)

    # B. Image to classify -----------------------------------------------------
    image_box = ttk.LabelFrame(main, text="B. Image to classify", padding=10)
    image_box.pack(fill="x", pady=(0, 8))
    if document.is_loaded:
        ttk.Checkbutton(
            image_box, text="Use the image open in the app", variable=use_current_var
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
    ttk.Label(image_box, text="Or image file:").grid(row=1, column=0, sticky="w")
    ttk.Entry(image_box, textvariable=path_var, width=60).grid(
        row=1, column=1, padx=6, sticky="ew"
    )

    def browse() -> None:
        start = SAMPLE_IMAGES_DIR if SAMPLE_IMAGES_DIR.is_dir() else Path.cwd()
        chosen = filedialog.askopenfilename(
            parent=dialog,
            title="Select an image to classify",
            initialdir=str(start),
            filetypes=IMAGE_TYPES,
        )
        if chosen:
            path_var.set(chosen)
            use_current_var.set(False)

    ttk.Button(image_box, text="Browse…", command=browse).grid(row=1, column=2)
    image_box.columnconfigure(1, weight=1)

    # C. Options ---------------------------------------------------------------
    opts = ttk.LabelFrame(main, text="C. Options", padding=10)
    opts.pack(fill="x", pady=(0, 8))
    ttk.Label(opts, text="Show top-k classes:").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(opts, textvariable=top_k_var, from_=1, to=10, width=6).grid(
        row=0, column=1, sticky="w", padx=6
    )
    ttk.Checkbutton(
        opts,
        text="Match training resolution: anti-aliased downscale of larger images "
        "to the dataset's image size first (recommended)",
        variable=match_res_var,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

    # Buttons -------------------------------------------------------------------
    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(4, 0))
    result: dict[str, Any] | None = None

    def selected() -> CnnModelInfo:
        return models[max(0, combo.current())]

    def refresh(_event: object = None) -> None:
        info = selected()
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", passport_summary_text(info))
        text.configure(state="disabled")
        if info.has_weights:
            status_var.set("weights found")
            status_label.configure(foreground="#006300")
            classify_btn.state(["!disabled"])
        else:
            status_var.set("weights missing: train the CNN first")
            status_label.configure(foreground="#b42318")
            classify_btn.state(["disabled"])

    def finish(action: str) -> None:
        nonlocal result
        info = selected()
        use_current = bool(use_current_var.get()) and document.is_loaded
        image_path = path_var.get().strip()
        if action == "classify" and not use_current:
            if not image_path or not resolve_path(image_path).is_file():
                messagebox.showerror(
                    "No image",
                    "Open an image in the app or choose an image file.",
                    parent=dialog,
                )
                return
        try:
            top_k = int(top_k_var.get())
        except (tk.TclError, ValueError):
            top_k = 5
        result = {
            "action": action,
            "model_dir": str(info.folder),
            "use_current_image": use_current,
            "image_path": image_path or None,
            "top_k": max(1, min(10, top_k)),
            "match_training_resolution": bool(match_res_var.get()),
        }
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    classify_btn = ttk.Button(
        buttons, text="Classify image", command=lambda: finish("classify")
    )
    classify_btn.pack(side="right", padx=(0, 8))
    ttk.Button(
        buttons,
        text="Show passport figures",
        command=lambda: finish("passport"),
    ).pack(side="right", padx=(0, 8))
    if len(models) > 1:
        ttk.Button(
            buttons,
            text=f"Compare all {len(models)} models",
            command=lambda: finish("compare"),
        ).pack(side="right", padx=(0, 8))

    combo.bind("<<ComboboxSelected>>", refresh)
    refresh()
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _classify_job(
    info: CnnModelInfo,
    image: Image.Image,
    model: Any | None,
    top_k: int,
    match_training_resolution: bool,
    image_path: str | Path | None,
) -> tuple[dict[str, Any], Image.Image]:
    """Classify and render; runs in a worker thread."""
    result = classify_image(
        info,
        image,
        model=model,
        top_k=top_k,
        match_training_resolution=match_training_resolution,
        image_path=image_path,
    )
    return result, render_classification_figure(result)


class CnnClassificationTool(ForensicsTool):
    """Load a trained CNN, show its passport and classify an image."""

    tool_id = "cnn_classification"
    title = "CNN Classification"
    category = "Deep Learning"
    description = (
        "Load a CNN trained in notebooks/train_cnn.ipynb, inspect its model "
        "passport and classify an image (CIFAR-10 classes)."
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
            parameters = ask_cnn_classification_parameters(parent, document)
            if parameters is None:
                return None

        if parameters.get("action") == "compare":
            models = list_cnn_models()
            if len(models) < 2:
                raise ValueError("At least two exported models are needed.")
            figure = run_with_progress(
                parent,
                "Compare models",
                f"Drawing the passports of {len(models)} models…",
                render_model_comparison,
                models,
            )
            return ToolResult(
                message="Compared models: "
                + ", ".join(m.display_name for m in models)
                + ".",
                image=figure,
                details=comparison_details(models),
            )

        info = load_passport(resolve_path(parameters["model_dir"]))
        if parameters.get("action", "classify") == "passport":
            figure = run_with_progress(
                parent,
                "Model passport",
                f"Drawing the passport of {info.display_name}…",
                render_passport_figure,
                info,
            )
            return ToolResult(
                message=f"Model passport: {info.display_name}.",
                image=figure,
                details=passport_details(info),
            )

        if not info.has_weights and parameters.get("model") is None:
            raise MissingWeightsError(missing_weights_message(info))

        image_path: str | Path | None = parameters.get("image_path")
        if parameters.get("image") is not None:
            image = parameters["image"]
        elif parameters.get("use_current_image") and document.current is not None:
            image = document.current
            image_path = document.path
        else:
            path = resolve_path(str(image_path))
            with Image.open(path) as img:
                image = img.copy()

        first_load = parameters.get("model") is None and not _is_cached(info)
        result, figure = run_with_progress(
            parent,
            "CNN classification",
            (f"Loading {info.display_name} (first use only)…\n" if first_load else "")
            + "Classifying the image…",
            _classify_job,
            info,
            image,
            parameters.get("model"),
            int(parameters.get("top_k", 5)),
            bool(parameters.get("match_training_resolution", True)),
            image_path,
        )

        details: dict[str, Any] = {
            "Model": info.display_name,
            "Prediction": f"{result['predicted_name']} ({result['confidence']:.1%})",
        }
        if result["true_name"]:
            correct = result["true_name"] == result["predicted_name"]
            details["True class (filename)"] = (
                f"{result['true_name']} ({'correct' if correct else 'wrong'})"
            )
        for rank, (name, prob) in enumerate(result["top_k"], 1):
            details[f"Top {rank}"] = f"{name}: {prob:.2%}"
        for i, step in enumerate(result["steps"], 1):
            details[f"Preprocessing {i}"] = step
        details["Time"] = f"{result['elapsed_ms']:.0f} ms"
        return ToolResult(
            message=(
                f"{info.display_name}: {result['predicted_name']} "
                f"({result['confidence']:.1%})."
            ),
            image=figure,
            details=details,
        )


def _is_cached(info: CnnModelInfo) -> bool:
    if not info.has_weights:
        return False
    path = info.model_path.resolve()
    return (str(path), path.stat().st_mtime_ns) in _MODEL_CACHE


def passport_details(info: CnnModelInfo) -> dict[str, Any]:
    """Return key passport facts for the results table."""
    p = info.passport
    data = p.get("dataset", {})
    prep = p.get("preprocessing", {})
    ev = p.get("evaluation", {})
    params = p.get("architecture", {}).get("parameters", {})
    details: dict[str, Any] = {
        "Model": info.display_name,
        "Weights": "found" if info.has_weights else "missing (see passport text)",
        "Dataset": data.get("name", "?"),
        "Samples": ", ".join(
            f"{k} {_fmt_int(v)}" for k, v in data.get("sample_sizes", {}).items()
        ),
        "Classes": ", ".join(info.class_names),
        "Input": f"{prep.get('input_size')} RGB, {prep.get('formula', '')}",
        "Parameters": _fmt_int(params.get("total", "?")),
    }
    if "test_accuracy" in ev:
        details["Test accuracy"] = f"{ev['test_accuracy']:.2%}"
    for name, acc in ev.get("per_class_accuracy", {}).items():
        details[f"  {name}"] = f"{acc:.1%}"
    return details
