"""Set 10 extra — evaluate CLIP zero-shot on a labelled folder.

Each sub-folder is a class and its name is the class text (``hot_chocolate``
becomes "hot chocolate"). Every template is evaluated on its own and, with
several templates, as an ensemble, which answers the notebook's exercise
"which prompts work best?". The best scheme gets a confusion matrix,
per-class precision / recall / F1 and its worst mistakes; predictions are
saved as CSV in ``outputs/set10_clip/``.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.paths import PROJECT_ROOT
from forensics_app.utils.background import run_with_progress

from .base import ToolResult
from .clip_common import (
    CLIP_MODELS,
    DEFAULT_MODEL,
    DEVICES,
    ProgressCallback,
    embed_image_paths,
    figure_to_image,
    folder_embeddings,
    load_clip,
    loading_message,
    logit_scale,
    open_rgb,
    parse_templates,
    resolve_device,
    scan_images,
    softmax_probabilities,
    template_and_ensemble_embeddings,
    thumbnail,
)
from .clip_zero_shot import ENSEMBLE_TEMPLATES

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "set10_clip"
MAX_MISTAKES_SHOWN = 6


def class_text(folder_name: str) -> str:
    return " ".join(folder_name.replace("_", " ").replace("-", " ").split())


def _collect_dataset_with_stats(
    root: str | Path, max_per_class: int = 0
) -> tuple[list[str], list[str], list[Path], list[tuple[int, int]], np.ndarray]:
    """Return folder names, class texts, image paths, file stats and targets."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found:\n{root}")
    folders, paths, stats, targets = [], [], [], []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name.startswith("."):
            continue
        scanned = scan_images(sub, recursive=True)
        if max_per_class > 0:
            scanned = scanned[:max_per_class]
        if not scanned:
            continue
        index = len(folders)
        folders.append(sub.name)
        paths.extend(p for p, _, _ in scanned)
        stats.extend((mtime_ns, size) for _, mtime_ns, size in scanned)
        targets.extend([index] * len(scanned))
    if len(folders) < 2:
        raise ValueError(
            "Put the images in at least two sub-folders, one per class "
            "(the folder name is the class text)."
        )
    return folders, [class_text(f) for f in folders], paths, stats, np.asarray(targets)


def collect_dataset(
    root: str | Path, max_per_class: int = 0
) -> tuple[list[str], list[str], list[Path], np.ndarray]:
    """Return folder names, class texts, image paths and true class indices."""
    folders, labels, paths, _stats, targets = _collect_dataset_with_stats(
        root, max_per_class
    )
    return folders, labels, paths, targets


@dataclass(frozen=True)
class SchemeScore:
    name: str
    templates: list[str]
    probabilities: np.ndarray  # (N, C)
    accuracy: float
    top_k_accuracy: float


def score(
    name: str, templates: list[str], probs: np.ndarray, targets: np.ndarray, k: int
) -> SchemeScore:
    predictions = probs.argmax(axis=1)
    accuracy = float((predictions == targets).mean())
    if k <= 1:
        top_k_accuracy = accuracy
    elif k >= probs.shape[1]:
        top_k_accuracy = 1.0
    else:
        top_k = np.argpartition(probs, -k, axis=1)[:, -k:]
        top_k_accuracy = float((top_k == targets[:, None]).any(axis=1).mean())
    return SchemeScore(
        name=name,
        templates=list(templates),
        probabilities=probs,
        accuracy=accuracy,
        top_k_accuracy=top_k_accuracy,
    )


def confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, classes: int
) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def per_class_metrics(matrix: np.ndarray) -> list[tuple[float, float, float, int]]:
    """``(precision, recall, F1, support)`` per class (rows = true class)."""
    metrics = []
    for c in range(len(matrix)):
        tp = matrix[c, c]
        predicted = matrix[:, c].sum()
        support = int(matrix[c].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        metrics.append((float(precision), float(recall), float(f1), support))
    return metrics


@dataclass(frozen=True)
class Evaluation:
    folders: list[str]
    labels: list[str]
    paths: list[Path]
    targets: np.ndarray
    schemes: list[SchemeScore]
    best: SchemeScore
    matrix: np.ndarray
    metrics: list[tuple[float, float, float, int]]
    k: int
    elapsed_ms: float
    from_cache: bool = False


def evaluate_folder(
    model: Any,
    processor: Any,
    root: str | Path,
    templates: list[str],
    device: str = "cpu",
    *,
    max_per_class: int = 0,
    model_name: str = "",
    progress: ProgressCallback | None = None,
) -> Evaluation:
    started = time.perf_counter()
    folders, labels, paths, stats, targets = _collect_dataset_with_stats(
        root, max_per_class
    )
    if model_name:
        image_emb, from_cache = folder_embeddings(
            model,
            processor,
            paths,
            device,
            model_name=model_name,
            stats=stats,
            progress=progress,
        )
    else:
        image_emb = embed_image_paths(
            model, processor, paths, device, progress=progress
        )
        from_cache = False

    per_template, ensemble = template_and_ensemble_embeddings(
        model, processor, labels, templates, device
    )
    scale = logit_scale(model)
    k = min(5, len(labels) - 1) if len(labels) > 2 else 1

    schemes = []
    for template, text_emb in zip(templates, per_template):
        probs = softmax_probabilities((image_emb @ text_emb.T).numpy(), scale)
        schemes.append(score(f'"{template}"', [template], probs, targets, k))
    if len(templates) > 1:
        probs = softmax_probabilities((image_emb @ ensemble.T).numpy(), scale)
        schemes.append(
            score(f"ensemble ({len(templates)})", templates, probs, targets, k)
        )
    best = max(schemes, key=lambda s: (s.accuracy, s.top_k_accuracy))
    matrix = confusion_matrix(targets, best.probabilities.argmax(axis=1), len(labels))
    return Evaluation(
        folders,
        labels,
        paths,
        targets,
        schemes,
        best,
        matrix,
        per_class_metrics(matrix),
        k,
        (time.perf_counter() - started) * 1000.0,
        from_cache,
    )


def mistakes(evaluation: Evaluation) -> list[tuple[int, int, float]]:
    """``(image, predicted class, p(true class))``, most confident errors first."""
    probs = evaluation.best.probabilities
    predictions = probs.argmax(axis=1)
    wrong = np.flatnonzero(predictions != evaluation.targets)
    p_true = probs[wrong, evaluation.targets[wrong]]
    order = wrong[np.argsort(p_true, kind="stable")]
    return [
        (int(i), int(predictions[i]), float(probs[i, evaluation.targets[i]]))
        for i in order
    ]


def save_csv(evaluation: Evaluation, root: Path, out_dir: Path = OUTPUT_DIR) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = root.name or root.resolve().name or "dataset"
    folder = out_dir / f"{dataset_name}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    probs = evaluation.best.probabilities
    predictions = probs.argmax(axis=1)
    with open(folder / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["image", "true", "predicted", "p_predicted", "p_true", "correct"]
        )
        for i, path in enumerate(evaluation.paths):
            t, p = int(evaluation.targets[i]), int(predictions[i])
            writer.writerow(
                [
                    str(path.relative_to(root)),
                    evaluation.labels[t],
                    evaluation.labels[p],
                    f"{probs[i, p]:.4f}",
                    f"{probs[i, t]:.4f}",
                    int(t == p),
                ]
            )
    with open(folder / "templates.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scheme", "accuracy", f"top{evaluation.k}_accuracy"])
        for s in evaluation.schemes:
            writer.writerow([s.name, f"{s.accuracy:.4f}", f"{s.top_k_accuracy:.4f}"])
    return folder


def render_evaluation(evaluation: Evaluation, title: str = "") -> Image.Image:
    n = len(evaluation.labels)
    errors = mistakes(evaluation)[:MAX_MISTAKES_SHOWN]
    size = max(4.5, 0.45 * n + 2.5)
    fig = Figure(figsize=(size + 6.5, size + (3.0 if errors else 0.6)), dpi=100)
    rows = 2 if errors else 1
    grid = fig.add_gridspec(
        rows,
        MAX_MISTAKES_SHOWN,
        height_ratios=(size, 2.6) if errors else (1,),
    )

    ax = fig.add_subplot(grid[0, :4])
    row_sums = np.maximum(evaluation.matrix.sum(axis=1, keepdims=True), 1)
    normalised = evaluation.matrix / row_sums
    ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), evaluation.labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), evaluation.labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for (r, c), value in np.ndenumerate(evaluation.matrix):
        if value:
            ax.text(
                c,
                r,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalised[r, c] > 0.6 else "black",
            )
    ax.set_title(
        f"Confusion matrix — {evaluation.best.name}: "
        f"accuracy {evaluation.best.accuracy:.1%}",
        fontsize=10,
    )

    bar_ax = fig.add_subplot(grid[0, 4:])
    names = [s.name for s in evaluation.schemes][::-1]
    acc = [s.accuracy for s in evaluation.schemes][::-1]
    top = [s.top_k_accuracy for s in evaluation.schemes][::-1]
    y = np.arange(len(names))
    bar_ax.barh(y + 0.2, acc, height=0.4, color="#3b6ea8", label="top-1")
    bar_ax.barh(y - 0.2, top, height=0.4, color="#9ec3e6", label=f"top-{evaluation.k}")
    bar_ax.set_yticks(y, names, fontsize=8)
    bar_ax.set_xlim(0, 1.12)
    bar_ax.set_xlabel("accuracy")
    bar_ax.set_title("Accuracy per prompt template", fontsize=10)
    for yi, value in zip(y, acc):
        bar_ax.text(value + 0.01, yi + 0.2, f"{value:.2f}", va="center", fontsize=8)
    bar_ax.legend(fontsize=8, loc="lower right")

    for col, (i, predicted, p_true) in enumerate(errors):
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(thumbnail(open_rgb(evaluation.paths[i]), 200))
        ax.set_title(
            f"true: {evaluation.labels[evaluation.targets[i]]}\n"
            f"pred: {evaluation.labels[predicted]}\np(true) = {p_true:.2f}",
            fontsize=8,
            color="#b42318",
        )
        ax.axis("off")
    fig.suptitle(
        title or f"CLIP zero-shot on {len(evaluation.paths)} images, {n} classes",
        fontsize=11,
    )
    fig.tight_layout()
    return figure_to_image(fig)


def evaluation_details(
    evaluation: Evaluation, model_name: str, device: str, csv_folder: Path | None
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "Model": model_name,
        "Device": device,
        "Images": len(evaluation.paths),
        "Classes": len(evaluation.labels),
        "Best scheme": evaluation.best.name,
        "Accuracy": f"{evaluation.best.accuracy:.1%}",
        f"Top-{evaluation.k} accuracy": f"{evaluation.best.top_k_accuracy:.1%}",
    }
    for s in evaluation.schemes:
        details[f"  {s.name}"] = (
            f"{s.accuracy:.1%} (top-{evaluation.k} {s.top_k_accuracy:.1%})"
        )
    for label, (precision, recall, f1, support) in zip(
        evaluation.labels, evaluation.metrics
    ):
        details[f"{label} (n={support})"] = (
            f"P {precision:.2f}  R {recall:.2f}  F1 {f1:.2f}"
        )
    details["Mistakes"] = int(len(evaluation.targets) - np.trace(evaluation.matrix))
    if evaluation.from_cache:
        details["Embeddings"] = "from cache"
    if csv_folder is not None:
        details["CSV"] = str(csv_folder)
    details["Time"] = f"{evaluation.elapsed_ms / 1000:.1f} s"
    return details


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def ask_evaluation_parameters(parent: tk.Misc) -> dict[str, Any] | None:
    dialog = tk.Toplevel(parent)
    dialog.title("CLIP Folder Evaluation")
    dialog.transient(parent)
    dialog.grab_set()

    folder_var = tk.StringVar()
    max_var = tk.IntVar(value=0)
    model_var = tk.StringVar(value=DEFAULT_MODEL)
    device_var = tk.StringVar(value="auto")
    csv_var = tk.BooleanVar(value=True)

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    folder_box = ttk.LabelFrame(
        main, text="A. Labelled folder (one sub-folder per class)", padding=10
    )
    folder_box.pack(fill="x", pady=(0, 8))
    ttk.Entry(folder_box, textvariable=folder_var, width=52).grid(
        row=0, column=0, sticky="ew"
    )

    def browse() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Labelled folder", initialdir=str(PROJECT_ROOT)
        )
        if chosen:
            folder_var.set(chosen)

    ttk.Button(folder_box, text="Browse…", command=browse).grid(
        row=0, column=1, padx=(6, 0)
    )
    ttk.Label(
        folder_box,
        text="e.g. photos/cat/*.jpg, photos/hot_chocolate/*.jpg → classes "
        '"cat", "hot chocolate"',
        foreground="#555",
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
    ttk.Label(folder_box, text="Max images per class (0 = all):").grid(
        row=2, column=0, sticky="w", pady=(4, 0)
    )
    ttk.Spinbox(folder_box, textvariable=max_var, from_=0, to=10000, width=6).grid(
        row=2, column=1, sticky="w", pady=(4, 0)
    )
    folder_box.columnconfigure(0, weight=1)

    template_box = ttk.LabelFrame(
        main, text="B. Templates to compare ({} = class, one per line)", padding=10
    )
    template_box.pack(fill="both", expand=True, pady=(0, 8))
    templates_text = tk.Text(template_box, width=52, height=6, wrap="none")
    templates_text.pack(fill="both", expand=True)
    templates_text.insert("1.0", "{}\n" + "\n".join(ENSEMBLE_TEMPLATES))

    model_box = ttk.LabelFrame(main, text="C. Model", padding=10)
    model_box.pack(fill="x", pady=(0, 8))
    ttk.Combobox(model_box, textvariable=model_var, values=CLIP_MODELS, width=34).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(model_box, text="Device:").grid(row=0, column=1, padx=(12, 4))
    ttk.Combobox(
        model_box, textvariable=device_var, values=DEVICES, state="readonly", width=7
    ).grid(row=0, column=2)
    ttk.Checkbutton(
        model_box,
        text="Save predictions as CSV (outputs/set10_clip/)",
        variable=csv_var,
    ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(4, 0))
    result: dict[str, Any] | None = None

    def submit() -> None:
        nonlocal result
        folder = Path(folder_var.get().strip()).expanduser()
        if not folder.is_dir():
            messagebox.showerror("Folder", "Choose an existing folder.", parent=dialog)
            return
        try:
            templates = parse_templates(templates_text.get("1.0", "end"))
            max_per_class = max(0, int(max_var.get()))
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid input", str(exc), parent=dialog)
            return
        result = {
            "folder": str(folder),
            "templates": templates,
            "max_per_class": max_per_class,
            "model": model_var.get().strip() or DEFAULT_MODEL,
            "device": device_var.get(),
            "save_csv": bool(csv_var.get()),
        }
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Evaluate", command=submit).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _evaluate_job(
    parameters: dict[str, Any],
    set_status: Callable[[str], None] | None = None,
) -> tuple[Evaluation, Image.Image, str, Path | None]:
    device = resolve_device(str(parameters.get("device", "auto")))
    model_name = str(parameters.get("model") or DEFAULT_MODEL)
    injected = parameters.get("clip") is not None
    if injected:
        model, processor = parameters["clip"]
    else:
        if set_status is not None:
            set_status(f"Loading {model_name.split('/')[-1]}…")
        model, processor = load_clip(model_name, device)
    root = Path(parameters["folder"])
    progress = (
        (lambda done, total: set_status(f"Embedding images: {done:,} / {total:,}…"))
        if set_status is not None
        else None
    )
    evaluation = evaluate_folder(
        model,
        processor,
        root,
        list(parameters["templates"]),
        device,
        max_per_class=int(parameters.get("max_per_class", 0)),
        model_name="" if injected else model_name,
        progress=progress,
    )
    csv_folder = None
    if parameters.get("save_csv"):
        if set_status is not None:
            set_status("Saving CSV…")
        csv_folder = save_csv(
            evaluation, root, Path(parameters.get("output_dir") or OUTPUT_DIR)
        )
    if set_status is not None:
        set_status("Rendering report…")
    return evaluation, render_evaluation(evaluation), device, csv_folder


def run_folder_evaluation(
    parent: tk.Misc,
    document: ImageDocument,
    parameters: dict[str, Any] | None = None,
) -> ToolResult | None:
    """Ask for a labelled folder, then evaluate CLIP zero-shot on it."""
    if parameters is None:
        parameters = ask_evaluation_parameters(parent)
        if parameters is None:
            return None
    device = resolve_device(str(parameters.get("device", "auto")))
    message = (
        ""
        if parameters.get("clip") is not None
        else loading_message([str(parameters.get("model") or DEFAULT_MODEL)], device)
    ) + "Embedding the images and scoring every template…"
    status_line = [""]

    def set_status(text: str) -> None:
        status_line[0] = text

    evaluation, figure, device, csv_folder = run_with_progress(
        parent,
        "CLIP folder evaluation",
        message,
        _evaluate_job,
        parameters,
        set_status,
        status=lambda: status_line[0],
    )
    model_name = str(parameters.get("model") or DEFAULT_MODEL)
    return ToolResult(
        message=(
            f"CLIP zero-shot: {evaluation.best.accuracy:.1%} accuracy on "
            f"{len(evaluation.paths)} images ({evaluation.best.name})"
            + ("; CSV saved." if csv_folder else ".")
        ),
        image=figure,
        details=evaluation_details(evaluation, model_name, device, csv_folder),
    )
