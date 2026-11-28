"""Set 11.1 — Segmentation using a U-Net (TrayDataset).

One sidebar tool with two tasks:

* **Segment the open image** with a saved model: original image, segmentation
  mask and the mask overlaid on the image, plus the ground truth and its Dice
  when the image comes from the dataset (``XTest/a.jpg`` → ``yTest/a.png``).
* **Train a new model** on TrayDataset (binary food vs. not food, or all 43
  classes) and save it under ``models/unet/``.

The network, training and metrics are in ``unet_common`` (shared with
``notebooks/train_unet.ipynb``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.paths import PROJECT_ROOT
from forensics_app.utils.background import run_with_progress

from . import unet_common as uc
from .base import ForensicsTool, ToolResult

TASK_SEGMENT = "segment"
TASK_TRAIN = "train"
DEVICES = ("auto", "cpu", "mps", "cuda")
DEFAULT_MODEL = uc.UNET_MODELS_DIR / "tray_multiclass"
CHART_MARKER = "forensics_unet_chart"
MAX_LEGEND = 12


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _resolve(text: str) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _figure_to_image(fig: Figure) -> Image.Image:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    image = Image.fromarray(rgba[:, :, :3].copy(), mode="RGB")
    image.info[CHART_MARKER] = True
    return image


def source_image(document: ImageDocument) -> tuple[Image.Image, bool]:
    """The image to segment; the original one if the canvas shows our chart."""
    if document.current is None:
        raise ValueError("Open an image first.")
    current, original = document.current, document.original
    if current.info.get(CHART_MARKER) and original is not None:
        if not original.info.get(CHART_MARKER):
            return original, True
    return current, False


# ---------------------------------------------------------------------------
# Segmentation figure
# ---------------------------------------------------------------------------


def _show_labels(ax: Any, labels: np.ndarray, vmax: int, title: str) -> None:
    ax.imshow(labels, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)


MAX_DISPLAY_EDGE = 900


def _display_pair(
    rgb: Image.Image, mask: np.ndarray, ground_truth: np.ndarray | None
) -> tuple[Image.Image, np.ndarray, np.ndarray | None]:
    """Downscale huge photos/masks for Matplotlib display without altering metrics."""
    longest = max(rgb.size)
    if longest <= MAX_DISPLAY_EDGE:
        return rgb, mask, ground_truth
    scale = MAX_DISPLAY_EDGE / longest
    size = (
        max(1, int(round(rgb.width * scale))),
        max(1, int(round(rgb.height * scale))),
    )
    disp_rgb = rgb.resize(size, Image.BILINEAR)
    disp_mask = np.asarray(
        Image.fromarray(mask.astype(np.uint8, copy=False)).resize(size, Image.NEAREST)
    )
    disp_gt = (
        np.asarray(
            Image.fromarray(ground_truth.astype(np.uint8, copy=False)).resize(
                size, Image.NEAREST
            )
        )
        if ground_truth is not None
        else None
    )
    return disp_rgb, disp_mask, disp_gt


def render_segmentation(
    image: Image.Image,
    mask: np.ndarray,
    passport: dict[str, Any],
    *,
    ground_truth: np.ndarray | None = None,
    scores: dict[str, float] | None = None,
    summary: list[tuple[str, float]] | None = None,
) -> Image.Image:
    """Original image, (ground truth,) segmentation mask and overlay, as on the
    slide, with a legend of the classes found."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    names = passport["class_names"]
    name_to_idx = {name: i for i, name in enumerate(names)}
    vmax = int(passport["num_classes"]) - 1
    found = (
        summary if summary is not None else uc.label_summary(mask, names)
    )[:MAX_LEGEND]
    disp_rgb, disp_mask, disp_gt = _display_pair(rgb, mask, ground_truth)

    panels = 4 if disp_gt is not None else 3
    aspect = disp_rgb.height / disp_rgb.width
    fig = Figure(figsize=(4.6 * panels, 4.6 * aspect + 1.6), dpi=100)
    axes = fig.subplots(1, panels)

    axes[0].imshow(disp_rgb)
    axes[0].set_title("Original Image", fontsize=10)
    col = 1
    if disp_gt is not None:
        _show_labels(axes[1], disp_gt, vmax, "Ground Truth")
        col = 2
    title = "Segmentation Mask"
    if scores:
        title += "\n" + ", ".join(f"{k} {v:.2f}" for k, v in scores.items())
    _show_labels(axes[col], disp_mask, vmax, title)
    axes[col + 1].imshow(disp_rgb)
    axes[col + 1].imshow(
        disp_mask, cmap="viridis", vmin=0, vmax=vmax, alpha=0.5, interpolation="nearest"
    )
    axes[col + 1].set_title("Overlayed Mask", fontsize=10)

    cmap = colormaps["viridis"]
    denom = max(vmax, 1)
    handles = [
        Patch(
            color=cmap(name_to_idx[name] / denom),
            label=f"{name} ({share:.0%})",
        )
        for name, share in found
        if name in name_to_idx
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(6, len(handles)) or 1,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return _figure_to_image(fig)


def segmentation_scores(
    mask: np.ndarray, ground_truth: np.ndarray, task: str
) -> dict[str, float]:
    if task == uc.TASK_BINARY:
        return {
            "Dice": uc.dice_score(mask, ground_truth),
            "IoU": uc.iou_score(mask, ground_truth),
        }
    result = uc.score_predictions(mask[None], ground_truth[None], task)
    return {
        "mean class Dice": result["mean_dice"],
        "pixel accuracy": result["pixel_accuracy"],
        "food Dice": result["food_dice"],
    }


def _segment_job(
    image: Image.Image,
    image_path: Path | None,
    parameters: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any], str]:
    started = time.perf_counter()
    device = uc.resolve_device(str(parameters.get("device", "auto")))
    if parameters.get("model") is not None:
        model, passport = parameters["model"]
        folder = Path(parameters.get("model_dir") or "(in memory)")
    else:
        folder = _resolve(str(parameters.get("model_dir") or DEFAULT_MODEL))
        model, passport = uc.load_model(folder, device)
    task = passport["task"]
    threshold = parameters.get("threshold")
    mask = uc.segment(model, passport, image, device, threshold)

    gt_path = (
        uc.ground_truth_path(image_path) if parameters.get("compare", True) else None
    )
    ground_truth = scores = None
    if gt_path is not None:
        ground_truth = uc.load_ground_truth(gt_path, task, image.size)
        scores = segmentation_scores(mask, ground_truth, task)
    summary_items = uc.label_summary(mask, passport["class_names"])
    figure = render_segmentation(
        image,
        mask,
        passport,
        ground_truth=ground_truth,
        scores=scores,
        summary=summary_items,
    )

    details: dict[str, Any] = {
        "Model": display_path(folder),
        "Task": f"{task} ({passport['num_classes']} classes)",
        "Model input": "{} × {}".format(*passport["input_size"]),
        "Device": device,
        "Image": f"{image.width} × {image.height}",
    }
    if task == uc.TASK_BINARY:
        details["Threshold"] = (
            threshold if threshold is not None else passport.get("threshold", 0.5)
        )
    details["Ground truth"] = display_path(gt_path) if gt_path else "not found"
    for name, value in (scores or {}).items():
        details[f"  {name}"] = f"{value:.3f}"
    for name, share in summary_items:
        details[f"Class: {name}"] = f"{share:.1%} of pixels"
    details["Time"] = f"{(time.perf_counter() - started) * 1000:.0f} ms"

    summary = f"UNet ({folder.name}): {len(summary_items)} classes found"
    if scores:
        name, value = next(iter(scores.items()))
        summary += f", {name} {value:.2f} against the ground truth"
    return figure, details, summary + "."


# ---------------------------------------------------------------------------
# Training figure
# ---------------------------------------------------------------------------


def render_training(run: uc.TrainingRun, device: str, samples: int = 3) -> Image.Image:
    """Training curves and a few test predictions next to their ground truth."""
    passport = run.passport
    task = passport["task"]
    vmax = int(passport["num_classes"]) - 1
    n = min(samples, len(run.test_images))
    pred = uc.predict_arrays(run.model, run.test_images[:n], task, device)

    fig = Figure(figsize=(13, 3.6 + 2.6 * n), dpi=100)
    grid = fig.add_gridspec(1 + n, 3, height_ratios=(1.3,) + (1,) * n)
    epochs = range(1, len(run.history["train_loss"]) + 1)
    ax = fig.add_subplot(grid[0, :2])
    ax.plot(epochs, run.history["train_loss"], marker="o", markersize=3)
    ax.set_title("Training loss", fontsize=10)
    ax.set_xlabel("epoch")
    ax = fig.add_subplot(grid[0, 2])
    ax.plot(epochs, run.history["val_score"], marker="o", markersize=3, color="#2f855a")
    best = passport["training"]["best_epoch"]
    ax.axvline(best, color="#999", linestyle="--", linewidth=1)
    ax.set_title(
        f"Validation {passport['metrics']['validation']['score_name']}", fontsize=10
    )
    ax.set_xlabel(f"epoch (best {best})")
    for r in range(n):
        score = segmentation_scores(pred[r], run.test_labels[r], task)
        name, value = next(iter(score.items()))
        for c, (data, title) in enumerate(
            (
                (run.test_images[r], "Test image"),
                (run.test_labels[r], "Ground truth"),
                (pred[r], f"Prediction ({name} {value:.2f})"),
            )
        ):
            ax = fig.add_subplot(grid[1 + r, c])
            if c == 0:
                ax.imshow(data)
            else:
                _show_labels(ax, data, vmax, title)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle(f"UNet {task}: {passport['parameters']:,} parameters", fontsize=11)
    fig.tight_layout()
    return _figure_to_image(fig)


def training_details(run: uc.TrainingRun, folder: Path) -> dict[str, Any]:
    p = run.passport
    tr, test = p["training"], p["metrics"]["test"]
    details: dict[str, Any] = {
        "Saved to": display_path(folder),
        "Task": f"{p['task']} ({p['num_classes']} classes)",
        "Input size": "{} × {}".format(*p["input_size"]),
        "Parameters": f"{p['parameters']:,}",
        "Images (train / val / test)": (
            f"{tr['train_images']} / {tr['val_images']} / {tr['test_images']}"
        ),
        "Epochs (best)": f"{tr['epochs']} ({tr['best_epoch']})",
        "Device": tr["device"],
        "Best validation": (
            f"{p['metrics']['validation']['score_name']} "
            f"{p['metrics']['validation']['best']:.3f}"
        ),
    }
    if p["task"] == uc.TASK_BINARY:
        details["Test Dice"] = f"{test['dice']:.3f} ± {test['dice_std']:.3f}"
        details["Test IoU"] = f"{test['iou']:.3f}"
    else:
        details["Test mean class Dice"] = f"{test['mean_dice']:.3f}"
        details["Test pixel accuracy"] = f"{test['pixel_accuracy']:.3f}"
        details["Test food Dice"] = f"{test['food_dice']:.3f}"
    details["Time"] = f"{tr['seconds']:.0f} s"
    return details


def _train_job(
    parameters: dict[str, Any], status: dict[str, str]
) -> tuple[uc.TrainingRun, Path, Image.Image]:
    def report(message: str) -> None:
        status["text"] = message

    run = uc.train_unet(
        _resolve(str(parameters.get("dataset") or uc.TRAY_DATASET_DIR)),
        parameters.get("task", uc.TASK_BINARY),
        epochs=parameters.get("epochs"),
        size=parameters.get("size"),
        base=parameters.get("base"),
        batch_size=parameters.get("batch_size"),
        lr=parameters.get("lr"),
        device=str(parameters.get("device", "auto")),
        max_images=int(parameters.get("max_images", 0)),
        progress=report,
    )
    folder = uc.save_model(
        run.model, run.passport, _resolve(str(parameters["save_to"]))
    )
    report("Drawing the results…")
    figure = render_training(run, run.passport["training"]["device"])
    return run, folder, figure


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------


def _modal(parent: tk.Misc, title: str) -> tk.Toplevel:
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    return dialog


def ask_task(parent: tk.Misc, has_image: bool) -> str | None:
    dialog = _modal(parent, "UNet Segmentation")
    task_var = tk.StringVar(value=TASK_SEGMENT if has_image else TASK_TRAIN)
    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)
    box = ttk.LabelFrame(main, text="What do you want to do?", padding=10)
    box.pack(fill="x", pady=(0, 8))
    ttk.Radiobutton(
        box,
        text="Segment the open image with a saved model"
        + ("" if has_image else " (open an image first)"),
        value=TASK_SEGMENT,
        variable=task_var,
        state="normal" if has_image else "disabled",
    ).grid(row=0, column=0, sticky="w", pady=2)
    ttk.Radiobutton(
        box,
        text="Train a new model on TrayDataset and save it",
        value=TASK_TRAIN,
        variable=task_var,
    ).grid(row=1, column=0, sticky="w", pady=2)
    result: str | None = None

    def submit() -> None:
        nonlocal result
        result = task_var.get()
        dialog.destroy()

    buttons = ttk.Frame(main)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Next…", command=submit).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


def _passport_summary(folder: Path) -> tuple[str, str | None]:
    try:
        import json

        with open(folder / uc.PASSPORT_FILE, encoding="utf-8") as fh:
            p = json.load(fh)
    except (OSError, ValueError):
        return "model not found (needs unet.pt and passport.json)", None
    test = p.get("metrics", {}).get("test", {})
    score = (
        f"test Dice {test['dice']:.3f}"
        if "dice" in test
        else f"test mean class Dice {test.get('mean_dice', 0):.3f}"
    )
    h, w = p["input_size"]
    return f"{p['task']}, {p['num_classes']} classes, input {h}×{w}, {score}", p["task"]


def ask_segment_parameters(
    parent: tk.Misc, image_path: Path | None
) -> dict[str, Any] | None:
    dialog = _modal(parent, "UNet Segmentation: segment the image")
    models = uc.list_models()
    default = (
        DEFAULT_MODEL
        if DEFAULT_MODEL in models
        else (models[0] if models else DEFAULT_MODEL)
    )
    model_var = tk.StringVar(value=display_path(default))
    device_var = tk.StringVar(value="auto")
    threshold_var = tk.DoubleVar(value=0.5)
    compare_var = tk.BooleanVar(value=True)
    info_var = tk.StringVar()

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)
    box = ttk.LabelFrame(main, text="A. Saved model", padding=10)
    box.pack(fill="x", pady=(0, 8))
    combo = ttk.Combobox(
        box, textvariable=model_var, values=[display_path(m) for m in models], width=46
    )
    combo.grid(row=0, column=0, sticky="ew")

    def browse() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Model folder", initialdir=str(uc.UNET_MODELS_DIR)
        )
        if chosen:
            model_var.set(display_path(chosen))
            refresh()

    ttk.Button(box, text="Browse…", command=browse).grid(row=0, column=1, padx=(6, 0))
    info_label = ttk.Label(box, textvariable=info_var)
    info_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
    box.columnconfigure(0, weight=1)

    options = ttk.LabelFrame(main, text="B. Options", padding=10)
    options.pack(fill="x", pady=(0, 8))
    ttk.Label(options, text="Device:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(
        options, textvariable=device_var, values=DEVICES, state="readonly", width=7
    ).grid(row=0, column=1, sticky="w", padx=6)
    ttk.Label(options, text="Food threshold (binary models):").grid(
        row=1, column=0, sticky="w", pady=(4, 0)
    )
    threshold_box = ttk.Spinbox(
        options,
        textvariable=threshold_var,
        from_=0.05,
        to=0.95,
        increment=0.05,
        width=6,
    )
    threshold_box.grid(row=1, column=1, sticky="w", padx=6, pady=(4, 0))
    gt = uc.ground_truth_path(image_path)
    ttk.Checkbutton(
        options,
        text="Compare with the ground truth mask"
        + (f" ({display_path(gt)})" if gt else " (none found for this image)"),
        variable=compare_var,
        state="normal" if gt else "disabled",
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def refresh(_event: object = None) -> None:
        text, task = _passport_summary(_resolve(model_var.get().strip()))
        info_var.set(text)
        info_label.configure(foreground="#006300" if task else "#b42318")
        threshold_box.configure(
            state="normal" if task == uc.TASK_BINARY else "disabled"
        )

    combo.bind("<<ComboboxSelected>>", refresh)
    combo.bind("<FocusOut>", refresh)
    refresh()

    buttons = ttk.Frame(main)
    buttons.pack(fill="x")
    result: dict[str, Any] | None = None

    def submit() -> None:
        nonlocal result
        folder = _resolve(model_var.get().strip())
        if not (folder / uc.WEIGHTS_FILE).is_file():
            messagebox.showerror(
                "Model", f"No trained model in:\n{folder}", parent=dialog
            )
            return
        try:
            threshold = min(max(float(threshold_var.get()), 0.01), 0.99)
        except (tk.TclError, ValueError):
            threshold = 0.5
        result = {
            "model_dir": str(folder),
            "device": device_var.get(),
            "threshold": threshold,
            "compare": bool(compare_var.get()),
        }
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Segment", command=submit).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


def ask_train_parameters(parent: tk.Misc) -> dict[str, Any] | None:
    dialog = _modal(parent, "UNet Segmentation: train a model")
    task_var = tk.StringVar(value=uc.TASK_BINARY)
    dataset_var = tk.StringVar(value=display_path(uc.TRAY_DATASET_DIR))
    device_var = tk.StringVar(value="auto")
    save_var = tk.StringVar()
    fields = {
        name: tk.StringVar()
        for name in ("epochs", "height", "width", "base", "batch_size", "lr")
    }

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)
    data_box = ttk.LabelFrame(
        main, text="A. Dataset (folder with XTrain, yTrain, XTest, yTest)", padding=10
    )
    data_box.pack(fill="x", pady=(0, 8))
    ttk.Entry(data_box, textvariable=dataset_var, width=50).grid(
        row=0, column=0, sticky="ew"
    )

    def browse() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="TrayDataset folder", initialdir=str(PROJECT_ROOT)
        )
        if chosen:
            dataset_var.set(display_path(chosen))

    ttk.Button(data_box, text="Browse…", command=browse).grid(
        row=0, column=1, padx=(6, 0)
    )
    data_box.columnconfigure(0, weight=1)

    task_box = ttk.LabelFrame(main, text="B. Task", padding=10)
    task_box.pack(fill="x", pady=(0, 8))

    def apply_defaults() -> None:
        task = task_var.get()
        d = uc.TASK_DEFAULTS[task]
        fields["epochs"].set(str(d["epochs"]))
        fields["height"].set(str(d["size"][0]))
        fields["width"].set(str(d["size"][1]))
        fields["base"].set(str(d["base"]))
        fields["batch_size"].set(str(d["batch"]))
        fields["lr"].set(str(d["lr"]))
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        name = "tray_food_binary" if task == uc.TASK_BINARY else "tray_multiclass"
        save_var.set(display_path(uc.UNET_MODELS_DIR / f"{name}_{stamp}"))

    ttk.Radiobutton(
        task_box,
        text="Food vs. not food",
        value=uc.TASK_BINARY,
        variable=task_var,
        command=apply_defaults,
    ).grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(
        task_box,
        text="All 43 classes",
        value=uc.TASK_MULTICLASS,
        variable=task_var,
        command=apply_defaults,
    ).grid(row=1, column=0, sticky="w")

    settings = ttk.LabelFrame(main, text="C. Training settings", padding=10)
    settings.pack(fill="x", pady=(0, 8))
    labels = (
        ("Epochs:", "epochs"),
        ("Input height:", "height"),
        ("Input width:", "width"),
        ("Base channels:", "base"),
        ("Batch size:", "batch_size"),
        ("Learning rate:", "lr"),
    )
    for i, (text, key) in enumerate(labels):
        ttk.Label(settings, text=text).grid(row=i // 3, column=2 * (i % 3), sticky="w")
        ttk.Entry(settings, textvariable=fields[key], width=8).grid(
            row=i // 3, column=2 * (i % 3) + 1, sticky="w", padx=(4, 12), pady=2
        )
    ttk.Label(settings, text="Device:").grid(row=2, column=0, sticky="w")
    ttk.Combobox(
        settings, textvariable=device_var, values=DEVICES, state="readonly", width=7
    ).grid(row=2, column=1, sticky="w", padx=(4, 12), pady=2)
    ttk.Label(
        settings,
        text="Height and width must be divisible by 8.",
        foreground="#555",
    ).grid(row=3, column=0, columnspan=6, sticky="w", pady=(4, 0))

    save_box = ttk.LabelFrame(main, text="D. Save the model to", padding=10)
    save_box.pack(fill="x", pady=(0, 8))
    ttk.Entry(save_box, textvariable=save_var, width=56).pack(fill="x")
    apply_defaults()

    buttons = ttk.Frame(main)
    buttons.pack(fill="x")
    result: dict[str, Any] | None = None

    def submit() -> None:
        nonlocal result
        dataset = _resolve(dataset_var.get().strip())
        try:
            uc.tray_splits(dataset)
        except FileNotFoundError as exc:
            messagebox.showerror("Dataset", str(exc), parent=dialog)
            return
        try:
            epochs = int(fields["epochs"].get())
            size = (int(fields["height"].get()), int(fields["width"].get()))
            base = int(fields["base"].get())
            batch = int(fields["batch_size"].get())
            lr = float(fields["lr"].get())
            if min(epochs, base, batch) < 1 or lr <= 0 or size[0] % 8 or size[1] % 8:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Settings",
                "Use positive numbers; height and width must be divisible by 8.",
                parent=dialog,
            )
            return
        target = _resolve(save_var.get().strip())
        if (target / uc.WEIGHTS_FILE).exists() and not messagebox.askyesno(
            "Overwrite?",
            f"A model already exists in:\n{target}\nReplace it?",
            parent=dialog,
        ):
            return
        result = {
            "task": task_var.get(),
            "dataset": str(dataset),
            "epochs": epochs,
            "size": size,
            "base": base,
            "batch_size": batch,
            "lr": lr,
            "device": device_var.get(),
            "save_to": str(target),
        }
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Train", command=submit).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class UNetSegmentationTool(ForensicsTool):
    """Segment the open image with a saved U-Net, or train and save a new one."""

    tool_id = "unet_segmentation"
    title = "UNet Segmentation"
    category = "Deep Learning"
    description = (
        "Semantic segmentation of meal trays with a U-Net: segment the open "
        "image (mask and overlay, Dice against the ground truth when known) "
        "or train and save a new model on TrayDataset."
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
            task = ask_task(parent, document.current is not None)
            if task is None:
                return None
        else:
            task = parameters.get("task_choice", TASK_SEGMENT)
        if task == TASK_TRAIN:
            return self._train(parent, parameters)
        return self._segment(parent, document, parameters)

    def _segment(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        parameters: dict[str, Any] | None,
    ) -> ToolResult | None:
        image, used_original = source_image(document)
        if parameters is None:
            parameters = ask_segment_parameters(parent, document.path)
            if parameters is None:
                return None
        figure, details, summary = run_with_progress(
            parent,
            "UNet segmentation",
            "Segmenting the image…",
            _segment_job,
            image,
            document.path,
            parameters,
        )
        if used_original:
            details = {"Image used": "original (the canvas showed a chart)", **details}
        return ToolResult(message=summary, image=figure, details=details)

    def _train(
        self, parent: tk.Misc, parameters: dict[str, Any] | None
    ) -> ToolResult | None:
        if parameters is None:
            parameters = ask_train_parameters(parent)
            if parameters is None:
                return None
        status = {"text": "Starting…"}
        run, folder, figure = run_with_progress(
            parent,
            "Training the UNet",
            "Training on TrayDataset; this can take several minutes.",
            _train_job,
            parameters,
            status,
            status=lambda: status["text"],
        )
        details = training_details(run, folder)
        test = run.passport["metrics"]["test"]
        score = (
            f"test Dice {test['dice']:.3f}"
            if run.passport["task"] == uc.TASK_BINARY
            else f"test mean class Dice {test['mean_dice']:.3f}"
        )
        return ToolResult(
            message=f"UNet trained and saved to {display_path(folder)} ({score}).",
            image=figure,
            details=details,
        )
