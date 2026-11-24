"""Set 8.2 — Load Pretrained CNN Model, View Passport & Classify Images.

Loads a trained CNN (``weights.pt`` + ``passport.json``) or an off-the-shelf
ImageNet-1K pretrained backbone, displays its full Model Passport, and performs
classification on:
  1. An individual image from the held-out test set
  2. A group (or full batch) of images from the held-out test set
  3. A completely new image (currently open canvas image or browsed file)
"""

from __future__ import annotations

import io
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, Dataset

from forensics_app.core import ImageDocument
from forensics_app.paths import resolve_path
from forensics_app.utils.background import run_with_progress

from .base import ForensicsTool, ToolResult
from .cnn_common import (
    DEFAULT_CNN_ROOT,
    describe_compute_device,
    find_latest_cnn_model_dir,
    format_passport_text,
    list_saved_cnn_models,
    load_cnn_bundle,
    load_imagenet_pretrained_bundle,
    load_test_record_image,
    preprocess_pil_image,
)
from .codebook_construction import ensure_document_has_image, portable_path
from .hog_person_detection import IMAGE_TYPES


class _PreloadedImageListDataset(Dataset):
    """Dataset wrapping passport test records for batched GPU/CPU inference."""

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


def classify_single_cnn_image(
    image: Image.Image,
    bundle: dict[str, Any],
    *,
    true_label: str | None = None,
    render: bool = True,
) -> dict[str, Any]:
    """Classify one PIL image using the loaded CNN bundle and passport preprocessing."""
    t0 = time.perf_counter()
    passport = bundle["passport"]
    model = bundle["model"]
    device = bundle["device"]
    class_names = list(passport["class_names"])
    prep = passport["preprocessing"]

    tensor = preprocess_pil_image(image, prep).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    top_class = class_names[top_idx]
    top_prob = float(probs[top_idx])
    class_probs = [
        {"class_name": class_names[int(i)], "probability": float(probs[int(i)])}
        for i in order
    ]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    vis_image = None
    if render:
        vis_image = _render_single_classification_figure(
            image=image,
            class_probs=class_probs,
            top_class=top_class,
            top_prob=top_prob,
            true_label=true_label,
            passport=passport,
            elapsed_ms=elapsed_ms,
        )

    return {
        "mode": "single",
        "predicted_class": top_class,
        "top_probability": top_prob,
        "true_label": true_label,
        "is_correct": (true_label == top_class) if true_label is not None else None,
        "class_probabilities": class_probs,
        "processing_time_ms": elapsed_ms,
        "visualization_image": vis_image,
    }


def _render_single_classification_figure(
    *,
    image: Image.Image,
    class_probs: list[dict[str, Any]],
    top_class: str,
    top_prob: float,
    true_label: str | None,
    passport: dict[str, Any],
    elapsed_ms: float,
) -> Image.Image:
    """Render side-by-side input image and class probability bar chart."""
    fig = Figure(figsize=(10.5, 4.6), dpi=115)
    ax_img = fig.add_subplot(1, 2, 1)
    ax_bar = fig.add_subplot(1, 2, 2)

    ax_img.imshow(image.convert("RGB"))
    if true_label is not None:
        status = "CORRECT" if true_label == top_class else "MISCLASSIFIED"
        title_color = "#2f855a" if true_label == top_class else "#c53030"
        ax_img.set_title(
            f"Pred: {top_class} ({top_prob * 100:.1f}%)\nTrue: {true_label} [{status}]",
            fontsize=10,
            fontweight="bold",
            color=title_color,
        )
    else:
        ax_img.set_title(
            f"Prediction: {top_class} ({top_prob * 100:.1f}%)",
            fontsize=10.5,
            fontweight="bold",
            color="#2b6cb0",
        )
    ax_img.axis("off")

    top_items = class_probs[:10][::-1]
    labels = [item["class_name"] for item in top_items]
    values = [item["probability"] * 100.0 for item in top_items]
    colors = [
        "#2f855a" if item["class_name"] == top_class else "#4299e1"
        for item in top_items
    ]
    bars = ax_bar.barh(labels, values, color=colors)
    ax_bar.set_xlim(0, 108)
    ax_bar.set_xlabel("Probability (%)")
    ax_bar.set_title("Class Probability Distribution", fontsize=10, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.25)
    for bar, val in zip(bars, values):
        ax_bar.text(
            min(103.0, val + 1.5),
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            va="center",
            fontsize=8,
        )

    arch = passport.get("architecture", {})
    prep = passport.get("preprocessing", {})
    fig.suptitle(
        f"Model: {passport.get('model_name', 'CNN')} ({arch.get('arch_key', '')}) | "
        f"Input: {prep.get('input_size')} ({prep.get('normalization')}) | "
        f"{elapsed_ms:.1f} ms",
        fontsize=9.5,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def classify_held_out_group(
    bundle: dict[str, Any],
    *,
    class_filter: str = "All",
    max_images: int = 0,
    batch_size: int = 32,
    render: bool = True,
) -> dict[str, Any]:
    """Classify a group (or all) of held-out test images from the model passport."""
    t0 = time.perf_counter()
    passport = bundle["passport"]
    model = bundle["model"]
    device = bundle["device"]
    class_names = list(passport["class_names"])
    test_records = list(passport.get("dataset", {}).get("test_records", []))
    if not test_records:
        raise ValueError("This model passport does not contain held-out test records.")

    if class_filter and class_filter != "All":
        selected = [r for r in test_records if r.get("class_name") == class_filter]
    else:
        selected = list(test_records)

    if max_images and int(max_images) > 0:
        selected = selected[: int(max_images)]

    if not selected:
        raise ValueError(
            f"No held-out test images match class filter {class_filter!r}."
        )

    ds = _PreloadedImageListDataset(selected, passport)
    loader = DataLoader(
        ds,
        batch_size=max(1, min(int(batch_size), len(ds))),
        shuffle=False,
    )

    model.eval()
    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(targets.numpy())

    probs_mat = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.argmax(probs_mat, axis=1)
    confidences = probs_mat[np.arange(len(y_pred)), y_pred]
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    predictions: list[dict[str, Any]] = []
    for rec, t_id, p_id, conf in zip(selected, y_true, y_pred, confidences):
        predictions.append(
            {
                "filename": rec.get("filename", ""),
                "true_class": class_names[int(t_id)],
                "predicted_class": class_names[int(p_id)],
                "confidence": float(conf),
                "correct": int(t_id) == int(p_id),
            }
        )

    vis_image = None
    if render:
        vis_image = _render_group_classification_figure(
            selected_records=selected,
            predictions=predictions,
            cm=np.asarray(cm, dtype=np.int64),
            class_names=class_names,
            accuracy=acc,
            passport=passport,
            elapsed_ms=elapsed_ms,
        )

    return {
        "mode": "group",
        "count": len(selected),
        "correct_count": int(np.sum(y_true == y_pred)),
        "accuracy": acc,
        "confusion_matrix": cm,
        "predictions": predictions,
        "processing_time_ms": elapsed_ms,
        "visualization_image": vis_image,
    }


def _render_group_classification_figure(
    *,
    selected_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    cm: np.ndarray,
    class_names: list[str],
    accuracy: float,
    passport: dict[str, Any],
    elapsed_ms: float,
) -> Image.Image:
    """Render a grid of sample predictions alongside the group confusion matrix."""
    sample_count = min(12, len(selected_records))
    cols = 4
    rows = max(1, (sample_count + cols - 1) // cols)

    fig = Figure(figsize=(13.0, max(4.8, 2.25 * rows)), dpi=115)
    gs = fig.add_gridspec(rows, cols + 2, wspace=0.28, hspace=0.45)

    for idx in range(sample_count):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(gs[r, c])
        pil_img = load_test_record_image(selected_records[idx], passport).convert("RGB")
        ax.imshow(pil_img)
        pred_info = predictions[idx]
        ok = pred_info["correct"]
        color = "#2f855a" if ok else "#c53030"
        conf_pct = pred_info["confidence"] * 100
        ax.set_title(
            f"P: {pred_info['predicted_class']} ({conf_pct:.0f}%)\n"
            f"T: {pred_info['true_class']}",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
        ax.set_xticks(())
        ax.set_yticks(())

    ax_cm = fig.add_subplot(gs[:, cols:])
    if cm.size > 0:
        im = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
        ax_cm.set_title(
            f"Group Confusion Matrix\nAccuracy: {accuracy * 100:.2f}% "
            f"({sum(p['correct'] for p in predictions)}/{len(predictions)})",
            fontsize=9.5,
            fontweight="bold",
        )
        n_cls = cm.shape[0]
        ticks = np.arange(n_cls)
        short = [(n[:10] + "…" if len(n) > 11 else n) for n in class_names[:n_cls]]
        ax_cm.set_xticks(ticks)
        ax_cm.set_yticks(ticks)
        ax_cm.set_xticklabels(short, rotation=35, ha="right", fontsize=7.5)
        ax_cm.set_yticklabels(short, fontsize=7.5)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        if n_cls <= 12:
            for rr in range(n_cls):
                for cc in range(n_cls):
                    ax_cm.text(
                        cc,
                        rr,
                        str(cm[rr, cc]),
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if cm[rr, cc] > thresh else "black",
                    )
        fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Held-Out Test Group Evaluation — {passport.get('model_name', 'CNN')} "
        f"({len(predictions)} images, {elapsed_ms:.1f} ms)",
        fontsize=10.5,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.03, right=0.97)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _load_bundle_from_source(
    source_kind: str,
    model_dir: str,
    imagenet_arch: str,
    device_pref: str,
) -> dict[str, Any]:
    if source_kind == "imagenet":
        return load_imagenet_pretrained_bundle(
            imagenet_arch, device_preference=device_pref
        )
    return load_cnn_bundle(model_dir, device_preference=device_pref)


def ask_cnn_classification_parameters(
    parent: tk.Misc,
    document: ImageDocument,
) -> dict[str, Any] | None:
    """Interactive dialog to load a CNN model, view its passport, and classify."""
    dialog = tk.Toplevel(parent)
    dialog.title("Set 8.2 — CNN Model Passport & Image Classification")
    dialog.transient(parent.winfo_toplevel())
    dialog.grab_set()
    dialog.resizable(True, True)
    dialog.minsize(760, 620)

    saved_dirs = list_saved_cnn_models()
    default_dir = (
        portable_path(find_latest_cnn_model_dir())
        if saved_dirs
        else portable_path(DEFAULT_CNN_ROOT / "cnn_model_001")
    )

    model_source_var = tk.StringVar(value="saved" if saved_dirs else "imagenet")
    model_dir_var = tk.StringVar(value=default_dir)
    imagenet_arch_var = tk.StringVar(value="resnet50")
    device_var = tk.StringVar(value="auto")

    classify_mode_var = tk.StringVar(
        value="new_canvas" if document.current is not None else "held_out_individual"
    )
    class_filter_var = tk.StringVar(value="All")
    group_max_var = tk.IntVar(value=0)
    custom_img_var = tk.StringVar(value="")

    model_status_var = tk.StringVar(value=describe_compute_device("auto"))
    pred_headline_var = tk.StringVar(
        value="Load a model to view its passport and classify images."
    )
    pred_dist_var = tk.StringVar(value="")

    cached_bundle: dict[str, Any] | None = None
    test_records: list[dict[str, Any]] = []

    main = ttk.Frame(dialog, padding=10)
    main.pack(fill="both", expand=True)

    # 1. Load Pretrained Model
    top = ttk.LabelFrame(main, text="1. Load Pretrained CNN Model", padding=8)
    top.pack(fill="x", pady=(0, 6))
    top.columnconfigure(1, weight=1)

    src_row = ttk.Frame(top)
    src_row.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))

    saved_combo = ttk.Combobox(
        top,
        textvariable=model_dir_var,
        values=[portable_path(p) for p in saved_dirs] or [default_dir],
        width=48,
    )
    saved_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=2)

    imagenet_combo = ttk.Combobox(
        top,
        textvariable=imagenet_arch_var,
        values=("resnet50", "inception_v3"),
        state="readonly",
        width=18,
    )
    imagenet_combo.grid(row=2, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(top, text="Device:").grid(row=2, column=2, sticky="e", padx=4)
    dev_combo = ttk.Combobox(
        top,
        textvariable=device_var,
        values=("auto", "cuda", "cpu"),
        state="readonly",
        width=10,
    )
    dev_combo.grid(row=2, column=3, sticky="w", pady=2)

    # 2. Model Passport Viewer
    passport_frame = ttk.LabelFrame(main, text="2. Model Passport", padding=6)
    passport_frame.pack(fill="both", expand=True, pady=(0, 6))

    passport_text = tk.Text(
        passport_frame,
        height=11,
        wrap="word",
        font=("TkFixedFont", 9),
        background="#f7fafc",
        state="disabled",
    )
    passport_scroll = ttk.Scrollbar(
        passport_frame, orient="vertical", command=passport_text.yview
    )
    passport_text.configure(yscrollcommand=passport_scroll.set)
    passport_text.pack(side="left", fill="both", expand=True)
    passport_scroll.pack(side="right", fill="y")

    def _set_passport_display(text: str) -> None:
        passport_text.configure(state="normal")
        passport_text.delete("1.0", "end")
        passport_text.insert("1.0", text)
        passport_text.configure(state="disabled")

    # 3. Classification Target (Individual, Group, New Image)
    cls_frame = ttk.LabelFrame(
        main,
        text="3. Classification Target (Held-out Test Set or Completely New Image)",
        padding=8,
    )
    cls_frame.pack(fill="x", pady=(0, 6))
    cls_frame.columnconfigure(1, weight=1)

    test_combo = ttk.Combobox(cls_frame, state="readonly", width=44)
    filter_combo = ttk.Combobox(
        cls_frame, textvariable=class_filter_var, state="readonly", width=18
    )

    def _refresh_model() -> None:
        nonlocal cached_bundle, test_records
        try:
            cached_bundle = run_with_progress(
                dialog,
                "Loading CNN Model",
                "Loading model weights and passport…",
                _load_bundle_from_source,
                model_source_var.get(),
                model_dir_var.get().strip(),
                imagenet_arch_var.get(),
                device_var.get(),
            )
            passport = cached_bundle["passport"]
            _set_passport_display(format_passport_text(passport))
            dev_info = cached_bundle["device_info"]
            dev_desc = dev_info.get("gpu_name") or dev_info.get("device")
            model_status_var.set(
                f"Loaded: {passport.get('model_name')} | "
                f"{len(passport.get('class_names', []))} classes | Device: {dev_desc}"
            )
            test_records = list(passport.get("dataset", {}).get("test_records", []))
            if test_records:
                test_combo.configure(
                    values=[
                        f"[{r['class_name']}] {r['filename']}" for r in test_records
                    ]
                )
                test_combo.current(0)
                classes = ["All"] + list(passport.get("class_names", []))
                filter_combo.configure(values=classes)
                class_filter_var.set("All")
            else:
                test_combo.configure(values=("(No held-out test records in passport)",))
                test_combo.current(0)
                filter_combo.configure(values=("All",))
                class_filter_var.set("All")
        except Exception as exc:
            cached_bundle = None
            test_records = []
            _set_passport_display(f"Could not load model:\n{exc}")
            model_status_var.set(f"Load error: {exc}")

    def _on_source_toggle() -> None:
        is_saved = model_source_var.get() == "saved"
        saved_combo.configure(state="normal" if is_saved else "disabled")
        browse_model_btn.configure(state="normal" if is_saved else "disabled")
        imagenet_combo.configure(state="disabled" if is_saved else "readonly")
        _refresh_model()

    ttk.Radiobutton(
        src_row,
        text="Saved Trained Model (weights.pt + passport.json)",
        variable=model_source_var,
        value="saved",
        command=_on_source_toggle,
    ).pack(side="left", padx=(0, 12))
    ttk.Radiobutton(
        src_row,
        text="ImageNet-1K Pretrained Backbone",
        variable=model_source_var,
        value="imagenet",
        command=_on_source_toggle,
    ).pack(side="left")

    ttk.Label(top, text="Saved model folder:").grid(row=1, column=0, sticky="w", pady=2)

    def _browse_model_dir() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select Saved CNN Model Folder"
        )
        if chosen:
            model_dir_var.set(chosen)
            model_source_var.set("saved")
            _refresh_model()

    browse_model_btn = ttk.Button(top, text="Browse…", command=_browse_model_dir)
    browse_model_btn.grid(row=1, column=3, sticky="w", pady=2)

    ttk.Label(top, text="ImageNet backbone:").grid(row=2, column=0, sticky="w", pady=2)
    ttk.Label(top, textvariable=model_status_var, foreground="#2b6cb0").grid(
        row=3, column=0, columnspan=4, sticky="w", pady=(2, 0)
    )

    saved_combo.bind("<<ComboboxSelected>>", lambda _e: _refresh_model())
    imagenet_combo.bind("<<ComboboxSelected>>", lambda _e: _refresh_model())
    dev_combo.bind("<<ComboboxSelected>>", lambda _e: _refresh_model())

    def _show_training_curves() -> None:
        if cached_bundle is None or cached_bundle.get("is_builtin_imagenet"):
            messagebox.showinfo(
                "Training Curves",
                "No local training_curves.png for this model.",
                parent=dialog,
            )
            return
        curve_path = Path(cached_bundle["model_dir"]) / "training_curves.png"
        if not curve_path.is_file():
            messagebox.showinfo(
                "Training Curves",
                f"File not found: {curve_path}",
                parent=dialog,
            )
            return
        win = tk.Toplevel(dialog)
        win.title("Saved Training Curves")
        with Image.open(curve_path) as im:
            im_copy = im.copy()
        im_copy.thumbnail((960, 560), Image.Resampling.LANCZOS)
        from PIL import ImageTk

        photo = ImageTk.PhotoImage(im_copy)
        lbl = ttk.Label(win, image=photo)
        lbl.image = photo  # keep reference alive
        lbl.pack(padx=8, pady=8)

    ttk.Button(passport_frame, text="View Curves", command=_show_training_curves).pack(
        side="bottom", anchor="e", pady=(4, 0)
    )

    # Classification target radio rows
    ttk.Radiobutton(
        cls_frame,
        text="Held-out test image (individual):",
        variable=classify_mode_var,
        value="held_out_individual",
    ).grid(row=0, column=0, sticky="w", pady=2)
    test_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=6, pady=2)

    ttk.Radiobutton(
        cls_frame,
        text="Held-out test set (group):",
        variable=classify_mode_var,
        value="held_out_group",
    ).grid(row=1, column=0, sticky="w", pady=2)
    grp_box = ttk.Frame(cls_frame)
    grp_box.grid(row=1, column=1, columnspan=2, sticky="w", padx=6, pady=2)
    ttk.Label(grp_box, text="Class:").pack(side="left")
    filter_combo.pack(in_=grp_box, side="left", padx=(4, 12))
    ttk.Label(grp_box, text="Max images (0=all):").pack(side="left")
    ttk.Spinbox(
        grp_box, textvariable=group_max_var, from_=0, to=10000, increment=10, width=7
    ).pack(side="left", padx=4)

    rb_canvas = ttk.Radiobutton(
        cls_frame,
        text="New image: currently open canvas image",
        variable=classify_mode_var,
        value="new_canvas",
    )
    rb_canvas.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
    if document.current is None:
        rb_canvas.configure(state="disabled")

    ttk.Radiobutton(
        cls_frame,
        text="New image: browse file…",
        variable=classify_mode_var,
        value="new_file",
    ).grid(row=3, column=0, sticky="w", pady=2)
    ttk.Entry(cls_frame, textvariable=custom_img_var, width=42).grid(
        row=3, column=1, sticky="ew", padx=6, pady=2
    )

    def _browse_query_img() -> None:
        chosen = filedialog.askopenfilename(
            parent=dialog, title="Select New Image to Classify", filetypes=IMAGE_TYPES
        )
        if chosen:
            custom_img_var.set(chosen)
            classify_mode_var.set("new_file")

    ttk.Button(cls_frame, text="Browse…", command=_browse_query_img).grid(
        row=3, column=2, sticky="w", pady=2
    )

    # Preview readout
    prev_bar = ttk.Frame(cls_frame)
    prev_bar.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 2))
    headline_font = tkfont.nametofont("TkDefaultFont").copy()
    headline_font.configure(weight="bold")

    def _build_params_from_ui() -> dict[str, Any]:
        mode = classify_mode_var.get()
        params: dict[str, Any] = {
            "model_source": model_source_var.get(),
            "model_dir": model_dir_var.get().strip(),
            "imagenet_arch": imagenet_arch_var.get(),
            "device": device_var.get(),
            "classify_mode": mode,
            "class_filter": class_filter_var.get(),
            "group_max_images": int(group_max_var.get()),
        }
        if mode == "held_out_individual":
            if not test_records:
                raise ValueError("Loaded model has no held-out test records.")
            idx = max(0, test_combo.current())
            params["test_record_index"] = idx
        elif mode == "new_canvas":
            if document.current is None:
                raise ValueError("No image is currently open on the canvas.")
            params["use_current_image"] = True
        elif mode == "new_file":
            path_txt = custom_img_var.get().strip()
            if not path_txt or not resolve_path(path_txt).is_file():
                raise ValueError("Please select a valid image file.")
            params["custom_image_path"] = path_txt
        return params

    def _on_preview_classify() -> None:
        try:
            if cached_bundle is None:
                _refresh_model()
            if cached_bundle is None:
                return
            params = _build_params_from_ui()
            out = run_with_progress(
                dialog,
                "CNN Classification",
                "Running CNN inference…",
                run_cnn_classification,
                params,
                document.current.copy() if document.current is not None else None,
                cached_bundle,
                False,
            )
            cls_res = out["classification"]
            if cls_res["mode"] == "group":
                pred_headline_var.set(
                    f"Group Accuracy: {cls_res['accuracy'] * 100:.2f}% "
                    f"({cls_res['correct_count']}/{cls_res['count']} correct)"
                )
                pred_dist_var.set(
                    f"Evaluated {cls_res['count']} held-out images in "
                    f"{cls_res['processing_time_ms']:.1f} ms."
                )
            else:
                true_str = (
                    f" | True: {cls_res['true_label']}"
                    if cls_res.get("true_label") is not None
                    else ""
                )
                pred_headline_var.set(
                    f"Prediction: {cls_res['predicted_class']} "
                    f"({cls_res['top_probability'] * 100:.1f}%){true_str}"
                )
                top_k = cls_res["class_probabilities"][:5]
                pred_dist_var.set(
                    "Top probabilities: "
                    + "  |  ".join(
                        f"{x['class_name']}: {x['probability'] * 100:.1f}%"
                        for x in top_k
                    )
                )
        except Exception as exc:
            messagebox.showerror("Classification Failed", str(exc), parent=dialog)

    ttk.Button(prev_bar, text="Classify (Preview)", command=_on_preview_classify).pack(
        side="left", padx=(0, 10)
    )
    ttk.Label(
        prev_bar,
        textvariable=pred_headline_var,
        font=headline_font,
        foreground="#2f855a",
    ).pack(side="left")
    ttk.Label(
        cls_frame, textvariable=pred_dist_var, foreground="#2d3748", wraplength=700
    ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(2, 0))

    _on_source_toggle()

    result: dict[str, Any] | None = None

    def _on_apply() -> None:
        nonlocal result
        try:
            result = _build_params_from_ui()
            dialog.destroy()
        except Exception as exc:
            messagebox.showerror("Invalid Selection", str(exc), parent=dialog)

    btns = ttk.Frame(main)
    btns.pack(fill="x")
    ttk.Button(btns, text="Classify & Render Result", command=_on_apply).pack(
        side="right", padx=(6, 0)
    )
    ttk.Button(btns, text="Cancel", command=dialog.destroy).pack(side="right")

    dialog.wait_window()
    return result


def run_cnn_classification(
    parameters: dict[str, Any],
    canvas_image: Image.Image | None = None,
    preloaded_bundle: dict[str, Any] | None = None,
    render: bool = True,
) -> dict[str, Any]:
    """Load model (if needed) and classify individual, group, or new image."""
    bundle = preloaded_bundle or _load_bundle_from_source(
        str(parameters.get("model_source", "saved")),
        str(parameters.get("model_dir", find_latest_cnn_model_dir())),
        str(parameters.get("imagenet_arch", "resnet50")),
        str(parameters.get("device", "auto")),
    )
    passport = bundle["passport"]
    mode = str(parameters.get("classify_mode", "new_canvas"))

    if mode == "held_out_group":
        cls_out = classify_held_out_group(
            bundle,
            class_filter=str(parameters.get("class_filter", "All")),
            max_images=int(parameters.get("group_max_images", 0)),
            render=render,
        )
        return {"bundle": bundle, "classification": cls_out}

    if mode == "held_out_individual":
        records = list(passport.get("dataset", {}).get("test_records", []))
        if not records:
            raise ValueError("Model passport contains no held-out test records.")
        idx = min(max(0, int(parameters.get("test_record_index", 0))), len(records) - 1)
        rec = records[idx]
        query_img = load_test_record_image(rec, passport)
        cls_out = classify_single_cnn_image(
            query_img, bundle, true_label=str(rec["class_name"]), render=render
        )
        cls_out["filename"] = rec.get("filename", "")
        return {"bundle": bundle, "classification": cls_out}

    if parameters.get("query_image") is not None:
        query_img = parameters["query_image"]
    elif mode == "new_file" and parameters.get("custom_image_path"):
        qpath = resolve_path(parameters["custom_image_path"])
        with Image.open(qpath) as im:
            query_img = im.copy()
    elif canvas_image is not None:
        query_img = canvas_image
    else:
        raise ValueError("No query image provided for classification.")

    cls_out = classify_single_cnn_image(query_img, bundle, render=render)
    return {"bundle": bundle, "classification": cls_out}


class CNNClassificationTool(ForensicsTool):
    """Set 8.2 — Load pretrained CNN, view model passport, and classify images."""

    tool_id = "cnn_classifier"
    title = "CNN Image Classification"
    category = "CNN Classification"
    description = (
        "Load a pretrained CNN model, inspect its passport, and classify a held-out "
        "test image, a held-out test group, or a completely new image."
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

        canvas_copy = document.current.copy() if document.current is not None else None
        out = run_with_progress(
            parent,
            "CNN Classification",
            "Loading CNN model and classifying…",
            run_cnn_classification,
            parameters,
            canvas_copy,
            None,
            True,
        )
        bundle = out["bundle"]
        passport = bundle["passport"]
        cls_out = out["classification"]
        vis_img = cls_out["visualization_image"]
        ensure_document_has_image(document, vis_img)

        arch = passport.get("architecture", {})
        ds = passport.get("dataset", {})
        prep = passport.get("preprocessing", {})

        details: dict[str, Any] = {
            "Model": passport.get("model_name"),
            "Model ID": passport.get("model_id"),
            "Architecture": arch.get("display_name", arch.get("arch_key")),
            "Dataset": ds.get("dataset_name"),
            "Preprocessing": f"{prep.get('input_size')} ({prep.get('normalization')})",
        }

        if cls_out["mode"] == "group":
            details["Evaluation mode"] = "Held-out test group"
            details["Images classified"] = cls_out["count"]
            details["Correct"] = f"{cls_out['correct_count']} / {cls_out['count']}"
            details["Group accuracy"] = f"{cls_out['accuracy'] * 100:.2f}%"
            details["Processing time"] = f"{cls_out['processing_time_ms']:.1f} ms"
            acc_pct = cls_out["accuracy"] * 100
            msg = (
                f"Classified {cls_out['count']} held-out test images with "
                f"{passport.get('model_name')}: accuracy = {acc_pct:.2f}%."
            )
        else:
            details["Predicted class"] = cls_out["predicted_class"]
            details["Confidence"] = f"{cls_out['top_probability'] * 100:.2f}%"
            if cls_out.get("true_label") is not None:
                details["True class"] = cls_out["true_label"]
                details["Match"] = "Correct" if cls_out["is_correct"] else "Incorrect"
            for item in cls_out["class_probabilities"][:5]:
                details[f"P({item['class_name']})"] = (
                    f"{item['probability'] * 100:.2f}%"
                )
            details["Processing time"] = f"{cls_out['processing_time_ms']:.1f} ms"
            top_pct = cls_out["top_probability"] * 100
            msg = (
                f"CNN predicted '{cls_out['predicted_class']}' "
                f"({top_pct:.1f}%) using {passport.get('model_name')}."
            )

        return ToolResult(image=vis_img, message=msg, details=details)
