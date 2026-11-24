"""Set 8.1 — Train Classification CNN, Save Weights & Model Passport.

Trains either from-scratch CNN architectures (Shallow CNN, Deeper VGG, Complete VGG)
or ImageNet-pretrained transfer-learning architectures (InceptionV3, ResNet50) using
local NVIDIA CUDA GPU acceleration when available, then saves ``weights.pt``,
``passport.json`` (dataset, sample size, class names, preprocessing instructions,
training history), and ``training_curves.png``.
"""

from __future__ import annotations

import io
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from forensics_app.core import ImageDocument
from forensics_app.paths import resolve_path
from forensics_app.utils.background import run_with_progress

from .base import ForensicsTool, ToolResult
from .cnn_common import (
    CNN_ARCHITECTURES,
    DEFAULT_CNN_ROOT,
    build_cnn_model,
    build_preprocessing_spec,
    describe_compute_device,
    get_compute_device,
    inspect_cnn_dataset,
    list_saved_cnn_models,
    prepare_cnn_dataset,
    resolve_arch_key,
    save_cnn_bundle,
)
from .codebook_construction import (
    DEFAULT_DATASET_DIR,
    ensure_document_has_image,
    portable_path,
)


def _next_default_model_dir() -> Path:
    """Return a clean non-clobbering default save directory under ``models/cnn``."""
    existing = {p.name for p in list_saved_cnn_models()}
    idx = 1
    while f"cnn_model_{idx:03d}" in existing:
        idx += 1
    return DEFAULT_CNN_ROOT / f"cnn_model_{idx:03d}"


def _make_optimizer(
    name: str,
    params: list[nn.Parameter],
    lr: float,
) -> torch.optim.Optimizer:
    opt_name = str(name).strip().lower()
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=float(lr))
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=float(lr), momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name!r}. Choose 'SGD' or 'Adam'.")


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader | None,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> tuple[float | None, float | None, np.ndarray, np.ndarray]:
    """Run forward evaluation without gradients; return (loss, acc, y_true, y_pred)."""
    if loader is None or len(loader.dataset) == 0:
        return (
            None,
            None,
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_count = 0
    y_true_all: list[torch.Tensor] = []
    y_pred_all: list[torch.Tensor] = []
    non_blocking = device.type == "cuda"
    amp_enabled = use_amp and device.type == "cuda"

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, dtype=torch.long, non_blocking=non_blocking)
            logits = model(inputs)
            loss = criterion(logits, targets)
            bs = int(targets.size(0))
            total_loss += loss.detach().to(dtype=torch.float64) * bs
            total_count += bs
            preds = torch.argmax(logits, dim=1)
            y_true_all.append(targets.detach())
            y_pred_all.append(preds.detach())

    if total_count == 0:
        return (
            None,
            None,
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    y_true = torch.cat(y_true_all).cpu().numpy()
    y_pred = torch.cat(y_pred_all).cpu().numpy()
    acc = float(accuracy_score(y_true, y_pred))
    return float(total_loss.item()) / total_count, acc, y_true, y_pred


def _run_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    phase_name: str,
    history: dict[str, list[Any]],
    *,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Execute ``epochs`` training passes and append metrics to ``history``."""
    non_blocking = device.type == "cuda"
    amp_enabled = use_amp and device.type == "cuda"
    for _ in range(int(epochs)):
        ep_t0 = time.perf_counter()
        model.train()
        running_loss = torch.zeros((), device=device, dtype=torch.float64)
        correct = torch.zeros((), device=device, dtype=torch.long)
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, dtype=torch.long, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = int(targets.size(0))
            running_loss += loss.detach().to(dtype=torch.float64) * bs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum()
            total += bs

        tr_loss = float(running_loss.item()) / max(1, total)
        tr_acc = float(correct.item()) / max(1, total)
        va_loss, va_acc, _, _ = _evaluate_loader(
            model, val_loader, criterion, device, use_amp=amp_enabled
        )
        ep_dt = time.perf_counter() - ep_t0

        prev_tr_acc = history["train_acc"][-1] if history["train_acc"] else None
        prev_va_acc = history["val_acc"][-1] if history["val_acc"] else None
        ep_num = len(history["epoch"]) + 1

        history["epoch"].append(ep_num)
        history["phase"].append(phase_name)
        history["train_loss"].append(round(float(tr_loss), 6))
        history["train_acc"].append(round(float(tr_acc), 6))
        history["val_loss"].append(
            round(float(va_loss), 6) if va_loss is not None else None
        )
        history["val_acc"].append(
            round(float(va_acc), 6) if va_acc is not None else None
        )

        tr_gain = (
            f" ({(tr_acc - prev_tr_acc) * 100:+.2f}%)"
            if prev_tr_acc is not None
            else ""
        )
        val_str = ""
        if va_loss is not None and va_acc is not None:
            va_gain = (
                f" ({(va_acc - prev_va_acc) * 100:+.2f}%)"
                if prev_va_acc is not None
                else ""
            )
            val_str = f" | val_loss={va_loss:.4f} val_acc={va_acc * 100:.2f}%{va_gain}"
        print(
            f"  [Epoch {ep_num:02d} | {phase_name}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc * 100:.2f}%{tr_gain}"
            f"{val_str} | {ep_dt:.2f}s",
            flush=True,
        )


def render_training_summary_image(passport: dict[str, Any]) -> Image.Image:
    """Render training/validation curves and test confusion matrix as a PIL image."""
    hist = passport.get("training_history", {})
    ev = passport.get("evaluation", {})
    arch = passport.get("architecture", {})
    cfg = passport.get("training_config", {})
    ds = passport.get("dataset", {})
    class_names = list(passport.get("class_names", []))

    epochs = hist.get("epoch", [])
    tr_loss = hist.get("train_loss", [])
    va_loss = [v if v is not None else np.nan for v in hist.get("val_loss", [])]
    tr_acc = [v * 100.0 for v in hist.get("train_acc", [])]
    va_acc = [(v * 100.0) if v is not None else np.nan for v in hist.get("val_acc", [])]
    phases = hist.get("phase", [])

    fig = Figure(figsize=(13.2, 4.6), dpi=115)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    if epochs:
        ax1.plot(
            epochs, tr_loss, marker="o", markersize=3, color="#2b6cb0", label="Train"
        )
        if any(np.isfinite(va_loss)):
            ax1.plot(
                epochs, va_loss, marker="s", markersize=3, color="#dd6b20", label="Val"
            )
        ax2.plot(
            epochs, tr_acc, marker="o", markersize=3, color="#2f855a", label="Train"
        )
        if any(np.isfinite(va_acc)):
            ax2.plot(
                epochs, va_acc, marker="s", markersize=3, color="#d53f8c", label="Val"
            )

        if "fine_tune" in phases:
            ft_start = epochs[phases.index("fine_tune")] - 0.5
            ax1.axvline(
                ft_start,
                color="#718096",
                linestyle="--",
                linewidth=1.0,
                label="Fine-tune",
            )
            ax2.axvline(
                ft_start,
                color="#718096",
                linestyle="--",
                linewidth=1.0,
                label="Fine-tune",
            )

    ax1.set_title("Cross-Entropy Loss", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)
    if epochs:
        ax1.legend(fontsize=8)

    ax2.set_title("Classification Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.25)
    if epochs:
        ax2.legend(fontsize=8)

    cm = np.asarray(ev.get("confusion_matrix", []), dtype=np.int64)
    test_acc = ev.get("test_accuracy")
    acc_str = f"{test_acc * 100:.1f}%" if test_acc is not None else "N/A"
    if cm.ndim == 2 and cm.size > 0:
        im = ax3.imshow(cm, cmap="Blues", interpolation="nearest")
        ax3.set_title(
            f"Test Confusion Matrix (Acc: {acc_str})", fontsize=10, fontweight="bold"
        )
        n_cls = cm.shape[0]
        ticks = np.arange(n_cls)
        short_names = [
            (n[:10] + "…" if len(n) > 11 else n) for n in class_names[:n_cls]
        ]
        ax3.set_xticks(ticks)
        ax3.set_yticks(ticks)
        ax3.set_xticklabels(short_names, rotation=35, ha="right", fontsize=7.5)
        ax3.set_yticklabels(short_names, fontsize=7.5)
        ax3.set_ylabel("True")
        ax3.set_xlabel("Predicted")
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        if n_cls <= 12:
            for r in range(n_cls):
                for c in range(n_cls):
                    ax3.text(
                        c,
                        r,
                        str(cm[r, c]),
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white" if cm[r, c] > thresh else "black",
                    )
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, "No test confusion matrix", ha="center", va="center")
        ax3.axis("off")

    dev_label = cfg.get("gpu_name") or cfg.get("device_used", "cpu")
    fig.suptitle(
        f"{arch.get('display_name', 'CNN')} | Dataset: {ds.get('dataset_name', '—')} | "
        f"Device: {dev_label} | Time: {cfg.get('elapsed_seconds', 0):.1f}s",
        fontsize=10.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def train_cnn_model(parameters: dict[str, Any]) -> dict[str, Any]:
    """Train a classification CNN, evaluate on held-out test set, and save passport."""
    arch_key = resolve_arch_key(parameters.get("arch_key", "deeper_vgg"))
    spec = CNN_ARCHITECTURES[arch_key]
    input_size = int(parameters.get("input_size", spec.default_input_size))
    normalization = str(parameters.get("normalization", spec.default_normalization))
    source_type = str(parameters.get("source_type", "image_folder"))
    dataset_dir = parameters.get("dataset_dir", DEFAULT_DATASET_DIR)
    test_size = float(parameters.get("test_size", 0.2))
    val_split = float(parameters.get("val_split", 0.1))
    max_samples_per_class = parameters.get("max_samples_per_class")
    if max_samples_per_class is not None and int(max_samples_per_class) <= 0:
        max_samples_per_class = None

    epochs = max(1, int(parameters.get("epochs", 15)))
    fine_tune_epochs = max(0, int(parameters.get("fine_tune_epochs", 0)))
    batch_size = max(2, int(parameters.get("batch_size", 32)))
    optimizer_name = str(parameters.get("optimizer", spec.default_optimizer))
    learning_rate = float(parameters.get("learning_rate", spec.default_lr))
    fine_tune_lr = float(parameters.get("fine_tune_lr", 1e-4))
    pretrained = bool(parameters.get("pretrained", spec.is_transfer_learning))
    freeze_backbone = bool(parameters.get("freeze_backbone", spec.is_transfer_learning))
    seed = int(parameters.get("random_state", 42))
    device_pref = str(parameters.get("device", "auto"))
    model_dir = resolve_path(parameters.get("model_dir") or _next_default_model_dir())

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, device_info = get_compute_device(device_pref)
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    preprocessing = build_preprocessing_spec(
        input_size=input_size, normalization=normalization
    )

    ds_bundle = prepare_cnn_dataset(
        source_type=source_type,
        dataset_dir=dataset_dir,
        test_size=test_size,
        val_split=val_split,
        random_state=seed,
        max_samples_per_class=max_samples_per_class,
        preprocessing=preprocessing,
    )
    class_names = list(ds_bundle["class_names"])
    num_classes = len(class_names)

    train_ds = ds_bundle["train_dataset"]
    val_ds = ds_bundle["val_dataset"]
    test_ds = ds_bundle["test_dataset"]
    if len(train_ds) == 0:
        raise ValueError("Training split is empty.")

    pin_mem = device.type == "cuda"
    eff_batch = min(batch_size, len(train_ds))
    # Avoid trailing batch of size 1 which fails BatchNorm2d in training mode.
    drop_last = len(train_ds) > eff_batch and (len(train_ds) % eff_batch == 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=eff_batch,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=pin_mem,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem
    )

    model = build_cnn_model(
        arch_key,
        num_classes,
        input_size=input_size,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[Any]] = {
        "epoch": [],
        "phase": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    t0 = time.perf_counter()
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params_list:
        for p in model.parameters():
            p.requires_grad = True
        trainable_params_list = list(model.parameters())

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in trainable_params_list)
    dev_label = device_info["gpu_name"] or device_info["device"]
    print(
        "[CNNTrainer] Training parameters:\n"
        f"  arch={spec.display_name} ({arch_key}) | input={input_size}x{input_size} | "
        f"norm={normalization} | params={n_trainable_params:,}/{n_total_params:,} trainable\n"
        f"  device={device_info['device']} ({dev_label}) | amp={use_amp} | "
        f"cudnn_benchmark={bool(use_amp and torch.backends.cudnn.benchmark)}\n"
        f"  dataset={ds_bundle['dataset_name']} (train={len(train_ds)}, "
        f"val={len(val_ds) if val_ds is not None else 0}, test={len(test_ds)}, "
        f"classes={num_classes})\n"
        f"  epochs={epochs}"
        + (
            f" + fine_tune={fine_tune_epochs} (ft_lr={fine_tune_lr})"
            if spec.is_transfer_learning and fine_tune_epochs > 0
            else ""
        )
        + f" | batch_size={eff_batch} | optimizer={optimizer_name} | lr={learning_rate} | "
        f"seed={seed}\n"
        f"  output_dir={model_dir}",
        flush=True,
    )

    optimizer = _make_optimizer(optimizer_name, trainable_params_list, learning_rate)
    _run_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs,
        "train",
        history,
        use_amp=use_amp,
        scaler=scaler,
    )

    if spec.is_transfer_learning and fine_tune_epochs > 0:
        if hasattr(model, "set_backbone_trainable"):
            model.set_backbone_trainable(True)
        if hasattr(model, "set_head_trainable"):
            model.set_head_trainable(True)
        ft_params = [p for p in model.parameters() if p.requires_grad]
        ft_opt = _make_optimizer("Adam", ft_params, fine_tune_lr)
        _run_epochs(
            model,
            train_loader,
            val_loader,
            criterion,
            ft_opt,
            device,
            fine_tune_epochs,
            "fine_tune",
            history,
            use_amp=use_amp,
            scaler=scaler,
        )

    elapsed = time.perf_counter() - t0

    _, train_eval_acc, _, _ = _evaluate_loader(
        model, train_loader, criterion, device, use_amp=use_amp
    )
    _, val_eval_acc, _, _ = _evaluate_loader(
        model, val_loader, criterion, device, use_amp=use_amp
    )
    test_loss, test_acc, y_true, y_pred = _evaluate_loader(
        model, test_loader, criterion, device, use_amp=use_amp
    )

    cm = (
        confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
        if len(y_true) > 0
        else []
    )
    per_class_acc: dict[str, float] = {}
    for idx, cname in enumerate(class_names):
        mask = y_true == idx
        if np.any(mask):
            per_class_acc[cname] = round(float(np.mean(y_pred[mask] == idx)), 4)

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    passport: dict[str, Any] = {
        "passport_version": 1,
        "model_name": model_dir.name,
        "framework": "pytorch",
        "architecture": {
            "arch_key": arch_key,
            "display_name": spec.display_name,
            "num_classes": num_classes,
            "pretrained_backbone": pretrained if spec.is_transfer_learning else False,
            "freeze_backbone": freeze_backbone if spec.is_transfer_learning else False,
            "feature_layer": spec.feature_layer,
            "feature_dim": spec.feature_dim,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "dataset": {
            "dataset_name": ds_bundle["dataset_name"],
            "source_type": ds_bundle["source_type"],
            "dataset_dir": ds_bundle["dataset_dir"],
            "split_mode": ds_bundle["split_mode"],
            "sample_size": ds_bundle["sample_size"],
            "train_records": ds_bundle["train_records"],
            "val_records": ds_bundle["val_records"],
            "test_records": ds_bundle["test_records"],
        },
        "class_names": class_names,
        "preprocessing": preprocessing,
        "training_config": {
            "epochs": epochs,
            "fine_tune_epochs": fine_tune_epochs if spec.is_transfer_learning else 0,
            "batch_size": eff_batch,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "fine_tune_lr": fine_tune_lr if spec.is_transfer_learning else None,
            "loss_function": "CrossEntropyLoss",
            "val_split": val_split,
            "test_size": test_size,
            "max_samples_per_class": max_samples_per_class,
            "random_state": seed,
            "device_requested": device_pref,
            "device_used": device_info["device"],
            "gpu_name": device_info["gpu_name"],
            "vram_mb": device_info["vram_mb"],
            "torch_version": device_info["torch_version"],
            "cuda_version": device_info["cuda_version"],
            "elapsed_seconds": round(elapsed, 3),
        },
        "training_history": history,
        "evaluation": {
            "train_accuracy": round(float(train_eval_acc or 0.0), 6),
            "val_accuracy": (
                round(float(val_eval_acc), 6) if val_eval_acc is not None else None
            ),
            "test_accuracy": round(float(test_acc), 6)
            if test_acc is not None
            else None,
            "test_loss": round(float(test_loss), 6) if test_loss is not None else None,
            "confusion_matrix": cm,
            "per_class_accuracy": per_class_acc,
        },
    }

    saved_dir = save_cnn_bundle(model, passport, model_dir)
    saved_passport = json_load(saved_dir / "passport.json")
    summary_img = render_training_summary_image(saved_passport)
    summary_img.save(saved_dir / "training_curves.png")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_dir": str(saved_dir),
        "passport": saved_passport,
        "summary_image": summary_img,
    }


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def ask_cnn_training_parameters(parent: tk.Misc) -> dict[str, Any] | None:
    """Open the CNN training configuration dialog."""
    dialog = tk.Toplevel(parent)
    dialog.title("Set 8.1 — Train Classification CNN & Save Passport")
    dialog.transient(parent.winfo_toplevel())
    dialog.grab_set()
    dialog.resizable(False, False)

    device_var = tk.StringVar(value="auto")
    device_status_var = tk.StringVar(value=describe_compute_device("auto"))

    source_var = tk.StringVar(value="image_folder")
    dataset_dir_var = tk.StringVar(value=portable_path(DEFAULT_DATASET_DIR))
    max_per_class_var = tk.IntVar(value=0)
    test_size_var = tk.DoubleVar(value=0.2)
    val_split_var = tk.DoubleVar(value=0.1)
    dataset_status_var = tk.StringVar(value="")

    arch_display_var = tk.StringVar(value=CNN_ARCHITECTURES["deeper_vgg"].display_name)
    input_size_var = tk.IntVar(value=32)
    norm_var = tk.StringVar(value="unit_0_1")
    pretrained_var = tk.BooleanVar(value=True)
    freeze_backbone_var = tk.BooleanVar(value=True)

    epochs_var = tk.IntVar(value=15)
    ft_epochs_var = tk.IntVar(value=0)
    batch_size_var = tk.IntVar(value=128)
    optimizer_var = tk.StringVar(value="SGD")
    lr_var = tk.DoubleVar(value=0.01)
    ft_lr_var = tk.DoubleVar(value=0.0001)
    seed_var = tk.IntVar(value=42)
    model_dir_var = tk.StringVar(value=portable_path(_next_default_model_dir()))

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    # 1. Hardware / GPU
    hw_frame = ttk.LabelFrame(
        main, text="1. Compute Device (Local GPU / CPU)", padding=8
    )
    hw_frame.pack(fill="x", pady=(0, 8))
    ttk.Label(hw_frame, textvariable=device_status_var, foreground="#2f855a").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
    )
    ttk.Label(hw_frame, text="Device:").grid(row=1, column=0, sticky="w")
    dev_combo = ttk.Combobox(
        hw_frame,
        textvariable=device_var,
        values=("auto", "cuda", "cpu"),
        state="readonly",
        width=14,
    )
    dev_combo.grid(row=1, column=1, sticky="w", padx=6)

    def _on_device_change(*_args: object) -> None:
        device_status_var.set(describe_compute_device(device_var.get()))

    dev_combo.bind("<<ComboboxSelected>>", _on_device_change)

    # 2. Dataset
    ds_frame = ttk.LabelFrame(main, text="2. Dataset & Split", padding=8)
    ds_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(ds_frame, text="Source:").grid(row=0, column=0, sticky="w", pady=2)
    src_row = ttk.Frame(ds_frame)
    src_row.grid(row=0, column=1, columnspan=3, sticky="w", pady=2)

    ds_entry = ttk.Entry(ds_frame, textvariable=dataset_dir_var, width=42)
    ds_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=2)

    def _inspect_ds() -> None:
        try:
            info = inspect_cnn_dataset(source_var.get(), dataset_dir_var.get().strip())
            classes = info["class_names"]
            if info["split_mode"] == "pre_split":
                tr = sum(info["train_counts"].values())
                te = sum(info["test_counts"].values())
                counts_str = f"train={tr}, test={te}"
            else:
                tot = sum(info["total_counts"].values())
                counts_str = f"total={tot} (stratified split)"
            cls_preview = ", ".join(classes[:5])
            dataset_status_var.set(
                f"{info['dataset_name']}: {len(classes)} classes ({cls_preview}"
                f"{'…' if len(classes) > 5 else ''}) | {counts_str}"
            )
        except Exception as exc:
            dataset_status_var.set(f"Dataset error: {exc}")

    def _on_source_change() -> None:
        is_cifar = source_var.get() == "cifar10"
        ds_entry.configure(state="disabled" if is_cifar else "normal")
        browse_ds_btn.configure(state="disabled" if is_cifar else "normal")
        if is_cifar and max_per_class_var.get() == 0:
            max_per_class_var.set(500)
        elif not is_cifar and max_per_class_var.get() == 500:
            max_per_class_var.set(0)
        _inspect_ds()

    ttk.Radiobutton(
        src_row,
        text="Image Folder",
        variable=source_var,
        value="image_folder",
        command=_on_source_change,
    ).pack(side="left", padx=(0, 10))
    ttk.Radiobutton(
        src_row,
        text="CIFAR-10 (auto-download)",
        variable=source_var,
        value="cifar10",
        command=_on_source_change,
    ).pack(side="left")

    ttk.Label(ds_frame, text="Dataset folder:").grid(
        row=1, column=0, sticky="w", pady=2
    )

    def _browse_ds() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select Dataset Directory"
        )
        if chosen:
            dataset_dir_var.set(chosen)
            _inspect_ds()

    browse_ds_btn = ttk.Button(ds_frame, text="Browse…", command=_browse_ds)
    browse_ds_btn.grid(row=1, column=3, sticky="w", pady=2)

    ttk.Label(ds_frame, text="Max/class (0=all):").grid(
        row=2, column=0, sticky="w", pady=2
    )
    ttk.Spinbox(
        ds_frame,
        textvariable=max_per_class_var,
        from_=0,
        to=5000,
        increment=50,
        width=8,
    ).grid(row=2, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(ds_frame, text="Test / Val split:").grid(
        row=2, column=2, sticky="e", pady=2
    )
    split_box = ttk.Frame(ds_frame)
    split_box.grid(row=2, column=3, sticky="w", pady=2)
    ttk.Spinbox(
        split_box,
        textvariable=test_size_var,
        from_=0.05,
        to=0.5,
        increment=0.05,
        width=5,
    ).pack(side="left")
    ttk.Label(split_box, text=" / ").pack(side="left")
    ttk.Spinbox(
        split_box,
        textvariable=val_split_var,
        from_=0.0,
        to=0.4,
        increment=0.05,
        width=5,
    ).pack(side="left")

    ttk.Label(ds_frame, textvariable=dataset_status_var, foreground="#2b6cb0").grid(
        row=3, column=0, columnspan=4, sticky="w", pady=(2, 0)
    )

    # 3. Architecture & Preprocessing
    arch_frame = ttk.LabelFrame(
        main, text="3. CNN Architecture & Preprocessing", padding=8
    )
    arch_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(arch_frame, text="Architecture:").grid(
        row=0, column=0, sticky="w", pady=2
    )
    arch_combo = ttk.Combobox(
        arch_frame,
        textvariable=arch_display_var,
        values=[spec.display_name for spec in CNN_ARCHITECTURES.values()],
        state="readonly",
        width=36,
    )
    arch_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=2)

    ttk.Label(arch_frame, text="Input size (px):").grid(
        row=1, column=0, sticky="w", pady=2
    )
    ttk.Combobox(
        arch_frame,
        textvariable=input_size_var,
        values=(32, 64, 96, 128, 224, 299),
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(arch_frame, text="Normalization:").grid(
        row=1, column=2, sticky="e", pady=2
    )
    ttk.Combobox(
        arch_frame,
        textvariable=norm_var,
        values=("unit_0_1", "imagenet"),
        state="readonly",
        width=12,
    ).grid(row=1, column=3, sticky="w", padx=6, pady=2)

    chk_pretrained = ttk.Checkbutton(
        arch_frame,
        text="Use ImageNet pretrained backbone (Transfer Learning)",
        variable=pretrained_var,
    )
    chk_pretrained.grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
    chk_freeze = ttk.Checkbutton(
        arch_frame,
        text="Freeze backbone weights",
        variable=freeze_backbone_var,
    )
    chk_freeze.grid(row=2, column=2, columnspan=2, sticky="w", pady=2)

    # 4. Training Hyperparameters & Output
    hp_frame = ttk.LabelFrame(
        main, text="4. Training Hyperparameters & Output Folder", padding=8
    )
    hp_frame.pack(fill="x", pady=(0, 10))

    ttk.Label(hp_frame, text="Epochs:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Spinbox(
        hp_frame, textvariable=epochs_var, from_=1, to=512, increment=5, width=7
    ).grid(row=0, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(hp_frame, text="Batch size:").grid(row=0, column=2, sticky="w", pady=2)
    ttk.Spinbox(
        hp_frame, textvariable=batch_size_var, from_=4, to=512, increment=16, width=7
    ).grid(row=0, column=3, sticky="w", padx=6, pady=2)

    ttk.Label(hp_frame, text="Optimizer:").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Combobox(
        hp_frame,
        textvariable=optimizer_var,
        values=("SGD", "Adam"),
        state="readonly",
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(hp_frame, text="Learning rate:").grid(row=1, column=2, sticky="w", pady=2)
    ttk.Entry(hp_frame, textvariable=lr_var, width=9).grid(
        row=1, column=3, sticky="w", padx=6, pady=2
    )

    ttk.Label(hp_frame, text="Fine-tune epochs:").grid(
        row=2, column=0, sticky="w", pady=2
    )
    ft_ep_spin = ttk.Spinbox(
        hp_frame, textvariable=ft_epochs_var, from_=0, to=100, increment=1, width=7
    )
    ft_ep_spin.grid(row=2, column=1, sticky="w", padx=6, pady=2)

    ttk.Label(hp_frame, text="Fine-tune LR:").grid(row=2, column=2, sticky="w", pady=2)
    ft_lr_entry = ttk.Entry(hp_frame, textvariable=ft_lr_var, width=9)
    ft_lr_entry.grid(row=2, column=3, sticky="w", padx=6, pady=2)

    ttk.Label(hp_frame, text="Save model dir:").grid(
        row=3, column=0, sticky="w", pady=2
    )
    ttk.Entry(hp_frame, textvariable=model_dir_var, width=34).grid(
        row=3, column=1, columnspan=2, sticky="ew", padx=6, pady=2
    )

    def _browse_out() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select Model Output Folder"
        )
        if chosen:
            model_dir_var.set(chosen)

    ttk.Button(hp_frame, text="Browse…", command=_browse_out).grid(
        row=3, column=3, sticky="w", pady=2
    )

    def _on_arch_change(*_args: object) -> None:
        spec = CNN_ARCHITECTURES[resolve_arch_key(arch_display_var.get())]
        input_size_var.set(spec.default_input_size)
        norm_var.set(spec.default_normalization)
        optimizer_var.set(spec.default_optimizer)
        lr_var.set(spec.default_lr)
        state = "normal" if spec.is_transfer_learning else "disabled"
        chk_pretrained.configure(state=state)
        chk_freeze.configure(state=state)
        ft_ep_spin.configure(state=state)
        ft_lr_entry.configure(state=state)
        if spec.is_transfer_learning:
            epochs_var.set(8)
        else:
            ft_epochs_var.set(0)
            epochs_var.set(20)

    arch_combo.bind("<<ComboboxSelected>>", _on_arch_change)
    _on_arch_change()
    _inspect_ds()

    result: dict[str, Any] | None = None

    def _on_train() -> None:
        nonlocal result
        try:
            arch_key = resolve_arch_key(arch_display_var.get())
            spec = CNN_ARCHITECTURES[arch_key]
            in_sz = int(input_size_var.get())
            if in_sz < spec.min_input_size:
                raise ValueError(
                    f"{spec.display_name} requires input size >= {spec.min_input_size}."
                )
            inspect_cnn_dataset(source_var.get(), dataset_dir_var.get().strip())
            result = {
                "device": device_var.get(),
                "source_type": source_var.get(),
                "dataset_dir": dataset_dir_var.get().strip(),
                "max_samples_per_class": int(max_per_class_var.get()),
                "test_size": float(test_size_var.get()),
                "val_split": float(val_split_var.get()),
                "arch_key": arch_key,
                "input_size": in_sz,
                "normalization": norm_var.get(),
                "pretrained": bool(pretrained_var.get()),
                "freeze_backbone": bool(freeze_backbone_var.get()),
                "epochs": int(epochs_var.get()),
                "fine_tune_epochs": int(ft_epochs_var.get()),
                "batch_size": int(batch_size_var.get()),
                "optimizer": optimizer_var.get(),
                "learning_rate": float(lr_var.get()),
                "fine_tune_lr": float(ft_lr_var.get()),
                "random_state": int(seed_var.get()),
                "model_dir": model_dir_var.get().strip(),
            }
            dialog.destroy()
        except Exception as exc:
            messagebox.showerror("Invalid Configuration", str(exc), parent=dialog)

    btns = ttk.Frame(main)
    btns.pack(fill="x")
    ttk.Button(btns, text="Train & Save Model + Passport", command=_on_train).pack(
        side="right", padx=(6, 0)
    )
    ttk.Button(btns, text="Cancel", command=dialog.destroy).pack(side="right")

    dialog.wait_window()
    return result


class CNNTrainerTool(ForensicsTool):
    """Set 8.1 — Train Classification CNN and save weights + model passport."""

    tool_id = "cnn_trainer"
    title = "Train Classification CNN"
    category = "CNN Classification"
    description = (
        "Train a CNN (from scratch or InceptionV3/ResNet50 transfer learning) on "
        "NVIDIA GPU or CPU and save weights.pt + passport.json."
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
            parameters = ask_cnn_training_parameters(parent)
            if parameters is None:
                return None

        out = run_with_progress(
            parent,
            "CNN Training",
            "Training classification CNN, evaluating on held-out test set, "
            "and saving weights & model passport…",
            train_cnn_model,
            parameters,
        )
        passport = out["passport"]
        arch = passport["architecture"]
        ds = passport["dataset"]
        sample = ds["sample_size"]
        cfg = passport["training_config"]
        ev = passport["evaluation"]
        vis_img = out["summary_image"]
        ensure_document_has_image(document, vis_img)

        te_acc = ev.get("test_accuracy")
        te_str = f"{te_acc * 100:.2f}%" if te_acc is not None else "N/A"
        tr_acc = ev.get("train_accuracy", 0.0)
        dev_str = cfg.get("gpu_name") or cfg.get("device_used")

        details = {
            "Model ID": passport.get("model_id"),
            "Saved to": portable_path(out["model_dir"]),
            "Architecture": arch["display_name"],
            "Device": dev_str,
            "Dataset": ds["dataset_name"],
            "Sample size": (
                f"total={sample['total']} (train={sample['train']}, "
                f"val={sample['val']}, test={sample['test']})"
            ),
            "Classes": ", ".join(passport["class_names"]),
            "Preprocessing": passport["preprocessing"]["instructions"],
            "Epochs trained": len(passport["training_history"].get("epoch", [])),
            "Train accuracy": f"{tr_acc * 100:.2f}%",
            "Test accuracy": te_str,
            "Training time": f"{cfg['elapsed_seconds']:.2f} s",
        }
        return ToolResult(
            image=vis_img,
            message=(
                f"Trained {arch['display_name']} on {ds['dataset_name']} "
                f"({dev_str}) — Test accuracy: {te_str}."
            ),
            details=details,
        )
