"""Set 11 — U-Net semantic segmentation of hospital meal trays (TrayDataset).

Shared by the app tool and ``notebooks/train_unet.ipynb``: the U-Net from the
course notebook, data loading, the Dice coefficient, training, saving/loading a
model folder and predicting a mask for any image.

Two tasks:

* ``binary``: food vs. not food (classes 5+ are food), as in the notebook,
  with BCE + soft Dice loss.
* ``multiclass``: all 43 classes (as on the slide), with cross-entropy + soft
  Dice loss (notebook exercise 6).

A model folder holds ``unet.pt`` (the state dict) and ``passport.json`` (task,
classes, input size, architecture, training settings and metrics).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from forensics_app.paths import DATASETS_DIR, MODELS_DIR, PROJECT_ROOT

UNET_MODELS_DIR = MODELS_DIR / "unet"
TRAY_DATASET_DIR = DATASETS_DIR / "tray" / "TrayDataset"
SAMPLES_DIR = UNET_MODELS_DIR / "tray_samples"
WEIGHTS_FILE = "unet.pt"
PASSPORT_FILE = "passport.json"

TASK_BINARY = "binary"
TASK_MULTICLASS = "multiclass"
TASKS = (TASK_BINARY, TASK_MULTICLASS)

CLASS_NAMES = [
    "background", "tray", "cutlery", "form", "straw", "meatball", "beef",
    "roastlamb", "beeftomatocasserole", "ham", "bean", "cucumber", "leaf",
    "tomato", "boiledrice", "beefmexicanmeatballs", "spinachandpumpkinrisotto",
    "bakedfish", "gravy", "zucchini", "carrot", "broccoli", "pumpkin", "celery",
    "sandwich", "sidesalad", "tartaresauce", "jacketpotato", "creamedpotato",
    "bread", "margarine", "soup", "apple", "cannedfruit", "milk",
    "vanillayogurt", "jelly", "custard", "lemonsponge", "juice", "applejuice",
    "orangejuice", "water",
]  # fmt: skip
BINARY_NAMES = ["not food", "food"]
FIRST_FOOD = 5  # class index of 'meatball', the first food class

# Notebook settings for the binary task; the multi-class task keeps the full
# 256 × 416 resolution of the slide and a wider network for 43 classes.
TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    TASK_BINARY: {
        "size": (128, 208),
        "base": 16,
        "epochs": 15,
        "batch": 16,
        "lr": 1e-3,
    },
    TASK_MULTICLASS: {
        "size": (256, 416),
        "base": 32,
        "epochs": 30,
        "batch": 8,
        "lr": 1e-3,
    },
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def find_dir(root: str | Path, name: str) -> Path:
    """Return the first sub-folder called ``name`` anywhere inside ``root``."""
    root = Path(root)
    if root.name == name and root.is_dir():
        return root
    for p in sorted(root.rglob(name)):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No folder called {name} inside {root}")


def list_images(folder: str | Path) -> list[Path]:
    return sorted(
        p
        for p in Path(folder).iterdir()
        if p.suffix.lower() in IMG_EXT and not p.name.startswith(".")
    )


def pair_files(img_dir: str | Path, mask_dir: str | Path) -> list[tuple[Path, Path]]:
    """Match each image with its mask using the file name (without extension)."""
    imgs = list_images(img_dir)
    masks = {p.stem: p for p in list_images(mask_dir)}
    pairs = [(p, masks[p.stem]) for p in imgs if p.stem in masks]
    if not pairs:  # names differ -> fall back to sorted order
        pairs = list(zip(imgs, list_images(mask_dir)))
    return pairs


def read_mask(path: str | Path) -> np.ndarray:
    """Class index per pixel (some masks are saved with 3 identical channels)."""
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m


def num_classes(task: str) -> int:
    return 2 if task == TASK_BINARY else len(CLASS_NAMES)


def class_names(task: str) -> list[str]:
    return list(BINARY_NAMES if task == TASK_BINARY else CLASS_NAMES)


def to_task(mask: np.ndarray, task: str) -> np.ndarray:
    """Map raw class indices to the task's labels."""
    if task == TASK_BINARY:
        return (mask >= FIRST_FOOD).astype(np.uint8)
    return np.clip(mask, 0, len(CLASS_NAMES) - 1).astype(np.uint8)


def load_pairs(
    pairs: list[tuple[Path, Path]], size: tuple[int, int], task: str
) -> tuple[np.ndarray, np.ndarray]:
    """Resized images ``(N, H, W, 3)`` and task labels ``(N, H, W)`` in memory."""
    h, w = size
    X = np.zeros((len(pairs), h, w, 3), dtype=np.uint8)
    Y = np.zeros((len(pairs), h, w), dtype=np.uint8)
    for i, (ip, mp) in enumerate(pairs):
        X[i] = np.array(Image.open(ip).convert("RGB").resize((w, h), Image.BILINEAR))
        # NEAREST interpolation is essential for masks: no invented class values
        mask = Image.fromarray(read_mask(mp)).resize((w, h), Image.NEAREST)
        Y[i] = to_task(np.array(mask), task)
    return X, Y


def tray_splits(root: str | Path) -> dict[str, list[tuple[Path, Path]]]:
    """Image/mask pairs of the dataset's training and test folders."""
    return {
        "train": pair_files(find_dir(root, "XTrain"), find_dir(root, "yTrain")),
        "test": pair_files(find_dir(root, "XTest"), find_dir(root, "yTest")),
    }


def make_dataset(X: np.ndarray, Y: np.ndarray, task: str, augment: bool) -> Any:
    """The notebook's ``TrayDataset``: tensors in [0, 1], random horizontal flip."""
    import torch
    from torch.utils.data import Dataset

    class TrayDataset(Dataset):
        def __len__(self) -> int:
            return len(X)

        def __getitem__(self, i: int):
            x = torch.from_numpy(X[i]).permute(2, 0, 1).float() / 255.0
            if task == TASK_BINARY:
                y = torch.from_numpy(Y[i]).float().unsqueeze(0)  # (1, H, W)
            else:
                y = torch.from_numpy(Y[i]).long()  # (H, W)
            if augment and random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            return x, y

    return TrayDataset()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_unet(in_ch: int = 3, out_ch: int = 1, base: int = 16) -> Any:
    """The notebook's U-Net: 3 poolings, so H and W must be divisible by 8."""
    import torch
    import torch.nn as nn

    def conv_block(cin: int, cout: int) -> nn.Sequential:
        # Two 3x3 convolutions, each followed by BatchNorm and ReLU
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
            # Encoder
            self.enc1 = conv_block(in_ch, c1)
            self.enc2 = conv_block(c1, c2)
            self.enc3 = conv_block(c2, c3)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = conv_block(c3, c4)
            # Decoder: upsample, concatenate the skip connection, then convolve
            self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
            self.dec3 = conv_block(c3 * 2, c3)
            self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
            self.dec2 = conv_block(c2 * 2, c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
            self.dec1 = conv_block(c1 * 2, c1)
            self.head = nn.Conv2d(c1, out_ch, 1)  # one score per pixel and class

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    return UNet()


def output_channels(task: str) -> int:
    return 1 if task == TASK_BINARY else len(CLASS_NAMES)


def count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


def available_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(device: str) -> str:
    return available_device() if device in ("", "auto") else device


# ---------------------------------------------------------------------------
# Dice and losses
# ---------------------------------------------------------------------------


def dice_coef(pred: Any, target: Any, eps: float = 1e-6) -> Any:
    """Dice per image for ``(B, 1, H, W)`` tensors with values in [0, 1]."""
    inter = (pred * target).sum(dim=(1, 2, 3))
    total = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * inter + eps) / (total + eps)


def binary_loss(logits: Any, target: Any) -> Any:
    """BCE + (1 − soft Dice), as in the notebook."""
    import torch
    import torch.nn.functional as F

    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + (1 - dice_coef(probs, target).mean())


def multiclass_loss(logits: Any, target: Any, eps: float = 1e-6) -> Any:
    """Cross-entropy + (1 − soft Dice averaged over the classes in the batch)."""
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, target)
    probs = logits.softmax(dim=1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(0, 2, 3))
    total = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (total + eps)
    present = onehot.sum(dim=(0, 2, 3)) > 0
    return ce + (1 - dice[present].mean())


def loss_function(task: str) -> Callable[[Any, Any], Any]:
    return binary_loss if task == TASK_BINARY else multiclass_loss


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Hard Dice of two binary masks; 1.0 when both are empty."""
    pred, target = pred.astype(bool), target.astype(bool)
    total = pred.sum() + target.sum()
    return 1.0 if total == 0 else float(2 * (pred & target).sum() / total)


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection over Union (notebook exercise 4); 1.0 when both are empty."""
    pred, target = pred.astype(bool), target.astype(bool)
    union = (pred | target).sum()
    return 1.0 if union == 0 else float((pred & target).sum() / union)


@dataclass
class ClassCounts:
    """Per-class pixel counts for multi-class Dice, accumulated over images."""

    classes: int
    inter: np.ndarray = field(init=False)
    pred: np.ndarray = field(init=False)
    true: np.ndarray = field(init=False)
    correct: int = 0
    pixels: int = 0

    def __post_init__(self) -> None:
        self.inter = np.zeros(self.classes, dtype=np.int64)
        self.pred = np.zeros(self.classes, dtype=np.int64)
        self.true = np.zeros(self.classes, dtype=np.int64)

    def add(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred = pred.ravel().astype(np.int64)
        true = true.ravel().astype(np.int64)
        n = self.classes
        self.pred += np.bincount(pred, minlength=n)[:n]
        self.true += np.bincount(true, minlength=n)[:n]
        self.inter += np.bincount(pred[pred == true], minlength=n)[:n]
        self.correct += int((pred == true).sum())
        self.pixels += pred.size

    def per_class_dice(self) -> dict[int, float]:
        """Dice of every class that occurs in the ground truth."""
        return {
            c: float(2 * self.inter[c] / (self.pred[c] + self.true[c]))
            for c in range(self.classes)
            if self.true[c] > 0
        }

    def mean_dice(self) -> float:
        values = list(self.per_class_dice().values())
        return float(np.mean(values)) if values else 0.0

    def pixel_accuracy(self) -> float:
        return self.correct / self.pixels if self.pixels else 0.0


# ---------------------------------------------------------------------------
# Prediction and evaluation
# ---------------------------------------------------------------------------


def predict_batch(model: Any, x: Any, task: str, threshold: float = 0.5) -> Any:
    """Labels ``(B, H, W)`` for a batch of images ``(B, 3, H, W)`` in [0, 1]."""
    import torch

    with torch.no_grad():
        logits = model(x)
    if task == TASK_BINARY:
        return (torch.sigmoid(logits)[:, 0] > threshold).long()
    return logits.argmax(dim=1)


def predict_arrays(
    model: Any,
    X: np.ndarray,
    task: str,
    device: str = "cpu",
    *,
    threshold: float = 0.5,
    batch_size: int = 8,
) -> np.ndarray:
    """Labels ``(N, H, W)`` for in-memory images ``(N, H, W, 3)``."""
    import torch

    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        x = torch.from_numpy(X[start : start + batch_size]).permute(0, 3, 1, 2)
        x = x.float().div(255).to(device)
        preds.append(predict_batch(model, x, task, threshold).cpu().numpy())
    if not preds:
        return np.zeros(X.shape[:3], dtype=np.int64)
    return np.concatenate(preds)


def score_predictions(pred: np.ndarray, Y: np.ndarray, task: str) -> dict[str, Any]:
    """Binary: hard Dice and IoU per image (the notebook's evaluation).

    Multi-class: Dice per class over all pixels, its mean, pixel accuracy and
    the food-vs-not-food Dice of the same predictions (comparable to binary).
    """
    if task == TASK_BINARY:
        dices = np.array([dice_score(p, y) for p, y in zip(pred, Y)])
        ious = np.array([iou_score(p, y) for p, y in zip(pred, Y)])
        return {
            "score": float(dices.mean()),
            "score_name": "Dice",
            "dice": float(dices.mean()),
            "dice_std": float(dices.std()),
            "iou": float(ious.mean()),
            "per_image_dice": dices.tolist(),
        }
    counts = ClassCounts(len(CLASS_NAMES))
    food = []
    for p, y in zip(pred, Y):
        counts.add(p, y)
        food.append(dice_score(p >= FIRST_FOOD, y >= FIRST_FOOD))
    return {
        "score": counts.mean_dice(),
        "score_name": "mean class Dice",
        "mean_dice": counts.mean_dice(),
        "pixel_accuracy": counts.pixel_accuracy(),
        "food_dice": float(np.mean(food)),
        "per_class_dice": {
            CLASS_NAMES[c]: v for c, v in counts.per_class_dice().items()
        },
    }


def evaluate(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    task: str,
    device: str = "cpu",
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    pred = predict_arrays(model, X, task, device, threshold=threshold)
    return score_predictions(pred, Y, task)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainingRun:
    model: Any
    passport: dict[str, Any]
    history: dict[str, list[float]]
    test_images: np.ndarray
    test_labels: np.ndarray


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_unet(
    root: str | Path,
    task: str = TASK_BINARY,
    *,
    epochs: int | None = None,
    size: tuple[int, int] | None = None,
    base: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
    device: str = "auto",
    max_images: int = 0,
    progress: Callable[[str], None] | None = None,
) -> TrainingRun:
    """Train on ``XTrain``/``yTrain`` (keeping ``val_fraction`` for validation),
    keep the epoch with the best validation score and evaluate on the test set.
    """
    import torch
    from torch.utils.data import DataLoader

    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; use one of {TASKS}.")
    defaults = TASK_DEFAULTS[task]
    epochs = int(epochs or defaults["epochs"])
    size = tuple(int(v) for v in (size or defaults["size"]))
    base = int(base or defaults["base"])
    batch_size = int(batch_size or defaults["batch"])
    lr = float(lr or defaults["lr"])
    if size[0] % 8 or size[1] % 8:
        raise ValueError("Height and width must be divisible by 8 (3 poolings).")
    device = resolve_device(device)
    report = progress or (lambda _msg: None)
    seed_everything(seed)
    started = time.time()

    report("Loading the dataset…")
    splits = tray_splits(root)
    train_pairs, test_pairs = splits["train"], splits["test"]
    if max_images:
        train_pairs = train_pairs[:max_images]
    if len(train_pairs) < 2 or not test_pairs:
        raise ValueError(f"No image/mask pairs found in {root}.")
    X_all, Y_all = load_pairs(train_pairs, size, task)
    X_test, Y_test = load_pairs(test_pairs, size, task)

    idx = np.random.permutation(len(X_all))
    n_val = max(1, int(val_fraction * len(idx)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_dl = DataLoader(
        make_dataset(X_all[train_idx], Y_all[train_idx], task, augment=True),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val, Y_val = X_all[val_idx], Y_all[val_idx]

    model = build_unet(out_ch=output_channels(task), base=base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = loss_function(task)
    history: dict[str, list[float]] = {"train_loss": [], "val_score": []}
    best_score, best_epoch = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    val: dict[str, Any] = {"score_name": ""}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)
        train_loss = running / len(train_idx)
        val = evaluate(model, X_val, Y_val, task, device)
        history["train_loss"].append(train_loss)
        history["val_score"].append(val["score"])
        if val["score"] > best_score:
            best_score, best_epoch = val["score"], epoch
            best_state = copy.deepcopy(model.state_dict())
        report(
            f"Epoch {epoch}/{epochs} | loss {train_loss:.3f} | "
            f"val {val['score_name']} {val['score']:.3f} (best {best_score:.3f})"
        )

    model.load_state_dict(best_state)
    report("Evaluating on the test set…")
    test = evaluate(model, X_test, Y_test, task, device)
    passport = {
        "task": task,
        "class_names": class_names(task),
        "num_classes": num_classes(task),
        "first_food": FIRST_FOOD,
        "input_size": list(size),
        "architecture": {
            "name": "UNet",
            "base": base,
            "out_ch": output_channels(task),
        },
        "parameters": count_parameters(model),
        "threshold": 0.5,
        "training": {
            "dataset": _project_relative(root),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": "Adam",
            "loss": (
                "BCE + soft Dice"
                if task == TASK_BINARY
                else "cross-entropy + soft Dice"
            ),
            "augmentation": "random horizontal flip",
            "seed": seed,
            "device": device,
            "train_images": int(len(train_idx)),
            "val_images": int(n_val),
            "test_images": int(len(X_test)),
            "best_epoch": best_epoch,
            "seconds": round(time.time() - started, 1),
        },
        "metrics": {
            "validation": {"score_name": val["score_name"], "best": best_score},
            "test": {k: v for k, v in test.items() if k != "score_name"},
        },
        "history": history,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    return TrainingRun(model.eval(), passport, history, X_test, Y_test)


def _project_relative(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def save_model(model: Any, passport: dict[str, Any], folder: str | Path) -> Path:
    import torch

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, folder / WEIGHTS_FILE)
    with open(folder / PASSPORT_FILE, "w", encoding="utf-8") as fh:
        json.dump(passport, fh, indent=2)
    return folder


def list_models(root: str | Path = UNET_MODELS_DIR) -> list[Path]:
    """Model folders (with weights and passport) under ``root``."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p.parent
        for p in root.rglob(PASSPORT_FILE)
        if (p.parent / WEIGHTS_FILE).is_file()
    )


_MODEL_CACHE: dict[tuple[str, int, str], tuple[Any, dict[str, Any]]] = {}


def load_model(folder: str | Path, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Return ``(model, passport)``; cached until the weights file changes."""
    import torch

    folder = Path(folder)
    weights = folder / WEIGHTS_FILE
    if not weights.is_file():
        raise FileNotFoundError(f"No {WEIGHTS_FILE} in {folder}")
    key = (str(weights.resolve()), weights.stat().st_mtime_ns, device)
    if key not in _MODEL_CACHE:
        with open(folder / PASSPORT_FILE, encoding="utf-8") as fh:
            passport = json.load(fh)
        arch = passport["architecture"]
        model = build_unet(out_ch=int(arch["out_ch"]), base=int(arch["base"]))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = (model.to(device).eval(), passport)
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------------------
# Segmenting one image
# ---------------------------------------------------------------------------


def segment(
    model: Any,
    passport: dict[str, Any],
    image: Image.Image,
    device: str = "cpu",
    threshold: float | None = None,
) -> np.ndarray:
    """Label per pixel at the image's own size."""
    h, w = passport["input_size"]
    rgb = image.convert("RGB")
    x = np.array(rgb.resize((w, h), Image.BILINEAR))[None]
    if threshold is None:
        threshold = float(passport.get("threshold", 0.5))
    labels = predict_arrays(model, x, passport["task"], device, threshold=threshold)
    small = Image.fromarray(labels[0].astype(np.uint8))
    return np.array(small.resize(rgb.size, Image.NEAREST))


def ground_truth_path(image_path: str | Path | None) -> Path | None:
    """The mask of a TrayDataset image: ``XTest/a.jpg`` → ``yTest/a.png`` and
    ``images/a.jpg`` → ``masks/a.png``."""
    if image_path is None:
        return None
    path = Path(image_path)
    folder = path.parent.name
    if folder.startswith("X"):
        mask_dir = path.parent.parent / ("y" + folder[1:])
    elif folder == "images":
        mask_dir = path.parent.parent / "masks"
    else:
        return None
    for ext in (".png", ".bmp"):
        mask = mask_dir / (path.stem + ext)
        if mask.is_file():
            return mask
    return None


def load_ground_truth(
    mask_path: str | Path, task: str, size: tuple[int, int]
) -> np.ndarray:
    """Task labels of a mask file, resized (nearest) to ``size`` = (W, H)."""
    mask = to_task(read_mask(mask_path), task)
    if (mask.shape[1], mask.shape[0]) != tuple(size):
        mask = np.array(Image.fromarray(mask).resize(tuple(size), Image.NEAREST))
    return mask


def label_summary(labels: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    """``(class name, fraction of pixels)``, largest first."""
    values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    out = []
    for i in order:
        value = int(values[i])
        name = names[value] if value < len(names) else str(value)
        out.append((name, counts[i] / labels.size))
    return out
