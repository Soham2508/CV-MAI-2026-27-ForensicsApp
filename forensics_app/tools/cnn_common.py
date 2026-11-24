"""Shared PyTorch CNN architectures, GPU device helpers, datasets, and model passports.

Supports local NVIDIA CUDA GPU acceleration as well as CPU execution for:
  - Set 8.1: Training classification CNNs (from scratch & transfer learning)
  - Set 8.2: Loading pretrained models, viewing passports, and classifying images
  - Set 8.3: Extracting deep feature vectors and 2D projection (t-SNE / UMAP)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import models as tv_models

from forensics_app.paths import DATASETS_DIR, MODELS_DIR, resolve_path

from .codebook_construction import (
    DEFAULT_DATASET_DIR,
    inspect_dataset_directory,
    load_visual_words_dataset,
    portable_path,
    resolve_record_path,
)

DEFAULT_CNN_ROOT = MODELS_DIR / "cnn"
DEFAULT_CNN_MODEL_DIR = DEFAULT_CNN_ROOT / "cnn_model_001"
DEFAULT_CIFAR10_DIR = DATASETS_DIR / "cifar10"

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class ArchitectureSpec:
    """Metadata describing a supported CNN classification architecture."""

    arch_key: str
    display_name: str
    default_input_size: int
    min_input_size: int
    default_normalization: str
    default_optimizer: str
    default_lr: float
    is_transfer_learning: bool
    feature_layer: str
    feature_dim: int


CNN_ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "shallow_cnn": ArchitectureSpec(
        arch_key="shallow_cnn",
        display_name="Shallow CNN (Conv32 -> Dense100)",
        default_input_size=32,
        min_input_size=16,
        default_normalization="unit_0_1",
        default_optimizer="SGD",
        default_lr=0.01,
        is_transfer_learning=False,
        feature_layer="fc1",
        feature_dim=100,
    ),
    "deeper_vgg": ArchitectureSpec(
        arch_key="deeper_vgg",
        display_name="Deeper VGG (2xConv32 -> Dense128)",
        default_input_size=32,
        min_input_size=16,
        default_normalization="unit_0_1",
        default_optimizer="SGD",
        default_lr=0.01,
        is_transfer_learning=False,
        feature_layer="fc1",
        feature_dim=128,
    ),
    "complete_vgg": ArchitectureSpec(
        arch_key="complete_vgg",
        display_name="Complete VGG (64-128-256 -> Dense128)",
        default_input_size=64,
        min_input_size=32,
        default_normalization="unit_0_1",
        default_optimizer="SGD",
        default_lr=0.01,
        is_transfer_learning=False,
        feature_layer="fc1",
        feature_dim=128,
    ),
    "inception_v3": ArchitectureSpec(
        arch_key="inception_v3",
        display_name="InceptionV3 (Transfer Learning)",
        default_input_size=299,
        min_input_size=75,
        default_normalization="imagenet",
        default_optimizer="Adam",
        default_lr=0.001,
        is_transfer_learning=True,
        feature_layer="custom_dense",
        feature_dim=1024,
    ),
    "resnet50": ArchitectureSpec(
        arch_key="resnet50",
        display_name="ResNet50 (Transfer Learning / avgpool)",
        default_input_size=224,
        min_input_size=32,
        default_normalization="imagenet",
        default_optimizer="Adam",
        default_lr=0.001,
        is_transfer_learning=True,
        feature_layer="avgpool",
        feature_dim=2048,
    ),
}

DISPLAY_TO_ARCH_KEY: dict[str, str] = {
    spec.display_name: key for key, spec in CNN_ARCHITECTURES.items()
}


def resolve_arch_key(name: str) -> str:
    """Resolve an internal key or UI display label to a canonical ``arch_key``."""
    raw = str(name).strip()
    if raw in CNN_ARCHITECTURES:
        return raw
    if raw in DISPLAY_TO_ARCH_KEY:
        return DISPLAY_TO_ARCH_KEY[raw]
    lowered = raw.lower().replace("-", "_").replace(" ", "_")
    for key in CNN_ARCHITECTURES:
        if key == lowered or key in lowered:
            return key
    raise ValueError(
        f"Unsupported CNN architecture: {name!r}. "
        f"Choose one of: {', '.join(CNN_ARCHITECTURES)}."
    )


def get_compute_device(preferred: str = "auto") -> tuple[torch.device, dict[str, Any]]:
    """Select CUDA GPU or CPU and return diagnostic hardware metadata."""
    pref = str(preferred).strip().lower()
    cuda_available = bool(torch.cuda.is_available())
    if pref in ("cuda", "gpu"):
        if not cuda_available:
            raise RuntimeError(
                "CUDA GPU was requested, but torch.cuda.is_available() is False. "
                "Verify an NVIDIA CUDA PyTorch build is installed."
            )
        device = torch.device("cuda:0")
    elif pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if cuda_available else "cpu")

    gpu_name = None
    vram_mb = None
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        gpu_name = str(props.name)
        vram_mb = int(round(props.total_memory / (1024 * 1024)))
    elif cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu_name = str(props.name)
        vram_mb = int(round(props.total_memory / (1024 * 1024)))

    info = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_mb": vram_mb,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    return device, info


def describe_compute_device(preferred: str = "auto") -> str:
    """Return a concise single-line description of the active compute device."""
    try:
        device, info = get_compute_device(preferred)
    except Exception as exc:
        return f"Device unavailable: {exc}"
    if device.type == "cuda":
        vram = f" ({info['vram_mb']} MB VRAM)" if info["vram_mb"] else ""
        return (
            f"GPU Active: {info['gpu_name']}{vram} | "
            f"CUDA {info['cuda_version']} | PyTorch {info['torch_version']}"
        )
    if info["cuda_available"]:
        return f"CPU selected (GPU available: {info['gpu_name']})"
    return f"CPU mode (CUDA not available in PyTorch {info['torch_version']})"


def _init_he_uniform(module: nn.Module) -> None:
    """Apply Kaiming uniform initialization matching Keras ``he_uniform``."""
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class ShallowCNN(nn.Module):
    """Set 8.1 Shallow CNN matching the exercise notebook."""

    def __init__(self, num_classes: int, input_size: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        feat_side = max(1, (int(input_size) - 2) // 2)
        flat_dim = 32 * feat_side * feat_side
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_dim, 100)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(100, int(num_classes))
        _init_he_uniform(self)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.flatten(self.features(x))
        return self.relu(self.fc1(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class DeeperVGG(nn.Module):
    """Deeper 2xConv32 VGG-style CNN matching the exercise notebook."""

    def __init__(self, num_classes: int, input_size: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        feat_side = max(1, int(input_size) // 2)
        flat_dim = 32 * feat_side * feat_side
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_dim, 128)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(128, int(num_classes))
        _init_he_uniform(self)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.flatten(self.features(x))
        return self.relu(self.fc1(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class CompleteVGG(nn.Module):
    """Complete VGG block CNN (64 -> 128 -> 256 -> Dense128) from the notebook."""

    def __init__(self, num_classes: int, input_size: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        feat_side = max(1, int(input_size) // 8)
        flat_dim = 256 * feat_side * feat_side
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_dim, 128)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(128, int(num_classes))
        _init_he_uniform(self)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.flatten(self.features(x))
        return self.relu(self.fc1(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class InceptionV3Classifier(nn.Module):
    """InceptionV3 transfer learning classifier with GlobalAvgPool -> Dense1024."""

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            weights = tv_models.Inception_V3_Weights.DEFAULT
            backbone = tv_models.inception_v3(weights=weights)
            backbone.aux_logits = False
            backbone.AuxLogits = None
        else:
            backbone = tv_models.inception_v3(
                weights=None, aux_logits=False, init_weights=False
            )

        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.custom_dense = nn.Linear(in_features, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(1024, int(num_classes))
        _init_he_uniform(self.custom_dense)
        _init_he_uniform(self.classifier)
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = bool(trainable)

    def set_head_trainable(self, trainable: bool) -> None:
        for layer in (self.custom_dense, self.classifier):
            for param in layer.parameters():
                param.requires_grad = bool(trainable)

    def extract_features(
        self, x: torch.Tensor, layer_name: str = "custom_dense"
    ) -> torch.Tensor:
        pooled = self.backbone(x)
        if layer_name == "avgpool":
            return pooled
        return self.relu(self.custom_dense(pooled))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x, "custom_dense")
        return self.classifier(self.dropout(feat))


class ResNet50Classifier(nn.Module):
    """ResNet50 transfer learning classifier exposing 2048-D ``avgpool`` features."""

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        in_features = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, int(num_classes))
        _init_he_uniform(self.classifier)
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = bool(trainable)

    def set_head_trainable(self, trainable: bool) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = bool(trainable)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def build_cnn_model(
    arch_key: str,
    num_classes: int,
    *,
    input_size: int | None = None,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Construct one of the supported CNN architectures."""
    key = resolve_arch_key(arch_key)
    spec = CNN_ARCHITECTURES[key]
    size = int(input_size if input_size is not None else spec.default_input_size)
    if size < spec.min_input_size:
        raise ValueError(
            f"{spec.display_name} requires input_size >= {spec.min_input_size}, "
            f"got {size}."
        )
    if num_classes < 2:
        raise ValueError(f"Expected at least 2 classes, got {num_classes}.")

    if key == "shallow_cnn":
        return ShallowCNN(num_classes, input_size=size)
    if key == "deeper_vgg":
        return DeeperVGG(num_classes, input_size=size)
    if key == "complete_vgg":
        return CompleteVGG(num_classes, input_size=size)
    if key == "inception_v3":
        return InceptionV3Classifier(
            num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone
        )
    if key == "resnet50":
        return ResNet50Classifier(
            num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone
        )
    raise ValueError(f"Unsupported architecture: {key!r}")


def build_preprocessing_spec(
    *,
    input_size: int,
    normalization: str = "unit_0_1",
) -> dict[str, Any]:
    """Create canonical preprocessing metadata saved in the model passport."""
    size = int(input_size)
    norm = str(normalization).strip().lower()
    if norm in ("imagenet", "imagenet_standard", "zscore"):
        norm_key = "imagenet"
        mean = list(IMAGENET_MEAN)
        std = list(IMAGENET_STD)
        instructions = (
            f"1) Convert image to RGB; "
            f"2) Resize with bilinear interpolation to {size}x{size}; "
            f"3) Scale pixels to [0.0, 1.0] (float32 / 255.0); "
            f"4) Standardize channels with ImageNet mean={mean} and std={std}; "
            f"5) Arrange tensor as NCHW (3, {size}, {size})."
        )
    elif norm in ("unit_0_1", "0_1", "rescale", "unit"):
        norm_key = "unit_0_1"
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        instructions = (
            f"1) Convert image to RGB; "
            f"2) Resize with bilinear interpolation to {size}x{size}; "
            f"3) Scale pixels from [0, 255] to [0.0, 1.0] (float32 / 255.0); "
            f"4) Arrange tensor as NCHW (3, {size}, {size})."
        )
    else:
        raise ValueError(
            f"Unsupported normalization mode: {normalization!r}. "
            "Use 'unit_0_1' or 'imagenet'."
        )

    return {
        "input_size": [size, size],
        "color_mode": "RGB",
        "channel_order": "CHW",
        "normalization": norm_key,
        "mean": mean,
        "std": std,
        "resize_interpolation": "bilinear",
        "instructions": instructions,
    }


def preprocess_pil_image(
    image: Image.Image,
    preprocessing: dict[str, Any],
) -> torch.Tensor:
    """Preprocess a PIL image strictly following a passport's preprocessing spec."""
    h, w = [int(v) for v in preprocessing["input_size"]]
    rgb = image.convert("RGB")
    if rgb.size != (w, h):
        rgb = rgb.resize((w, h), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    if preprocessing.get("normalization") == "imagenet":
        mean = torch.tensor(
            preprocessing.get("mean", IMAGENET_MEAN), dtype=torch.float32
        ).view(3, 1, 1)
        std = torch.tensor(
            preprocessing.get("std", IMAGENET_STD), dtype=torch.float32
        ).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


class ImageRecordDataset(Dataset):
    """PyTorch Dataset backed by disk records and a passport preprocessing spec."""

    _MAX_CACHE_BYTES = 512 * 1024 * 1024

    def __init__(
        self,
        records: list[dict[str, Any]],
        preprocessing: dict[str, Any],
    ) -> None:
        self.records = list(records)
        self.preprocessing = dict(preprocessing)
        h, w = [int(v) for v in self.preprocessing["input_size"]]
        est_bytes = len(self.records) * 3 * h * w * 4
        self._use_cache = est_bytes <= self._MAX_CACHE_BYTES
        self._cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rec = self.records[index]
        tensor = self._cache.get(index)
        if tensor is None:
            img_path = resolve_record_path(rec["path"])
            with Image.open(img_path) as im:
                tensor = preprocess_pil_image(im, self.preprocessing)
            if self._use_cache:
                self._cache[index] = tensor
        return tensor, int(rec["class_id"])


class CIFAR10IndexedDataset(Dataset):
    """PyTorch Dataset backed by torchvision CIFAR-10 arrays and selected indices."""

    _MAX_PRELOAD_BYTES = 768 * 1024 * 1024

    def __init__(
        self,
        data: np.ndarray,
        targets: list[int] | np.ndarray,
        indices: list[int] | np.ndarray,
        preprocessing: dict[str, Any],
    ) -> None:
        self.data = data
        self.targets = targets
        self.indices = [int(i) for i in indices]
        self.preprocessing = dict(preprocessing)
        self._labels = torch.tensor(
            [int(targets[i]) for i in self.indices], dtype=torch.long
        )
        self._tensors: torch.Tensor | None = self._maybe_preload_tensors()

    def _maybe_preload_tensors(self) -> torch.Tensor | None:
        if not self.indices:
            return None
        h, w = [int(v) for v in self.preprocessing["input_size"]]
        est_bytes = len(self.indices) * 3 * h * w * 4
        if est_bytes > self._MAX_PRELOAD_BYTES:
            return None

        sub = np.ascontiguousarray(self.data[self.indices], dtype=np.uint8)
        tensors = (
            torch.from_numpy(sub)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        if tensors.shape[2] != h or tensors.shape[3] != w:
            tensors = torch.nn.functional.interpolate(
                tensors, size=(h, w), mode="bilinear", align_corners=False
            )
        if self.preprocessing.get("normalization") == "imagenet":
            mean = torch.tensor(
                self.preprocessing.get("mean", IMAGENET_MEAN), dtype=torch.float32
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                self.preprocessing.get("std", IMAGENET_STD), dtype=torch.float32
            ).view(1, 3, 1, 1)
            tensors.sub_(mean).div_(std)
        return tensors

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self._tensors is not None:
            return self._tensors[index], int(self._labels[index])
        src_idx = self.indices[index]
        arr = self.data[src_idx]
        img = Image.fromarray(arr, mode="RGB")
        tensor = preprocess_pil_image(img, self.preprocessing)
        return tensor, int(self.targets[src_idx])


def _stratified_subset_indices(
    targets: list[int] | np.ndarray,
    max_per_class: int | None,
    *,
    num_classes: int,
    seed: int,
) -> list[int]:
    """Select all indices or a deterministic stratified subset per class."""
    y = np.asarray(targets, dtype=np.int64)
    if max_per_class is None or int(max_per_class) <= 0:
        return list(range(len(y)))
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    cap = int(max_per_class)
    for cid in range(num_classes):
        idx = np.where(y == cid)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        chosen.extend(sorted(int(i) for i in idx))
    return chosen


def _split_train_val(
    items: list[Any],
    labels: list[int],
    val_split: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    """Split training items into train and validation partitions safely."""
    n = len(items)
    if n < 4 or val_split <= 0.0:
        return list(items), []
    val_fraction = min(0.45, max(0.05, float(val_split)))
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    val_count = max(n_classes, int(round(n * val_fraction)))
    if len(unique) >= 2 and int(counts.min()) >= 2 and val_count < n:
        try:
            tr_idx, va_idx = train_test_split(
                np.arange(n),
                test_size=val_count,
                stratify=labels,
                random_state=int(seed),
            )
            return [items[i] for i in tr_idx], [items[i] for i in va_idx]
        except ValueError:
            pass
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    split_at = max(1, min(n - 1, int(round(n * (1.0 - val_fraction)))))
    return [items[i] for i in perm[:split_at]], [items[i] for i in perm[split_at:]]


def _count_per_class(labels: list[int], class_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for cid in labels:
        if 0 <= int(cid) < len(class_names):
            cname = class_names[int(cid)]
            counts[cname] = counts.get(cname, 0) + 1
    return counts


def prepare_cnn_dataset(
    *,
    source_type: str = "image_folder",
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    test_size: float = 0.2,
    val_split: float = 0.1,
    random_state: int = 42,
    max_samples_per_class: int | None = None,
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    """Load an Image Folder or CIFAR-10 dataset and build train/val/test Datasets."""
    src = str(source_type).strip().lower()
    if src == "cifar10":
        root = resolve_path(DEFAULT_CIFAR10_DIR)
        root.mkdir(parents=True, exist_ok=True)
        train_ds = torchvision.datasets.CIFAR10(
            root=str(root), train=True, download=True
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=str(root), train=False, download=True
        )
        class_names = list(getattr(train_ds, "classes", CIFAR10_CLASS_NAMES))
        num_classes = len(class_names)

        cap_train = (
            int(max_samples_per_class)
            if max_samples_per_class and int(max_samples_per_class) > 0
            else None
        )
        cap_test = max(10, cap_train // 4) if cap_train is not None else None
        train_pool = _stratified_subset_indices(
            train_ds.targets, cap_train, num_classes=num_classes, seed=random_state
        )
        test_indices = _stratified_subset_indices(
            test_ds.targets, cap_test, num_classes=num_classes, seed=random_state + 1
        )
        pool_labels = [int(train_ds.targets[i]) for i in train_pool]
        train_indices, val_indices = _split_train_val(
            train_pool, pool_labels, val_split=val_split, seed=random_state
        )

        train_labels = [int(train_ds.targets[i]) for i in train_indices]
        val_labels = [int(train_ds.targets[i]) for i in val_indices]
        test_labels = [int(test_ds.targets[i]) for i in test_indices]

        return {
            "source_type": "cifar10",
            "dataset_name": "CIFAR-10",
            "dataset_dir": portable_path(root),
            "split_mode": "pre_split",
            "class_names": class_names,
            "train_dataset": CIFAR10IndexedDataset(
                train_ds.data, train_ds.targets, train_indices, preprocessing
            ),
            "val_dataset": (
                CIFAR10IndexedDataset(
                    train_ds.data, train_ds.targets, val_indices, preprocessing
                )
                if val_indices
                else None
            ),
            "test_dataset": CIFAR10IndexedDataset(
                test_ds.data, test_ds.targets, test_indices, preprocessing
            ),
            "train_labels": train_labels,
            "val_labels": val_labels,
            "test_labels": test_labels,
            "train_records": [],
            "val_records": [],
            "test_records": [
                {
                    "cifar_index": int(idx),
                    "split": "test",
                    "filename": (
                        f"cifar10_test_{idx:05d}_"
                        f"{class_names[int(test_ds.targets[idx])]}.png"
                    ),
                    "class_name": class_names[int(test_ds.targets[idx])],
                    "class_id": int(test_ds.targets[idx]),
                }
                for idx in test_indices
            ],
            "sample_size": {
                "total": len(train_indices) + len(val_indices) + len(test_indices),
                "train": len(train_indices),
                "val": len(val_indices),
                "test": len(test_indices),
                "per_class_train": _count_per_class(train_labels, class_names),
                "per_class_val": _count_per_class(val_labels, class_names),
                "per_class_test": _count_per_class(test_labels, class_names),
            },
        }

    # Image Folder Dataset (reusing codebook_construction dataset loader)
    loaded = load_visual_words_dataset(
        dataset_dir,
        test_size=float(test_size),
        random_state=int(random_state),
        max_images_per_class=max_samples_per_class,
    )
    class_names = list(loaded["class_names"])
    train_pool_recs = list(loaded["train_records"])
    test_recs = list(loaded["test_records"])
    pool_labels = [int(r["class_id"]) for r in train_pool_recs]
    train_recs, val_recs = _split_train_val(
        train_pool_recs, pool_labels, val_split=val_split, seed=random_state
    )

    train_labels = [int(r["class_id"]) for r in train_recs]
    val_labels = [int(r["class_id"]) for r in val_recs]
    test_labels = [int(r["class_id"]) for r in test_recs]
    ds_root = Path(loaded["dataset_dir"])

    return {
        "source_type": "image_folder",
        "dataset_name": ds_root.name,
        "dataset_dir": portable_path(ds_root),
        "split_mode": loaded["split_mode"],
        "class_names": class_names,
        "train_dataset": ImageRecordDataset(train_recs, preprocessing),
        "val_dataset": ImageRecordDataset(val_recs, preprocessing)
        if val_recs
        else None,
        "test_dataset": ImageRecordDataset(test_recs, preprocessing),
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "train_records": train_recs,
        "val_records": val_recs,
        "test_records": test_recs,
        "sample_size": {
            "total": len(train_recs) + len(val_recs) + len(test_recs),
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
            "per_class_train": _count_per_class(train_labels, class_names),
            "per_class_val": _count_per_class(val_labels, class_names),
            "per_class_test": _count_per_class(test_labels, class_names),
        },
    }


def inspect_cnn_dataset(
    source_type: str,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
) -> dict[str, Any]:
    """Quick summary of a dataset without running full training."""
    if str(source_type).strip().lower() == "cifar10":
        return {
            "source_type": "cifar10",
            "dataset_name": "CIFAR-10",
            "dataset_dir": portable_path(DEFAULT_CIFAR10_DIR),
            "split_mode": "pre_split",
            "class_names": list(CIFAR10_CLASS_NAMES),
            "num_classes": 10,
            "train_counts": {c: 5000 for c in CIFAR10_CLASS_NAMES},
            "test_counts": {c: 1000 for c in CIFAR10_CLASS_NAMES},
        }
    info = inspect_dataset_directory(dataset_dir)
    info["source_type"] = "image_folder"
    info["dataset_name"] = Path(info["dataset_dir"]).name
    return info


def _compute_model_id(passport: dict[str, Any]) -> str:
    """Compute a short deterministic ID for a trained CNN model passport."""
    payload = {
        "model_name": passport.get("model_name", ""),
        "architecture": passport.get("architecture", {}),
        "class_names": passport.get("class_names", []),
        "preprocessing": passport.get("preprocessing", {}),
        "created_at": passport.get("created_at", ""),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def save_cnn_bundle(
    model: nn.Module,
    passport: dict[str, Any],
    model_dir: str | Path,
) -> Path:
    """Save model weights (``weights.pt``) and model passport (``passport.json``)."""
    out_dir = resolve_path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    passport_out = dict(passport)
    passport_out.setdefault("passport_version", 1)
    passport_out.setdefault("framework", "pytorch")
    passport_out.setdefault(
        "created_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    passport_out.setdefault("model_name", out_dir.name)
    if not passport_out.get("model_id"):
        passport_out["model_id"] = _compute_model_id(passport_out)

    weights_path = out_dir / "weights.pt"
    passport_path = out_dir / "passport.json"

    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_cpu, weights_path)
    passport_path.write_text(json.dumps(passport_out, indent=2), encoding="utf-8")
    return out_dir


def load_cnn_bundle(
    model_dir: str | Path,
    *,
    device_preference: str = "auto",
) -> dict[str, Any]:
    """Load saved CNN weights and passport, reconstructing the architecture."""
    resolved = resolve_record_path(model_dir)
    passport_path = resolved / "passport.json"
    weights_path = resolved / "weights.pt"
    if not passport_path.is_file():
        raise FileNotFoundError(f"Model passport not found: {passport_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    passport = json.loads(passport_path.read_text(encoding="utf-8"))
    arch = passport.get("architecture", {})
    prep = passport.get("preprocessing", {})
    class_names = list(passport.get("class_names", []))
    num_classes = int(arch.get("num_classes") or len(class_names))
    input_size = int(prep.get("input_size", [32, 32])[0])

    # Never hit the network when reconstructing a saved model whose weights are on disk.
    model = build_cnn_model(
        str(arch["arch_key"]),
        num_classes,
        input_size=input_size,
        pretrained=False,
        freeze_backbone=bool(arch.get("freeze_backbone", False)),
    )
    device, device_info = get_compute_device(device_preference)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model_dir": str(resolved),
        "model_id": passport.get("model_id", resolved.name),
        "passport": passport,
        "model": model,
        "device": device,
        "device_info": device_info,
        "is_builtin_imagenet": False,
    }


def load_imagenet_pretrained_bundle(
    arch_key: str,
    *,
    device_preference: str = "auto",
) -> dict[str, Any]:
    """Load an off-the-shelf torchvision ImageNet-1K pretrained CNN with a passport."""
    key = resolve_arch_key(arch_key)
    if key == "resnet50":
        weights = tv_models.ResNet50_Weights.DEFAULT
        model = tv_models.resnet50(weights=weights)
        input_size = 224
        feature_layer = "avgpool"
        feature_dim = 2048
    elif key == "inception_v3":
        weights = tv_models.Inception_V3_Weights.DEFAULT
        model = tv_models.inception_v3(weights=weights)
        model.aux_logits = False
        model.AuxLogits = None
        input_size = 299
        feature_layer = "avgpool"
        feature_dim = 2048
    else:
        raise ValueError(
            "Only 'resnet50' and 'inception_v3' support off-the-shelf ImageNet-1K."
        )

    categories = list(weights.meta.get("categories", []))
    if not categories:
        categories = [f"class_{i}" for i in range(1000)]

    device, device_info = get_compute_device(device_preference)
    model.to(device)
    model.eval()

    prep = build_preprocessing_spec(input_size=input_size, normalization="imagenet")
    total_params = sum(p.numel() for p in model.parameters())
    passport = {
        "passport_version": 1,
        "model_id": f"imagenet_{key}",
        "model_name": f"ImageNet-1K {CNN_ARCHITECTURES[key].display_name}",
        "created_at": "ImageNet Pretrained (torchvision)",
        "framework": "pytorch",
        "architecture": {
            "arch_key": key,
            "display_name": f"{key} (ImageNet-1K Pretrained)",
            "num_classes": len(categories),
            "pretrained_backbone": True,
            "freeze_backbone": True,
            "feature_layer": feature_layer,
            "feature_dim": feature_dim,
            "total_params": total_params,
            "trainable_params": 0,
        },
        "dataset": {
            "dataset_name": "ImageNet-1K (ILSVRC2012)",
            "source_type": "imagenet_pretrained",
            "dataset_dir": "torchvision.models",
            "split_mode": "pretrained",
            "sample_size": {
                "total": 1281167,
                "train": 1281167,
                "val": 50000,
                "test": 0,
            },
            "test_records": [],
        },
        "class_names": categories,
        "preprocessing": prep,
        "training_config": {
            "epochs": 0,
            "optimizer": "Pretrained (torchvision weights)",
            "device_used": str(device),
            "gpu_name": device_info.get("gpu_name"),
        },
        "training_history": {},
        "evaluation": {},
    }
    return {
        "model_dir": f"pretrained://{key}",
        "model_id": passport["model_id"],
        "passport": passport,
        "model": model,
        "device": device,
        "device_info": device_info,
        "is_builtin_imagenet": True,
    }


def list_saved_cnn_models(root: str | Path = DEFAULT_CNN_ROOT) -> list[Path]:
    """Return sorted saved CNN model directories containing ``passport.json``."""
    base = resolve_path(root)
    if not base.exists() or not base.is_dir():
        return []
    found = [
        p
        for p in sorted(base.iterdir(), key=lambda item: item.name.lower())
        if p.is_dir()
        and (p / "passport.json").is_file()
        and (p / "weights.pt").is_file()
    ]
    return found


def find_latest_cnn_model_dir(root: str | Path = DEFAULT_CNN_ROOT) -> Path:
    """Return the most recently modified saved CNN model dir or the default path."""
    saved = list_saved_cnn_models(root)
    if saved:
        return max(saved, key=lambda p: (p / "passport.json").stat().st_mtime)
    return DEFAULT_CNN_MODEL_DIR


def load_test_record_image(
    record: dict[str, Any],
    passport: dict[str, Any],
) -> Image.Image:
    """Load an individual test-set PIL image from either disk or cached CIFAR-10."""
    if "cifar_index" in record:
        ds_dir = resolve_record_path(
            passport.get("dataset", {}).get("dataset_dir", DEFAULT_CIFAR10_DIR)
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=str(ds_dir), train=False, download=True
        )
        idx = int(record["cifar_index"])
        return Image.fromarray(test_ds.data[idx], mode="RGB")

    img_path = resolve_record_path(record["path"])
    if not img_path.is_file():
        raise FileNotFoundError(f"Test image not found: {img_path}")
    with Image.open(img_path) as im:
        return im.copy()


def format_passport_text(passport: dict[str, Any]) -> str:
    """Format a model passport into a clean, readable report for the UI."""
    arch = passport.get("architecture", {})
    ds = passport.get("dataset", {})
    sample = ds.get("sample_size", {})
    prep = passport.get("preprocessing", {})
    cfg = passport.get("training_config", {})
    hist = passport.get("training_history", {})
    ev = passport.get("evaluation", {})
    classes = list(passport.get("class_names", []))

    lines: list[str] = [
        f"MODEL PASSPORT  [{passport.get('model_name', 'unnamed')}]",
        (
            f"Model ID: {passport.get('model_id', '—')}  |  "
            f"Created: {passport.get('created_at', '—')}"
        ),
        f"Architecture: {arch.get('display_name', arch.get('arch_key', '—'))}",
        (
            f"Parameters: {int(arch.get('total_params', 0)):,} total "
            f"({int(arch.get('trainable_params', 0)):,} trainable)  |  "
            f"Feature layer: {arch.get('feature_layer', '—')} "
            f"({arch.get('feature_dim', '—')}-D)"
        ),
        "",
        "--- Dataset & Sample Size ---",
        (
            f"Dataset: {ds.get('dataset_name', '—')} "
            f"({ds.get('source_type', '—')}, split={ds.get('split_mode', '—')})"
        ),
        (
            f"Samples: total={sample.get('total', 0)} | "
            f"train={sample.get('train', 0)} | "
            f"val={sample.get('val', 0)} | "
            f"test={sample.get('test', 0)}"
        ),
    ]
    if sample.get("per_class_train"):
        lines.append(
            "Per-class train: "
            + ", ".join(f"{k}={v}" for k, v in sample["per_class_train"].items())
        )
    if sample.get("per_class_test"):
        lines.append(
            "Per-class test:  "
            + ", ".join(f"{k}={v}" for k, v in sample["per_class_test"].items())
        )

    lines.append("")
    lines.append(f"--- Class Names ({len(classes)}) ---")
    if len(classes) <= 20:
        lines.append(", ".join(f"{i}: {name}" for i, name in enumerate(classes)))
    else:
        preview = ", ".join(f"{i}: {name}" for i, name in enumerate(classes[:10]))
        lines.append(f"{preview}, ... ({len(classes)} total classes)")

    lines.append("")
    lines.append("--- Preprocessing Instructions ---")
    lines.append(
        f"Input size: {prep.get('input_size')} | Mode: {prep.get('color_mode')} | "
        f"Order: {prep.get('channel_order')} | Norm: {prep.get('normalization')}"
    )
    lines.append(str(prep.get("instructions", "—")))

    lines.append("")
    lines.append("--- Training History & Evaluation ---")
    gpu_str = cfg.get("gpu_name") or cfg.get("device_used", "—")
    lines.append(
        f"Device: {gpu_str} | Optimizer: {cfg.get('optimizer', '—')} "
        f"(lr={cfg.get('learning_rate', '—')}) | "
        f"Batch: {cfg.get('batch_size', '—')} | Epochs: {cfg.get('epochs', 0)}"
    )
    epochs_ran = hist.get("epoch", [])
    if epochs_ran:
        tr_acc = hist.get("train_acc", [])
        va_acc = hist.get("val_acc", [])
        tr_loss = hist.get("train_loss", [])
        va_loss = hist.get("val_loss", [])
        last_tr_a = f"{tr_acc[-1] * 100:.1f}%" if tr_acc else "—"
        last_tr_l = f"{tr_loss[-1]:.4f}" if tr_loss else "—"
        last_va_a = (
            f"{va_acc[-1] * 100:.1f}%" if va_acc and va_acc[-1] is not None else "—"
        )
        last_va_l = f"{va_loss[-1]:.4f}" if va_loss and va_loss[-1] is not None else "—"
        lines.append(
            f"Final epoch ({len(epochs_ran)}): "
            f"train_acc={last_tr_a}, train_loss={last_tr_l} | "
            f"val_acc={last_va_a}, val_loss={last_va_l}"
        )
    if ev:
        te_acc = ev.get("test_accuracy")
        te_loss = ev.get("test_loss")
        te_acc_str = f"{te_acc * 100:.2f}%" if te_acc is not None else "—"
        te_loss_str = f"{te_loss:.4f}" if te_loss is not None else "—"
        lines.append(f"Held-out test accuracy: {te_acc_str} | Test loss: {te_loss_str}")

    return "\n".join(lines)
