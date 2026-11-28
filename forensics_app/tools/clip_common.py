"""Shared CLIP code for the Set 10 tools: loading, embeddings and helpers.

Weights come from Hugging Face and are cached in ``models/clip/`` on first use.
``transformers`` and PyTorch are imported only when they are needed.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageOps

from forensics_app.core import ImageDocument
from forensics_app.paths import MODELS_DIR

CLIP_CACHE_DIR = MODELS_DIR / "clip"
CLIP_MODELS = (
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
)
DEFAULT_MODEL = CLIP_MODELS[0]
DEVICES = ("auto", "cpu", "mps", "cuda")
MODEL_CACHE_SIZE = 2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Set on every chart these tools return, so a re-run uses the photo instead.
CHART_MARKER = "forensics_clip_chart"


# ---------------------------------------------------------------------------
# Device and model loading
# ---------------------------------------------------------------------------


def _available_accelerator() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device: str) -> str:
    device = device if device in DEVICES else "auto"
    return _available_accelerator() if device == "auto" else device


def _import_clip() -> tuple[Any, Any]:
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face transformers is not installed. Install it with:\n"
            "    pip install transformers"
        ) from exc
    return CLIPModel, CLIPProcessor


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def is_model_downloaded(name: str) -> bool:
    folder = CLIP_CACHE_DIR / ("models--" + name.replace("/", "--"))
    return any(folder.glob("snapshots/*/config.json"))


def is_model_cached(name: str, device: str) -> bool:
    return (name, device) in _MODEL_CACHE


def load_clip(name: str, device: str) -> tuple[Any, Any]:
    """Return ``(model, processor)``, downloading the weights on first use."""
    key = (name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    clip_model, clip_processor = _import_clip()
    CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = clip_model.from_pretrained(name, cache_dir=CLIP_CACHE_DIR)
    processor = clip_processor.from_pretrained(name, cache_dir=CLIP_CACHE_DIR)
    while len(_MODEL_CACHE) >= MODEL_CACHE_SIZE:
        del _MODEL_CACHE[next(iter(_MODEL_CACHE))]
    _MODEL_CACHE[key] = (model.to(device).eval(), processor)
    return _MODEL_CACHE[key]


def loading_message(names: list[str], device: str) -> str:
    """Progress text announcing a download or load, empty if all are in memory."""
    missing = [n for n in names if not is_model_cached(n, device)]
    if not missing:
        return ""
    if any(not is_model_downloaded(n) for n in missing):
        return "Downloading and loading CLIP (first use only)…\n"
    return "Loading CLIP…\n"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _as_tensor(output: Any) -> Any:
    # Newer transformers versions may return an output object instead of a tensor.
    import torch

    return output if torch.is_tensor(output) else output.pooler_output


def embed_images(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    device: str = "cpu",
    *,
    batch_size: int = 32,
) -> Any:
    """Return L2-normalised image embeddings, shape ``(N, D)``, on the CPU."""
    import torch
    import torch.nn.functional as F

    chunks = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[start : start + batch_size]]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            emb = _as_tensor(model.get_image_features(**inputs))
            chunks.append(F.normalize(emb.float(), dim=-1).cpu())
    return torch.cat(chunks)


def embed_texts(
    model: Any, processor: Any, texts: list[str], device: str = "cpu"
) -> Any:
    """Return L2-normalised text embeddings, shape ``(N, D)``, on the CPU."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        inputs = processor(text=list(texts), return_tensors="pt", padding=True)
        emb = _as_tensor(model.get_text_features(**inputs.to(device)))
    return F.normalize(emb.float(), dim=-1).cpu()


def build_prompts(labels: list[str], templates: list[str]) -> list[list[str]]:
    """Return, for each label, its sentence for every template."""
    return [[t.replace("{}", label) for t in templates] for label in labels]


def class_embeddings(
    model: Any,
    processor: Any,
    labels: list[str],
    templates: list[str],
    device: str = "cpu",
) -> Any:
    """One normalised embedding per class, averaged over the templates.

    Averaging normalised sentence embeddings is the CLIP paper's prompt
    ensembling; with one template it is just that sentence's embedding.
    """
    import torch.nn.functional as F

    flat = [p for group in build_prompts(labels, templates) for p in group]
    emb = embed_texts(model, processor, flat, device)
    emb = emb.reshape(len(labels), len(templates), -1).mean(dim=1)
    return F.normalize(emb, dim=-1)


def logit_scale(model: Any) -> float:
    return float(model.logit_scale.detach().exp())


def softmax_probabilities(similarities: Any, scale: float) -> np.ndarray:
    """Softmax over the last axis of scaled cosine similarities."""
    logits = np.asarray(similarities, dtype=np.float64) * scale
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def parse_labels(text: str) -> list[str]:
    """Return the class texts: one per line, or comma-separated on a single line.

    Blank entries and repeats (case-insensitive) are dropped, order is kept.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) == 1 and "," in lines[0]:
        lines = [part.strip() for part in lines[0].split(",")]
    return _unique(lines)


def validate_template(template: str) -> str:
    """Return the template if it has exactly one ``{}`` placeholder."""
    template = template.strip()
    if template.count("{}") != 1:
        raise ValueError(
            f'Template "{template}" must contain "{{}}" exactly once, '
            'e.g. "a photo of a {}."'
        )
    return template


def parse_templates(text: str) -> list[str]:
    """One template per non-empty line, each validated; repeats dropped."""
    templates = [validate_template(t) for t in text.splitlines() if t.strip()]
    if not templates:
        raise ValueError('Enter at least one template, e.g. "a photo of a {}."')
    return _unique(templates)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Images, folders and figures
# ---------------------------------------------------------------------------


def list_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found:\n{folder}")
    pattern = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p
        for p in pattern
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.startswith(".")
    )


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return (ImageOps.exif_transpose(image) or image).convert("RGB")


def mark_chart(image: Image.Image) -> Image.Image:
    image.info[CHART_MARKER] = True
    return image


def is_chart(image: Image.Image | None) -> bool:
    return image is not None and bool(image.info.get(CHART_MARKER))


def source_image(document: ImageDocument) -> tuple[Image.Image, bool]:
    """The image to analyse, and whether the original replaced a CLIP chart."""
    if document.current is None:
        raise ValueError("Open an image first.")
    if is_chart(document.current) and document.original is not None:
        if not is_chart(document.original):
            return document.original, True
    return document.current, False


def figure_to_image(fig: Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return mark_chart(Image.open(buf).convert("RGB"))


def thumbnail(image: Image.Image, size: int = 256) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail((size, size))
    return copy
