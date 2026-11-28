"""Shared CLIP code for the Set 10 tools: loading, embeddings and helpers.

Weights come from Hugging Face and are cached in ``models/clip/`` on first use.
``transformers`` and PyTorch are imported only when they are needed.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import os
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageOps

from forensics_app.core import ImageDocument
from forensics_app.paths import MODELS_DIR

CLIP_CACHE_DIR = MODELS_DIR / "clip"
DISK_INDEX_CACHE_DIR = CLIP_CACHE_DIR / "index_cache"
CLIP_MODELS = (
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
)
DEFAULT_MODEL = CLIP_MODELS[0]
DEVICES = ("auto", "cpu", "mps", "cuda")
MODEL_CACHE_SIZE = 2
INDEX_CACHE_SIZE = 4
MAX_DISK_CACHE_FILES = 24
DEFAULT_TEXT_BATCH_SIZE = 256
EXIF_ORIENTATION_TAG = 274

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Set on every chart these tools return, so a re-run uses the photo instead.
CHART_MARKER = "forensics_clip_chart"

ProgressCallback = Callable[[int, int], None]


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


def _default_image_batch_size(device: str) -> int:
    return 64 if device in ("cuda", "mps") else 32


def _is_torch_module(model: Any) -> bool:
    try:
        import torch

        return isinstance(model, torch.nn.Module)
    except Exception:
        return False


def _autocast_context(model: Any, device: str) -> Any:
    """Use fp16 autocast on MPS/CUDA for real PyTorch models; no-op otherwise."""
    if device not in ("cuda", "mps") or not _is_torch_module(model):
        return contextlib.nullcontext()
    import torch

    try:
        return torch.autocast(device_type=device, dtype=torch.float16)
    except (RuntimeError, TypeError):
        return contextlib.nullcontext()


def _release_device_memory(model: Any, device: str) -> None:
    if device not in ("cuda", "mps") or not _is_torch_module(model):
        return
    import torch

    try:
        model.to("cpu")
    except Exception:
        pass
    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _import_clip() -> tuple[Any, Any]:
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face transformers is not installed. Install it with:\n"
            "    pip install transformers"
        ) from exc
    return CLIPModel, CLIPProcessor


_MODEL_CACHE: OrderedDict[tuple[str, str], tuple[Any, Any]] = OrderedDict()
_INDEX_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()


def is_model_downloaded(name: str) -> bool:
    folder = CLIP_CACHE_DIR / ("models--" + name.replace("/", "--"))
    return any(folder.glob("snapshots/*/config.json"))


def is_model_cached(name: str, device: str) -> bool:
    return (name, device) in _MODEL_CACHE


def _load_pretrained(
    loader: Any,
    name: str,
    *,
    local_only: bool,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load from Hugging Face cache first when available, falling back cleanly."""
    extra = extra_kwargs or {}
    attempts: list[dict[str, Any]] = []
    if local_only:
        if extra:
            attempts.append({"local_files_only": True, **extra})
        attempts.append({"local_files_only": True})
    if extra:
        attempts.append(dict(extra))
    attempts.append({})

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return loader.from_pretrained(name, cache_dir=CLIP_CACHE_DIR, **kwargs)
        except (OSError, TypeError, ValueError) as exc:
            last_error = exc
            if not kwargs:
                raise
    raise last_error  # pragma: no cover


def load_clip(name: str, device: str) -> tuple[Any, Any]:
    """Return ``(model, processor)``, downloading the weights on first use."""
    key = (name, device)
    if key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(key)
        return _MODEL_CACHE[key]
    clip_model, clip_processor = _import_clip()
    CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_only = is_model_downloaded(name)
    model = _load_pretrained(clip_model, name, local_only=local_only)
    processor = _load_pretrained(
        clip_processor,
        name,
        local_only=local_only,
        extra_kwargs={"backend": "torchvision"},
    )
    while len(_MODEL_CACHE) >= MODEL_CACHE_SIZE:
        (_, evicted_device), (evicted_model, _) = _MODEL_CACHE.popitem(last=False)
        _release_device_memory(evicted_model, evicted_device)
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


def _ensure_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def _encode_image_batch(
    model: Any,
    processor: Any,
    batch: list[Image.Image],
    device: str,
) -> Any:
    import torch.nn.functional as F

    inputs = processor(images=batch, return_tensors="pt").to(device)
    with _autocast_context(model, device):
        emb = _as_tensor(model.get_image_features(**inputs))
    return F.normalize(emb.float(), dim=-1).cpu()


def embed_images(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    device: str = "cpu",
    *,
    batch_size: int | None = None,
    progress: ProgressCallback | None = None,
) -> Any:
    """Return L2-normalised image embeddings, shape ``(N, D)``, on the CPU."""
    import torch

    bs = batch_size if batch_size is not None else _default_image_batch_size(device)
    total = len(images)
    chunks = []
    with torch.inference_mode():
        for start in range(0, total, bs):
            batch = [_ensure_rgb(im) for im in images[start : start + bs]]
            chunks.append(_encode_image_batch(model, processor, batch, device))
            if progress is not None:
                progress(min(start + len(batch), total), total)
    return torch.cat(chunks)


def embed_image_paths(
    model: Any,
    processor: Any,
    paths: list[Path],
    device: str = "cpu",
    *,
    batch_size: int | None = None,
    progress: ProgressCallback | None = None,
) -> Any:
    """Stream images from disk in batches with overlapped CPU decoding.

    Peak memory is bounded to at most two batches of decoded images instead of
    holding the whole folder in RAM at once.
    """
    import torch

    bs = batch_size if batch_size is not None else _default_image_batch_size(device)
    total = len(paths)
    if total == 0:
        raise ValueError("No images to embed.")

    batches = [paths[start : start + bs] for start in range(0, total, bs)]
    workers = min(4, os.cpu_count() or 2) if total > 1 else 1
    chunks = []
    done = 0

    with torch.inference_mode():
        if workers <= 1:
            for chunk_paths in batches:
                batch = [open_rgb(p) for p in chunk_paths]
                chunks.append(_encode_image_batch(model, processor, batch, device))
                done += len(batch)
                if progress is not None:
                    progress(done, total)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = [pool.submit(open_rgb, p) for p in batches[0]]
                for idx in range(len(batches)):
                    next_pending = (
                        [pool.submit(open_rgb, p) for p in batches[idx + 1]]
                        if idx + 1 < len(batches)
                        else None
                    )
                    batch = [fut.result() for fut in pending]
                    chunks.append(_encode_image_batch(model, processor, batch, device))
                    done += len(batch)
                    if progress is not None:
                        progress(done, total)
                    pending = next_pending or []
    return torch.cat(chunks)


def embed_texts(
    model: Any,
    processor: Any,
    texts: list[str],
    device: str = "cpu",
    *,
    batch_size: int = DEFAULT_TEXT_BATCH_SIZE,
) -> Any:
    """Return L2-normalised text embeddings, shape ``(N, D)``, on the CPU."""
    import torch
    import torch.nn.functional as F

    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            try:
                inputs = processor(
                    text=batch, return_tensors="pt", padding=True, truncation=True
                )
            except TypeError:
                inputs = processor(text=batch, return_tensors="pt", padding=True)
            with _autocast_context(model, device):
                emb = _as_tensor(model.get_text_features(**inputs.to(device)))
            chunks.append(F.normalize(emb.float(), dim=-1).cpu())
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def build_prompts(labels: list[str], templates: list[str]) -> list[list[str]]:
    """Return, for each label, its sentence for every template."""
    return [[t.replace("{}", label) for t in templates] for label in labels]


def template_and_ensemble_embeddings(
    model: Any,
    processor: Any,
    labels: list[str],
    templates: list[str],
    device: str = "cpu",
) -> tuple[list[Any], Any]:
    """Encode every ``(label, template)`` prompt once and derive both views.

    Returns ``(per_template, ensemble)``, where ``per_template[t]`` has shape
    ``(C, D)`` for template ``t`` and ``ensemble`` is the L2-normalised mean
    across all templates, also shape ``(C, D)``.
    """
    import torch.nn.functional as F

    flat = [p for group in build_prompts(labels, templates) for p in group]
    emb = embed_texts(model, processor, flat, device).reshape(
        len(labels), len(templates), -1
    )
    per_template = [emb[:, t, :] for t in range(len(templates))]
    if len(templates) == 1:
        ensemble = per_template[0]
    else:
        ensemble = F.normalize(emb.mean(dim=1), dim=-1)
    return per_template, ensemble


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
    _, ensemble = template_and_ensemble_embeddings(
        model, processor, labels, templates, device
    )
    return ensemble


def logit_scale(model: Any) -> float:
    return float(model.logit_scale.detach().exp())


def softmax_probabilities(similarities: Any, scale: float) -> np.ndarray:
    """Softmax over the last axis of scaled cosine similarities."""
    logits = np.asarray(similarities, dtype=np.float64) * scale
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Shared folder embedding cache (in-memory LRU + incremental on-disk cache)
# ---------------------------------------------------------------------------


def _path_entries(
    paths: list[Path], stats: list[tuple[int, int]] | None = None
) -> tuple[tuple[str, int, int], ...]:
    if stats is not None and len(stats) == len(paths):
        return tuple(
            (str(p), mtime_ns, size) for p, (mtime_ns, size) in zip(paths, stats)
        )
    entries = []
    for p in paths:
        st = p.stat()
        entries.append((str(p), st.st_mtime_ns, st.st_size))
    return tuple(entries)


def _disk_cache_path(model_name: str, paths: list[Path]) -> Path:
    try:
        common = os.path.commonpath([str(p.parent) for p in paths])
    except ValueError:
        common = ""
    digest = hashlib.sha256(f"{model_name}\0{common}".encode("utf-8")).hexdigest()[:24]
    return DISK_INDEX_CACHE_DIR / f"{digest}.npz"


def _load_disk_cache(cache_path: Path) -> dict[tuple[str, int, int], np.ndarray]:
    if not cache_path.is_file():
        return {}
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_paths = data["paths"]
            mtimes = data["mtimes"]
            sizes = data["sizes"]
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            if (
                len(cached_paths) != len(embeddings)
                or len(mtimes) != len(embeddings)
                or len(sizes) != len(embeddings)
            ):
                return {}
            return {
                (str(p), int(m), int(s)): embeddings[i]
                for i, (p, m, s) in enumerate(zip(cached_paths, mtimes, sizes))
            }
    except Exception:
        return {}


def _save_disk_cache(
    cache_path: Path,
    entries: tuple[tuple[str, int, int], ...],
    embeddings: np.ndarray,
) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp.npz")
        np.savez(
            tmp_path,
            paths=np.array([e[0] for e in entries], dtype=np.str_),
            mtimes=np.array([e[1] for e in entries], dtype=np.int64),
            sizes=np.array([e[2] for e in entries], dtype=np.int64),
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )
        os.replace(tmp_path, cache_path)
        _prune_disk_cache(cache_path.parent)
    except OSError:
        pass


def _prune_disk_cache(cache_dir: Path) -> None:
    try:
        files = sorted(
            (p for p in cache_dir.iterdir() if p.suffix == ".npz" and p.is_file()),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in files[MAX_DISK_CACHE_FILES:]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


def folder_embeddings(
    model: Any,
    processor: Any,
    paths: list[Path],
    device: str = "cpu",
    *,
    model_name: str,
    stats: list[tuple[int, int]] | None = None,
    progress: ProgressCallback | None = None,
    use_disk_cache: bool | None = None,
) -> tuple[Any, bool]:
    """Return embeddings for ``paths`` and whether every item came from cache.

    Uses a shared in-memory LRU cache across Search and Evaluation, plus an
    incremental on-disk cache under ``models/clip/index_cache/`` for real
    PyTorch models so adding or editing files only embeds the changed images.
    """
    import torch

    entries = _path_entries(paths, stats)
    key = (model_name, entries)
    if key in _INDEX_CACHE:
        _INDEX_CACHE.move_to_end(key)
        return _INDEX_CACHE[key], True

    disk_enabled = (
        _is_torch_module(model) if use_disk_cache is None else use_disk_cache
    ) and bool(model_name)
    disk_map: dict[tuple[str, int, int], np.ndarray] = {}
    cache_path: Path | None = None
    if disk_enabled and paths:
        cache_path = _disk_cache_path(model_name, paths)
        disk_map = _load_disk_cache(cache_path)

    missing_indices = [i for i, e in enumerate(entries) if e not in disk_map]
    if not missing_indices and entries:
        emb = torch.from_numpy(np.stack([disk_map[e] for e in entries]))
        from_cache = True
    else:
        if len(missing_indices) == len(paths):
            emb = embed_image_paths(
                model, processor, paths, device, progress=progress
            )
        else:
            missing_paths = [paths[i] for i in missing_indices]
            new_emb = embed_image_paths(
                model, processor, missing_paths, device, progress=progress
            ).numpy()
            dim = new_emb.shape[1]
            merged = np.empty((len(paths), dim), dtype=np.float32)
            for i, e in enumerate(entries):
                if e in disk_map:
                    merged[i] = disk_map[e]
            for pos, i in enumerate(missing_indices):
                merged[i] = new_emb[pos]
            emb = torch.from_numpy(merged)
        from_cache = False
        if cache_path is not None:
            _save_disk_cache(cache_path, entries, emb.numpy())

    while len(_INDEX_CACHE) >= INDEX_CACHE_SIZE:
        _INDEX_CACHE.popitem(last=False)
    _INDEX_CACHE[key] = emb
    return emb, from_cache


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


def scan_images(
    folder: str | Path, recursive: bool = False
) -> list[tuple[Path, int, int]]:
    """Return sorted ``(path, mtime_ns, size)`` tuples via a single scandir walk."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found:\n{root}")

    found: list[tuple[Path, int, int]] = []
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as it:
            for entry in it:
                name = entry.name
                if name.startswith("."):
                    continue
                if entry.is_dir(follow_symlinks=True):
                    if recursive:
                        stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=True):
                    continue
                _, ext = os.path.splitext(name)
                if ext.lower() not in IMAGE_EXTENSIONS:
                    continue
                st = entry.stat(follow_symlinks=True)
                found.append((Path(entry.path), st.st_mtime_ns, st.st_size))
    found.sort(key=lambda item: item[0])
    return found


def list_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    return [path for path, _, _ in scan_images(folder, recursive=recursive)]


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        exif = image.getexif()
        orientation = exif.get(EXIF_ORIENTATION_TAG, 1) if exif else 1
        if orientation != 1:
            transposed = ImageOps.exif_transpose(image)
            if transposed is not None:
                image = transposed
        return image.convert("RGB") if image.mode != "RGB" else image.copy()


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
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return mark_chart(Image.fromarray(rgba[:, :, :3].copy(), mode="RGB"))


def thumbnail(image: Image.Image, size: int = 256) -> Image.Image:
    copy = image.copy() if image.mode == "RGB" else image.convert("RGB")
    if max(copy.size) > size:
        copy.thumbnail((size, size))
    return copy
