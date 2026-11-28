"""Set 10 extra — CLIP text-to-image search in a folder.

Images and text live in the same CLIP space, so a folder of images can be
ranked by a text query ("something you drink in the morning"). Folder
embeddings are kept in memory per model, so repeated searches in the same
folder only embed the query.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    _INDEX_CACHE as _INDEX_CACHE,
)
from .clip_common import (
    CLIP_MODELS,
    DEFAULT_MODEL,
    DEVICES,
    ProgressCallback,
    embed_texts,
    figure_to_image,
    folder_embeddings,
    load_clip,
    loading_message,
    open_rgb,
    resolve_device,
    scan_images,
    thumbnail,
)
from .clip_common import (
    INDEX_CACHE_SIZE as INDEX_CACHE_SIZE,
)

DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class SearchHit:
    path: Path
    similarity: float


def rank(similarities: Any, paths: list[Path], top_k: int) -> list[SearchHit]:
    sims = np.asarray(similarities, dtype=np.float64)
    k = min(max(1, top_k), len(sims))
    if k < len(sims):
        nth = np.partition(sims, len(sims) - k)[len(sims) - k]
        above = np.flatnonzero(sims > nth)
        equal = np.flatnonzero(sims == nth)[: k - len(above)]
        candidates = np.concatenate((above, equal))
        order = candidates[np.argsort(-sims[candidates], kind="stable")]
    else:
        order = np.argsort(-sims, kind="stable")[:k]
    return [SearchHit(paths[i], float(sims[i])) for i in order]


def search(
    model: Any,
    processor: Any,
    paths: list[Path],
    query: str,
    device: str = "cpu",
    *,
    model_name: str,
    top_k: int = DEFAULT_TOP_K,
    stats: list[tuple[int, int]] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[SearchHit], dict[str, Any]]:
    """Rank the images by cosine similarity to a text query."""
    if not paths:
        raise ValueError("The folder has no images.")
    if not query.strip():
        raise ValueError("Type a text query.")
    started = time.perf_counter()
    gallery, cached = folder_embeddings(
        model,
        processor,
        paths,
        device,
        model_name=model_name,
        stats=stats,
        progress=progress,
    )
    query_emb = embed_texts(model, processor, [query.strip()], device)
    sims = (query_emb @ gallery.T)[0].numpy()
    info = {
        "indexed": len(paths),
        "from_cache": cached,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "mean_similarity": float(np.mean(sims)),
    }
    return rank(sims, paths, top_k), info


def _short(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return path.name


def render_hits(
    hits: list[SearchHit], query: str, root: Path | None = None
) -> Image.Image:
    """The best matches side by side, as in the notebook."""
    cols = min(4, len(hits))
    rows = -(-len(hits) // cols)
    fig = Figure(figsize=(3.4 * cols, 3.3 * rows + 0.5), dpi=100)
    for i, hit in enumerate(hits, 1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(thumbnail(open_rgb(hit.path)))
        ax.set_title(f"{_short(hit.path, root)}  sim={hit.similarity:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f'query: "{query}"', fontsize=12)
    fig.tight_layout()
    return figure_to_image(fig)


def search_details(
    hits: list[SearchHit],
    info: dict[str, Any],
    parameters: dict[str, Any],
    device: str,
    root: Path,
) -> dict[str, Any]:
    indexed = info["indexed"]
    details: dict[str, Any] = {
        "Folder": str(root),
        "Model": parameters.get("model", DEFAULT_MODEL),
        "Device": device,
        "Query": f'"{parameters.get("text", "")}"',
        "Images indexed": f"{indexed} (from memory)" if info["from_cache"] else indexed,
        "Mean similarity (folder)": f"{info['mean_similarity']:.3f}",
    }
    for i, hit in enumerate(hits, 1):
        details[f"#{i} {_short(hit.path, root)}"] = f"{hit.similarity:.3f}"
    details["Time"] = f"{info['elapsed_ms']:.0f} ms"
    return details


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def ask_search_parameters(
    parent: tk.Misc, start_folder: Path | None
) -> dict[str, Any] | None:
    dialog = tk.Toplevel(parent)
    dialog.title("CLIP Image Search")
    dialog.transient(parent)
    dialog.grab_set()

    folder_var = tk.StringVar(value=str(start_folder) if start_folder else "")
    recursive_var = tk.BooleanVar(value=True)
    text_var = tk.StringVar(value="something you drink in the morning")
    top_k_var = tk.IntVar(value=DEFAULT_TOP_K)
    model_var = tk.StringVar(value=DEFAULT_MODEL)
    device_var = tk.StringVar(value="auto")

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    folder_box = ttk.LabelFrame(main, text="A. Folder to search", padding=10)
    folder_box.pack(fill="x", pady=(0, 8))
    ttk.Entry(folder_box, textvariable=folder_var, width=52).grid(
        row=0, column=0, sticky="ew"
    )

    def browse() -> None:
        chosen = filedialog.askdirectory(
            parent=dialog,
            title="Folder of images",
            initialdir=folder_var.get() or str(PROJECT_ROOT),
        )
        if chosen:
            folder_var.set(chosen)

    ttk.Button(folder_box, text="Browse…", command=browse).grid(
        row=0, column=1, padx=(6, 0)
    )
    ttk.Checkbutton(
        folder_box, text="Include sub-folders", variable=recursive_var
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))
    folder_box.columnconfigure(0, weight=1)

    query_box = ttk.LabelFrame(main, text="B. Text query", padding=10)
    query_box.pack(fill="x", pady=(0, 8))
    ttk.Entry(query_box, textvariable=text_var, width=52).grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    ttk.Label(query_box, text="Results:").grid(row=1, column=0, sticky="w", pady=(4, 0))
    ttk.Spinbox(query_box, textvariable=top_k_var, from_=1, to=20, width=5).grid(
        row=1, column=1, sticky="w", padx=6, pady=(4, 0)
    )

    model_box = ttk.LabelFrame(main, text="C. Model", padding=10)
    model_box.pack(fill="x", pady=(0, 8))
    ttk.Combobox(model_box, textvariable=model_var, values=CLIP_MODELS, width=34).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(model_box, text="Device:").grid(row=0, column=1, padx=(12, 4))
    ttk.Combobox(
        model_box, textvariable=device_var, values=DEVICES, state="readonly", width=7
    ).grid(row=0, column=2)

    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(4, 0))
    result: dict[str, Any] | None = None

    def submit() -> None:
        nonlocal result
        folder = Path(folder_var.get().strip()).expanduser()
        if not folder.is_dir():
            messagebox.showerror("Folder", "Choose an existing folder.", parent=dialog)
            return
        if not text_var.get().strip():
            messagebox.showerror("Query", "Type a text query.", parent=dialog)
            return
        try:
            top_k = int(top_k_var.get())
        except (tk.TclError, ValueError):
            top_k = DEFAULT_TOP_K
        result = {
            "folder": str(folder),
            "recursive": bool(recursive_var.get()),
            "text": text_var.get().strip(),
            "top_k": min(max(top_k, 1), 20),
            "model": model_var.get().strip() or DEFAULT_MODEL,
            "device": device_var.get(),
        }
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Search", command=submit).pack(side="right", padx=(0, 8))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _search_job(
    parameters: dict[str, Any],
    set_status: Callable[[str], None] | None = None,
) -> tuple[list[SearchHit], dict[str, Any], Image.Image, str, Path]:
    root = Path(parameters["folder"])
    scanned = scan_images(root, bool(parameters.get("recursive", True)))
    if not scanned:
        raise ValueError(f"No images found in:\n{root}")
    paths = [p for p, _, _ in scanned]
    stats = [(mtime_ns, size) for _, mtime_ns, size in scanned]
    device = resolve_device(str(parameters.get("device", "auto")))
    model_name = str(parameters.get("model") or DEFAULT_MODEL)
    if parameters.get("clip") is not None:
        model, processor = parameters["clip"]
    else:
        if set_status is not None:
            set_status(f"Loading {model_name.split('/')[-1]}…")
        model, processor = load_clip(model_name, device)
    query = str(parameters.get("text") or "")
    progress = (
        (lambda done, total: set_status(f"Embedding images: {done:,} / {total:,}…"))
        if set_status is not None
        else None
    )
    hits, info = search(
        model,
        processor,
        paths,
        query,
        device,
        model_name=model_name,
        top_k=int(parameters.get("top_k", DEFAULT_TOP_K)),
        stats=stats,
        progress=progress,
    )
    if set_status is not None:
        set_status("Rendering matches…")
    return hits, info, render_hits(hits, query, root), device, root


def run_image_search(
    parent: tk.Misc,
    document: ImageDocument,
    parameters: dict[str, Any] | None = None,
) -> ToolResult | None:
    """Ask for a folder and a text query, then show the best matches."""
    if parameters is None:
        start = document.path.parent if document.path else None
        parameters = ask_search_parameters(parent, start)
        if parameters is None:
            return None
    device = resolve_device(str(parameters.get("device", "auto")))
    message = (
        ""
        if parameters.get("clip") is not None
        else loading_message([str(parameters.get("model") or DEFAULT_MODEL)], device)
    ) + "Embedding the folder (first search only) and ranking…"
    status_line = [""]

    def set_status(text: str) -> None:
        status_line[0] = text

    hits, info, figure, device, root = run_with_progress(
        parent,
        "CLIP image search",
        message,
        _search_job,
        parameters,
        set_status,
        status=lambda: status_line[0],
    )
    best = hits[0]
    return ToolResult(
        message=(
            f"Best match: {_short(best.path, root)} "
            f"(similarity {best.similarity:.3f}) of {info['indexed']} images."
        ),
        image=figure,
        details=search_details(hits, info, parameters, device, root),
    )
