"""Set 10.1 — CLIP zero-shot classification.

CLIP embeds the image and one sentence per class (a prompt template filled with
the class text) in a shared space; the scaled cosine similarities, passed
through a softmax, give one probability per class. The classes are just text,
so any number of classes with any wording can be used without training.

Modes:

* **Classify**: one bar chart; several templates are averaged (prompt
  ensembling, as in the CLIP paper).
* **Compare templates**: one chart per template plus their ensemble.
* **Compare models**: the same prompts with several CLIP models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.utils.background import run_with_progress

from .base import ForensicsTool, ToolResult
from .clip_common import (
    CLIP_MODELS,
    DEFAULT_MODEL,
    DEVICES,
    class_embeddings,
    embed_images,
    figure_to_image,
    is_model_downloaded,
    load_clip,
    loading_message,
    logit_scale,
    parse_labels,
    parse_templates,
    resolve_device,
    softmax_probabilities,
    source_image,
    template_and_ensemble_embeddings,
    thumbnail,
)

MODE_CLASSIFY = "classify"
MODE_TEMPLATES = "templates"
MODE_MODELS = "models"
MODES = {
    MODE_CLASSIFY: "Classify",
    MODE_TEMPLATES: "Compare templates",
    MODE_MODELS: "Compare models",
}

DEFAULT_TEMPLATE = "a photo of a {}."
# Prompt ensembling as in the CLIP paper: average the text embeddings per class.
ENSEMBLE_TEMPLATES = (
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a good photo of a {}.",
    "a cropped photo of a {}.",
    "a picture of a {}.",
)
COMPARE_TEMPLATES = ("{}", "a photo of a {}.", "a photo of a cup of {}.")

LABEL_EXAMPLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "Animals": (
        ("a photo of a {}.",),
        ("cat", "dog", "tiger", "rabbit", "fox", "owl"),
    ),
    "Counting cats": (
        ("a photo of {}.",),
        ("one cat", "two cats", "three cats", "no cats"),
    ),
    "Cats: where": (
        ("a photo of cats {}.",),
        ("on a sofa", "on the grass", "in a car", "on a bed"),
    ),
    "Drinks": (COMPARE_TEMPLATES, ("coffee", "tea", "soup", "hot chocolate")),
    "Image style": (
        ("a {} of an object.",),
        ("photo", "painting", "cartoon", "pencil sketch"),
    ),
}

BAR_COLOR = "#3b6ea8"
TOP_COLOR = "#2f855a"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroShotResult:
    labels: list[str]
    probabilities: np.ndarray
    similarities: np.ndarray
    templates: list[str]
    logit_scale: float
    elapsed_ms: float

    def ranked(self) -> list[tuple[str, float, float]]:
        """``(label, probability, cosine similarity)``, most probable first."""
        order = np.argsort(-self.probabilities, kind="stable")
        return [
            (self.labels[i], float(self.probabilities[i]), float(self.similarities[i]))
            for i in order
        ]

    @property
    def top(self) -> tuple[str, float]:
        idx = int(np.argmax(self.probabilities))
        return self.labels[idx], float(self.probabilities[idx])


def _check_labels(labels: list[str]) -> None:
    if len(labels) < 2:
        raise ValueError("Give at least two classes to choose between.")


def _result(
    labels: list[str],
    templates: list[str],
    similarities: np.ndarray,
    scale: float,
    started: float,
) -> ZeroShotResult:
    return ZeroShotResult(
        labels=list(labels),
        probabilities=softmax_probabilities(similarities, scale),
        similarities=np.asarray(similarities, dtype=np.float64),
        templates=list(templates),
        logit_scale=scale,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def classify(
    model: Any,
    processor: Any,
    image: Image.Image,
    labels: list[str],
    templates: list[str],
    device: str = "cpu",
) -> ZeroShotResult:
    """Return a probability and a cosine similarity for every label.

    With several templates the class embeddings are averaged (ensembling).
    """
    _check_labels(labels)
    started = time.perf_counter()
    image_emb = embed_images(model, processor, [image], device)
    text_emb = class_embeddings(model, processor, labels, templates, device)
    sims = (image_emb @ text_emb.T)[0].numpy()
    return _result(labels, templates, sims, logit_scale(model), started)


def compare_templates(
    model: Any,
    processor: Any,
    image: Image.Image,
    labels: list[str],
    templates: list[str],
    device: str = "cpu",
) -> list[tuple[str, ZeroShotResult]]:
    """One result per template, then the ensemble of all of them."""
    _check_labels(labels)
    started = time.perf_counter()
    image_emb = embed_images(model, processor, [image], device)
    per_template, ensemble = template_and_ensemble_embeddings(
        model, processor, labels, templates, device
    )
    scale = logit_scale(model)
    panels: list[tuple[str, ZeroShotResult]] = []
    for template, text_emb in zip(templates, per_template):
        sims = (image_emb @ text_emb.T)[0].numpy()
        panels.append(
            (f'"{template}"', _result(labels, [template], sims, scale, started))
        )
    if len(templates) > 1:
        sims = (image_emb @ ensemble.T)[0].numpy()
        panels.append(
            (
                f"ensemble of {len(templates)} templates",
                _result(labels, templates, sims, scale, started),
            )
        )
    return panels


Loader = Callable[[str, str], tuple[Any, Any]]


def compare_models(
    loader: Loader,
    model_names: list[str],
    image: Image.Image,
    labels: list[str],
    templates: list[str],
    device: str = "cpu",
    *,
    on_model: Callable[[int, int, str], None] | None = None,
) -> list[tuple[str, ZeroShotResult]]:
    """The same prompts classified by each model."""
    if not model_names:
        raise ValueError("Choose at least one model to compare.")
    panels = []
    total = len(model_names)
    for index, name in enumerate(model_names, start=1):
        short_name = name.split("/")[-1]
        if on_model is not None:
            on_model(index, total, short_name)
        model, processor = loader(name, device)
        result = classify(model, processor, image, labels, templates, device)
        panels.append((short_name, result))
    return panels


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def template_caption(templates: list[str]) -> str:
    if len(templates) == 1:
        return f'template: "{templates[0]}"'
    return f"ensemble of {len(templates)} templates"


def _bars(ax: Any, result: ZeroShotResult, order: list[str], title: str) -> None:
    index = {label: i for i, label in enumerate(result.labels)}
    values = [float(result.probabilities[index[label]]) for label in order]
    best = max(values)
    colors = [TOP_COLOR if v == best else BAR_COLOR for v in values]
    ax.barh(order, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title(title, fontsize=10)
    for i, value in enumerate(values):
        if value > 0.85:  # inside the bar end, so it never overlaps the frame
            ax.text(
                value - 0.02,
                i,
                f"{value:.2f}",
                va="center",
                ha="right",
                fontsize=9,
                color="white",
            )
        else:
            ax.text(value + 0.02, i, f"{value:.2f}", va="center", fontsize=9)


def _show_image(ax: Any, image: Image.Image, title: str) -> None:
    ax.imshow(thumbnail(image, 800))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def render_result(
    image: Image.Image,
    result: ZeroShotResult,
    title: str = "",
) -> Image.Image:
    """Image on the left, probability bars (highest on top) on the right."""
    order = [label for label, _, _ in result.ranked()][::-1]
    height = max(3.2, 0.42 * len(order) + 1.4)
    fig = Figure(figsize=(10, height), dpi=110)
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.4))
    _show_image(fig.add_subplot(grid[0, 0]), image, title)
    _bars(
        fig.add_subplot(grid[0, 1]), result, order, template_caption(result.templates)
    )
    fig.tight_layout()
    return figure_to_image(fig)


def render_comparison(
    image: Image.Image,
    panels: list[tuple[str, ZeroShotResult]],
    title: str = "",
    suptitle: str = "",
) -> Image.Image:
    """The image, then one bar chart per panel with the same class order."""
    first = panels[0][1]
    order = [label for label, _, _ in first.ranked()][::-1]
    count = len(panels) + 1
    cols = min(count, 3)
    rows = -(-count // cols)
    row_h = max(3.0, 0.38 * len(order) + 1.3)
    fig = Figure(figsize=(4.6 * cols, row_h * rows + 0.5), dpi=100)
    _show_image(fig.add_subplot(rows, cols, 1), image, title)
    for i, (caption, result) in enumerate(panels, start=2):
        _bars(fig.add_subplot(rows, cols, i), result, order, caption)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return figure_to_image(fig)


# ---------------------------------------------------------------------------
# Details
# ---------------------------------------------------------------------------


def result_details(
    result: ZeroShotResult, model_name: str, device: str
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "Model": model_name,
        "Device": device,
        "Classes": len(result.labels),
        "Templates": " | ".join(result.templates),
        "Logit scale": f"{result.logit_scale:.1f}",
    }
    for rank, (label, prob, sim) in enumerate(result.ranked(), 1):
        details[f"#{rank} {label}"] = f"p = {prob:.3f}   cos = {sim:.3f}"
    details["Time"] = f"{result.elapsed_ms:.0f} ms"
    return details


def comparison_details(
    panels: list[tuple[str, ZeroShotResult]], heading: str
) -> dict[str, Any]:
    details: dict[str, Any] = {heading: len(panels)}
    for caption, result in panels:
        label, prob = result.top
        details[caption] = f"{label} ({prob:.3f})"
    labels = panels[0][1].labels
    for i, label in enumerate(labels):
        details[f"  p({label})"] = "  ".join(
            f"{float(r.probabilities[i]):.2f}" for _, r in panels
        )
    return details


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def ask_clip_parameters(parent: tk.Misc) -> dict[str, Any] | None:
    dialog = tk.Toplevel(parent)
    dialog.title("CLIP Zero-Shot Classification")
    dialog.transient(parent)
    dialog.grab_set()

    mode_var = tk.StringVar(value=MODE_CLASSIFY)
    model_var = tk.StringVar(value=DEFAULT_MODEL)
    device_var = tk.StringVar(value="auto")
    status_var = tk.StringVar()
    example_var = tk.StringVar(value="Animals")
    title_var = tk.StringVar()
    compare_vars = {name: tk.BooleanVar(value=True) for name in CLIP_MODELS}

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    # A. Mode ------------------------------------------------------------------------
    mode_box = ttk.LabelFrame(main, text="A. What to do", padding=10)
    mode_box.pack(fill="x", pady=(0, 8))
    for i, (key, text) in enumerate(MODES.items()):
        ttk.Radiobutton(mode_box, text=text, value=key, variable=mode_var).grid(
            row=i // 2, column=i % 2, sticky="w", padx=(0, 16)
        )

    # B. Model -----------------------------------------------------------------------
    model_box = ttk.LabelFrame(main, text="B. Model", padding=10)
    model_box.pack(fill="x", pady=(0, 8))
    combo = ttk.Combobox(
        model_box, textvariable=model_var, values=CLIP_MODELS, width=34
    )
    combo.grid(row=0, column=0, sticky="w")
    ttk.Label(model_box, text="Device:").grid(row=0, column=1, padx=(12, 4))
    ttk.Combobox(
        model_box, textvariable=device_var, values=DEVICES, state="readonly", width=7
    ).grid(row=0, column=2)
    status_label = ttk.Label(model_box, textvariable=status_var)
    status_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
    compare_frame = ttk.Frame(model_box)
    compare_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Label(compare_frame, text="Compare models:").pack(side="left")
    for name, var in compare_vars.items():
        size = " (1.7 GB)" if "large" in name else ""
        ttk.Checkbutton(
            compare_frame, text=name.split("/")[-1] + size, variable=var
        ).pack(side="left", padx=(6, 0))

    def refresh(_event: object = None) -> None:
        if is_model_downloaded(model_var.get().strip()):
            status_var.set("weights downloaded")
            status_label.configure(foreground="#006300")
        else:
            status_var.set("not downloaded yet: downloaded on first use")
            status_label.configure(foreground="#8a5a00")

    combo.bind("<<ComboboxSelected>>", refresh)
    combo.bind("<FocusOut>", refresh)

    # C. Classes and prompts -----------------------------------------------------------
    text_box = ttk.LabelFrame(main, text="C. Classes and prompts", padding=10)
    text_box.pack(fill="both", expand=True, pady=(0, 8))
    ttk.Label(text_box, text="Load example:").grid(row=0, column=0, sticky="w")
    example_combo = ttk.Combobox(
        text_box,
        textvariable=example_var,
        values=list(LABEL_EXAMPLES),
        state="readonly",
        width=24,
    )
    example_combo.grid(row=0, column=1, sticky="w", padx=6)
    ttk.Label(text_box, text="Classes (one per line):").grid(
        row=1, column=0, sticky="w", pady=(6, 0)
    )
    ttk.Label(text_box, text="Templates ({} = class, one per line):").grid(
        row=1, column=1, sticky="w", pady=(6, 0), padx=6
    )
    labels_text = tk.Text(text_box, width=30, height=8, wrap="none")
    labels_text.grid(row=2, column=0, sticky="nsew")
    templates_text = tk.Text(text_box, width=34, height=8, wrap="none")
    templates_text.grid(row=2, column=1, sticky="nsew", padx=6)
    ttk.Label(
        text_box,
        text="Several templates are averaged (Classify) "
        "or compared (Compare templates).",
        foreground="#555",
    ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
    template_buttons = ttk.Frame(text_box)
    template_buttons.grid(row=4, column=1, sticky="w", padx=6, pady=(4, 0))

    def set_templates(templates: tuple[str, ...]) -> None:
        templates_text.delete("1.0", "end")
        templates_text.insert("1.0", "\n".join(templates))

    ttk.Button(
        template_buttons,
        text="CLIP-paper ensemble",
        command=lambda: set_templates(ENSEMBLE_TEMPLATES),
    ).pack(side="left")
    ttk.Button(
        template_buttons,
        text="Prompt comparison",
        command=lambda: set_templates(COMPARE_TEMPLATES),
    ).pack(side="left", padx=(6, 0))
    text_box.columnconfigure(0, weight=1)
    text_box.columnconfigure(1, weight=1)
    text_box.rowconfigure(2, weight=1)

    def load_example(_event: object = None) -> None:
        templates, labels = LABEL_EXAMPLES[example_var.get()]
        labels_text.delete("1.0", "end")
        labels_text.insert("1.0", "\n".join(labels))
        set_templates(templates)

    example_combo.bind("<<ComboboxSelected>>", load_example)

    # D. Options ---------------------------------------------------------------------
    options = ttk.LabelFrame(main, text="D. Options", padding=10)
    options.pack(fill="x", pady=(0, 8))
    ttk.Label(options, text="Image title (optional):").grid(row=0, column=0, sticky="w")
    ttk.Entry(options, textvariable=title_var, width=30).grid(
        row=0, column=1, columnspan=5, sticky="w", padx=6
    )
    # Buttons ------------------------------------------------------------------------
    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(4, 0))
    result: dict[str, Any] | None = None

    def submit() -> None:
        nonlocal result
        labels = parse_labels(labels_text.get("1.0", "end"))
        if len(labels) < 2:
            messagebox.showerror(
                "Classes", "Enter at least two classes.", parent=dialog
            )
            return
        try:
            templates = parse_templates(templates_text.get("1.0", "end"))
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=dialog)
            return
        models = [name for name, var in compare_vars.items() if var.get()]
        if mode_var.get() == MODE_MODELS and not models:
            messagebox.showerror("Models", "Tick at least one model.", parent=dialog)
            return
        result = {
            "mode": mode_var.get(),
            "model": model_var.get().strip() or DEFAULT_MODEL,
            "models": models,
            "device": device_var.get(),
            "labels": labels,
            "templates": templates,
            "title": title_var.get().strip(),
        }
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Run", command=submit).pack(side="right", padx=(0, 8))

    load_example()
    refresh()
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _loader(parameters: dict[str, Any]) -> Loader:
    if parameters.get("loader") is not None:
        return parameters["loader"]
    if parameters.get("clip") is not None:
        return lambda _name, _device: parameters["clip"]
    return load_clip


def _models_used(parameters: dict[str, Any]) -> list[str]:
    if parameters.get("mode") == MODE_MODELS:
        return list(parameters.get("models") or CLIP_MODELS)
    return [str(parameters.get("model") or DEFAULT_MODEL)]


def _run_job(
    image: Image.Image,
    parameters: dict[str, Any],
    set_status: Callable[[str], None] | None = None,
) -> tuple[Image.Image, dict[str, Any], str]:
    """Run the chosen mode; executes in a worker thread."""
    mode = parameters.get("mode", MODE_CLASSIFY)
    device = resolve_device(str(parameters.get("device", "auto")))
    load = _loader(parameters)
    labels = list(parameters["labels"])
    templates = list(parameters.get("templates") or [DEFAULT_TEMPLATE])
    title = str(parameters.get("title") or "")
    model_name = str(parameters.get("model") or DEFAULT_MODEL)

    if mode == MODE_MODELS:
        on_model = (
            (lambda i, n, name: set_status(f"Model {i} / {n}: {name}…"))
            if set_status is not None
            else None
        )
        panels = compare_models(
            load,
            _models_used(parameters),
            image,
            labels,
            templates,
            device,
            on_model=on_model,
        )
        figure = render_comparison(image, panels, title, template_caption(templates))
        details = {"Device": device, **comparison_details(panels, "Models")}
        summary = "; ".join(f"{c}: {r.top[0]} ({r.top[1]:.0%})" for c, r in panels)
        return figure, details, f"CLIP model comparison: {summary}."

    model, processor = load(model_name, device)
    if mode == MODE_TEMPLATES:
        panels = compare_templates(model, processor, image, labels, templates, device)
        figure = render_comparison(image, panels, title, model_name)
        details = {
            "Model": model_name,
            "Device": device,
            **comparison_details(panels, "Templates"),
        }
        winners = {r.top[0] for _, r in panels}
        agreement = "all agree" if len(winners) == 1 else "they disagree"
        return figure, details, f"CLIP template comparison: {agreement}."

    result = classify(model, processor, image, labels, templates, device)
    figure = render_result(image, result, title)
    label, prob = result.top
    return (
        figure,
        result_details(result, model_name, device),
        f"CLIP: {label} ({prob:.0%}) out of {len(labels)} classes.",
    )


TASK_CLASSIFY = "classify"
TASK_SEARCH = "search"
TASK_EVALUATE = "evaluate"
TASKS = {
    TASK_CLASSIFY: "Classify the open image (zero-shot, compare templates or models)",
    TASK_SEARCH: "Search a folder of images with a text query",
    TASK_EVALUATE: "Evaluate zero-shot accuracy on a labelled folder",
}


def ask_clip_task(parent: tk.Misc, has_image: bool) -> str | None:
    """First step of the tool: choose what to do with CLIP."""
    dialog = tk.Toplevel(parent)
    dialog.title("CLIP Zero-Shot")
    dialog.transient(parent)
    dialog.grab_set()

    task_var = tk.StringVar(value=TASK_CLASSIFY if has_image else TASK_SEARCH)
    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)
    box = ttk.LabelFrame(main, text="What do you want to do?", padding=10)
    box.pack(fill="x", pady=(0, 8))
    for row, (key, text) in enumerate(TASKS.items()):
        disabled = key == TASK_CLASSIFY and not has_image
        ttk.Radiobutton(
            box,
            text=text + (" (open an image first)" if disabled else ""),
            value=key,
            variable=task_var,
            state="disabled" if disabled else "normal",
        ).grid(row=row, column=0, sticky="w", pady=2)

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


class ClipZeroShotTool(ForensicsTool):
    """Set 10 CLIP: zero-shot classification, text search and evaluation."""

    tool_id = "clip_zero_shot"
    title = "CLIP Zero-Shot"
    category = "Deep Learning"
    description = (
        "Classify the image into any classes you type, without training "
        "(also compares prompt templates and models), search a folder with a "
        "text query, or evaluate zero-shot accuracy on a labelled folder."
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
            task = ask_clip_task(parent, document.current is not None)
            if task is None:
                return None
        else:
            task = parameters.get("task", TASK_CLASSIFY)
        if task == TASK_SEARCH:
            from .clip_image_search import run_image_search

            return run_image_search(parent, document, parameters)
        if task == TASK_EVALUATE:
            from .clip_folder_evaluation import run_folder_evaluation

            return run_folder_evaluation(parent, document, parameters)
        return self._classify(parent, document, parameters)

    def _classify(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        parameters: dict[str, Any] | None,
    ) -> ToolResult | None:
        if parameters is None:
            parameters = ask_clip_parameters(parent)
            if parameters is None:
                return None
        image, used_original = source_image(document)

        device = resolve_device(str(parameters.get("device", "auto")))
        injected = parameters.get("clip") or parameters.get("loader")
        message = "" if injected else loading_message(_models_used(parameters), device)
        message += (
            "Classifying with each model…"
            if parameters.get("mode") == MODE_MODELS
            else "Classifying…"
        )
        status_line = [""]

        def set_status(text: str) -> None:
            status_line[0] = text

        figure, details, summary = run_with_progress(
            parent,
            "CLIP zero-shot",
            message,
            _run_job,
            image,
            parameters,
            set_status,
            status=(lambda: status_line[0])
            if parameters.get("mode") == MODE_MODELS
            else None,
        )
        if used_original:
            details = {"Image": "original (the canvas showed a CLIP chart)", **details}
        return ToolResult(message=summary, image=figure, details=details)
