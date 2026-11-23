"""Set 9.1 — YOLO Object Detection (pretrained Ultralytics models).

Runs an Ultralytics YOLO detector on the open image and draws every detected
box with its class name and confidence. The default model is YOLO11n
pretrained on COCO (80 classes), kept in ``models/yolo/pretrained/`` and
downloaded on first use; any other ``.pt`` weights under ``models/yolo/`` (or
chosen with Browse) can be used instead. Ultralytics and PyTorch are imported
only when a model is loaded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from forensics_app.core import ImageDocument
from forensics_app.paths import MODELS_DIR, PROJECT_ROOT, resolve_path
from forensics_app.utils.background import run_with_progress

from .base import ForensicsTool, ToolResult

YOLO_MODELS_DIR = MODELS_DIR / "yolo"
PRETRAINED_DIR = YOLO_MODELS_DIR / "pretrained"
DEFAULT_WEIGHTS = PRETRAINED_DIR / "yolo11n.pt"

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMAGE_SIZE = 640
DEVICES = ("auto", "cpu", "mps", "cuda")
MAX_LISTED_DETECTIONS = 30

# Ultralytics' class palette (same colours as ``Results.plot()``), RGB hex.
CLASS_PALETTE = (
    "042AFF",
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",
    "FF6FDD",
    "FF444F",
    "CCED00",
    "00F344",
    "BD00FF",
    "00B4FF",
    "DD00BA",
    "00FFFF",
    "26C000",
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C",
    "FC6D2F",
    "A2FF0B",
)


# ---------------------------------------------------------------------------
# Weights discovery, device and model loading
# ---------------------------------------------------------------------------


def list_yolo_weights(root: str | Path = YOLO_MODELS_DIR) -> list[Path]:
    """Return the default weights followed by every other ``.pt`` under ``root``.

    The default YOLO11n weights are listed even when not downloaded yet.
    """
    root = Path(root)
    found = sorted(root.rglob("*.pt")) if root.is_dir() else []
    default = DEFAULT_WEIGHTS
    return [default] + [p for p in found if p.resolve() != default.resolve()]


def display_path(path: str | Path) -> str:
    """Return ``path`` relative to the project folder when inside it."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def is_downloadable(path: str | Path) -> bool:
    """Return True for missing weights that Ultralytics downloads by name."""
    path = Path(path)
    return path.parent.resolve() == PRETRAINED_DIR.resolve() and path.name.startswith(
        ("yolo11", "yolov8")
    )


def _available_accelerator() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device: str) -> str:
    """Return the Ultralytics device string for a ``DEVICES`` choice."""
    device = device if device in DEVICES else "auto"
    if device == "auto":
        device = _available_accelerator()
    return "0" if device == "cuda" else device


def _import_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with:\n"
            "    pip install ultralytics"
        ) from exc
    return YOLO


# Loaded model, keyed by (path, modification time).
_MODEL_CACHE: dict[tuple[str, int], Any] = {}


def _cache_key(path: Path) -> tuple[str, int]:
    path = path.resolve()
    return str(path), path.stat().st_mtime_ns


def is_model_cached(path: str | Path) -> bool:
    path = Path(path)
    return path.is_file() and _cache_key(path) in _MODEL_CACHE


def load_yolo_model(path: str | Path) -> Any:
    """Load YOLO weights (cached until the file changes).

    Missing pretrained weights in ``models/yolo/pretrained/`` are downloaded.
    """
    path = Path(path)
    if not path.is_file() and not is_downloadable(path):
        raise FileNotFoundError(f"YOLO weights not found:\n{path}")
    if path.is_file() and is_model_cached(path):
        return _MODEL_CACHE[_cache_key(path)]

    yolo = _import_yolo()
    path.parent.mkdir(parents=True, exist_ok=True)
    model = yolo(str(path))
    _MODEL_CACHE.clear()
    if path.is_file():
        _MODEL_CACHE[_cache_key(path)] = model
    return model


def class_names(model: Any) -> dict[int, str]:
    """Return the model's ``{class id: name}`` mapping."""
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def parse_class_filter(text: str, names: dict[int, str]) -> list[int] | None:
    """Return class ids for comma-separated names or ids; ``None`` keeps all."""
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if not tokens:
        return None
    by_name = {n.lower(): i for i, n in names.items()}
    ids: list[int] = []
    unknown: list[str] = []
    for token in tokens:
        if token.isdigit() and int(token) in names:
            ids.append(int(token))
        elif token.lower() in by_name:
            ids.append(by_name[token.lower()])
        else:
            unknown.append(token)
    if unknown:
        raise ValueError(
            f"Unknown class(es): {', '.join(unknown)}.\n"
            f"This model knows: {', '.join(names.values())}"
        )
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One predicted box in original image pixels (x1, y1, x2, y2)."""

    class_id: int
    name: str
    confidence: float
    box: tuple[float, float, float, float]


def _to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def detect_objects(
    model: Any,
    image: Image.Image,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: str = "cpu",
    classes: list[int] | None = None,
) -> tuple[list[Detection], dict[str, Any]]:
    """Run the detector on an image and return the detections and timings.

    Detections are sorted by decreasing confidence.
    """
    started = time.perf_counter()
    rgb = image.convert("RGB")
    # Ultralytics reads NumPy arrays as BGR (OpenCV order).
    bgr = np.ascontiguousarray(np.asarray(rgb)[..., ::-1])
    results = model.predict(
        bgr,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        device=device,
        classes=classes,
        verbose=False,
    )
    result = results[0]
    names = class_names(model)

    detections: list[Detection] = []
    boxes = result.boxes
    if boxes is not None and len(boxes):
        xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
        confs = _to_numpy(boxes.conf).reshape(-1)
        ids = _to_numpy(boxes.cls).reshape(-1).astype(int)
        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, ids):
            detections.append(
                Detection(
                    class_id=int(cid),
                    name=names.get(int(cid), str(cid)),
                    confidence=float(conf),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    info = {
        "speed_ms": dict(getattr(result, "speed", {}) or {}),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "image_size": (rgb.width, rgb.height),
    }
    return detections, info


def count_by_class(detections: list[Detection]) -> list[tuple[str, int]]:
    """Return ``(class name, count)`` pairs, most frequent first."""
    return Counter(d.name for d in detections).most_common()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return the RGB colour Ultralytics uses for a class id."""
    hex_color = CLASS_PALETTE[int(class_id) % len(CLASS_PALETTE)]
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return (0, 0, 0) if luminance > 0.6 else (255, 255, 255)


def _load_font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has a single fixed-size font
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    *,
    show_labels: bool = True,
    show_confidence: bool = True,
) -> Image.Image:
    """Return a copy of the image with boxes and labels drawn in class colours.

    Line width and font size scale with the image size.
    """
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    width = max(2, round((output.width + output.height) / 2 * 0.003))
    font = _load_font(max(11, width * 6))

    for det in detections:
        color = class_color(det.class_id)
        x1, y1, x2, y2 = det.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        if not show_labels:
            continue
        label = det.name + (f" {det.confidence:.2f}" if show_confidence else "")
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        pad = max(2, width)
        label_h = text_h + 2 * pad
        # Above the box when there is room, otherwise inside its top edge.
        ly = y1 - label_h if y1 - label_h >= 0 else y1
        lx = min(max(0.0, x1), max(0.0, output.width - text_w - 2 * pad))
        draw.rectangle((lx, ly, lx + text_w + 2 * pad, ly + label_h), fill=color)
        draw.text(
            (lx + pad - left, ly + pad - top),
            label,
            fill=_text_color(color),
            font=font,
        )
    return output


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def ask_yolo_detection_parameters(parent: tk.Misc) -> dict[str, Any] | None:
    """Display the model, detection settings and display options dialog."""
    weights = list_yolo_weights()

    dialog = tk.Toplevel(parent)
    dialog.title("YOLO Object Detection")
    dialog.transient(parent)
    dialog.grab_set()

    weights_var = tk.StringVar(value=display_path(weights[0]))
    status_var = tk.StringVar()
    conf_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
    iou_var = tk.DoubleVar(value=DEFAULT_IOU)
    size_var = tk.IntVar(value=DEFAULT_IMAGE_SIZE)
    device_var = tk.StringVar(value="auto")
    classes_var = tk.StringVar()
    labels_var = tk.BooleanVar(value=True)
    conf_text_var = tk.BooleanVar(value=True)

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    # A. Model --------------------------------------------------------------------
    model_box = ttk.LabelFrame(main, text="A. Model weights", padding=10)
    model_box.pack(fill="x", pady=(0, 8))
    combo = ttk.Combobox(
        model_box,
        textvariable=weights_var,
        values=[display_path(p) for p in weights],
        width=56,
    )
    combo.grid(row=0, column=0, sticky="ew")

    def browse() -> None:
        start = YOLO_MODELS_DIR if YOLO_MODELS_DIR.is_dir() else PROJECT_ROOT
        chosen = filedialog.askopenfilename(
            parent=dialog,
            title="Select YOLO weights",
            initialdir=str(start),
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if chosen:
            weights_var.set(display_path(chosen))
            refresh()

    ttk.Button(model_box, text="Browse…", command=browse).grid(
        row=0, column=1, padx=(6, 0)
    )
    status_label = ttk.Label(model_box, textvariable=status_var)
    status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
    model_box.columnconfigure(0, weight=1)

    # B. Detection settings ---------------------------------------------------------
    settings = ttk.LabelFrame(main, text="B. Detection settings", padding=10)
    settings.pack(fill="x", pady=(0, 8))
    rows = (
        ("Confidence threshold:", conf_var, 0.01, 0.99, 0.05),
        ("NMS IoU threshold:", iou_var, 0.05, 0.95, 0.05),
        ("Input size (pixels):", size_var, 160, 1280, 32),
    )
    for r, (label, var, lo, hi, step) in enumerate(rows):
        ttk.Label(settings, text=label).grid(row=r, column=0, sticky="w")
        ttk.Spinbox(
            settings, textvariable=var, from_=lo, to=hi, increment=step, width=8
        ).grid(row=r, column=1, sticky="w", padx=6, pady=2)
    ttk.Label(settings, text="Device:").grid(row=3, column=0, sticky="w")
    ttk.Combobox(
        settings, textvariable=device_var, values=DEVICES, state="readonly", width=8
    ).grid(row=3, column=1, sticky="w", padx=6, pady=2)
    ttk.Label(settings, text="Only classes (optional):").grid(
        row=4, column=0, sticky="w"
    )
    ttk.Entry(settings, textvariable=classes_var, width=36).grid(
        row=4, column=1, sticky="w", padx=6, pady=2
    )
    ttk.Label(settings, text="comma-separated names or ids, e.g. person, bus").grid(
        row=5, column=1, sticky="w", padx=6
    )

    # C. Display ----------------------------------------------------------------------
    display = ttk.LabelFrame(main, text="C. Display", padding=10)
    display.pack(fill="x", pady=(0, 8))
    ttk.Checkbutton(display, text="Draw class labels", variable=labels_var).grid(
        row=0, column=0, sticky="w"
    )
    ttk.Checkbutton(
        display, text="Show confidence in labels", variable=conf_text_var
    ).grid(row=0, column=1, sticky="w", padx=(12, 0))

    # Buttons -------------------------------------------------------------------------
    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(4, 0))
    result: dict[str, Any] | None = None

    def refresh(_event: object = None) -> None:
        path = resolve_path(weights_var.get().strip())
        if path.is_file():
            status_var.set(f"weights found ({path.stat().st_size / 1e6:.1f} MB)")
            status_label.configure(foreground="#006300")
        elif is_downloadable(path):
            status_var.set("not downloaded yet: downloaded on first use")
            status_label.configure(foreground="#8a5a00")
        else:
            status_var.set("file not found")
            status_label.configure(foreground="#b42318")

    def detect() -> None:
        nonlocal result
        path = resolve_path(weights_var.get().strip())
        if not path.is_file() and not is_downloadable(path):
            messagebox.showerror(
                "No weights", f"YOLO weights not found:\n{path}", parent=dialog
            )
            return
        try:
            conf = float(conf_var.get())
            iou = float(iou_var.get())
            size = int(size_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror(
                "Invalid settings",
                "Thresholds must be numbers and the size an integer.",
                parent=dialog,
            )
            return
        result = {
            "weights": str(path),
            "confidence": min(max(conf, 0.01), 0.99),
            "iou": min(max(iou, 0.05), 0.95),
            "image_size": max(32, round(size / 32) * 32),
            "device": device_var.get(),
            "classes": classes_var.get().strip(),
            "show_labels": bool(labels_var.get()),
            "show_confidence": bool(conf_text_var.get()),
        }
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Detect objects", command=detect).pack(
        side="right", padx=(0, 8)
    )

    combo.bind("<<ComboboxSelected>>", refresh)
    combo.bind("<FocusOut>", refresh)
    refresh()
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _detect_job(
    weights: Path,
    model: Any | None,
    image: Image.Image,
    parameters: dict[str, Any],
) -> tuple[list[Detection], dict[str, Any], Image.Image, dict[int, str]]:
    """Load the model, detect and draw; runs in a worker thread."""
    if model is None:
        model = load_yolo_model(weights)
    names = class_names(model)
    classes = parse_class_filter(str(parameters.get("classes") or ""), names)
    device = resolve_device(str(parameters.get("device", "auto")))
    detections, info = detect_objects(
        model,
        image,
        confidence=float(parameters.get("confidence", DEFAULT_CONFIDENCE)),
        iou=float(parameters.get("iou", DEFAULT_IOU)),
        image_size=int(parameters.get("image_size", DEFAULT_IMAGE_SIZE)),
        device=device,
        classes=classes,
    )
    info["device"] = device
    info["classes"] = classes
    annotated = draw_detections(
        image,
        detections,
        show_labels=bool(parameters.get("show_labels", True)),
        show_confidence=bool(parameters.get("show_confidence", True)),
    )
    return detections, info, annotated, names


def detection_details(
    weights: Path,
    detections: list[Detection],
    info: dict[str, Any],
    parameters: dict[str, Any],
    names: dict[int, str],
) -> dict[str, Any]:
    """Return the settings, class counts and boxes for the results table."""
    classes = info.get("classes")
    details: dict[str, Any] = {
        "Model": display_path(weights),
        "Model classes": len(names),
        "Device": info.get("device", "?"),
        "Image": "{}x{}".format(*info["image_size"]),
        "Input size": parameters.get("image_size", DEFAULT_IMAGE_SIZE),
        "Confidence ≥": parameters.get("confidence", DEFAULT_CONFIDENCE),
        "NMS IoU": parameters.get("iou", DEFAULT_IOU),
        "Class filter": ", ".join(names[i] for i in classes) if classes else "all",
        "Detections": len(detections),
    }
    for name, count in count_by_class(detections):
        details[f"  {name}"] = count
    for i, det in enumerate(detections[:MAX_LISTED_DETECTIONS], 1):
        x1, y1, x2, y2 = det.box
        details[f"#{i} {det.name}"] = (
            f"{det.confidence:.2f} at ({x1:.0f}, {y1:.0f})–({x2:.0f}, {y2:.0f})"
        )
    if len(detections) > MAX_LISTED_DETECTIONS:
        details["…"] = f"{len(detections) - MAX_LISTED_DETECTIONS} more boxes"
    speed = info.get("speed_ms", {})
    if speed:
        details["Time (pre / inf / post)"] = (
            " / ".join(
                f"{speed.get(k, 0):.1f}"
                for k in ("preprocess", "inference", "postprocess")
            )
            + " ms"
        )
    details["Time (total)"] = f"{info['elapsed_ms']:.0f} ms"
    return details


class YoloDetectionTool(ForensicsTool):
    """Detect objects in the open image with a pretrained YOLO model."""

    tool_id = "yolo_detection"
    title = "YOLO Object Detection"
    category = "Deep Learning"
    description = (
        "Detect objects in the image with an Ultralytics YOLO model "
        "(default: YOLO11n pretrained on the 80 COCO classes)."
    )
    requires_image = True

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> ToolResult | None:
        if parameters is None:
            parameters = ask_yolo_detection_parameters(parent)
            if parameters is None:
                return None
        if document.current is None:
            raise ValueError("Open an image first.")

        weights = resolve_path(parameters.get("weights") or DEFAULT_WEIGHTS)
        model = parameters.get("model")
        message = "Detecting objects…"
        if model is None and not is_model_cached(weights):
            message = (
                "Downloading and loading the weights (first use only)…\n"
                if not weights.is_file()
                else "Loading the weights (first use only)…\n"
            ) + message
        detections, info, annotated, names = run_with_progress(
            parent,
            "YOLO detection",
            message,
            _detect_job,
            weights,
            model,
            document.current,
            parameters,
        )

        counts = ", ".join(f"{name} ×{n}" for name, n in count_by_class(detections))
        return ToolResult(
            message=(
                f"{weights.name}: {len(detections)} object(s) detected"
                + (f" ({counts})." if counts else ".")
            ),
            image=annotated,
            details=detection_details(weights, detections, info, parameters, names),
        )
