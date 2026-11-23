"""Set 9.3 — Blood Cell Detection (trained YOLO11n, ground truth vs predictions).

Loads a YOLO model trained in ``notebooks/train_yolo.ipynb`` (exported to
``models/yolo/<model>/`` with ``best.pt`` and ``passport.json``), runs it on the
open image or a chosen file and draws the ground-truth boxes (dashed) and the
predictions (solid, class colours) on the same image. When a YOLO label file is
found for the image, predictions are matched to the ground truth per class and
false positives and missed cells are marked and counted. A whole folder of
labelled images can be evaluated at once (per-class precision and recall, the
hardest images, a CSV per model), unmatched cells cut by the image border can be
left out of the counts, and both exported models can be compared side by side.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from forensics_app.paths import PROJECT_ROOT, resolve_path
from forensics_app.utils.background import run_with_progress

from .base import ForensicsTool, ToolResult
from .hog_person_detection import IMAGE_TYPES
from .yolo_detection import (
    DEVICES,
    YOLO_MODELS_DIR,
    Detection,
    _load_font,
    _text_color,
    detect_objects,
    display_path,
    is_model_cached,
    load_yolo_model,
    resolve_device,
)

SAMPLES_DIR = YOLO_MODELS_DIR / "blood_cell_samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "set9_3_detection"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_BORDER_MARGIN = 3
WORST_SHOWN = 4
WORST_LISTED = 10
DEFAULT_MODEL = "blood_cell_yolo11n"
DEFAULT_MATCH_IOU = 0.5
FALLBACK_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d8458f")
GT_DASH_COLORS = ((255, 255, 255), (31, 31, 30))
TAG_BACKGROUND = (31, 31, 30)


# ---------------------------------------------------------------------------
# Exported models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YoloModelInfo:
    """An exported YOLO model folder and its passport."""

    folder: Path
    passport: dict[str, Any] = field(hash=False, compare=False)

    @property
    def name(self) -> str:
        return str(self.passport.get("model_name", self.folder.name))

    @property
    def display_name(self) -> str:
        return str(self.passport.get("display_name", self.name))

    @property
    def class_names(self) -> list[str]:
        return [str(n) for n in self.passport.get("class_names", [])]

    @property
    def class_colors(self) -> list[tuple[int, int, int]]:
        colors = list(self.passport.get("class_colors", []))
        n = max(len(self.class_names), len(colors))
        colors += [
            FALLBACK_COLORS[i % len(FALLBACK_COLORS)] for i in range(len(colors), n)
        ]
        return [hex_to_rgb(c) for c in colors]

    @property
    def weights_path(self) -> Path:
        name = self.passport.get("files", {}).get("weights", "best.pt")
        return self.folder / name

    @property
    def has_weights(self) -> bool:
        return self.weights_path.is_file()

    @property
    def inference(self) -> dict[str, Any]:
        return dict(self.passport.get("inference", {}))


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Return ``(r, g, b)`` for ``#rrggbb``."""
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def load_yolo_passport(folder: str | Path) -> dict[str, Any]:
    """Read ``passport.json`` from an exported model folder."""
    with open(Path(folder) / "passport.json", encoding="utf-8") as fh:
        return json.load(fh)


def list_detection_models(root: str | Path = YOLO_MODELS_DIR) -> list[YoloModelInfo]:
    """Return exported detection models (folders with a passport), default first."""
    root = Path(root)
    models = []
    if root.is_dir():
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            if (folder / "passport.json").is_file():
                passport = load_yolo_passport(folder)
                if passport.get("task") == "object detection":
                    models.append(YoloModelInfo(folder, passport))
    return sorted(models, key=lambda m: (m.name != DEFAULT_MODEL, m.name))


def passport_summary(info: YoloModelInfo) -> str:
    """Return a short multi-line description of an exported model."""
    p = info.passport
    arch = p.get("architecture", {})
    train = p.get("training", {})
    ul = p.get("evaluation", {}).get("ultralytics", {})
    lines = [
        info.display_name,
        p.get("description", ""),
        "",
        f"Classes: {', '.join(info.class_names)}",
        f"Architecture: {arch.get('type', '?')}, "
        f"{arch.get('parameters', 0):,} parameters, {arch.get('gflops', '?')} GFLOPs",
        f"Initial weights: {arch.get('initial_weights', '?')}",
        f"Training: {train.get('epochs_run', '?')} epochs in "
        f"{train.get('minutes', '?')} min on {train.get('device', '?')} "
        f"(best epoch {train.get('best_epoch', '?')})",
    ]
    if ul:
        lines.append(
            f"Test: mAP50 {ul.get('mAP50', 0):.3f}, "
            f"mAP50-95 {ul.get('mAP50-95', 0):.3f}"
        )
        for name, v in ul.get("per_class", {}).items():
            lines.append(
                f"  {name}: AP50 {v.get('AP50', 0):.3f}, "
                f"AP50-95 {v.get('AP50-95', 0):.3f}"
            )
    if not info.has_weights:
        lines += ["", f"Weights missing: {display_path(info.weights_path)}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthBox:
    """One labelled box in image pixels (x1, y1, x2, y2)."""

    class_id: int
    name: str
    box: tuple[float, float, float, float]


def find_label_file(image_path: str | Path | None) -> Path | None:
    """Return the YOLO label file of an image, or ``None``.

    Looks in the sibling ``labels/`` folder when the image is inside an
    ``images/`` folder, then next to the image.
    """
    if image_path is None:
        return None
    image_path = Path(image_path)
    candidates = []
    if image_path.parent.name == "images":
        candidates.append(
            image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        )
    candidates.append(image_path.with_suffix(".txt"))
    return next((c for c in candidates if c.is_file()), None)


def read_yolo_labels(
    path: str | Path, width: int, height: int, names: list[str]
) -> list[GroundTruthBox]:
    """Read normalised ``class x_c y_c w h`` lines as pixel boxes."""
    boxes = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{Path(path).name}, line {line_no}: expected 5 values")
        cls = int(float(parts[0]))
        xc, yc, w, h = (float(v) for v in parts[1:])
        boxes.append(
            GroundTruthBox(
                class_id=cls,
                name=names[cls] if 0 <= cls < len(names) else str(cls),
                box=(
                    (xc - w / 2) * width,
                    (yc - h / 2) * height,
                    (xc + w / 2) * width,
                    (yc + h / 2) * height,
                ),
            )
        )
    return boxes


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the IoU matrix of (N, 4) and (M, 4) boxes (x1, y1, x2, y2)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


@dataclass(frozen=True)
class MatchResult:
    """Indices of matched pairs, false positives and missed ground-truth boxes.

    Unmatched boxes cut by the image border are listed as ignored instead when
    border ignoring is on.
    """

    matches: list[tuple[int, int, float]]  # (detection, ground truth, IoU)
    false_positives: list[int]
    missed: list[int]
    ignored_detections: list[int] = field(default_factory=list)
    ignored_ground_truth: list[int] = field(default_factory=list)


def touches_border(
    box: tuple[float, float, float, float], image_size: tuple[int, int], margin: float
) -> bool:
    """Return True when a box comes within ``margin`` pixels of the image edge."""
    x1, y1, x2, y2 = box
    width, height = image_size
    return min(x1, y1) <= margin or x2 >= width - margin or y2 >= height - margin


def match_detections(
    detections: list[Detection],
    ground_truth: list[GroundTruthBox],
    iou_threshold: float = DEFAULT_MATCH_IOU,
    *,
    image_size: tuple[int, int] | None = None,
    border_margin: float = 0.0,
) -> MatchResult:
    """Match detections to ground truth per class (greedy by confidence).

    Each detection, in decreasing confidence, takes the unmatched box of its
    class with the highest IoU if it reaches ``iou_threshold``. With a
    ``border_margin`` > 0, unmatched detections and labels that touch the image
    border are ignored rather than counted as false positives or missed.
    """
    det_boxes = np.array([d.box for d in detections], dtype=float).reshape(-1, 4)
    gt_boxes = np.array([g.box for g in ground_truth], dtype=float).reshape(-1, 4)
    gt_cls = np.array([g.class_id for g in ground_truth], dtype=int)
    ious = box_iou(det_boxes, gt_boxes)
    used = np.zeros(len(ground_truth), dtype=bool)
    order = sorted(range(len(detections)), key=lambda i: -detections[i].confidence)

    matches: list[tuple[int, int, float]] = []
    false_positives: list[int] = []
    for i in order:
        candidates = np.where((gt_cls == detections[i].class_id) & ~used, ious[i], -1.0)
        j = int(np.argmax(candidates)) if len(candidates) else -1
        if j >= 0 and candidates[j] >= iou_threshold:
            used[j] = True
            matches.append((i, j, float(candidates[j])))
        else:
            false_positives.append(i)
    missed = [j for j in range(len(ground_truth)) if not used[j]]
    ignored_det: list[int] = []
    ignored_gt: list[int] = []
    if border_margin > 0 and image_size is not None:

        def at_border(box: tuple[float, float, float, float]) -> bool:
            return touches_border(box, image_size, border_margin)

        ignored_det = [i for i in false_positives if at_border(detections[i].box)]
        ignored_gt = [j for j in missed if at_border(ground_truth[j].box)]
        false_positives = [i for i in false_positives if i not in set(ignored_det)]
        missed = [j for j in missed if j not in set(ignored_gt)]
    return MatchResult(
        matches, sorted(false_positives), missed, sorted(ignored_det), ignored_gt
    )


def match_summary(
    detections: list[Detection],
    ground_truth: list[GroundTruthBox],
    match: MatchResult,
    names: list[str],
) -> dict[str, dict[str, float]]:
    """Return per-class and overall counts, precision and recall.

    Ignored (border) boxes are counted separately and left out of the
    ground-truth and prediction totals.
    """
    rows: dict[str, dict[str, float]] = {}
    keys = list(range(len(names))) + [None]
    for c in keys:
        label = names[c] if c is not None else "all"

        def in_class(cid: int, c: int | None = c) -> bool:
            return c is None or cid == c

        tp = sum(in_class(detections[i].class_id) for i, _, _ in match.matches)
        fp = sum(in_class(detections[i].class_id) for i in match.false_positives)
        fn = sum(in_class(ground_truth[j].class_id) for j in match.missed)
        ignored = sum(
            in_class(detections[i].class_id) for i in match.ignored_detections
        ) + sum(in_class(ground_truth[j].class_id) for j in match.ignored_ground_truth)
        rows[label] = counts_row(tp, fp, fn, ignored)
    return rows


def counts_row(tp: int, fp: int, missed: int, ignored: int = 0) -> dict[str, float]:
    """Return the counts with ground truth, predictions, precision, recall, F1."""
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + missed) if tp + missed else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if tp and precision + recall
        else (0.0 if tp + fp + missed else float("nan"))
    )
    return {
        "ground_truth": tp + missed,
        "predicted": tp + fp,
        "tp": tp,
        "fp": fp,
        "missed": missed,
        "ignored": ignored,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    width: int,
    dash: int,
    colors: tuple[tuple[int, int, int], ...] = GT_DASH_COLORS,
) -> None:
    """Draw a rectangle whose dashes alternate between ``colors``."""
    x1, y1, x2, y2 = box
    edges = (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    )
    for (ax, ay), (bx, by) in edges:
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(1, int(length // dash))
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            start = (ax + (bx - ax) * t0, ay + (by - ay) * t0)
            end = (ax + (bx - ax) * t1, ay + (by - ay) * t1)
            draw.line((start, end), fill=colors[k % len(colors)], width=width)


def _tag(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    image_size: tuple[int, int],
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left + 4, bottom - top + 4
    x = min(max(0.0, xy[0]), image_size[0] - w)
    y = min(max(0.0, xy[1]), image_size[1] - h)
    draw.rectangle((x, y, x + w, y + h), fill=TAG_BACKGROUND)
    draw.text((x + 2 - left, y + 2 - top), text, fill=(255, 255, 255), font=font)


def draw_ground_truth_and_predictions(
    image: Image.Image,
    detections: list[Detection],
    ground_truth: list[GroundTruthBox],
    match: MatchResult | None,
    colors: list[tuple[int, int, int]],
    *,
    show_ground_truth: bool = True,
    show_predictions: bool = True,
    mark_errors: bool = True,
    show_confidence: bool = False,
) -> Image.Image:
    """Return the image with dashed ground truth and solid predicted boxes.

    False positives are drawn thicker with an "FP" tag and missed ground-truth
    boxes get a "missed" tag.
    """
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    unit = max(1, round((output.width + output.height) / 2 / 400))
    font = _load_font(max(10, unit * 9))
    fp_set = set(match.false_positives) if (match and mark_errors) else set()
    missed_set = set(match.missed) if (match and mark_errors) else set()

    def color_of(cid: int) -> tuple[int, int, int]:
        return colors[cid] if 0 <= cid < len(colors) else (200, 200, 200)

    if show_ground_truth:
        for j, gt in enumerate(ground_truth):
            width = unit * 2 if j in missed_set else unit
            _dashed_rectangle(draw, gt.box, width=width, dash=unit * 6)
    if show_predictions:
        for i, det in enumerate(detections):
            width = unit * 3 if i in fp_set else unit * 2
            draw.rectangle(det.box, outline=color_of(det.class_id), width=width)
            if show_confidence:
                label = f"{det.name} {det.confidence:.2f}"
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                x1, y1 = det.box[0], det.box[1]
                draw.rectangle(
                    (x1, y1, x1 + right - left + 4, y1 + bottom - top + 4),
                    fill=color_of(det.class_id),
                )
                draw.text(
                    (x1 + 2 - left, y1 + 2 - top),
                    label,
                    fill=_text_color(color_of(det.class_id)),
                    font=font,
                )
    if show_predictions:
        for i in sorted(fp_set):
            x1, y1, _, y2 = detections[i].box
            _tag(
                draw,
                (x1 + unit * 2, y2 - unit * 14),
                f"FP {detections[i].confidence:.2f}",
                font,
                output.size,
            )
    if show_ground_truth:
        for j in sorted(missed_set):
            x1, y1, _, _ = ground_truth[j].box
            _tag(
                draw,
                (x1 + unit * 2, y1 + unit * 2),
                f"missed {ground_truth[j].name}",
                font,
                output.size,
            )
    return output


def side_by_side(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    """Return the panels next to each other with a title strip above each."""
    return image_grid(panels, columns=len(panels))


def image_grid(panels: list[tuple[str, Image.Image]], columns: int = 2) -> Image.Image:
    """Return the panels in a grid, each with a title strip above it."""
    cell_w = max(img.width for _, img in panels)
    cell_h = max(img.height for _, img in panels)
    font = _load_font(max(12, cell_h // 40))
    gap = max(8, cell_w // 80)
    title_h = int(font.size * 1.8) if hasattr(font, "size") else 24
    rows = -(-len(panels) // columns)
    width = columns * cell_w + gap * (columns - 1)
    height = rows * (cell_h + title_h) + gap * (rows - 1)
    canvas = Image.new("RGB", (width, height), (252, 252, 251))
    draw = ImageDraw.Draw(canvas)
    for k, (title, img) in enumerate(panels):
        x = (k % columns) * (cell_w + gap)
        y = (k // columns) * (cell_h + title_h + gap)
        canvas.paste(img.convert("RGB"), (x, y + title_h))
        draw.text(
            (x + 4, y + title_h // 2), title, fill=(31, 31, 30), font=font, anchor="lm"
        )
    return canvas


# ---------------------------------------------------------------------------
# Detection job
# ---------------------------------------------------------------------------


@dataclass
class EvaluationRun:
    """Detections, ground truth, matching and the drawn image for one model."""

    info: YoloModelInfo
    detections: list[Detection]
    ground_truth: list[GroundTruthBox]
    match: MatchResult | None
    summary: dict[str, dict[str, float]] | None
    annotated: Image.Image | None
    timing: dict[str, Any]


def evaluate_image(
    info: YoloModelInfo,
    image: Image.Image,
    label_path: Path | None,
    parameters: dict[str, Any],
    model: Any | None = None,
    *,
    draw: bool = True,
) -> EvaluationRun:
    """Detect cells with one model and compare them with the ground truth."""
    if model is None:
        if not info.has_weights:
            raise FileNotFoundError(f"Weights not found:\n{info.weights_path}")
        model = load_yolo_model(info.weights_path)
    defaults = info.inference
    device = resolve_device(str(parameters.get("device", "auto")))
    detections, timing = detect_objects(
        model,
        image,
        confidence=float(
            parameters.get("confidence", defaults.get("confidence", 0.25))
        ),
        iou=float(parameters.get("iou", defaults.get("nms_iou", 0.7))),
        image_size=int(parameters.get("image_size", defaults.get("image_size", 640))),
        device=device,
    )
    timing["device"] = device

    names = info.class_names or sorted({d.name for d in detections})
    ground_truth = (
        read_yolo_labels(label_path, image.width, image.height, names)
        if label_path is not None
        else []
    )
    match = summary = None
    if label_path is not None:
        match = match_detections(
            detections,
            ground_truth,
            float(parameters.get("match_iou", DEFAULT_MATCH_IOU)),
            image_size=image.size,
            border_margin=float(parameters.get("border_margin", 0)),
        )
        summary = match_summary(detections, ground_truth, match, names)

    annotated = (
        draw_annotations(image, detections, ground_truth, match, info, parameters)
        if draw
        else None
    )
    return EvaluationRun(
        info, detections, ground_truth, match, summary, annotated, timing
    )


def draw_annotations(
    image: Image.Image,
    detections: list[Detection],
    ground_truth: list[GroundTruthBox],
    match: MatchResult | None,
    info: YoloModelInfo,
    parameters: dict[str, Any],
) -> Image.Image:
    """Draw one evaluation with the display options from ``parameters``."""
    return draw_ground_truth_and_predictions(
        image,
        detections,
        ground_truth,
        match,
        info.class_colors,
        show_ground_truth=bool(parameters.get("show_ground_truth", True)),
        show_predictions=bool(parameters.get("show_predictions", True)),
        mark_errors=bool(parameters.get("mark_errors", True)),
        show_confidence=bool(parameters.get("show_confidence", False)),
    )


def _evaluate_job(
    infos: list[YoloModelInfo],
    image: Image.Image,
    label_path: Path | None,
    parameters: dict[str, Any],
    models: dict[str, Any],
) -> list[EvaluationRun]:
    """Evaluate one or more models; runs in a worker thread."""
    return [
        evaluate_image(info, image, label_path, parameters, models.get(info.name))
        for info in infos
    ]


# ---------------------------------------------------------------------------
# Folder evaluation
# ---------------------------------------------------------------------------


def list_labelled_images(folder: str | Path) -> tuple[list[tuple[Path, Path]], int]:
    """Return (image, label) pairs in a folder and the number without labels."""
    images = sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    pairs = [(p, label) for p in images if (label := find_label_file(p)) is not None]
    return pairs, len(images) - len(pairs)


def aggregate_summaries(
    summaries: list[dict[str, dict[str, float]]], names: list[str]
) -> dict[str, dict[str, float]]:
    """Sum per-image counts per class and recompute precision, recall and F1."""
    totals = {}
    for label in [*names, "all"]:
        sums = {
            key: int(sum(s[label][key] for s in summaries if label in s))
            for key in ("tp", "fp", "missed", "ignored")
        }
        totals[label] = counts_row(
            sums["tp"], sums["fp"], sums["missed"], sums["ignored"]
        )
    return totals


@dataclass
class FolderEvaluation:
    """One model evaluated on every labelled image of a folder."""

    info: YoloModelInfo
    folder: Path
    per_image: list[tuple[Path, Path, dict[str, dict[str, float]]]]
    totals: dict[str, dict[str, float]]
    skipped: int
    mean_iou: float
    elapsed_ms: float
    device: str

    def errors(self, image: Path) -> int:
        row = next(s for p, _, s in self.per_image if p == image)["all"]
        return int(row["fp"] + row["missed"])

    def worst(self, count: int = WORST_LISTED) -> list[Path]:
        """Images with the most false positives + missed cells, worst first."""
        ranked = sorted(self.per_image, key=lambda r: (-self.errors(r[0]), r[0].name))
        return [p for p, _, _ in ranked[:count]]


def evaluate_folder(
    info: YoloModelInfo,
    folder: str | Path,
    parameters: dict[str, Any],
    model: Any | None = None,
) -> FolderEvaluation:
    """Detect and match on every labelled image of a folder with one model."""
    folder = Path(folder)
    pairs, skipped = list_labelled_images(folder)
    if not pairs:
        raise FileNotFoundError(f"No images with YOLO labels found in:\n{folder}")
    if model is None:
        if not info.has_weights:
            raise FileNotFoundError(f"Weights not found:\n{info.weights_path}")
        model = load_yolo_model(info.weights_path)
    started = time.perf_counter()
    per_image = []
    ious: list[float] = []
    device = "?"
    for image_path, label_path in pairs:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        run = evaluate_image(info, image, label_path, parameters, model, draw=False)
        per_image.append((image_path, label_path, run.summary))
        ious += [iou for _, _, iou in run.match.matches]
        device = run.timing["device"]
    names = info.class_names or sorted(
        {k for _, _, s in per_image for k in s if k != "all"}
    )
    return FolderEvaluation(
        info=info,
        folder=folder,
        per_image=per_image,
        totals=aggregate_summaries([s for _, _, s in per_image], names),
        skipped=skipped,
        mean_iou=float(np.mean(ious)) if ious else float("nan"),
        elapsed_ms=(time.perf_counter() - started) * 1000,
        device=device,
    )


def folder_label(folder: Path) -> str:
    """Return a short name for a folder (``test`` for ``…/test/images``)."""
    return folder.parent.name if folder.name == "images" else folder.name


def save_evaluation_csv(
    evaluation: FolderEvaluation, output_dir: str | Path, border_margin: float
) -> Path:
    """Write per-image, per-class counts to a CSV file and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_border" if border_margin > 0 else ""
    path = output_dir / (
        f"{evaluation.info.name}_{folder_label(evaluation.folder)}{suffix}.csv"
    )
    keys = ("ground_truth", "predicted", "tp", "fp", "missed", "ignored")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "class", *keys, "precision", "recall"])
        for image_path, _, summary in evaluation.per_image:
            for label, row in summary.items():
                writer.writerow(
                    [image_path.name, label, *(int(row[k]) for k in keys)]
                    + [_csv_number(row["precision"]), _csv_number(row["recall"])]
                )
        for label, row in evaluation.totals.items():
            writer.writerow(
                ["TOTAL", label, *(int(row[k]) for k in keys)]
                + [_csv_number(row["precision"]), _csv_number(row["recall"])]
            )
    return path


def _csv_number(value: float) -> str:
    return "" if value != value else f"{value:.4f}"  # NaN -> empty cell


def _evaluate_folder_job(
    infos: list[YoloModelInfo],
    folder: Path,
    parameters: dict[str, Any],
    models: dict[str, Any],
) -> tuple[list[FolderEvaluation], list[tuple[str, Image.Image]]]:
    """Evaluate a folder with one or more models and draw the hardest images.

    One model: its worst images in a grid. Several: the image with the most
    errors over all models, once per model. Runs in a worker thread.
    """
    evaluations = [
        evaluate_folder(info, folder, parameters, models.get(info.name))
        for info in infos
    ]
    panels: list[tuple[str, Image.Image]] = []
    if len(evaluations) == 1:
        targets = [(evaluations[0], p) for p in evaluations[0].worst(WORST_SHOWN)]
    else:
        images = [p for p, _, _ in evaluations[0].per_image]
        hardest = max(images, key=lambda p: sum(e.errors(p) for e in evaluations))
        targets = [(e, hardest) for e in evaluations]
    for k, (evaluation, image_path) in enumerate(targets, 1):
        label_path = find_label_file(image_path)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        model = models.get(evaluation.info.name)
        if model is None:
            model = load_yolo_model(evaluation.info.weights_path)
        run = evaluate_image(evaluation.info, image, label_path, parameters, model)
        row = run.summary["all"]
        who = evaluation.info.name if len(evaluations) > 1 else f"#{k}"
        title = (
            f"{who} {image_path.name[:14]}…  FP {row['fp']} · missed {row['missed']}"
        )
        panels.append((title, run.annotated))
    return evaluations, panels


def folder_details(evaluation: FolderEvaluation, prefix: str = "") -> dict[str, Any]:
    """Return the Results-table rows for one folder evaluation."""
    n = len(evaluation.per_image)
    details: dict[str, Any] = {f"{prefix}Images evaluated": n}
    if evaluation.skipped:
        details[f"{prefix}Skipped (no labels)"] = evaluation.skipped
    for label, row in evaluation.totals.items():
        text = (
            f"TP {row['tp']}/{row['ground_truth']} · FP {row['fp']} · "
            f"missed {row['missed']} · P {_fmt(row['precision'])} · "
            f"R {_fmt(row['recall'])}"
        )
        if row["ignored"]:
            text += f" · border {row['ignored']}"
        details[f"{prefix}  {label}"] = text
    overall = evaluation.totals["all"]
    details[f"{prefix}F1"] = _fmt(overall["f1"])
    details[f"{prefix}Mean IoU (matched)"] = _fmt(evaluation.mean_iou)
    details[f"{prefix}Time per image"] = (
        f"{evaluation.elapsed_ms / n:.0f} ms ({evaluation.device})"
    )
    for k, image_path in enumerate(evaluation.worst(), 1):
        row = next(s for p, _, s in evaluation.per_image if p == image_path)["all"]
        if row["fp"] + row["missed"] == 0:
            break
        details[f"{prefix}Worst #{k}"] = (
            f"{image_path.name[:20]}…: FP {row['fp']}, missed {row['missed']}"
        )
    return details


def _fmt(value: float) -> str:
    return "–" if value != value else f"{value:.3f}"  # NaN -> dash


def run_details(run: EvaluationRun, prefix: str = "") -> dict[str, Any]:
    """Return the Results-table rows for one model run."""
    details: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for det in run.detections:
        counts[det.name] = counts.get(det.name, 0) + 1
    details[f"{prefix}Predicted"] = len(run.detections)
    if run.summary is None:
        for name in run.info.class_names:
            details[f"{prefix}  {name}"] = counts.get(name, 0)
    else:
        for name, row in run.summary.items():
            details[f"{prefix}  {name}"] = (
                f"GT {row['ground_truth']} · pred {row['predicted']} · "
                f"TP {row['tp']} · FP {row['fp']} · missed {row['missed']}"
            ) + (f" · border {row['ignored']}" if row["ignored"] else "")
        overall = run.summary["all"]
        details[f"{prefix}Precision"] = _fmt(overall["precision"])
        details[f"{prefix}Recall"] = _fmt(overall["recall"])
        ious = [iou for _, _, iou in run.match.matches]
        if ious:
            details[f"{prefix}Mean IoU (matched)"] = f"{np.mean(ious):.3f}"
    details[f"{prefix}Time"] = (
        f"{run.timing['elapsed_ms']:.0f} ms ({run.timing['device']})"
    )
    return details


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def ask_blood_cell_parameters(
    parent: tk.Misc, document: ImageDocument
) -> dict[str, Any] | None:
    """Display the model, image, settings and display dialog."""
    models = list_detection_models()
    if not models:
        messagebox.showerror(
            "No models",
            f"No exported YOLO models found in:\n{YOLO_MODELS_DIR}\n\n"
            "Train and export a model with notebooks/train_yolo.ipynb first.",
            parent=parent,
        )
        return None

    dialog = tk.Toplevel(parent)
    dialog.title("Blood Cell Detection")
    dialog.transient(parent)
    dialog.grab_set()

    model_var = tk.StringVar(value=models[0].display_name)
    has_document = document.current is not None
    source_var = tk.StringVar(value="current" if has_document else "file")
    default_file = ""
    if SAMPLES_DIR.is_dir():
        samples = sorted((SAMPLES_DIR / "images").glob("*"))
        default_file = display_path(samples[0]) if samples else ""
    file_var = tk.StringVar(value=default_file)
    default_folder = SAMPLES_DIR / "images"
    folder_var = tk.StringVar(
        value=display_path(default_folder) if default_folder.is_dir() else ""
    )
    gt_status_var = tk.StringVar()
    conf_var = tk.DoubleVar(value=models[0].inference.get("confidence", 0.25))
    iou_var = tk.DoubleVar(value=models[0].inference.get("nms_iou", 0.7))
    match_var = tk.DoubleVar(value=DEFAULT_MATCH_IOU)
    device_var = tk.StringVar(value="auto")
    show_gt_var = tk.BooleanVar(value=True)
    show_pred_var = tk.BooleanVar(value=True)
    mark_var = tk.BooleanVar(value=True)
    conf_text_var = tk.BooleanVar(value=False)
    border_var = tk.BooleanVar(value=False)
    margin_var = tk.IntVar(value=DEFAULT_BORDER_MARGIN)

    main = ttk.Frame(dialog, padding=12)
    main.pack(fill="both", expand=True)

    # A. Model ------------------------------------------------------------------------
    model_box = ttk.LabelFrame(main, text="A. Model", padding=10)
    model_box.pack(fill="both", expand=True, pady=(0, 8))
    combo = ttk.Combobox(
        model_box,
        textvariable=model_var,
        values=[m.display_name for m in models],
        state="readonly",
        width=60,
    )
    combo.pack(fill="x")
    passport_text = tk.Text(model_box, height=11, width=78, wrap="word", relief="flat")
    passport_text.pack(fill="both", expand=True, pady=(6, 0))

    # B. Image ------------------------------------------------------------------------
    image_box = ttk.LabelFrame(main, text="B. Image or folder", padding=10)
    image_box.pack(fill="x", pady=(0, 8))
    current_radio = ttk.Radiobutton(
        image_box, text="Current image", variable=source_var, value="current"
    )
    current_radio.grid(row=0, column=0, sticky="w")
    if not has_document:
        current_radio.state(["disabled"])
    ttk.Radiobutton(image_box, text="File:", variable=source_var, value="file").grid(
        row=1, column=0, sticky="w"
    )
    ttk.Entry(image_box, textvariable=file_var, width=58).grid(
        row=1, column=1, sticky="ew", padx=4
    )

    def browse() -> None:
        start = SAMPLES_DIR / "images" if SAMPLES_DIR.is_dir() else YOLO_MODELS_DIR
        chosen = filedialog.askopenfilename(
            parent=dialog,
            title="Select an image",
            initialdir=str(start),
            filetypes=IMAGE_TYPES,
        )
        if chosen:
            file_var.set(display_path(chosen))
            source_var.set("file")
            refresh()

    ttk.Button(image_box, text="Browse…", command=browse).grid(row=1, column=2)
    ttk.Radiobutton(
        image_box, text="Folder:", variable=source_var, value="folder"
    ).grid(row=2, column=0, sticky="w")
    ttk.Entry(image_box, textvariable=folder_var, width=58).grid(
        row=2, column=1, sticky="ew", padx=4
    )

    def browse_folder() -> None:
        start = default_folder if default_folder.is_dir() else YOLO_MODELS_DIR
        chosen = filedialog.askdirectory(
            parent=dialog, title="Select a folder of images", initialdir=str(start)
        )
        if chosen:
            folder_var.set(display_path(chosen))
            source_var.set("folder")
            refresh()

    ttk.Button(image_box, text="Browse…", command=browse_folder).grid(row=2, column=2)
    ttk.Label(image_box, textvariable=gt_status_var).grid(
        row=3, column=0, columnspan=3, sticky="w", pady=(6, 0)
    )
    image_box.columnconfigure(1, weight=1)

    # C. Settings ---------------------------------------------------------------------
    settings = ttk.LabelFrame(main, text="C. Detection settings", padding=10)
    settings.pack(fill="x", pady=(0, 8))
    rows = (
        ("Confidence threshold:", conf_var, 0.01, 0.99),
        ("NMS IoU threshold:", iou_var, 0.05, 0.95),
        ("Match IoU (ground truth):", match_var, 0.1, 0.95),
    )
    for r, (label, var, lo, hi) in enumerate(rows):
        ttk.Label(settings, text=label).grid(row=r, column=0, sticky="w")
        ttk.Spinbox(
            settings, textvariable=var, from_=lo, to=hi, increment=0.05, width=8
        ).grid(row=r, column=1, sticky="w", padx=6, pady=2)
    ttk.Label(settings, text="Device:").grid(row=3, column=0, sticky="w")
    ttk.Combobox(
        settings, textvariable=device_var, values=DEVICES, state="readonly", width=8
    ).grid(row=3, column=1, sticky="w", padx=6, pady=2)
    border_row = ttk.Frame(settings)
    border_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
    ttk.Checkbutton(
        border_row,
        text="Don't count unmatched cells cut by the border, within",
        variable=border_var,
    ).pack(side="left")
    ttk.Spinbox(
        border_row, textvariable=margin_var, from_=0, to=50, increment=1, width=4
    ).pack(side="left", padx=4)
    ttk.Label(border_row, text="px").pack(side="left")

    # D. Display ----------------------------------------------------------------------
    display = ttk.LabelFrame(main, text="D. Display", padding=10)
    display.pack(fill="x", pady=(0, 8))
    checks = (
        ("Ground truth (dashed)", show_gt_var),
        ("Predictions (solid)", show_pred_var),
        ("Mark false positives and missed cells", mark_var),
        ("Class and confidence labels", conf_text_var),
    )
    for i, (text, var) in enumerate(checks):
        ttk.Checkbutton(display, text=text, variable=var).grid(
            row=i // 2, column=i % 2, sticky="w", padx=(0, 16)
        )

    # Buttons -------------------------------------------------------------------------
    buttons = ttk.Frame(main)
    buttons.pack(fill="x", pady=(4, 0))
    result: dict[str, Any] | None = None

    def selected() -> YoloModelInfo:
        return next(m for m in models if m.display_name == model_var.get())

    def refresh(_event: object = None) -> None:
        info = selected()
        passport_text.configure(state="normal")
        passport_text.delete("1.0", "end")
        passport_text.insert("1.0", passport_summary(info))
        passport_text.configure(state="disabled")
        if source_var.get() == "folder":
            folder = resolve_path(folder_var.get().strip() or ".")
            if not folder.is_dir():
                gt_status_var.set("Folder not found")
                return
            pairs, skipped = list_labelled_images(folder)
            gt_status_var.set(
                f"{len(pairs)} images with YOLO labels"
                + (f", {skipped} without (skipped)" if skipped else "")
                + "; results saved as CSV in outputs/set9_3_detection/"
            )
            return
        if source_var.get() == "current":
            path = document.path
        else:
            path = (
                resolve_path(file_var.get().strip()) if file_var.get().strip() else None
            )
        label = find_label_file(path)
        gt_status_var.set(
            f"Ground truth: {display_path(label)}"
            if label
            else "Ground truth: no YOLO label file found (predictions only)"
        )

    def finish(action: str) -> None:
        nonlocal result
        try:
            conf, iou, match_iou = (
                float(v.get()) for v in (conf_var, iou_var, match_var)
            )
            margin = max(0, int(margin_var.get())) if border_var.get() else 0
        except (tk.TclError, ValueError):
            messagebox.showerror(
                "Invalid settings",
                "Thresholds must be numbers and the border margin an integer.",
                parent=dialog,
            )
            return
        path = folder = None
        if source_var.get() == "folder" and action != "curves":
            folder = resolve_path(folder_var.get().strip() or ".")
            if not folder.is_dir() or not list_labelled_images(folder)[0]:
                messagebox.showerror(
                    "No labelled images",
                    f"No images with YOLO labels found in:\n{folder}",
                    parent=dialog,
                )
                return
        if source_var.get() == "file":
            path = resolve_path(file_var.get().strip())
            if action != "curves" and not path.is_file():
                messagebox.showerror(
                    "No image", f"File not found:\n{path}", parent=dialog
                )
                return
        info = selected()
        if action != "curves" and not info.has_weights and action != "compare":
            messagebox.showerror(
                "No weights",
                f"Weights not found:\n{info.weights_path}\n\n"
                "Train and export the model with notebooks/train_yolo.ipynb first.",
                parent=dialog,
            )
            return
        result = {
            "action": action,
            "model": info.name,
            "image_path": str(path) if path else None,
            "folder": str(folder) if folder else None,
            "border_margin": margin,
            "confidence": min(max(conf, 0.01), 0.99),
            "iou": min(max(iou, 0.05), 0.95),
            "match_iou": min(max(match_iou, 0.1), 0.95),
            "device": device_var.get(),
            "show_ground_truth": bool(show_gt_var.get()),
            "show_predictions": bool(show_pred_var.get()),
            "mark_errors": bool(mark_var.get()),
            "show_confidence": bool(conf_text_var.get()),
        }
        dialog.destroy()

    ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(buttons, text="Run", command=lambda: finish("detect")).pack(
        side="right", padx=(0, 8)
    )
    ttk.Button(
        buttons, text="Compare all models", command=lambda: finish("compare")
    ).pack(side="right", padx=(0, 8))
    ttk.Button(
        buttons, text="Show training curves", command=lambda: finish("curves")
    ).pack(side="left")

    combo.bind("<<ComboboxSelected>>", refresh)
    source_var.trace_add("write", lambda *_: refresh())
    file_var.trace_add("write", lambda *_: refresh())
    folder_var.trace_add("write", lambda *_: refresh())
    refresh()
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class BloodCellDetectionTool(ForensicsTool):
    """Detect blood cells with a trained YOLO model and compare with labels."""

    tool_id = "blood_cell_detection"
    title = "Blood Cell Detection"
    category = "Deep Learning"
    description = (
        "Detect platelets, red and white blood cells with the YOLO11n models "
        "trained in Set 9.2, compare them with the ground truth and evaluate "
        "whole folders of labelled images."
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
            parameters = ask_blood_cell_parameters(parent, document)
            if parameters is None:
                return None

        models = parameters.get("models_root")
        all_models = (
            list_detection_models(models) if models else list_detection_models()
        )
        if not all_models:
            raise FileNotFoundError(
                f"No exported YOLO models found in {YOLO_MODELS_DIR}"
            )
        by_name = {m.name: m for m in all_models}
        info = by_name.get(parameters.get("model", DEFAULT_MODEL), all_models[0])
        action = parameters.get("action", "detect")

        if action == "curves":
            return self._training_curves(info)

        loaded = dict(parameters.get("loaded_models", {}))
        infos = all_models if action == "compare" else [info]
        infos = [m for m in infos if m.has_weights or m.name in loaded]
        if not infos:
            raise FileNotFoundError("No exported model has its weights (best.pt).")
        loading = any(
            m.name not in loaded and not is_model_cached(m.weights_path) for m in infos
        )
        load_note = "Loading the model(s) (first use only)…\n" if loading else ""
        settings = {
            "Confidence ≥": parameters.get("confidence", 0.25),
            "NMS IoU": parameters.get("iou", 0.7),
            "Match IoU": parameters.get("match_iou", DEFAULT_MATCH_IOU),
            "Border cells": (
                f"not counted ({parameters['border_margin']:g} px)"
                if parameters.get("border_margin")
                else "counted"
            ),
        }
        if parameters.get("folder"):
            return self._evaluate_folder(
                parent, infos, Path(parameters["folder"]), parameters, loaded,
                load_note, settings,
            )  # fmt: skip

        image_path = parameters.get("image_path")
        if image_path:
            path = Path(image_path)
            with Image.open(path) as img:
                image = img.convert("RGB")
        else:
            if document.current is None:
                raise ValueError("Open an image or choose a file first.")
            path = Path(document.path) if document.path else None
            image = document.current
        label_path = find_label_file(path)

        started = time.perf_counter()
        runs = run_with_progress(
            parent,
            "Blood cell detection",
            load_note + "Detecting cells…",
            _evaluate_job,
            infos,
            image,
            label_path,
            parameters,
            loaded,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        details: dict[str, Any] = {
            "Image": (path.name if path else "current image")
            + f" ({image.width}x{image.height})",
            "Ground truth": display_path(label_path) if label_path else "not found",
            **settings,
        }
        if len(runs) == 1:
            run = runs[0]
            details = {"Model": run.info.display_name, **details, **run_details(run)}
            output = run.annotated
            message = self._message(run)
        else:
            for k, run in enumerate(runs, 1):
                details[f"[{k}] Model"] = run.info.display_name
                details.update(run_details(run, prefix=f"[{k}] "))
            output = side_by_side(
                [
                    (f"[{k}] {r.info.display_name}", r.annotated)
                    for k, r in enumerate(runs, 1)
                ]
            )
            message = " | ".join(self._message(r) for r in runs)
        details.update(self._legend(runs[0].info))
        details["Time (total)"] = f"{elapsed_ms:.0f} ms"
        return ToolResult(message=message, image=output, details=details)

    def _evaluate_folder(
        self,
        parent: tk.Misc,
        infos: list[YoloModelInfo],
        folder: Path,
        parameters: dict[str, Any],
        loaded: dict[str, Any],
        load_note: str,
        settings: dict[str, Any],
    ) -> ToolResult:
        """Evaluate every labelled image of a folder and save a CSV per model."""
        pairs, _ = list_labelled_images(folder)
        evaluations, panels = run_with_progress(
            parent,
            "Blood cell evaluation",
            load_note + f"Evaluating {len(pairs)} labelled images…",
            _evaluate_folder_job,
            infos,
            folder,
            parameters,
            loaded,
        )
        output_dir = Path(parameters.get("output_dir") or OUTPUT_DIR)
        margin = float(parameters.get("border_margin", 0))
        details: dict[str, Any] = {"Folder": display_path(folder), **settings}
        messages = []
        for k, evaluation in enumerate(evaluations, 1):
            prefix = f"[{k}] " if len(evaluations) > 1 else ""
            details[f"{prefix}Model"] = evaluation.info.display_name
            details.update(folder_details(evaluation, prefix))
            csv_path = save_evaluation_csv(evaluation, output_dir, margin)
            details[f"{prefix}CSV"] = display_path(csv_path)
            overall = evaluation.totals["all"]
            messages.append(
                f"{evaluation.info.name}: {overall['tp']}/{overall['ground_truth']} "
                f"cells found in {len(evaluation.per_image)} images, "
                f"precision {_fmt(overall['precision'])}, "
                f"recall {_fmt(overall['recall'])}."
            )
        details.update(self._legend(infos[0]))
        output = image_grid(panels, columns=2) if panels else None
        return ToolResult(message=" | ".join(messages), image=output, details=details)

    @staticmethod
    def _legend(info: YoloModelInfo) -> dict[str, Any]:
        return {
            "Legend": "dashed = ground truth, solid = prediction",
            "Legend (errors)": "thick + FP tag = false positive, 'missed' tag",
            "Colours": ", ".join(
                f"{n} #{''.join(f'{c:02x}' for c in rgb)}"
                for n, rgb in zip(info.class_names, info.class_colors)
            ),
        }

    @staticmethod
    def _message(run: EvaluationRun) -> str:
        if run.summary is None:
            return f"{run.info.name}: {len(run.detections)} cells detected (no labels)."
        overall = run.summary["all"]
        return (
            f"{run.info.name}: {overall['tp']}/{overall['ground_truth']} cells found, "
            f"{overall['fp']} false positive(s), {overall['missed']} missed."
        )

    @staticmethod
    def _training_curves(info: YoloModelInfo) -> ToolResult:
        name = info.passport.get("files", {}).get(
            "training_curves", "training_curves.png"
        )
        path = info.folder / name
        if not path.is_file():
            raise FileNotFoundError(f"Training curves not found:\n{path}")
        with Image.open(path) as img:
            curves = img.convert("RGB")
        details = {"Model": info.display_name}
        details.update(
            {
                line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
                for line in passport_summary(info).splitlines()[3:]
                if ":" in line
            }
        )
        return ToolResult(
            message=f"{info.display_name}: training curves (Ultralytics results.png).",
            image=curves,
            details=details,
        )
