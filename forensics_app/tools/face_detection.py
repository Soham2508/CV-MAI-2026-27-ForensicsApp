"""Tool 2 — Face Detection (Static Image) and Shared Detector Backend.

Provides the common ``FaceDetector`` abstraction, ``YuNetDetector`` (ONNX via
``cv2.FaceDetectorYN``), ``ViolaJonesDetector`` (custom Haar+AdaBoost XML or
OpenCV/scikit-image cascade XML), and shared overlay rendering reused by both
static-image and live webcam face detection tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any
import urllib.request

import cv2
import numpy as np
from PIL import Image
from skimage import data as skimage_data
from skimage.feature import Cascade as SkimageCascade

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult
from .viola_jones_trainer import (
    DEFAULT_CASCADE_XML,
    FeatureType,
    HaarLikeFeature,
    ViolaJonesCascadeModel,
    _normalize_patch,
    compute_integral_image,
)

YUNET_MIN_BYTES = 100_000
DOWNLOAD_TIMEOUT_SEC = 60

YUNET_MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_YUNET_MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")

LANDMARK_NAMES: tuple[str, ...] = (
    "Right eye",
    "Left eye",
    "Nose",
    "Right mouth corner",
    "Left mouth corner",
)


def _to_bgr_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or RGB/gray numpy array to 3-channel BGR uint8."""
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    arr = np.asarray(image)
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    return arr.astype(np.uint8).copy()


def _to_gray_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL image or BGR/RGB numpy array to a 2D grayscale uint8 array."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)


def ensure_yunet_model(model_path: str | Path | None = None) -> Path:
    """Locate or download the pretrained YuNet ONNX model."""
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path))
    candidates.extend(
        [
            DEFAULT_YUNET_MODEL_PATH,
            Path("face_detection_yunet_2023mar.onnx"),
        ]
    )
    for cand in candidates:
        if cand.exists() and cand.is_file() and cand.stat().st_size >= YUNET_MIN_BYTES:
            return cand

    target = Path(model_path) if model_path else DEFAULT_YUNET_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temporary file and only rename it into place once it is
    # complete, so an interrupted download is never mistaken for the model.
    partial = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(
            YUNET_MODEL_URL, timeout=DOWNLOAD_TIMEOUT_SEC
        ) as resp:
            expected = resp.headers.get("Content-Length")
            with open(partial, "wb") as out:
                while chunk := resp.read(1 << 16):
                    out.write(chunk)
        size = partial.stat().st_size
        if size < YUNET_MIN_BYTES or (expected and size != int(expected)):
            raise RuntimeError("YuNet model download is incomplete.")
        os.replace(partial, target)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the YuNet model ({exc}). Browse to a local "
            "face_detection_yunet_2023mar.onnx file instead."
        ) from exc
    return target


class FaceDetector(ABC):
    """Shared detector interface implemented by YuNet and Viola–Jones detectors."""

    @abstractmethod
    def load_model(self, model_path: str | Path | None = None) -> None:
        """Load the underlying detection model from disk."""
        raise NotImplementedError

    @abstractmethod
    def configure(self, **params: Any) -> None:
        """Configure detection hyperparameters."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
        """Detect faces and return a list of shared detection dicts:
        ``{"bbox": (x, y, width, height), "confidence": ..., "landmarks": ...}``.
        """
        raise NotImplementedError


class YuNetDetector(FaceDetector):
    """YuNet ONNX face detector backed by ``cv2.FaceDetectorYN``."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        confidence_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ) -> None:
        self.model_path: Path | None = Path(model_path) if model_path else None
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self._detector: Any = None
        self._loaded_path: Path | None = None

    def load_model(self, model_path: str | Path | None = None) -> None:
        resolved = ensure_yunet_model(model_path or self.model_path)
        self.model_path = resolved
        self._loaded_path = resolved
        self._detector = cv2.FaceDetectorYN.create(
            str(resolved),
            "",
            (320, 320),
            float(self.confidence_threshold),
            float(self.nms_threshold),
            int(self.top_k),
        )

    def configure(self, **params: Any) -> None:
        if (
            "confidence_threshold" in params
            and params["confidence_threshold"] is not None
        ):
            self.confidence_threshold = float(params["confidence_threshold"])
        if "score_threshold" in params and params["score_threshold"] is not None:
            self.confidence_threshold = float(params["score_threshold"])
        if "nms_threshold" in params and params["nms_threshold"] is not None:
            self.nms_threshold = float(params["nms_threshold"])
        if "top_k" in params and params["top_k"] is not None:
            self.top_k = int(params["top_k"])

        if self._detector is not None:
            if hasattr(self._detector, "setScoreThreshold"):
                self._detector.setScoreThreshold(float(self.confidence_threshold))
            if hasattr(self._detector, "setNMSThreshold"):
                self._detector.setNMSThreshold(float(self.nms_threshold))
            if hasattr(self._detector, "setTopK"):
                self._detector.setTopK(int(self.top_k))

    def detect(self, image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
        if self._detector is None:
            self.load_model(self.model_path)
        assert self._detector is not None

        bgr = _to_bgr_array(image)
        height, width = bgr.shape[:2]
        self._detector.setInputSize((int(width), int(height)))
        self.configure()

        _, faces = self._detector.detect(bgr)
        results: list[dict[str, Any]] = []
        if faces is None or len(faces) == 0:
            return results

        for face in faces:
            x, y, w, h = face[:4].astype(int)
            confidence = float(face[14])
            pts = face[4:14].reshape(5, 2).astype(int)
            landmarks = [(int(px), int(py)) for px, py in pts]
            results.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": confidence,
                    "landmarks": landmarks,
                }
            )
        return results


def _group_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    min_neighbors: int = 3,
    iou_threshold: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """Group nearby candidate windows and apply non-maximum suppression.

    Starting from the highest-scoring unused window, every unused window whose
    IoU with it is at least ``iou_threshold`` joins its group. A group with at
    least ``min_neighbors`` windows is reported as its average box; otherwise
    only the seed window is discarded. IoU is computed for all windows at once.
    """
    if not boxes:
        return []
    if min_neighbors <= 1 and len(boxes) == 1:
        return boxes

    arr = np.asarray(boxes, dtype=np.float64)
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = x1 + arr[:, 2], y1 + arr[:, 3]
    area = arr[:, 2] * arr[:, 3]

    kept: list[tuple[int, int, int, int]] = []
    used = np.zeros(len(boxes), dtype=bool)
    for idx in order:
        if used[idx]:
            continue
        inter_w = np.clip(np.minimum(x2[idx], x2) - np.maximum(x1[idx], x1), 0.0, None)
        inter_h = np.clip(np.minimum(y2[idx], y2) - np.maximum(y1[idx], y1), 0.0, None)
        inter = inter_w * inter_h
        union = area[idx] + area - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        members = (iou >= iou_threshold) & ~used
        members[idx] = True
        cluster = np.flatnonzero(members)
        if len(cluster) >= max(1, min_neighbors):
            used[cluster] = True
            mean_box = arr[cluster].mean(axis=0)
            kept.append(tuple(int(round(v)) for v in mean_box))
        else:
            used[idx] = True
    return kept


def _feature_rects(
    clf: HaarLikeFeature,
) -> list[tuple[int, int, int, int, int]]:
    """Return a Haar feature's rectangles as (top, left, bottom, right, sign).

    Matches ``HaarLikeFeature.get_score_batch`` exactly, so whole-image
    evaluation gives the same scores as evaluating one window at a time.
    """
    r, c = clf.top_left
    w, h = clf.width, clf.height
    if clf.type == FeatureType.TWO_VERTICAL:
        half = h // 2
        return [(r, c, r + half, c + w, 1), (r + half, c, r + h, c + w, -1)]
    if clf.type == FeatureType.TWO_HORIZONTAL:
        half = w // 2
        return [(r, c, r + h, c + half, 1), (r, c + half, r + h, c + w, -1)]
    if clf.type == FeatureType.THREE_HORIZONTAL:
        third = w // 3
        return [
            (r, c, r + h, c + third, 1),
            (r, c + third, r + h, c + 2 * third, -1),
            (r, c + 2 * third, r + h, c + w, 1),
        ]
    if clf.type == FeatureType.THREE_VERTICAL:
        third = h // 3
        return [
            (r, c, r + third, c + w, 1),
            (r + third, c, r + 2 * third, c + w, -1),
            (r + 2 * third, c, r + h, c + w, 1),
        ]
    if clf.type == FeatureType.FOUR:
        half_h, half_w = h // 2, w // 2
        return [
            (r, c, r + half_h, c + half_w, 1),
            (r, c + half_w, r + half_h, c + w, -1),
            (r + half_h, c, r + h, c + half_w, -1),
            (r + half_h, c + half_w, r + h, c + w, 1),
        ]
    raise ValueError(f"Unsupported Haar feature type: {clf.type}")


def _evaluate_cascade_windows(
    model: ViolaJonesCascadeModel,
    ii: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    threshold_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run every window whose top-left is (rows[i], cols[i]) through the cascade.

    All windows still alive are scored together, stage by stage, so early
    rejection is kept while Python overhead no longer grows with the window
    count. Returns the indices of windows passing every stage and the summed
    stage scores of all windows (same values as ``evaluate_window``).
    """
    alive = np.arange(len(rows))
    total = np.zeros(len(rows), dtype=np.float64)
    for stage in model.stages:
        r_idx, c_idx = rows[alive], cols[alive]
        stage_score = np.zeros(len(alive), dtype=np.float64)
        for clf in stage.classifiers:
            value = np.zeros(len(alive), dtype=np.float64)
            for top, left, bottom, right, sign in _feature_rects(clf):
                value += sign * (
                    ii[r_idx + bottom, c_idx + right]
                    - ii[r_idx + top, c_idx + right]
                    - ii[r_idx + bottom, c_idx + left]
                    + ii[r_idx + top, c_idx + left]
                )
            stage_score += clf.weight * clf.predict_from_score(value)
        passed = stage_score >= stage.stage_threshold + threshold_offset
        total[alive] += stage_score
        alive = alive[passed]
        if alive.size == 0:
            break
    return alive, total


class ViolaJonesDetector(FaceDetector):
    """Viola–Jones / Cascade face detector supporting custom and OpenCV/skimage XML."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 3,
        min_size: tuple[int, int] = (24, 24),
        max_size: tuple[int, int] = (0, 0),
        threshold: float = 0.0,
    ) -> None:
        self.model_path: Path | None = Path(model_path) if model_path else None
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = (int(min_size[0]), int(min_size[1]))
        self.max_size = (int(max_size[0]), int(max_size[1]))
        self.threshold = float(threshold)

        self._custom_model: ViolaJonesCascadeModel | None = None
        self._cv2_cascade: Any = None
        self._skimage_cascade: SkimageCascade | None = None
        self._backend_kind: str = "unloaded"

    def load_model(self, model_path: str | Path | None = None) -> None:
        chosen = Path(model_path) if model_path else self.model_path
        self._custom_model = None
        self._cv2_cascade = None
        self._skimage_cascade = None

        if chosen and not chosen.is_file():
            raise FileNotFoundError(
                f"Cascade file not found: {chosen}. Check the model path."
            )
        if chosen and chosen.exists() and chosen.is_file():
            self.model_path = chosen
            # First check if it's our custom Viola-Jones XML cascade
            try:
                self._custom_model = ViolaJonesCascadeModel.load_xml(chosen)
                self._backend_kind = "custom_viola_jones"
                return
            except ValueError:
                pass

            # Next check if cv2.CascadeClassifier is available in this OpenCV build
            if hasattr(cv2, "CascadeClassifier"):
                cascade = cv2.CascadeClassifier(str(chosen))
                if not cascade.empty():
                    self._cv2_cascade = cascade
                    self._backend_kind = "cv2_cascade"
                    return

            # Next check scikit-image Cascade XML loader
            try:
                self._skimage_cascade = SkimageCascade(str(chosen))
                self._backend_kind = "skimage_cascade"
                return
            except Exception as exc:
                raise ValueError(f"Could not load cascade XML {chosen}: {exc}") from exc

        # If default custom XML exists on disk, load it
        if DEFAULT_CASCADE_XML.exists():
            try:
                self._custom_model = ViolaJonesCascadeModel.load_xml(
                    DEFAULT_CASCADE_XML
                )
                self.model_path = DEFAULT_CASCADE_XML
                self._backend_kind = "custom_viola_jones"
                return
            except ValueError:
                pass

        # Fallback to scikit-image's built-in frontal face cascade XML
        builtin_xml = skimage_data.lbp_frontal_face_cascade_filename()
        self._skimage_cascade = SkimageCascade(builtin_xml)
        self.model_path = Path(builtin_xml)
        self._backend_kind = "skimage_cascade"

    def configure(self, **params: Any) -> None:
        if "scale_factor" in params and params["scale_factor"] is not None:
            scale_factor = float(params["scale_factor"])
            if not scale_factor > 1.0:
                raise ValueError("Scale factor must be greater than 1.0.")
            self.scale_factor = scale_factor
        if "min_neighbors" in params and params["min_neighbors"] is not None:
            self.min_neighbors = max(1, int(params["min_neighbors"]))
        if "min_size" in params and params["min_size"] is not None:
            ms = params["min_size"]
            self.min_size = (int(ms[0]), int(ms[1]))
        if "max_size" in params and params["max_size"] is not None:
            mxs = params["max_size"]
            self.max_size = (int(mxs[0]), int(mxs[1]))
        if "threshold" in params and params["threshold"] is not None:
            self.threshold = float(params["threshold"])

    def _detect_with_custom_cascade(self, gray: np.ndarray) -> list[dict[str, Any]]:
        assert self._custom_model is not None
        model = self._custom_model
        img_h, img_w = gray.shape[:2]
        win_w, win_h = model.window_width, model.window_height

        min_w = max(win_w, self.min_size[0])
        min_h = max(win_h, self.min_size[1])
        max_w = self.max_size[0] if self.max_size[0] > 0 else img_w
        max_h = self.max_size[1] if self.max_size[1] > 0 else img_h

        candidate_boxes: list[tuple[int, int, int, int]] = []
        candidate_scores: list[float] = []

        scale = max(1.0, max(min_w / win_w, min_h / win_h))
        # Limit number of pyramid levels for responsiveness
        while int(win_w * scale) <= min(img_w, max_w) and int(win_h * scale) <= min(
            img_h, max_h
        ):
            cur_w = int(round(win_w * scale))
            cur_h = int(round(win_h * scale))
            scaled_w = max(win_w, int(round(img_w / scale)))
            scaled_h = max(win_h, int(round(img_h / scale)))
            resized = cv2.resize(
                gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )

            scaled_norm = _normalize_patch(resized)
            ii_full = compute_integral_image(scaled_norm)

            step = max(2, min(win_w, win_h) // 4)
            grid_r, grid_c = np.meshgrid(
                np.arange(0, scaled_h - win_h + 1, step),
                np.arange(0, scaled_w - win_w + 1, step),
                indexing="ij",
            )
            rows, cols = grid_r.ravel(), grid_c.ravel()
            passed, scores = _evaluate_cascade_windows(
                model, ii_full, rows, cols, threshold_offset=self.threshold
            )
            for i in passed:
                orig_x = int(round(cols[i] * scale))
                orig_y = int(round(rows[i] * scale))
                candidate_boxes.append((orig_x, orig_y, cur_w, cur_h))
                candidate_scores.append(float(scores[i]))

            scale *= self.scale_factor

        grouped = _group_overlapping_boxes(
            candidate_boxes,
            candidate_scores,
            min_neighbors=self.min_neighbors,
        )
        return [{"bbox": box, "confidence": None, "landmarks": None} for box in grouped]

    def detect(self, image: Image.Image | np.ndarray) -> list[dict[str, Any]]:
        if self._backend_kind == "unloaded":
            self.load_model(self.model_path)

        gray = _to_gray_array(image)
        img_h, img_w = gray.shape[:2]

        if (
            self._backend_kind == "custom_viola_jones"
            and self._custom_model is not None
        ):
            return self._detect_with_custom_cascade(gray)

        if self._backend_kind == "cv2_cascade" and self._cv2_cascade is not None:
            max_sz = (
                self.max_size
                if (self.max_size[0] > 0 and self.max_size[1] > 0)
                else None
            )
            rects = self._cv2_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                maxSize=max_sz,
            )
            if rects is None or len(rects) == 0:
                return []
            return [
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": None,
                    "landmarks": None,
                }
                for (x, y, w, h) in rects
            ]

        if (
            self._backend_kind == "skimage_cascade"
            and self._skimage_cascade is not None
        ):
            min_r = max(12, self.min_size[1])
            min_c = max(12, self.min_size[0])
            max_r = self.max_size[1] if self.max_size[1] > 0 else img_h
            max_c = self.max_size[0] if self.max_size[0] > 0 else img_w
            detected = self._skimage_cascade.detect_multi_scale(
                img=gray,
                scale_factor=max(1.05, self.scale_factor),
                step_ratio=1,
                min_size=(min_r, min_c),
                max_size=(max(min_r, max_r), max(min_c, max_c)),
                min_neighbor_number=max(1, self.min_neighbors),
            )
            results: list[dict[str, Any]] = []
            for patch in detected:
                x = int(patch["c"])
                y = int(patch["r"])
                w = int(patch["width"])
                h = int(patch["height"])
                results.append(
                    {
                        "bbox": (x, y, w, h),
                        "confidence": None,
                        "landmarks": None,
                    }
                )
            return results

        return []


def create_face_detector(
    detector_type: str = "YuNet",
    model_path: str | Path | None = None,
    **params: Any,
) -> FaceDetector:
    """Factory returning a configured ``YuNetDetector`` or ``ViolaJonesDetector``."""
    norm = detector_type.strip().lower().replace("–", "-").replace("_", "-")
    if norm in ("yunet", "onnx"):
        det: FaceDetector = YuNetDetector(model_path=model_path)
    elif norm in ("viola-jones", "violajones", "haar", "cascade"):
        det = ViolaJonesDetector(model_path=model_path)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type!r}")
    det.configure(**params)
    det.load_model(model_path)
    return det


def draw_face_detections(
    image: Image.Image | np.ndarray,
    detections: list[dict[str, Any]],
    *,
    show_boxes: bool = True,
    show_landmarks: bool = True,
    show_confidence: bool = True,
    fps: float | None = None,
) -> Image.Image:
    """Draw bounding boxes, confidence scores, 5 landmarks, and optional FPS."""
    bgr = _to_bgr_array(image)

    for det in detections:
        x, y, w, h = det["bbox"]
        confidence = det.get("confidence")
        landmarks = det.get("landmarks")

        if show_boxes:
            cv2.rectangle(
                bgr,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (0, 255, 0),
                2,
            )

        if show_confidence and confidence is not None:
            cv2.putText(
                bgr,
                f"Face: {float(confidence):.2f}",
                (int(x), max(20, int(y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if show_landmarks and landmarks is not None:
            pts = (
                list(landmarks.values())
                if isinstance(landmarks, dict)
                else list(landmarks)
            )
            for px, py in pts:
                # Blue circle in BGR: (255, 0, 0) -> renders as blue in RGB
                cv2.circle(bgr, (int(px), int(py)), 4, (255, 0, 0), -1)

    if fps is not None:
        cv2.putText(
            bgr,
            f"FPS: {fps:.1f} | Faces: {len(detections)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB")


def detect_faces_in_image(
    image: Image.Image,
    *,
    detector_type: str = "YuNet",
    model_path: str | Path | None = None,
    show_boxes: bool = True,
    show_landmarks: bool = True,
    show_confidence: bool = True,
    **detector_params: Any,
) -> dict[str, Any]:
    """Run face detection on a static image and return detections and timing."""
    t0 = time.perf_counter()
    detector = create_face_detector(
        detector_type=detector_type, model_path=model_path, **detector_params
    )
    detections = detector.detect(image)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    is_yunet = detector_type.strip().lower().startswith("yunet")
    annotated = draw_face_detections(
        image,
        detections,
        show_boxes=show_boxes,
        show_landmarks=show_landmarks and is_yunet,
        show_confidence=show_confidence and is_yunet,
    )

    return {
        "detector_type": "YuNet" if is_yunet else "Viola–Jones",
        "model_path": getattr(detector, "model_path", model_path),
        "detections": detections,
        "annotated_image": annotated,
        "processing_time_ms": elapsed_ms,
    }


def ask_face_detection_parameters(
    parent: tk.Misc, initial_detector: str = "YuNet"
) -> dict[str, Any] | None:
    """Display the generic Face Detection configuration panel (Plan Section 3.A)."""
    dialog = tk.Toplevel(parent)
    dialog.title("Face Detection Configuration")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    detector_var = tk.StringVar(
        value="Viola–Jones" if "viola" in initial_detector.lower() else "YuNet"
    )
    yunet_default = str(DEFAULT_YUNET_MODEL_PATH)
    vj_candidates = (
        list(Path("models/viola_jones").glob("**/cascade.xml"))
        if Path("models/viola_jones").exists()
        else []
    )
    if vj_candidates:
        vj_default = str(max(vj_candidates, key=lambda p: p.stat().st_mtime))
    elif DEFAULT_CASCADE_XML.exists():
        vj_default = str(DEFAULT_CASCADE_XML)
    else:
        vj_default = ""
    model_path_var = tk.StringVar(
        value=yunet_default if detector_var.get() == "YuNet" else vj_default
    )

    # YuNet parameters
    conf_var = tk.DoubleVar(value=0.60)
    nms_var = tk.DoubleVar(value=0.30)
    topk_var = tk.IntVar(value=5000)
    show_landmarks_var = tk.BooleanVar(value=True)
    show_conf_var = tk.BooleanVar(value=True)

    # Viola–Jones parameters
    scale_factor_var = tk.DoubleVar(value=1.10)
    min_neighbors_var = tk.IntVar(value=3)
    min_size_var = tk.IntVar(value=24)
    max_size_var = tk.IntVar(value=0)
    vj_thresh_var = tk.DoubleVar(value=0.0)

    main_frame = ttk.Frame(dialog, padding=12)
    main_frame.pack(fill="both", expand=True)

    # Detector selection & model path
    top_frame = ttk.LabelFrame(main_frame, text="Detector & Model Path", padding=10)
    top_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(top_frame, text="Model type:").grid(row=0, column=0, sticky="w", pady=4)
    type_box = ttk.Combobox(
        top_frame,
        textvariable=detector_var,
        values=("YuNet", "Viola–Jones"),
        state="readonly",
        width=16,
    )
    type_box.grid(row=0, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(top_frame, text="Model path:").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(top_frame, textvariable=model_path_var, width=38).grid(
        row=1, column=1, padx=6, pady=4
    )

    def _browse_model() -> None:
        is_yn = detector_var.get() == "YuNet"
        ftypes = (
            [("ONNX model", "*.onnx"), ("All files", "*.*")]
            if is_yn
            else [("XML cascade", "*.xml"), ("All files", "*.*")]
        )
        chosen = filedialog.askopenfilename(
            parent=dialog, title="Select face detector model", filetypes=ftypes
        )
        if chosen:
            model_path_var.set(chosen)

    ttk.Button(top_frame, text="Browse…", command=_browse_model).grid(
        row=1, column=2, pady=4
    )

    # YuNet parameters frame
    yunet_frame = ttk.LabelFrame(main_frame, text="YuNet Parameters", padding=10)
    yunet_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(yunet_frame, text="Confidence threshold:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    tk.Scale(
        yunet_frame,
        from_=0.05,
        to=0.99,
        resolution=0.01,
        orient="horizontal",
        variable=conf_var,
        showvalue=False,
        length=180,
    ).grid(row=0, column=1, padx=4)
    ttk.Spinbox(
        yunet_frame,
        textvariable=conf_var,
        from_=0.05,
        to=0.99,
        increment=0.05,
        width=6,
    ).grid(row=0, column=2, padx=4)

    ttk.Label(yunet_frame, text="NMS threshold:").grid(
        row=1, column=0, sticky="w", pady=3
    )
    tk.Scale(
        yunet_frame,
        from_=0.05,
        to=0.99,
        resolution=0.01,
        orient="horizontal",
        variable=nms_var,
        showvalue=False,
        length=180,
    ).grid(row=1, column=1, padx=4)
    ttk.Spinbox(
        yunet_frame,
        textvariable=nms_var,
        from_=0.05,
        to=0.99,
        increment=0.05,
        width=6,
    ).grid(row=1, column=2, padx=4)

    ttk.Label(yunet_frame, text="Maximum candidates (top K):").grid(
        row=2, column=0, sticky="w", pady=3
    )
    tk.Scale(
        yunet_frame,
        from_=100,
        to=10000,
        resolution=100,
        orient="horizontal",
        variable=topk_var,
        showvalue=False,
        length=180,
    ).grid(row=2, column=1, padx=4)
    ttk.Spinbox(
        yunet_frame,
        textvariable=topk_var,
        from_=1,
        to=10000,
        increment=100,
        width=6,
    ).grid(row=2, column=2, padx=4)

    ttk.Checkbutton(
        yunet_frame,
        text="Display five facial landmarks",
        variable=show_landmarks_var,
    ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 2))
    ttk.Checkbutton(
        yunet_frame,
        text="Display confidence score",
        variable=show_conf_var,
    ).grid(row=4, column=0, columnspan=3, sticky="w", pady=2)

    # Viola–Jones parameters frame
    vj_frame = ttk.LabelFrame(
        main_frame, text="Viola–Jones / Haar Cascade Parameters", padding=10
    )
    vj_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(vj_frame, text="Scale factor:").grid(row=0, column=0, sticky="w", pady=3)
    ttk.Spinbox(
        vj_frame,
        textvariable=scale_factor_var,
        from_=1.01,
        to=2.0,
        increment=0.05,
        width=8,
    ).grid(row=0, column=1, sticky="w", padx=6, pady=3)

    ttk.Label(vj_frame, text="Min neighbours:").grid(
        row=0, column=2, sticky="w", padx=(12, 0), pady=3
    )
    ttk.Spinbox(
        vj_frame,
        textvariable=min_neighbors_var,
        from_=1,
        to=20,
        increment=1,
        width=8,
    ).grid(row=0, column=3, sticky="w", padx=6, pady=3)

    ttk.Label(vj_frame, text="Min face size (px):").grid(
        row=1, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(
        vj_frame,
        textvariable=min_size_var,
        from_=12,
        to=512,
        increment=4,
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=6, pady=3)

    ttk.Label(vj_frame, text="Max face size (0=auto):").grid(
        row=1, column=2, sticky="w", padx=(12, 0), pady=3
    )
    ttk.Spinbox(
        vj_frame,
        textvariable=max_size_var,
        from_=0,
        to=2048,
        increment=16,
        width=8,
    ).grid(row=1, column=3, sticky="w", padx=6, pady=3)

    ttk.Label(vj_frame, text="Stage threshold offset:").grid(
        row=2, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(
        vj_frame,
        textvariable=vj_thresh_var,
        from_=-5.0,
        to=5.0,
        increment=0.1,
        width=8,
    ).grid(row=2, column=1, sticky="w", padx=6, pady=3)

    def _set_children_state(frame: tk.Widget, state: str) -> None:
        for child in frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass

    def _on_detector_change(*_args: object) -> None:
        if detector_var.get() == "YuNet":
            _set_children_state(yunet_frame, "normal")
            _set_children_state(vj_frame, "disabled")
            if not model_path_var.get().endswith(".onnx"):
                model_path_var.set(yunet_default)
        else:
            _set_children_state(yunet_frame, "disabled")
            _set_children_state(vj_frame, "normal")
            if model_path_var.get().endswith(".onnx"):
                model_path_var.set(vj_default)

    detector_var.trace_add("write", _on_detector_change)
    _on_detector_change()

    result: dict[str, Any] | None = None

    def apply() -> None:
        nonlocal result
        det_type = detector_var.get()
        mpath = model_path_var.get().strip() or None
        try:
            if det_type == "YuNet":
                conf = float(conf_var.get())
                nms = float(nms_var.get())
                topk = int(topk_var.get())
                if not (0.0 < conf <= 1.0) or not (0.0 < nms <= 1.0) or topk < 1:
                    raise ValueError
                result = {
                    "detector_type": "YuNet",
                    "model_path": mpath,
                    "confidence_threshold": conf,
                    "nms_threshold": nms,
                    "top_k": topk,
                    "show_landmarks": bool(show_landmarks_var.get()),
                    "show_confidence": bool(show_conf_var.get()),
                }
            else:
                sf = float(scale_factor_var.get())
                mn = int(min_neighbors_var.get())
                min_s = int(min_size_var.get())
                max_s = int(max_size_var.get())
                th = float(vj_thresh_var.get())
                if sf <= 1.0 or mn < 1 or min_s < 8 or max_s < 0:
                    raise ValueError
                result = {
                    "detector_type": "Viola–Jones",
                    "model_path": mpath,
                    "scale_factor": sf,
                    "min_neighbors": mn,
                    "min_size": (min_s, min_s),
                    "max_size": (max_s, max_s),
                    "threshold": th,
                    "show_landmarks": False,
                    "show_confidence": False,
                }
        except (ValueError, tk.TclError):
            messagebox.showerror(
                "Invalid Parameters",
                "Please enter valid detection parameters.",
                parent=dialog,
            )
            return
        dialog.destroy()

    btn_bar = ttk.Frame(main_frame)
    btn_bar.pack(fill="x", pady=(6, 0))
    ttk.Button(btn_bar, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(btn_bar, text="Detect Faces", command=apply).pack(
        side="right", padx=(0, 8)
    )

    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class FaceDetectionTool(ForensicsTool):
    """Tool 2 — Generic static-image face detection supporting YuNet and Viola–Jones."""

    tool_id = "face_detection"
    title = "Face Detection (Image)"
    category = "Face Analysis"
    description = (
        "Select YuNet or Viola–Jones, configure detection parameters and detect faces."
    )
    requires_image = True

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> ToolResult | None:
        assert document.current is not None

        if parameters is None:
            parameters = ask_face_detection_parameters(parent)
            if parameters is None:
                return None

        out = detect_faces_in_image(document.current, **parameters)
        detections = out["detections"]
        det_name = out["detector_type"]
        elapsed_ms = out["processing_time_ms"]
        model_path = out.get("model_path")

        first_conf = (
            f"{detections[0]['confidence']:.2f}"
            if detections and detections[0].get("confidence") is not None
            else "N/A"
        )
        first_bbox = (
            f"{detections[0]['bbox'][0]}, {detections[0]['bbox'][1]}, "
            f"{detections[0]['bbox'][2]}, {detections[0]['bbox'][3]}"
            if detections
            else "None"
        )

        details: dict[str, Any] = {
            "Detector": det_name,
            "Model": Path(model_path).name if model_path else det_name,
            "Faces detected": len(detections),
            "Confidence": first_conf,
            "Processing time": f"{elapsed_ms:.1f} ms",
            "Bounding box": first_bbox,
        }

        for idx, det in enumerate(detections[:5]):
            bx, by, bw, bh = det["bbox"]
            conf = det.get("confidence")
            conf_str = f" (conf={conf:.2f})" if conf is not None else ""
            details[f"Face {idx + 1} bbox"] = f"({bx}, {by}, {bw}, {bh}){conf_str}"
            if idx == 0 and det.get("landmarks"):
                for name, (px, py) in zip(LANDMARK_NAMES, det["landmarks"]):
                    details[name] = f"({px}, {py})"

        return ToolResult(
            image=out["annotated_image"],
            message=(
                f"{det_name} detected {len(detections)} face(s) in {elapsed_ms:.1f} ms."
            ),
            details=details,
        )
