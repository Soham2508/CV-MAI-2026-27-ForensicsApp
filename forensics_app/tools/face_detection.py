"""Tool 2 — Face Detection (Static Image) and Shared Detector Backend.

Provides the common ``FaceDetector`` abstraction, ``YuNetDetector`` (ONNX via
``cv2.FaceDetectorYN``), ``ViolaJonesDetector`` (custom Haar+AdaBoost XML or
OpenCV/scikit-image cascade XML), and shared overlay rendering reused by both
static-image and live webcam face detection tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
    ViolaJonesCascadeModel,
    compute_integral_image,
)

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
        if cand.exists() and cand.is_file() and cand.stat().st_size >= 100_000:
            return cand

    target = Path(model_path) if model_path else DEFAULT_YUNET_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(YUNET_MODEL_URL, str(target))
    if not target.exists() or target.stat().st_size < 100_000:
        raise RuntimeError("YuNet model download failed or the file is incomplete.")
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
    """Group nearby candidate windows and apply non-maximum suppression."""
    if not boxes:
        return []
    if min_neighbors <= 1 and len(boxes) == 1:
        return boxes

    arr = np.asarray(boxes, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    order = np.argsort(sc)[::-1]

    kept: list[tuple[int, int, int, int]] = []
    used = np.zeros(len(boxes), dtype=bool)

    for idx in order:
        if used[idx]:
            continue
        x1, y1, w1, h1 = arr[idx]
        cluster = [idx]
        for other in order:
            if other == idx or used[other]:
                continue
            x2, y2, w2, h2 = arr[other]
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = w1 * h1 + w2 * h2 - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                cluster.append(other)

        if len(cluster) >= max(1, min_neighbors):
            for c_idx in cluster:
                used[c_idx] = True
            mean_box = np.mean(arr[cluster], axis=0)
            kept.append(
                (
                    int(round(mean_box[0])),
                    int(round(mean_box[1])),
                    int(round(mean_box[2])),
                    int(round(mean_box[3])),
                )
            )
    return kept


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
            self.scale_factor = max(1.02, float(params["scale_factor"]))
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

            step = max(2, min(win_w, win_h) // 3)
            for r in range(0, scaled_h - win_h + 1, step):
                for c in range(0, scaled_w - win_w + 1, step):
                    patch = resized[r : r + win_h, c : c + win_w].astype(np.float64)
                    pmax = float(np.max(patch))
                    if pmax > 0:
                        patch = patch / pmax
                    ii = compute_integral_image(patch)
                    iimax = float(np.max(ii))
                    if iimax > 0:
                        ii = ii / iimax
                    passed, score = model.evaluate_window(
                        ii, threshold_offset=self.threshold
                    )
                    if passed:
                        orig_x = int(round(c * scale))
                        orig_y = int(round(r * scale))
                        candidate_boxes.append((orig_x, orig_y, cur_w, cur_h))
                        candidate_scores.append(score)

            scale *= max(1.15, self.scale_factor)

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
    vj_default = str(DEFAULT_CASCADE_XML) if DEFAULT_CASCADE_XML.exists() else ""
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
        from_=1.05,
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
