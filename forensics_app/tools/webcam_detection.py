"""Tool 3 — Webcam Face Detection.

Reuses the shared detection backend and configuration from Tool 2 to run live
face detection on a webcam stream using a background worker thread and Tkinter's
``after()`` mechanism to render annotated frames directly in the central canvas.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from forensics_app.ui.image_view import ImageView

from .base import ForensicsTool, ToolResult
from .face_detection import (
    DEFAULT_YUNET_MODEL_PATH,
    LANDMARK_NAMES,
    FaceDetector,
    create_face_detector,
    draw_face_detections,
)
from .viola_jones_trainer import DEFAULT_CASCADE_XML

RESOLUTIONS: tuple[str, ...] = ("640x480", "1280x720", "320x240")
CAMERA_DEVICES: tuple[str, ...] = ("0", "1", "2")
SAVE_TYPES = [
    ("PNG image", "*.png"),
    ("JPEG image", "*.jpg"),
    ("TIFF image", "*.tiff"),
]


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse a 'WxH' resolution string into (width, height)."""
    parts = res_str.lower().split("x")
    if len(parts) != 2:
        return (640, 480)
    try:
        w, h = int(parts[0].strip()), int(parts[1].strip())
        return (max(64, w), max(64, h))
    except ValueError:
        return (640, 480)


def process_webcam_frame(
    frame_bgr: np.ndarray | Image.Image,
    detector: FaceDetector,
    *,
    show_boxes: bool = True,
    show_landmarks: bool = True,
    show_confidence: bool = True,
    show_fps: bool = True,
    fps: float | None = None,
) -> dict[str, Any]:
    """Run face detection on a webcam frame and return annotated image and metrics."""
    t0 = time.perf_counter()
    detections = detector.detect(frame_bgr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    measured_fps = (
        fps if fps is not None else (1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0)
    )

    annotated = draw_face_detections(
        frame_bgr,
        detections,
        show_boxes=show_boxes,
        show_landmarks=show_landmarks,
        show_confidence=show_confidence,
        fps=measured_fps if show_fps else None,
    )
    return {
        "annotated_image": annotated,
        "detections": detections,
        "processing_time_ms": elapsed_ms,
        "fps": measured_fps,
    }


def _find_central_image_view(widget: tk.Misc | None) -> ImageView | None:
    """Locate MainWindow's central ``ImageView`` widget from the parent hierarchy."""
    if widget is None or not hasattr(widget, "winfo_children"):
        return None
    root = widget.winfo_toplevel() if hasattr(widget, "winfo_toplevel") else widget
    queue: list[tk.Misc] = [root]
    while queue:
        current = queue.pop(0)
        if isinstance(current, ImageView):
            return current
        if hasattr(current, "winfo_children"):
            try:
                queue.extend(current.winfo_children())
            except Exception:
                pass
    return None


class WebcamDetectionController:
    """Dialog & background worker controller for live webcam face detection."""

    def __init__(self, parent: tk.Misc, document: ImageDocument) -> None:
        self.parent = parent
        self.document = document
        self.central_view = _find_central_image_view(parent)

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Webcam Face Detection")
        self.dialog.transient(parent)
        self.dialog.resizable(True, True)
        self.dialog.grab_set()

        self._running = False
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._latest_frame: Image.Image | None = None
        self._latest_detections: list[dict[str, Any]] = []
        self._latest_fps: float = 0.0
        self._latest_proc_ms: float = 0.0
        self._captured_result: ToolResult | None = None
        # Tkinter is not thread-safe: the capture thread never touches Tk
        # objects. The UI loop (main thread) copies the overlay options here,
        # and the worker reports errors and new frames through these fields.
        self._overlay: dict[str, bool] = {}
        self._worker_error: str | None = None
        self._frame_seq = 0
        self._shown_seq = -1
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._after_id: str | None = None

        self._init_vars()
        self._build_ui()
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_vars(self) -> None:
        self.detector_var = tk.StringVar(value="YuNet")
        self.model_path_var = tk.StringVar(value=str(DEFAULT_YUNET_MODEL_PATH))
        self.camera_var = tk.StringVar(value="0")
        self.resolution_var = tk.StringVar(value="640x480")

        # YuNet params
        self.conf_var = tk.DoubleVar(value=0.60)
        self.nms_var = tk.DoubleVar(value=0.30)
        self.topk_var = tk.IntVar(value=5000)

        # Viola–Jones params
        self.scale_factor_var = tk.DoubleVar(value=1.10)
        self.min_neighbors_var = tk.IntVar(value=3)
        self.min_size_var = tk.IntVar(value=24)

        # Overlay checkboxes
        self.show_boxes_var = tk.BooleanVar(value=True)
        self.show_landmarks_var = tk.BooleanVar(value=True)
        self.show_conf_var = tk.BooleanVar(value=True)
        self.show_fps_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Idle — Click 'Start webcam' to begin.")

    def _build_ui(self) -> None:
        container = ttk.Frame(self.dialog, padding=12)
        container.pack(fill="both", expand=True)

        left_col = ttk.Frame(container)
        left_col.pack(side="left", fill="y", padx=(0, 12))

        # A. Configuration
        cfg_frame = ttk.LabelFrame(left_col, text="A. Configuration", padding=10)
        cfg_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(cfg_frame, text="Detector type:").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Combobox(
            cfg_frame,
            textvariable=self.detector_var,
            values=("YuNet", "Viola–Jones"),
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(cfg_frame, text="Model path:").grid(
            row=1, column=0, sticky="w", pady=3
        )
        path_row = ttk.Frame(cfg_frame)
        path_row.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        ttk.Entry(path_row, textvariable=self.model_path_var, width=24).pack(
            side="left"
        )
        ttk.Button(path_row, text="…", width=3, command=self._browse_model).pack(
            side="left", padx=(4, 0)
        )

        ttk.Label(cfg_frame, text="Camera device:").grid(
            row=2, column=0, sticky="w", pady=3
        )
        ttk.Combobox(
            cfg_frame,
            textvariable=self.camera_var,
            values=CAMERA_DEVICES,
            state="readonly",
            width=14,
        ).grid(row=2, column=1, sticky="w", padx=6, pady=3)

        ttk.Label(cfg_frame, text="Webcam resolution:").grid(
            row=3, column=0, sticky="w", pady=3
        )
        ttk.Combobox(
            cfg_frame,
            textvariable=self.resolution_var,
            values=RESOLUTIONS,
            state="readonly",
            width=14,
        ).grid(row=3, column=1, sticky="w", padx=6, pady=3)

        # Dynamic detection parameters
        self.yunet_frame = ttk.LabelFrame(left_col, text="YuNet Parameters", padding=8)
        self.yunet_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(self.yunet_frame, text="Confidence threshold:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Spinbox(
            self.yunet_frame,
            textvariable=self.conf_var,
            from_=0.05,
            to=0.99,
            increment=0.05,
            width=7,
        ).grid(row=0, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(self.yunet_frame, text="NMS threshold:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Spinbox(
            self.yunet_frame,
            textvariable=self.nms_var,
            from_=0.05,
            to=0.99,
            increment=0.05,
            width=7,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(self.yunet_frame, text="Top K candidates:").grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Spinbox(
            self.yunet_frame,
            textvariable=self.topk_var,
            from_=10,
            to=10000,
            increment=100,
            width=7,
        ).grid(row=2, column=1, sticky="w", padx=6, pady=2)

        self.vj_frame = ttk.LabelFrame(
            left_col, text="Viola–Jones Parameters", padding=8
        )
        self.vj_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(self.vj_frame, text="Scale factor:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Spinbox(
            self.vj_frame,
            textvariable=self.scale_factor_var,
            from_=1.01,
            to=2.0,
            increment=0.05,
            width=7,
        ).grid(row=0, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(self.vj_frame, text="Min neighbours:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Spinbox(
            self.vj_frame,
            textvariable=self.min_neighbors_var,
            from_=1,
            to=20,
            increment=1,
            width=7,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(self.vj_frame, text="Min face size (px):").grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Spinbox(
            self.vj_frame,
            textvariable=self.min_size_var,
            from_=12,
            to=256,
            increment=4,
            width=7,
        ).grid(row=2, column=1, sticky="w", padx=6, pady=2)

        # Display overlay options
        overlay_frame = ttk.LabelFrame(left_col, text="Overlay Options", padding=8)
        overlay_frame.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(
            overlay_frame, text="Show bounding boxes", variable=self.show_boxes_var
        ).pack(anchor="w")
        self.chk_landmarks = ttk.Checkbutton(
            overlay_frame,
            text="Show landmarks (YuNet only)",
            variable=self.show_landmarks_var,
        )
        self.chk_landmarks.pack(anchor="w")
        self.chk_conf = ttk.Checkbutton(
            overlay_frame,
            text="Show confidence (YuNet only)",
            variable=self.show_conf_var,
        )
        self.chk_conf.pack(anchor="w")
        ttk.Checkbutton(
            overlay_frame, text="Show FPS", variable=self.show_fps_var
        ).pack(anchor="w")

        # Right column: Live webcam preview & B. Controls
        right_col = ttk.Frame(container)
        right_col.pack(side="left", fill="both", expand=True)

        preview_box = ttk.LabelFrame(
            right_col,
            text="Live Webcam Preview (mirrored in central canvas)",
            padding=8,
        )
        preview_box.pack(fill="both", expand=True)

        self.preview_canvas = tk.Canvas(
            preview_box,
            width=420,
            height=315,
            background="#181b20",
            highlightthickness=0,
        )
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.create_text(
            210,
            157,
            text="Camera image with detection overlays\nClick 'Start webcam' below",
            fill="#c7ccd4",
            justify="center",
        )

        ttk.Label(right_col, textvariable=self.status_var).pack(anchor="w", pady=(6, 4))

        controls_bar = ttk.Frame(right_col)
        controls_bar.pack(fill="x", pady=(4, 0))
        self.btn_start = ttk.Button(
            controls_bar, text="Start webcam", command=self.start_webcam
        )
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop = ttk.Button(
            controls_bar,
            text="Stop webcam",
            command=self.stop_webcam,
            state="disabled",
        )
        self.btn_stop.pack(side="left", padx=(0, 6))
        self.btn_capture = ttk.Button(
            controls_bar, text="Capture frame", command=self.capture_frame
        )
        self.btn_capture.pack(side="left", padx=(0, 6))
        self.btn_save = ttk.Button(
            controls_bar, text="Save current frame", command=self.save_current_frame
        )
        self.btn_save.pack(side="left", padx=(0, 6))
        ttk.Button(controls_bar, text="Close", command=self._on_close).pack(
            side="right"
        )

        self.detector_var.trace_add("write", self._on_detector_type_changed)
        self._on_detector_type_changed()

    def _browse_model(self) -> None:
        is_yn = self.detector_var.get() == "YuNet"
        ftypes = (
            [("ONNX model", "*.onnx"), ("All files", "*.*")]
            if is_yn
            else [("XML cascade", "*.xml"), ("All files", "*.*")]
        )
        selected = filedialog.askopenfilename(
            parent=self.dialog, title="Select face detector model", filetypes=ftypes
        )
        if selected:
            self.model_path_var.set(selected)

    def _on_detector_type_changed(self, *_args: object) -> None:
        is_yn = self.detector_var.get() == "YuNet"
        for child in self.yunet_frame.winfo_children():
            try:
                child.configure(state="normal" if is_yn else "disabled")
            except tk.TclError:
                pass
        for child in self.vj_frame.winfo_children():
            try:
                child.configure(state="disabled" if is_yn else "normal")
            except tk.TclError:
                pass
        self.chk_landmarks.configure(state="normal" if is_yn else "disabled")
        self.chk_conf.configure(state="normal" if is_yn else "disabled")
        if is_yn and not self.model_path_var.get().endswith(".onnx"):
            self.model_path_var.set(str(DEFAULT_YUNET_MODEL_PATH))
        elif not is_yn and self.model_path_var.get().endswith(".onnx"):
            self.model_path_var.set(
                str(DEFAULT_CASCADE_XML) if DEFAULT_CASCADE_XML.exists() else ""
            )

    def _build_detector(self) -> FaceDetector:
        det_type = self.detector_var.get()
        mpath = self.model_path_var.get().strip() or None
        if det_type == "YuNet":
            return create_face_detector(
                "YuNet",
                model_path=mpath,
                confidence_threshold=float(self.conf_var.get()),
                nms_threshold=float(self.nms_var.get()),
                top_k=int(self.topk_var.get()),
            )
        min_s = int(self.min_size_var.get())
        return create_face_detector(
            "Viola–Jones",
            model_path=mpath,
            scale_factor=float(self.scale_factor_var.get()),
            min_neighbors=int(self.min_neighbors_var.get()),
            min_size=(min_s, min_s),
        )

    def start_webcam(self) -> None:
        if self._running:
            return
        try:
            detector = self._build_detector()
        except Exception as exc:
            messagebox.showerror("Detector Load Failed", str(exc), parent=self.dialog)
            return

        cam_idx = int(self.camera_var.get())
        width, height = parse_resolution(self.resolution_var.get())

        self._sync_overlay_options()
        with self._lock:
            self._worker_error = None
        self._stop_event.clear()
        self._running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_var.set(f"Opening camera {cam_idx} ({width}×{height})…")

        self._worker_thread = threading.Thread(
            target=self._capture_loop,
            args=(detector, cam_idx, width, height),
            daemon=True,
        )
        self._worker_thread.start()
        self._schedule_ui_update()

    def _sync_overlay_options(self) -> None:
        """Copy Tk overlay settings into plain values (main thread only)."""
        is_yn = self.detector_var.get() == "YuNet"
        options = {
            "show_boxes": bool(self.show_boxes_var.get()),
            "show_landmarks": bool(self.show_landmarks_var.get()) and is_yn,
            "show_confidence": bool(self.show_conf_var.get()) and is_yn,
            "show_fps": bool(self.show_fps_var.get()),
        }
        with self._lock:
            self._overlay = options

    def _capture_loop(
        self, detector: FaceDetector, cam_idx: int, width: int, height: int
    ) -> None:
        """Worker thread: read, detect and annotate frames. Never touches Tk."""
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            with self._lock:
                self._worker_error = f"Could not open webcam device {cam_idx}."
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        prev_time = time.perf_counter()
        fps_ema = 0.0

        try:
            while not self._stop_event.is_set():
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    time.sleep(0.02)
                    continue

                now = time.perf_counter()
                dt = max(1e-6, now - prev_time)
                prev_time = now
                inst_fps = 1.0 / dt
                fps_ema = (
                    inst_fps if fps_ema == 0.0 else (0.8 * fps_ema + 0.2 * inst_fps)
                )

                with self._lock:
                    options = dict(self._overlay)
                out = process_webcam_frame(frame_bgr, detector, fps=fps_ema, **options)

                with self._lock:
                    self._latest_frame = out["annotated_image"]
                    self._latest_detections = out["detections"]
                    self._latest_fps = fps_ema
                    self._latest_proc_ms = out["processing_time_ms"]
                    self._frame_seq += 1
        except Exception as exc:  # e.g. a detector error on one frame
            with self._lock:
                self._worker_error = f"Webcam detection stopped: {exc}"
        finally:
            cap.release()
            self._running = False

    def _schedule_ui_update(self) -> None:
        if not self.dialog.winfo_exists():
            return

        self._sync_overlay_options()
        with self._lock:
            frame = self._latest_frame
            detections = list(self._latest_detections)
            fps = self._latest_fps
            seq = self._frame_seq
            error = self._worker_error
            self._worker_error = None

        if error is not None:
            self.status_var.set(error)
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")

        # Redraw only when the worker has produced a new frame.
        if frame is not None and seq != self._shown_seq:
            self._shown_seq = seq
            # Render in central application canvas
            if self.central_view is not None:
                self.central_view.show(frame)

            # Render in dialog preview canvas
            cw = max(1, self.preview_canvas.winfo_width())
            ch = max(1, self.preview_canvas.winfo_height())
            preview = frame.copy()
            preview.thumbnail((cw, ch), Image.Resampling.BILINEAR)
            self._preview_photo = ImageTk.PhotoImage(preview)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                cw // 2, ch // 2, image=self._preview_photo, anchor="center"
            )
            self.status_var.set(
                f"Detection running | FPS: {fps:.1f} | Faces: {len(detections)}"
            )

        worker = self._worker_thread
        worker_alive = worker is not None and worker.is_alive()
        if self._running or worker_alive:
            self._after_id = self.dialog.after(33, self._schedule_ui_update)

    def stop_webcam(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._after_id is not None:
            try:
                self.dialog.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        if self.dialog.winfo_exists():
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.status_var.set("Webcam stopped.")

    def _build_result_from_latest(self) -> ToolResult | None:
        with self._lock:
            frame = self._latest_frame.copy() if self._latest_frame else None
            detections = list(self._latest_detections)
            fps = self._latest_fps
            proc_ms = self._latest_proc_ms

        if frame is None:
            return None

        det_name = self.detector_var.get()
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
            "Source": f"Webcam {self.camera_var.get()} ({self.resolution_var.get()})",
            "Faces detected": len(detections),
            "Confidence": first_conf,
            "FPS": f"{fps:.1f}",
            "Processing time": f"{proc_ms:.1f} ms",
            "Bounding box": first_bbox,
        }
        if detections and detections[0].get("landmarks"):
            for name, (px, py) in zip(LANDMARK_NAMES, detections[0]["landmarks"]):
                details[name] = f"({px}, {py})"

        return ToolResult(
            image=frame,
            message=(
                f"Captured webcam frame ({len(detections)} face(s) detected, "
                f"{fps:.1f} FPS)."
            ),
            details=details,
        )

    def capture_frame(self) -> None:
        res = self._build_result_from_latest()
        if res is None:
            messagebox.showinfo(
                "No Frame Available",
                "Start the webcam and wait for a frame before capturing.",
                parent=self.dialog,
            )
            return
        self._captured_result = res
        self.stop_webcam()
        self.dialog.destroy()

    def save_current_frame(self) -> None:
        res = self._build_result_from_latest()
        if res is None or res.image is None:
            messagebox.showinfo(
                "No Frame Available",
                "Start the webcam and wait for a frame before saving.",
                parent=self.dialog,
            )
            return
        filename = filedialog.asksaveasfilename(
            parent=self.dialog,
            title="Save captured webcam frame",
            defaultextension=".png",
            initialfile="webcam_face_detection.png",
            filetypes=SAVE_TYPES,
        )
        if not filename:
            return
        try:
            res.image.save(filename)
            self._captured_result = res
            self.status_var.set(f"Saved frame to {Path(filename).name}")
        except OSError as exc:
            messagebox.showerror("Save Failed", str(exc), parent=self.dialog)

    def _on_close(self) -> None:
        if self._captured_result is None:
            self._captured_result = self._build_result_from_latest()
        self.stop_webcam()
        self.dialog.destroy()

    def show_and_wait(self) -> ToolResult | None:
        self.parent.wait_window(self.dialog)
        return self._captured_result


class WebcamFaceDetectionTool(ForensicsTool):
    """Tool 3 — Live webcam face detection reusing the shared detector backend."""

    tool_id = "webcam_face_detection"
    title = "Webcam Face Detection"
    category = "Face Analysis"
    description = (
        "Run the selected detector on a live webcam feed with configurable parameters."
    )
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        *,
        test_frame: Image.Image | np.ndarray | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> ToolResult | None:
        # Support headless single-frame execution for automated testing
        if test_frame is not None:
            params = parameters or {}
            det_type = params.get("detector_type", "YuNet")
            model_path = params.get("model_path")
            detector = create_face_detector(
                det_type,
                model_path=model_path,
                confidence_threshold=params.get("confidence_threshold", 0.60),
                nms_threshold=params.get("nms_threshold", 0.30),
                top_k=params.get("top_k", 5000),
                scale_factor=params.get("scale_factor", 1.10),
                min_neighbors=params.get("min_neighbors", 3),
            )
            is_yn = det_type.strip().lower().startswith("yunet")
            out = process_webcam_frame(
                test_frame,
                detector,
                show_boxes=params.get("show_boxes", True),
                show_landmarks=params.get("show_landmarks", True) and is_yn,
                show_confidence=params.get("show_confidence", True) and is_yn,
                show_fps=params.get("show_fps", True),
            )
            detections = out["detections"]
            return ToolResult(
                image=out["annotated_image"],
                message=f"Processed webcam frame ({len(detections)} face(s) detected).",
                details={
                    "Detector": "YuNet" if is_yn else "Viola–Jones",
                    "Faces detected": len(detections),
                    "FPS": f"{out['fps']:.1f}",
                    "Processing time": f"{out['processing_time_ms']:.1f} ms",
                },
            )

        controller = WebcamDetectionController(parent, document)
        return controller.show_and_wait()
