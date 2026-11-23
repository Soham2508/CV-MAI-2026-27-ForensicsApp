"""Unit tests for the Set 9.1 YOLO Object Detection tool (fake models, no torch)."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools import yolo_detection as yd

NAMES = {0: "person", 1: "bicycle", 5: "bus"}


class FakeTensor:
    """Mimics a torch tensor's ``.cpu().numpy()`` chain."""

    def __init__(self, values: list) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBoxes:
    def __init__(self, rows: list[tuple[float, float, float, float, float, int]]):
        self.xyxy = FakeTensor([r[:4] for r in rows])
        self.conf = FakeTensor([r[4] for r in rows])
        self.cls = FakeTensor([r[5] for r in rows])
        self._n = len(rows)

    def __len__(self) -> int:
        return self._n


class FakeResult:
    def __init__(self, rows) -> None:
        self.boxes = FakeBoxes(rows)
        self.speed = {"preprocess": 1.0, "inference": 3.0, "postprocess": 2.0}


class FakeYolo:
    """Stand-in for ``ultralytics.YOLO`` that records predict calls."""

    def __init__(self, rows=None, names=None) -> None:
        self.rows = rows if rows is not None else []
        self.names = names if names is not None else dict(NAMES)
        self.calls: list[tuple[np.ndarray, dict]] = []

    def predict(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return [FakeResult(self.rows)]


BUS_ROWS = [
    (2, 2, 18, 12, 0.62, 0),
    (1, 1, 30, 20, 0.94, 5),
    (20, 3, 28, 19, 0.89, 0),
]


def document_with(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.apply(image)
    return document


class YoloDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        yd._MODEL_CACHE.clear()

    def tearDown(self) -> None:
        yd._MODEL_CACHE.clear()
        self._tmp.cleanup()

    # -- weights, device and loading ---------------------------------------

    def test_list_weights_puts_default_first_and_finds_nested_files(self) -> None:
        (self.root / "blood" / "weights").mkdir(parents=True)
        (self.root / "blood" / "weights" / "best.pt").write_bytes(b"x")
        (self.root / "other.pt").write_bytes(b"x")

        weights = yd.list_yolo_weights(self.root)

        self.assertEqual(weights[0], yd.DEFAULT_WEIGHTS)
        self.assertEqual([p.name for p in weights[1:]], ["best.pt", "other.pt"])

    def test_list_weights_without_folder_still_offers_the_default(self) -> None:
        self.assertEqual(
            yd.list_yolo_weights(self.root / "missing"), [yd.DEFAULT_WEIGHTS]
        )

    def test_only_pretrained_ultralytics_names_are_downloadable(self) -> None:
        self.assertTrue(yd.is_downloadable(yd.PRETRAINED_DIR / "yolo11n.pt"))
        self.assertTrue(yd.is_downloadable(yd.PRETRAINED_DIR / "yolov8s.pt"))
        self.assertFalse(yd.is_downloadable(yd.PRETRAINED_DIR / "mine.pt"))
        self.assertFalse(yd.is_downloadable(self.root / "yolo11n.pt"))

    def test_resolve_device(self) -> None:
        self.assertEqual(yd.resolve_device("cpu"), "cpu")
        self.assertEqual(yd.resolve_device("mps"), "mps")
        self.assertEqual(yd.resolve_device("cuda"), "0")
        with patch.object(yd, "_available_accelerator", return_value="mps"):
            self.assertEqual(yd.resolve_device("auto"), "mps")
            self.assertEqual(yd.resolve_device("bogus"), "mps")

    def test_missing_custom_weights_raise_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            yd.load_yolo_model(self.root / "nothing.pt")

    def test_load_is_lazy_and_cached_until_file_changes(self) -> None:
        path = self.root / "w.pt"
        path.write_bytes(b"x")
        loads: list[str] = []

        def fake_yolo(p: str) -> FakeYolo:
            loads.append(p)
            return FakeYolo()

        with patch.object(yd, "_import_yolo", return_value=fake_yolo):
            first = yd.load_yolo_model(path)
            self.assertTrue(yd.is_model_cached(path))
            self.assertIs(yd.load_yolo_model(path), first)
            self.assertEqual(len(loads), 1)
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
            self.assertIsNot(yd.load_yolo_model(path), first)
        self.assertEqual(len(loads), 2)
        self.assertEqual(len(yd._MODEL_CACHE), 1)

    def test_cache_keeps_several_models_and_evicts_the_oldest(self) -> None:
        paths = [self.root / f"m{i}.pt" for i in range(yd.MODEL_CACHE_SIZE + 1)]
        for path in paths:
            path.write_bytes(b"x")
        with patch.object(yd, "_import_yolo", return_value=lambda p: FakeYolo()):
            models = [yd.load_yolo_model(path) for path in paths]
            self.assertFalse(yd.is_model_cached(paths[0]))
            for path, model in zip(paths[1:], models[1:]):
                self.assertIs(yd.load_yolo_model(path), model)

    # -- class names and filter --------------------------------------------

    def test_class_names_accepts_dict_or_list(self) -> None:
        self.assertEqual(yd.class_names(FakeYolo(names=NAMES)), NAMES)
        self.assertEqual(
            yd.class_names(FakeYolo(names=["PC", "RBC", "WBC"])),
            {0: "PC", 1: "RBC", 2: "WBC"},
        )

    def test_parse_class_filter(self) -> None:
        self.assertIsNone(yd.parse_class_filter("", NAMES))
        self.assertIsNone(yd.parse_class_filter(" , ", NAMES))
        self.assertEqual(yd.parse_class_filter("Bus, person, 0", NAMES), [0, 5])
        with self.assertRaises(ValueError) as ctx:
            yd.parse_class_filter("person, dragon", NAMES)
        self.assertIn("dragon", str(ctx.exception))

    # -- detection -----------------------------------------------------------

    def test_detect_sends_bgr_and_settings_and_sorts_by_confidence(self) -> None:
        image = Image.new("RGB", (32, 24), (200, 10, 50))
        model = FakeYolo(BUS_ROWS)

        detections, info = yd.detect_objects(
            model, image, confidence=0.3, iou=0.5, image_size=320, device="mps"
        )

        source, kwargs = model.calls[0]
        self.assertEqual(source.shape, (24, 32, 3))
        self.assertEqual(tuple(source[0, 0]), (50, 10, 200))
        self.assertEqual(kwargs["conf"], 0.3)
        self.assertEqual(kwargs["iou"], 0.5)
        self.assertEqual(kwargs["imgsz"], 320)
        self.assertEqual(kwargs["device"], "mps")
        self.assertFalse(kwargs["verbose"])
        self.assertEqual(
            [(d.name, round(d.confidence, 2)) for d in detections],
            [("bus", 0.94), ("person", 0.89), ("person", 0.62)],
        )
        self.assertEqual(detections[0].box, (1.0, 1.0, 30.0, 20.0))
        self.assertEqual(info["image_size"], (32, 24))
        self.assertEqual(info["speed_ms"]["inference"], 3.0)
        self.assertEqual(yd.count_by_class(detections), [("person", 2), ("bus", 1)])

    def test_detect_converts_grayscale_and_handles_no_boxes(self) -> None:
        model = FakeYolo([])
        detections, _ = yd.detect_objects(model, Image.new("L", (10, 8), 7))
        self.assertEqual(detections, [])
        self.assertEqual(model.calls[0][0].shape, (8, 10, 3))

    # -- drawing -------------------------------------------------------------

    def test_class_colors_match_the_ultralytics_palette(self) -> None:
        self.assertEqual(yd.class_color(0), (4, 42, 255))
        self.assertEqual(yd.class_color(5), (255, 111, 221))
        self.assertEqual(yd.class_color(20), yd.class_color(0))

    def test_draw_detections_outlines_boxes_in_class_colour(self) -> None:
        image = Image.new("L", (200, 150), 128)
        det = yd.Detection(0, "person", 0.9, (50.0, 60.0, 150.0, 140.0))

        plain = yd.draw_detections(image, [det], show_labels=False)
        labelled = yd.draw_detections(image, [det])

        self.assertEqual(plain.mode, "RGB")
        self.assertEqual(plain.size, image.size)
        self.assertEqual(plain.getpixel((50, 100)), (4, 42, 255))
        self.assertEqual(plain.getpixel((100, 100)), (128, 128, 128))
        self.assertEqual(image.getpixel((50, 100)), 128)
        # The label sits above the box, filled with the class colour.
        self.assertEqual(labelled.getpixel((51, 58)), (4, 42, 255))
        self.assertEqual(plain.getpixel((51, 58)), (128, 128, 128))

    def test_label_moves_inside_when_box_touches_the_top(self) -> None:
        image = Image.new("RGB", (120, 80), (0, 0, 0))
        det = yd.Detection(5, "bus", 0.9, (10.0, 0.0, 100.0, 70.0))
        out = np.asarray(yd.draw_detections(image, [det]))
        label_area = out[3:8, 14:40].reshape(-1, 3)
        pink = np.all(label_area == (255, 111, 221), axis=1).mean()
        self.assertGreater(pink, 0.3)

    # -- tool ----------------------------------------------------------------

    def test_tool_is_registered_under_deep_learning(self) -> None:
        tools = [
            t for t in build_tool_registry().all() if t.tool_id == "yolo_detection"
        ]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].category, "Deep Learning")
        self.assertTrue(tools[0].requires_image)

    def test_tool_detects_with_injected_model(self) -> None:
        image = Image.new("RGB", (32, 24), (90, 90, 90))
        model = FakeYolo(BUS_ROWS)

        result = yd.YoloDetectionTool().run(
            None,
            document_with(image),
            parameters={
                "weights": str(self.root / "fake.pt"),
                "model": model,
                "device": "cpu",
                "classes": "person, bus",
                "confidence": 0.4,
            },
        )

        self.assertIn("3 object(s) detected (person ×2, bus ×1)", result.message)
        self.assertEqual(result.image.size, (32, 24))
        self.assertEqual(model.calls[0][1]["classes"], [0, 5])
        self.assertEqual(model.calls[0][1]["conf"], 0.4)
        self.assertEqual(result.details["Detections"], 3)
        self.assertEqual(result.details["  person"], 2)
        self.assertEqual(result.details["Class filter"], "person, bus")
        self.assertEqual(result.details["#1 bus"], "0.94 at (1, 1)–(30, 20)")
        self.assertEqual(result.details["Device"], "cpu")
        self.assertIn("Time (total)", result.details)

    def test_tool_lists_at_most_the_configured_number_of_boxes(self) -> None:
        rows = [(1, 1, 5, 5, 0.5 + i / 1000, 0) for i in range(35)]
        result = yd.YoloDetectionTool().run(
            None,
            document_with(Image.new("RGB", (10, 10))),
            parameters={"model": FakeYolo(rows), "device": "cpu"},
        )
        listed = [k for k in result.details if k.startswith("#")]
        self.assertEqual(len(listed), yd.MAX_LISTED_DETECTIONS)
        self.assertEqual(result.details["…"], "5 more boxes")

    def test_tool_reports_unknown_filter_classes(self) -> None:
        with self.assertRaises(ValueError):
            yd.YoloDetectionTool().run(
                None,
                document_with(Image.new("RGB", (10, 10))),
                parameters={"model": FakeYolo(), "device": "cpu", "classes": "cat"},
            )


if __name__ == "__main__":
    unittest.main()
