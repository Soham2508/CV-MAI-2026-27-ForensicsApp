"""Unit tests for the Set 9.3 Blood Cell Detection tool (fake models, no torch)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import blood_cell_detection as bc
from forensics_app.tools import build_tool_registry
from forensics_app.tools.yolo_detection import Detection

NAMES = ["PC", "RBC", "WBC"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBoxes:
    def __init__(self, rows) -> None:
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        self.cls, self.conf = FakeTensor(rows[:, 0]), FakeTensor(rows[:, 1])
        self.xyxy = FakeTensor(rows[:, 2:])

    def __len__(self) -> int:
        return len(self.cls.values)


class FakeResult:
    def __init__(self, rows) -> None:
        self.boxes = FakeBoxes(rows)
        self.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 1.0}


class FakeYolo:
    """Returns fixed (class, confidence, x1, y1, x2, y2) rows."""

    names = dict(enumerate(NAMES))

    def __init__(self, rows) -> None:
        self.rows = rows
        self.calls: list[dict] = []

    def predict(self, source, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.rows)]


def det(cls: int, conf: float, box) -> Detection:
    return Detection(cls, NAMES[cls], conf, tuple(float(v) for v in box))


def gt(cls: int, box) -> bc.GroundTruthBox:
    return bc.GroundTruthBox(cls, NAMES[cls], tuple(float(v) for v in box))


# Ground truth on a 100x100 image: one PC, two RBC, one WBC.
GT_LINES = (
    "0 0.10 0.10 0.10 0.10\n"  # PC  (5, 5)-(15, 15)
    "1 0.40 0.40 0.20 0.20\n"  # RBC (30, 30)-(50, 50)
    "1 0.70 0.40 0.20 0.20\n"  # RBC (60, 30)-(80, 50)
    "2 0.50 0.80 0.30 0.20\n"  # WBC (35, 70)-(65, 90)
)
# Predictions: PC and first RBC correct, second RBC missed, WBC predicted as RBC,
# plus one extra RBC (false positive) at the border.
PRED_ROWS = [
    (0, 0.90, 5, 5, 15, 15),
    (1, 0.95, 31, 31, 51, 51),
    (1, 0.60, 35, 70, 65, 90),
    (1, 0.40, 0, 85, 12, 100),
]


class BloodCellTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_model(self, name: str, *, weights: bool = True, **extra) -> Path:
        folder = self.root / "models" / name
        folder.mkdir(parents=True)
        passport = {
            "model_name": name,
            "display_name": f"Model {name}",
            "description": "test model",
            "task": "object detection",
            "class_names": NAMES,
            "class_colors": COLORS,
            "inference": {"image_size": 640, "confidence": 0.25, "nms_iou": 0.7},
            "architecture": {"type": "Ultralytics YOLO11n", "parameters": 2590425},
            "training": {"epochs_run": 3, "minutes": 1.5, "device": "cpu"},
            "evaluation": {
                "ultralytics": {
                    "mAP50": 0.9,
                    "mAP50-95": 0.8,
                    "per_class": {"PC": {"AP50": 0.8, "AP50-95": 0.7}},
                }
            },
            "files": {"weights": "best.pt", "training_curves": "training_curves.png"},
            **extra,
        }
        (folder / "passport.json").write_text(json.dumps(passport))
        if weights:
            (folder / "best.pt").write_bytes(b"x")
        Image.new("RGB", (40, 20), (10, 20, 30)).save(folder / "training_curves.png")
        return folder

    def make_sample(self) -> Path:
        images = self.root / "samples" / "images"
        labels = self.root / "samples" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        path = images / "cell.png"
        Image.new("RGB", (100, 100), (230, 200, 210)).save(path)
        (labels / "cell.txt").write_text(GT_LINES)
        return path


class ModelDiscoveryTests(BloodCellTestCase):
    def test_lists_detection_models_with_default_first(self) -> None:
        self.make_model("zeta")
        self.make_model(bc.DEFAULT_MODEL)
        self.make_model("classifier", task="classification")
        (self.root / "models" / "pretrained").mkdir()

        models = bc.list_detection_models(self.root / "models")

        self.assertEqual([m.name for m in models], [bc.DEFAULT_MODEL, "zeta"])
        self.assertEqual(models[0].class_names, NAMES)
        self.assertEqual(models[0].class_colors[0], (42, 120, 214))
        self.assertTrue(models[0].has_weights)

    def test_missing_colours_fall_back_and_summary_mentions_missing_weights(
        self,
    ) -> None:
        self.make_model("m", weights=False, class_colors=["#000000"])
        info = bc.list_detection_models(self.root / "models")[0]

        self.assertEqual(len(info.class_colors), 3)
        self.assertEqual(info.class_colors[0], (0, 0, 0))
        summary = bc.passport_summary(info)
        self.assertIn("mAP50 0.900", summary)
        self.assertIn("Weights missing", summary)

    def test_missing_models_folder_gives_empty_list(self) -> None:
        self.assertEqual(bc.list_detection_models(self.root / "nothing"), [])


class GroundTruthTests(BloodCellTestCase):
    def test_label_file_found_in_sibling_labels_folder(self) -> None:
        image = self.make_sample()
        self.assertEqual(
            bc.find_label_file(image), self.root / "samples" / "labels" / "cell.txt"
        )

    def test_label_file_next_to_image_and_missing_cases(self) -> None:
        image = self.root / "loose.png"
        image.write_bytes(b"")
        self.assertIsNone(bc.find_label_file(image))
        self.assertIsNone(bc.find_label_file(None))
        image.with_suffix(".txt").write_text("")
        self.assertEqual(bc.find_label_file(image), image.with_suffix(".txt"))

    def test_read_labels_converts_to_pixels(self) -> None:
        label = self.make_sample().parent.parent / "labels" / "cell.txt"
        boxes = bc.read_yolo_labels(label, 100, 100, NAMES)

        self.assertEqual([b.name for b in boxes], ["PC", "RBC", "RBC", "WBC"])
        np.testing.assert_allclose(boxes[3].box, (35, 70, 65, 90))

    def test_read_labels_rejects_malformed_lines(self) -> None:
        bad = self.root / "bad.txt"
        bad.write_text("1 0.5 0.5\n")
        with self.assertRaises(ValueError):
            bc.read_yolo_labels(bad, 10, 10, NAMES)


class MatchingTests(unittest.TestCase):
    def test_box_iou(self) -> None:
        iou = bc.box_iou(
            np.array([[0, 0, 10, 10]]),
            np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]),
        )
        np.testing.assert_allclose(iou, [[1.0, 1 / 3, 0.0]])
        self.assertEqual(bc.box_iou(np.zeros((0, 4)), np.zeros((2, 4))).shape, (0, 2))

    def test_matching_is_per_class_and_greedy_by_confidence(self) -> None:
        truth = [gt(1, (0, 0, 10, 10)), gt(2, (20, 20, 30, 30))]
        preds = [
            det(1, 0.5, (0, 0, 10, 10)),  # duplicate, lower confidence
            det(1, 0.9, (1, 1, 10, 10)),  # takes the RBC first
            det(1, 0.8, (20, 20, 30, 30)),  # right place, wrong class
        ]
        match = bc.match_detections(preds, truth, 0.5)

        self.assertEqual([(i, j) for i, j, _ in match.matches], [(1, 0)])
        self.assertEqual(match.false_positives, [0, 2])
        self.assertEqual(match.missed, [1])

    def test_iou_threshold_controls_matches(self) -> None:
        truth = [gt(0, (0, 0, 10, 10))]
        preds = [det(0, 0.9, (5, 0, 15, 10))]  # IoU 1/3
        self.assertEqual(len(bc.match_detections(preds, truth, 0.3).matches), 1)
        self.assertEqual(len(bc.match_detections(preds, truth, 0.5).matches), 0)

    def test_no_ground_truth_or_no_predictions(self) -> None:
        self.assertEqual(
            bc.match_detections([det(0, 0.9, (0, 0, 1, 1))], []).false_positives, [0]
        )
        self.assertEqual(bc.match_detections([], [gt(0, (0, 0, 1, 1))]).missed, [0])

    def test_summary_counts_precision_and_recall(self) -> None:
        truth = [gt(0, (0, 0, 10, 10)), gt(1, (20, 20, 40, 40))]
        preds = [det(0, 0.9, (0, 0, 10, 10)), det(1, 0.7, (60, 60, 80, 80))]
        match = bc.match_detections(preds, truth)
        summary = bc.match_summary(preds, truth, match, NAMES)

        self.assertEqual(summary["all"]["tp"], 1)
        self.assertEqual(summary["all"]["fp"], 1)
        self.assertEqual(summary["all"]["missed"], 1)
        self.assertAlmostEqual(summary["all"]["precision"], 0.5)
        self.assertAlmostEqual(summary["all"]["recall"], 0.5)
        self.assertEqual(summary["RBC"]["missed"], 1)
        self.assertTrue(np.isnan(summary["WBC"]["precision"]))


class BorderTests(unittest.TestCase):
    def test_touches_border(self) -> None:
        self.assertTrue(bc.touches_border((0, 10, 20, 30), (100, 100), 3))
        self.assertTrue(bc.touches_border((10, 10, 98, 30), (100, 100), 3))
        self.assertTrue(bc.touches_border((10, 2, 20, 30), (100, 100), 3))
        self.assertFalse(bc.touches_border((10, 10, 90, 90), (100, 100), 3))

    def test_unmatched_border_boxes_are_ignored(self) -> None:
        truth = [
            gt(1, (0, 40, 10, 60)),  # border, matched
            gt(1, (90, 40, 100, 60)),  # border, missed -> ignored
            gt(1, (40, 40, 60, 60)),  # inside, missed -> counted
        ]
        preds = [
            det(1, 0.9, (0, 40, 10, 60)),
            det(1, 0.8, (0, 0, 8, 8)),  # border FP -> ignored
            det(1, 0.7, (20, 70, 30, 80)),  # inside FP -> counted
        ]
        plain = bc.match_detections(preds, truth)
        border = bc.match_detections(
            preds, truth, image_size=(100, 100), border_margin=3
        )

        self.assertEqual(plain.false_positives, [1, 2])
        self.assertEqual(plain.missed, [1, 2])
        self.assertEqual(len(border.matches), 1)
        self.assertEqual(border.false_positives, [2])
        self.assertEqual(border.missed, [2])
        self.assertEqual(border.ignored_detections, [1])
        self.assertEqual(border.ignored_ground_truth, [1])

        row = bc.match_summary(preds, truth, border, NAMES)["RBC"]
        self.assertEqual(
            (row["tp"], row["fp"], row["missed"], row["ignored"]), (1, 1, 1, 2)
        )
        self.assertEqual(row["ground_truth"], 2)
        self.assertEqual(row["predicted"], 2)
        self.assertAlmostEqual(row["precision"], 0.5)

    def test_margin_needs_an_image_size(self) -> None:
        preds = [det(1, 0.8, (0, 0, 8, 8))]
        match = bc.match_detections(preds, [], border_margin=3)
        self.assertEqual(match.false_positives, [0])

    def test_counts_row_f1(self) -> None:
        row = bc.counts_row(8, 2, 2)
        self.assertAlmostEqual(row["f1"], 0.8)
        self.assertEqual(bc.counts_row(0, 3, 0)["f1"], 0.0)
        self.assertTrue(np.isnan(bc.counts_row(0, 0, 0)["f1"]))


class DrawingTests(unittest.TestCase):
    COLORS = [bc.hex_to_rgb(c) for c in COLORS]

    def test_predictions_solid_ground_truth_dashed(self) -> None:
        image = Image.new("RGB", (200, 200), (128, 128, 128))
        truth = [gt(1, (100, 100, 180, 180))]
        preds = [det(1, 0.9, (20, 20, 80, 80))]

        out = bc.draw_ground_truth_and_predictions(
            image, preds, truth, None, self.COLORS, mark_errors=False
        )

        self.assertEqual(out.getpixel((20, 50)), (235, 104, 52))
        edge = [out.getpixel((x, 100)) for x in range(100, 181)]
        self.assertIn((255, 255, 255), edge)
        self.assertIn((31, 31, 30), edge)
        self.assertEqual(out.getpixel((140, 140)), (128, 128, 128))
        self.assertEqual(image.getpixel((20, 50)), (128, 128, 128))

    def test_toggles_hide_layers(self) -> None:
        image = Image.new("RGB", (100, 100), (128, 128, 128))
        truth = [gt(1, (10, 10, 90, 90))]
        preds = [det(1, 0.9, (10, 10, 90, 90))]
        hidden = bc.draw_ground_truth_and_predictions(
            image, preds, truth, None, self.COLORS,
            show_ground_truth=False, show_predictions=False,
        )  # fmt: skip
        self.assertEqual(hidden.getcolors(), [(10000, (128, 128, 128))])

    def test_errors_are_tagged(self) -> None:
        image = Image.new("RGB", (400, 400), (230, 230, 230))
        truth = [gt(2, (200, 200, 380, 380))]
        preds = [det(1, 0.8, (20, 20, 120, 120))]
        match = bc.match_detections(preds, truth)

        marked = np.asarray(
            bc.draw_ground_truth_and_predictions(
                image, preds, truth, match, self.COLORS
            )
        )
        plain = np.asarray(
            bc.draw_ground_truth_and_predictions(
                image, preds, truth, match, self.COLORS, mark_errors=False
            )
        )
        tag = np.all(marked == bc.TAG_BACKGROUND, axis=-1).sum()
        self.assertGreater(tag, 100)
        self.assertEqual(np.all(plain == bc.TAG_BACKGROUND, axis=-1).sum() < tag, True)

    def test_side_by_side_adds_title_strip(self) -> None:
        a = Image.new("RGB", (50, 40), (255, 0, 0))
        b = Image.new("RGB", (50, 40), (0, 0, 255))
        out = bc.side_by_side([("A", a), ("B", b)])
        self.assertGreater(out.width, 100)
        self.assertGreater(out.height, 40)
        self.assertEqual(out.getpixel((10, out.height - 5)), (255, 0, 0))
        self.assertEqual(out.getpixel((out.width - 5, out.height - 5)), (0, 0, 255))

    def test_image_grid_layout(self) -> None:
        panels = [
            (str(k), Image.new("RGB", (40, 30), (k * 60, 0, 0))) for k in range(3)
        ]
        out = bc.image_grid(panels, columns=2)
        self.assertGreater(out.width, 80)
        self.assertGreater(out.height, 60)
        self.assertEqual(out.getpixel((5, out.height - 5)), (120, 0, 0))


class ToolTests(BloodCellTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.make_model(bc.DEFAULT_MODEL)
        self.make_model("blood_cell_yolo11n_scratch")
        self.sample = self.make_sample()
        self.fine = FakeYolo(PRED_ROWS)
        self.scratch = FakeYolo(PRED_ROWS[:2])

    def run_tool(self, **parameters):
        params = {
            "models_root": str(self.root / "models"),
            "image_path": str(self.sample),
            "device": "cpu",
            "loaded_models": {
                bc.DEFAULT_MODEL: self.fine,
                "blood_cell_yolo11n_scratch": self.scratch,
            },
            **parameters,
        }
        return bc.BloodCellDetectionTool().run(None, ImageDocument(), parameters=params)

    def test_registered_under_deep_learning(self) -> None:
        tools = [
            t
            for t in build_tool_registry().all()
            if t.tool_id == "blood_cell_detection"
        ]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].category, "Deep Learning")
        self.assertFalse(tools[0].requires_image)

    def test_detect_compares_with_ground_truth(self) -> None:
        result = self.run_tool(confidence=0.3, iou=0.6)

        self.assertIn("2/4 cells found, 2 false positive(s), 2 missed", result.message)
        self.assertEqual(result.image.size, (100, 100))
        self.assertEqual(self.fine.calls[0]["conf"], 0.3)
        self.assertEqual(self.fine.calls[0]["iou"], 0.6)
        d = result.details
        self.assertEqual(d["Model"], f"Model {bc.DEFAULT_MODEL}")
        self.assertEqual(d["Ground truth"], str(self.root / "samples/labels/cell.txt"))
        self.assertEqual(d["Predicted"], 4)
        self.assertEqual(d["  RBC"], "GT 2 · pred 3 · TP 1 · FP 2 · missed 1")
        self.assertEqual(d["  WBC"], "GT 1 · pred 0 · TP 0 · FP 0 · missed 1")
        self.assertEqual(d["Precision"], "0.500")
        self.assertEqual(d["Recall"], "0.500")
        self.assertIn("Mean IoU (matched)", d)

    def test_border_option_moves_edge_errors_out_of_the_counts(self) -> None:
        result = self.run_tool(border_margin=3)

        self.assertIn("2/4 cells found, 1 false positive(s), 2 missed", result.message)
        self.assertEqual(
            result.details["  RBC"],
            "GT 2 · pred 2 · TP 1 · FP 1 · missed 1 · border 1",
        )
        self.assertEqual(result.details["Border cells"], "not counted (3 px)")

    def test_without_labels_only_counts_predictions(self) -> None:
        loose = self.root / "loose.png"
        Image.new("RGB", (100, 100)).save(loose)
        result = self.run_tool(image_path=str(loose))

        self.assertIn("4 cells detected (no labels)", result.message)
        self.assertEqual(result.details["Ground truth"], "not found")
        self.assertEqual(result.details["  RBC"], 3)
        self.assertNotIn("Precision", result.details)

    def test_uses_current_document_and_its_labels(self) -> None:
        document = ImageDocument()
        document.apply(Image.open(self.sample).convert("RGB"))
        document.path = self.sample
        result = bc.BloodCellDetectionTool().run(
            None,
            document,
            parameters={
                "models_root": str(self.root / "models"),
                "device": "cpu",
                "loaded_models": {bc.DEFAULT_MODEL: self.fine},
            },
        )
        self.assertIn("2/4 cells found", result.message)

    def test_compare_runs_every_model_side_by_side(self) -> None:
        result = self.run_tool(action="compare")

        self.assertGreater(result.image.width, 200)
        self.assertIn(" | ", result.message)
        self.assertEqual(result.details["[1] Predicted"], 4)
        self.assertEqual(result.details["[2] Predicted"], 2)
        self.assertEqual(result.details["[2] Recall"], "0.500")

    def test_training_curves_action(self) -> None:
        result = self.run_tool(action="curves")
        self.assertEqual(result.image.size, (40, 20))
        self.assertIn("training curves", result.message)
        self.assertEqual(result.details["Classes"], "PC, RBC, WBC")

    def test_missing_weights_are_skipped_or_reported(self) -> None:
        (self.root / "models" / bc.DEFAULT_MODEL / "best.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_tool(loaded_models={})


class FolderTests(BloodCellTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.make_model(bc.DEFAULT_MODEL)
        self.make_model("blood_cell_yolo11n_scratch")
        self.folder = self.make_sample().parent  # samples/images with cell.png
        Image.new("RGB", (100, 100), (220, 200, 210)).save(self.folder / "cell2.png")
        (self.folder.parent / "labels" / "cell2.txt").write_text(
            "1 0.40 0.40 0.20 0.20\n"
        )
        Image.new("RGB", (100, 100)).save(self.folder / "unlabelled.png")
        (self.folder / "notes.txt").write_text("not an image")
        self.fine = FakeYolo(PRED_ROWS)
        self.scratch = FakeYolo(PRED_ROWS[:2])
        self.output = self.root / "out"

    def info(self, name: str = bc.DEFAULT_MODEL) -> bc.YoloModelInfo:
        models = bc.list_detection_models(self.root / "models")
        return next(m for m in models if m.name == name)

    def test_list_labelled_images_skips_unlabelled(self) -> None:
        pairs, skipped = bc.list_labelled_images(self.folder)
        self.assertEqual([p.name for p, _ in pairs], ["cell.png", "cell2.png"])
        self.assertEqual(skipped, 1)

    def test_aggregate_summaries(self) -> None:
        a = {"PC": bc.counts_row(1, 0, 0), "all": bc.counts_row(1, 0, 0)}
        b = {"PC": bc.counts_row(2, 1, 1, 3), "all": bc.counts_row(2, 1, 1, 3)}
        totals = bc.aggregate_summaries([a, b], ["PC"])
        self.assertEqual(totals["PC"]["tp"], 3)
        self.assertEqual(totals["PC"]["ignored"], 3)
        self.assertAlmostEqual(totals["all"]["precision"], 0.75)
        self.assertAlmostEqual(totals["all"]["recall"], 0.75)

    def test_evaluate_folder_totals_and_worst_images(self) -> None:
        evaluation = bc.evaluate_folder(
            self.info(), self.folder, {"device": "cpu"}, self.fine
        )

        self.assertEqual(len(evaluation.per_image), 2)
        self.assertEqual(evaluation.skipped, 1)
        total = evaluation.totals["all"]
        # cell.png: TP 2, FP 2, missed 2; cell2.png (one RBC): TP 1, FP 3, missed 0
        self.assertEqual((total["tp"], total["fp"], total["missed"]), (3, 5, 2))
        self.assertEqual(
            [p.name for p in evaluation.worst()], ["cell.png", "cell2.png"]
        )
        self.assertEqual(evaluation.errors(self.folder / "cell2.png"), 3)
        self.assertGreater(evaluation.mean_iou, 0.8)

    def test_csv_has_per_image_and_total_rows(self) -> None:
        evaluation = bc.evaluate_folder(
            self.info(), self.folder, {"device": "cpu"}, self.fine
        )
        path = bc.save_evaluation_csv(evaluation, self.output, border_margin=3)

        self.assertEqual(path.name, f"{bc.DEFAULT_MODEL}_samples_border.csv")
        lines = path.read_text().splitlines()
        self.assertEqual(lines[0].split(",")[:3], ["image", "class", "ground_truth"])
        self.assertEqual(len(lines), 1 + 2 * 4 + 4)
        self.assertEqual(lines[-1], "TOTAL,all,5,8,3,5,2,0,0.3750,0.6000")
        self.assertIn("cell2.png,WBC,0,0,0,0,0,0,,", lines)

    def test_folder_label(self) -> None:
        self.assertEqual(bc.folder_label(Path("/data/yolo/test/images")), "test")
        self.assertEqual(bc.folder_label(Path("/data/mine")), "mine")

    def test_tool_evaluates_a_folder(self) -> None:
        result = bc.BloodCellDetectionTool().run(
            None,
            ImageDocument(),
            parameters={
                "models_root": str(self.root / "models"),
                "folder": str(self.folder),
                "output_dir": str(self.output),
                "device": "cpu",
                "loaded_models": {bc.DEFAULT_MODEL: self.fine},
            },
        )

        self.assertIn("3/5 cells found in 2 images", result.message)
        d = result.details
        self.assertEqual(d["Images evaluated"], 2)
        self.assertEqual(d["Skipped (no labels)"], 1)
        self.assertTrue(d["  all"].startswith("TP 3/5 · FP 5 · missed 2 · P 0.375"))
        self.assertEqual(d["Worst #1"][:8], "cell.png")
        self.assertTrue((self.output / f"{bc.DEFAULT_MODEL}_samples.csv").is_file())
        self.assertGreater(result.image.width, 200)  # two panels side by side

    def test_tool_compares_models_on_a_folder(self) -> None:
        result = bc.BloodCellDetectionTool().run(
            None,
            ImageDocument(),
            parameters={
                "action": "compare",
                "models_root": str(self.root / "models"),
                "folder": str(self.folder),
                "output_dir": str(self.output),
                "device": "cpu",
                "loaded_models": {
                    bc.DEFAULT_MODEL: self.fine,
                    "blood_cell_yolo11n_scratch": self.scratch,
                },
            },
        )

        self.assertIn(" | ", result.message)
        self.assertEqual(result.details["[1] Images evaluated"], 2)
        self.assertIn("[2] CSV", result.details)
        self.assertEqual(len(list(self.output.glob("*.csv"))), 2)

    def test_folder_without_labels_is_reported(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()
        Image.new("RGB", (10, 10)).save(empty / "x.png")
        with self.assertRaises(FileNotFoundError):
            bc.evaluate_folder(self.info(), empty, {"device": "cpu"}, self.fine)


if __name__ == "__main__":
    unittest.main()
