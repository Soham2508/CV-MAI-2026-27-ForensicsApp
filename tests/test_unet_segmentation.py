"""Unit tests for the Set 11 U-Net segmentation code and tool (tiny CPU models)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools import unet_common as uc
from forensics_app.tools import unet_segmentation as us
from forensics_app.utils.background import run_with_progress

TINY = {"size": (16, 24), "base": 4, "batch_size": 4, "epochs": 2, "device": "cpu"}


def make_tray_dataset(root: Path, train: int = 6, test: int = 2) -> Path:
    """Tray (class 1) with a red meatball (class 5) and cutlery (class 2)."""
    rng = np.random.default_rng(0)
    for split, count in (("Train", train), ("Test", test)):
        (root / f"X{split}").mkdir(parents=True)
        (root / f"y{split}").mkdir(parents=True)
        for i in range(count):
            img = np.zeros((32, 48, 3), dtype=np.uint8)
            mask = np.zeros((32, 48), dtype=np.uint8)
            img[4:28, 4:44] = (200, 180, 140)
            mask[4:28, 4:44] = 1
            y, x = rng.integers(6, 16), rng.integers(6, 30)
            img[y : y + 8, x : x + 8] = (220, 30, 30)
            mask[y : y + 8, x : x + 8] = 5
            img[24:26, 6:20] = (160, 160, 170)
            mask[24:26, 6:20] = 2
            Image.fromarray(img).save(root / f"X{split}" / f"{i:03d}a.jpg", quality=95)
            Image.fromarray(mask).save(root / f"y{split}" / f"{i:03d}a.png")
    return root


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = make_tray_dataset(Path(self._tmp.name) / "TrayDataset")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_pairs_match_by_name_and_find_nested_folders(self) -> None:
        splits = uc.tray_splits(self.root.parent)
        self.assertEqual(len(splits["train"]), 6)
        self.assertEqual(len(splits["test"]), 2)
        for image, mask in splits["train"]:
            self.assertEqual(image.stem, mask.stem)
        with self.assertRaises(FileNotFoundError):
            uc.find_dir(self.root, "XValid")

    def test_three_channel_masks_are_read_as_indices(self) -> None:
        path = Path(self._tmp.name) / "rgb.png"
        Image.fromarray(np.full((4, 4, 3), 7, dtype=np.uint8)).save(path)
        self.assertEqual(uc.read_mask(path).shape, (4, 4))
        self.assertTrue((uc.read_mask(path) == 7).all())

    def test_task_labels(self) -> None:
        raw = np.array([[0, 1, 4, 5, 42]])
        self.assertEqual(uc.to_task(raw, uc.TASK_BINARY).tolist(), [[0, 0, 0, 1, 1]])
        self.assertEqual(uc.to_task(raw, uc.TASK_MULTICLASS).tolist(), raw.tolist())
        self.assertEqual(uc.num_classes(uc.TASK_MULTICLASS), 43)
        self.assertEqual(uc.class_names(uc.TASK_BINARY), ["not food", "food"])

    def test_load_pairs_resizes_masks_without_new_values(self) -> None:
        pairs = uc.tray_splits(self.root)["train"]
        X, Y = uc.load_pairs(pairs, (16, 24), uc.TASK_MULTICLASS)
        self.assertEqual(X.shape, (6, 16, 24, 3))
        self.assertEqual(Y.shape, (6, 16, 24))
        self.assertTrue(set(np.unique(Y)) <= {0, 1, 2, 5})

    def test_ground_truth_path(self) -> None:
        image = self.root / "XTest" / "000a.jpg"
        self.assertEqual(uc.ground_truth_path(image), self.root / "yTest" / "000a.png")
        self.assertIsNone(uc.ground_truth_path(Path(self._tmp.name) / "a.jpg"))
        self.assertIsNone(uc.ground_truth_path(None))
        samples = uc.SAMPLES_DIR / "images" / "1005a.jpg"
        self.assertEqual(
            uc.ground_truth_path(samples), uc.SAMPLES_DIR / "masks" / "1005a.png"
        )


class MetricTests(unittest.TestCase):
    def test_notebook_dice_example(self) -> None:
        P = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        G = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(uc.dice_score(P, G), 0.5)
        self.assertAlmostEqual(uc.iou_score(P, G), 1 / 3)
        self.assertEqual(uc.dice_score(np.zeros(3), np.zeros(3)), 1.0)
        torch_dice = uc.dice_coef(
            torch.tensor(P, dtype=torch.float)[None, None],
            torch.tensor(G, dtype=torch.float)[None, None],
        )
        self.assertAlmostEqual(float(torch_dice[0]), 0.5, places=5)

    def test_per_class_dice(self) -> None:
        counts = uc.ClassCounts(4)
        counts.add(np.array([0, 0, 1, 1, 3]), np.array([0, 1, 1, 1, 2]))
        dice = counts.per_class_dice()
        self.assertEqual(set(dice), {0, 1, 2})  # class 3 is not in the ground truth
        self.assertAlmostEqual(dice[0], 2 / 3)
        self.assertAlmostEqual(dice[1], 0.8)
        self.assertEqual(dice[2], 0.0)
        self.assertAlmostEqual(counts.pixel_accuracy(), 0.6)

    def test_losses_are_finite(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 8, 8)
        target = (torch.rand(2, 1, 8, 8) > 0.5).float()
        self.assertTrue(torch.isfinite(uc.binary_loss(logits, target)))
        logits = torch.randn(2, 43, 8, 8)
        target = torch.randint(0, 43, (2, 8, 8))
        self.assertGreater(float(uc.multiclass_loss(logits, target)), 0)

    def test_score_predictions(self) -> None:
        Y = np.array([[[0, 5], [1, 5]]])
        multi = uc.score_predictions(Y.copy(), Y, uc.TASK_MULTICLASS)
        self.assertEqual(multi["mean_dice"], 1.0)
        self.assertEqual(multi["food_dice"], 1.0)
        binary = uc.score_predictions(
            np.array([[[0, 1], [0, 0]]]), np.array([[[0, 1], [0, 1]]]), uc.TASK_BINARY
        )
        self.assertAlmostEqual(binary["dice"], 2 / 3)


class ModelTests(unittest.TestCase):
    def test_notebook_unet_size_and_shapes(self) -> None:
        model = uc.build_unet()
        self.assertEqual(uc.count_parameters(model), 482_737)
        self.assertEqual(
            tuple(model(torch.zeros(1, 3, 128, 208)).shape), (1, 1, 128, 208)
        )
        multi = uc.build_unet(out_ch=43, base=4)
        self.assertEqual(tuple(multi(torch.zeros(1, 3, 16, 24)).shape), (1, 43, 16, 24))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = make_tray_dataset(Path(cls._tmp.name) / "TrayDataset")
        cls.messages: list[str] = []
        cls.runs = {
            task: uc.train_unet(cls.root, task, progress=cls.messages.append, **TINY)
            for task in uc.TASKS
        }

    @classmethod
    def tearDownClass(cls) -> None:
        uc._MODEL_CACHE.clear()
        cls._tmp.cleanup()

    def test_passport_and_history(self) -> None:
        for task, run in self.runs.items():
            p = run.passport
            self.assertEqual(p["task"], task)
            self.assertEqual(p["input_size"], [16, 24])
            self.assertEqual(len(run.history["train_loss"]), 2)
            self.assertEqual(p["training"]["test_images"], 2)
            self.assertIn(p["training"]["best_epoch"], (1, 2))
            self.assertIn("test", p["metrics"])
        self.assertTrue(any(m.startswith("Epoch 2/2") for m in self.messages))

    def test_save_load_and_segment_round_trip(self) -> None:
        run = self.runs[uc.TASK_MULTICLASS]
        folder = uc.save_model(run.model, run.passport, Path(self._tmp.name) / "m")
        self.assertEqual(uc.list_models(Path(self._tmp.name)), [folder])
        model, passport = uc.load_model(folder)
        image = Image.open(self.root / "XTest" / "000a.jpg")
        mask = uc.segment(model, passport, image)
        self.assertEqual(mask.shape, (32, 48))
        np.testing.assert_array_equal(mask, uc.segment(run.model, run.passport, image))

    def test_rejects_sizes_not_divisible_by_8(self) -> None:
        with self.assertRaises(ValueError):
            uc.train_unet(self.root, uc.TASK_BINARY, **{**TINY, "size": (20, 24)})


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = make_tray_dataset(Path(cls._tmp.name) / "TrayDataset")
        cls.models = Path(cls._tmp.name) / "models"
        cls.tool = us.UNetSegmentationTool()
        cls.trained = {}
        for task in uc.TASKS:
            result = cls.tool.run(
                None,
                ImageDocument(),
                parameters={
                    "task_choice": us.TASK_TRAIN,
                    "task": task,
                    "dataset": str(cls.root),
                    "save_to": str(cls.models / task),
                    **TINY,
                },
            )
            cls.trained[task] = result

    @classmethod
    def tearDownClass(cls) -> None:
        uc._MODEL_CACHE.clear()
        cls._tmp.cleanup()

    def segment(self, document: ImageDocument, task: str, **extra):
        return self.tool.run(
            None,
            document,
            parameters={
                "task_choice": us.TASK_SEGMENT,
                "model_dir": str(self.models / task),
                "device": "cpu",
                **extra,
            },
        )

    def test_tool_is_registered(self) -> None:
        ids = [t.tool_id for t in build_tool_registry().all()]
        self.assertIn(us.UNetSegmentationTool.tool_id, ids)

    def test_training_saves_a_model_and_shows_results(self) -> None:
        for task, result in self.trained.items():
            self.assertTrue((self.models / task / uc.WEIGHTS_FILE).is_file())
            self.assertTrue((self.models / task / uc.PASSPORT_FILE).is_file())
            self.assertIn("trained and saved", result.message)
            self.assertTrue(result.image.info.get(us.CHART_MARKER))
            self.assertIn("Epochs (best)", result.details)

    def test_segment_dataset_image_compares_with_ground_truth(self) -> None:
        document = ImageDocument()
        document.load(self.root / "XTest" / "001a.jpg")
        for task in uc.TASKS:
            result = self.segment(document, task)
            self.assertTrue(result.image.info.get(us.CHART_MARKER))
            self.assertIn("ground truth", result.message)
            self.assertTrue(result.details["Ground truth"].endswith("001a.png"))
            key = "  Dice" if task == uc.TASK_BINARY else "  mean class Dice"
            self.assertIn(key, result.details)

    def test_segment_other_image_and_rerun_on_chart(self) -> None:
        document = ImageDocument()
        document.apply(Image.new("RGB", (40, 30), "white"))
        result = self.segment(document, uc.TASK_BINARY, threshold=0.4)
        self.assertEqual(result.details["Ground truth"], "not found")
        self.assertEqual(result.details["Threshold"], 0.4)
        document.apply(result.image)
        again = self.segment(document, uc.TASK_BINARY, compare=False)
        self.assertIn("original", again.details["Image used"])
        self.assertEqual(again.details["Image"], "40 × 30")

    def test_segment_needs_an_image(self) -> None:
        with self.assertRaises(ValueError):
            self.segment(ImageDocument(), uc.TASK_BINARY)


class ProgressTests(unittest.TestCase):
    def test_status_is_not_passed_to_the_job(self) -> None:
        self.assertEqual(
            run_with_progress(None, "t", "m", lambda x: x * 2, 3, status=lambda: "s"),
            6,
        )


if __name__ == "__main__":
    unittest.main()
