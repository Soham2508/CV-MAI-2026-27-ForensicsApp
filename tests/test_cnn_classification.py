"""Unit tests for the Set 8.2 CNN Classification tool (fake models, no TensorFlow)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools import cnn_classification as cnn

CLASSES = ["cat", "dog", "frog"]


def make_passport(
    *,
    input_size: tuple[int, int] = (8, 8),
    scaling: str = "0_1",
    native: tuple[int, int] | None = (4, 4),
) -> dict:
    passport = {
        "model_name": f"fake_{scaling}",
        "display_name": f"Fake model {scaling}",
        "class_names": list(CLASSES),
        "preprocessing": {
            "input_size": list(input_size),
            "scaling": scaling,
            "formula": "x / 255.0" if scaling == "0_1" else "x / 127.5 - 1.0",
        },
        "dataset": {"name": "Toy", "sample_sizes": {"train": 30, "test": 9}},
        "architecture": {"type": "toy", "parameters": {"total": 123}},
        "training": {
            "history": {
                "loss": [1.0, 0.6, 0.4],
                "val_loss": [1.1, 0.7, 0.5],
                "accuracy": [0.4, 0.6, 0.8],
                "val_accuracy": [0.3, 0.5, 0.7],
            },
            "phase_boundaries": [1],
        },
        "evaluation": {
            "test_accuracy": 2 / 3,
            "test_loss": 0.5,
            "per_class_accuracy": {"cat": 1.0, "dog": 0.5, "frog": 0.5},
            "confusion_matrix": [[3, 0, 0], [1, 2, 0], [0, 1, 2]],
        },
        "files": {"model": "model.keras"},
    }
    if native is not None:
        passport["dataset"]["original_image_size"] = list(native)
    return passport


def write_model_folder(root: Path, name: str, passport: dict, *, weights: bool) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "passport.json").write_text(json.dumps(passport), encoding="utf-8")
    if weights:
        (folder / "model.keras").write_bytes(b"not a real model")
    return folder


class FakeModel:
    """Callable stand-in for a Keras model that records its calls."""

    def __init__(self, probs: list[float]) -> None:
        self.probs = np.asarray([probs], dtype=np.float32)
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def __call__(self, batch: np.ndarray, training: bool = True) -> np.ndarray:
        self.calls.append((batch.shape, training))
        return np.repeat(self.probs, len(batch), axis=0)


class CnnClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        cnn._MODEL_CACHE.clear()

    def tearDown(self) -> None:
        cnn._MODEL_CACHE.clear()
        self._tmp.cleanup()

    # -- passports ---------------------------------------------------------

    def test_list_models_skips_folders_without_or_with_broken_passport(self) -> None:
        write_model_folder(self.root, "good", make_passport(), weights=False)
        (self.root / "sample_images").mkdir()
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "passport.json").write_text("{not json", encoding="utf-8")
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        (incomplete / "passport.json").write_text('{"model_name": "x"}', "utf-8")

        models = cnn.list_cnn_models(self.root)

        self.assertEqual([m.folder.name for m in models], ["good"])
        self.assertEqual(models[0].class_names, CLASSES)
        self.assertFalse(models[0].has_weights)
        self.assertEqual(cnn.list_cnn_models(self.root / "missing"), [])

    def test_load_passport_rejects_unknown_scaling(self) -> None:
        folder = write_model_folder(
            self.root, "m", make_passport(scaling="0_255"), weights=False
        )
        with self.assertRaises(ValueError):
            cnn.load_passport(folder)

    def test_passport_text_and_details_mention_key_facts(self) -> None:
        folder = write_model_folder(self.root, "m", make_passport(), weights=False)
        info = cnn.load_passport(folder)

        text = cnn.passport_summary_text(info)
        details = cnn.passport_details(info)

        for fragment in ("Fake model 0_1", "cat, dog, frog", "MISSING", "66.67%"):
            self.assertIn(fragment, text)
        self.assertEqual(details["Test accuracy"], "66.67%")
        self.assertIn("missing", details["Weights"])

    # -- preprocessing -----------------------------------------------------

    def test_resize_bilinear_matches_tensorflow_half_pixel_formula(self) -> None:
        # Source x = -0.25, 0.25, 0.75, 1.25, clamped to [0, 1].
        x = np.array([[[0.0], [100.0]]], dtype=np.float32)
        out = cnn.resize_bilinear(x, 1, 4)
        np.testing.assert_allclose(out[0, :, 0], [0, 25, 75, 100], atol=1e-5)

    def test_resize_bilinear_keeps_same_size_and_constant_images(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.random((5, 7, 3), dtype=np.float32)
        np.testing.assert_allclose(cnn.resize_bilinear(x, 5, 7), x, atol=1e-6)
        flat = np.full((3, 3, 3), 42.0, dtype=np.float32)
        np.testing.assert_allclose(cnn.resize_bilinear(flat, 9, 6), 42.0)

    def test_preprocess_shapes_and_ranges_for_both_scalings(self) -> None:
        image = Image.new("RGB", (8, 8), (255, 0, 51))
        for scaling, low, high in (("0_1", 0.0, 1.0), ("-1_1", -1.0, 1.0)):
            batch, _steps = cnn.preprocess_image(
                image, make_passport(scaling=scaling, native=None)
            )
            self.assertEqual(batch.shape, (1, 8, 8, 3))
            self.assertEqual(batch.dtype, np.float32)
            self.assertAlmostEqual(float(batch.min()), low, places=6)
            self.assertAlmostEqual(float(batch.max()), high, places=6)

    def test_preprocess_converts_grayscale_and_rgba_to_rgb(self) -> None:
        passport = make_passport(native=None)
        for mode in ("L", "RGBA"):
            batch, _ = cnn.preprocess_image(Image.new(mode, (8, 8)), passport)
            self.assertEqual(batch.shape, (1, 8, 8, 3))

    def test_large_images_are_downscaled_to_dataset_resolution_first(self) -> None:
        passport = make_passport(input_size=(8, 8), native=(4, 4))
        big = Image.new("RGB", (64, 48), (10, 20, 30))

        _, steps = cnn.preprocess_image(big, passport)
        _, steps_off = cnn.preprocess_image(
            big, passport, match_training_resolution=False
        )
        _, steps_small = cnn.preprocess_image(Image.new("RGB", (4, 4)), passport)

        self.assertTrue(any("Anti-aliased" in s for s in steps))
        self.assertFalse(any("Anti-aliased" in s for s in steps_off))
        self.assertFalse(any("Anti-aliased" in s for s in steps_small))

    def test_batch_to_image_undoes_the_scaling(self) -> None:
        image = Image.new("RGB", (8, 8), (12, 200, 99))
        for scaling in ("0_1", "-1_1"):
            batch, _ = cnn.preprocess_image(
                image, make_passport(scaling=scaling, native=None)
            )
            view = cnn.batch_to_image(batch, scaling)
            self.assertEqual(view.getpixel((3, 3)), (12, 200, 99))

    def test_true_label_from_sample_filename(self) -> None:
        self.assertEqual(cnn.true_label_from_filename("dog_00012.png", CLASSES), "dog")
        self.assertIsNone(cnn.true_label_from_filename("holiday.jpg", CLASSES))
        self.assertIsNone(cnn.true_label_from_filename(None, CLASSES))

    # -- prediction --------------------------------------------------------

    def test_classify_image_ranks_classes_and_runs_in_inference_mode(self) -> None:
        folder = write_model_folder(self.root, "m", make_passport(), weights=False)
        info = cnn.load_passport(folder)
        model = FakeModel([0.2, 0.7, 0.1])

        result = cnn.classify_image(
            info,
            Image.new("RGB", (8, 8)),
            model=model,
            top_k=2,
            image_path="dog_00001.png",
        )

        self.assertEqual(result["predicted_name"], "dog")
        self.assertEqual([name for name, _ in result["top_k"]], ["dog", "cat"])
        self.assertAlmostEqual(result["confidence"], 0.7, places=5)
        self.assertEqual(result["true_name"], "dog")
        self.assertEqual(model.calls, [((1, 8, 8, 3), False)])

    def test_classify_image_rejects_output_size_mismatch(self) -> None:
        folder = write_model_folder(self.root, "m", make_passport(), weights=False)
        info = cnn.load_passport(folder)
        with self.assertRaises(ValueError):
            cnn.classify_image(
                info, Image.new("RGB", (8, 8)), model=FakeModel([0.5, 0.5])
            )

    def test_load_keras_model_is_lazy_and_cached_until_file_changes(self) -> None:
        folder = write_model_folder(self.root, "m", make_passport(), weights=True)
        info = cnn.load_passport(folder)
        loads: list[Path] = []
        fake_keras = types.SimpleNamespace(
            models=types.SimpleNamespace(
                load_model=lambda path, compile: loads.append(Path(path)) or object()
            )
        )

        with patch.dict(sys.modules, {"keras": fake_keras}):
            first = cnn.load_keras_model(info)
            second = cnn.load_keras_model(info)
            self.assertIs(first, second)
            self.assertEqual(len(loads), 1)

            stat = info.model_path.stat()
            os.utime(info.model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
            third = cnn.load_keras_model(info)

        self.assertIsNot(third, first)
        self.assertEqual(len(loads), 2)

    def test_missing_weights_raise_a_helpful_error(self) -> None:
        folder = write_model_folder(self.root, "m", make_passport(), weights=False)
        with self.assertRaises(cnn.MissingWeightsError) as ctx:
            cnn.load_keras_model(cnn.load_passport(folder))
        self.assertIn("Train the CNN", str(ctx.exception))

    # -- tool --------------------------------------------------------------

    def test_tool_is_registered_under_deep_learning(self) -> None:
        tools = [
            t for t in build_tool_registry().all() if t.tool_id == "cnn_classification"
        ]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].category, "Deep Learning")
        self.assertFalse(tools[0].requires_image)

    def test_tool_classifies_with_injected_model(self) -> None:
        folder = write_model_folder(self.root, "m", make_passport(), weights=False)
        result = cnn.CnnClassificationTool().run(
            None,
            ImageDocument(),
            parameters={
                "model_dir": str(folder),
                "image": Image.new("RGB", (20, 10), (5, 5, 5)),
                "image_path": "frog_00003.png",
                "model": FakeModel([0.1, 0.1, 0.8]),
                "top_k": 3,
            },
        )

        self.assertIn("frog", result.message)
        self.assertIsInstance(result.image, Image.Image)
        self.assertEqual(result.details["Top 1"], "frog: 80.00%")
        self.assertEqual(result.details["True class (filename)"], "frog (correct)")

    def test_tool_refuses_to_classify_without_weights(self) -> None:
        folder = write_model_folder(self.root, "m", make_passport(), weights=False)
        with self.assertRaises(cnn.MissingWeightsError):
            cnn.CnnClassificationTool().run(
                None,
                ImageDocument(),
                parameters={
                    "model_dir": str(folder),
                    "image": Image.new("RGB", (8, 8)),
                },
            )

    def test_tool_shows_passport_figure_even_without_weights(self) -> None:
        folder = write_model_folder(self.root, "m", make_passport(), weights=False)
        result = cnn.CnnClassificationTool().run(
            None,
            ImageDocument(),
            parameters={"action": "passport", "model_dir": str(folder)},
        )
        self.assertIsInstance(result.image, Image.Image)
        self.assertEqual(result.details["Model"], "Fake model 0_1")

    def test_tool_compares_all_models_side_by_side(self) -> None:
        write_model_folder(self.root, "a", make_passport(scaling="0_1"), weights=False)
        write_model_folder(self.root, "b", make_passport(scaling="-1_1"), weights=True)
        models = cnn.list_cnn_models(self.root)
        with patch.object(cnn, "list_cnn_models", return_value=models):
            result = cnn.CnnClassificationTool().run(
                None, ImageDocument(), parameters={"action": "compare"}
            )

        single = cnn.render_passport_figure(models[0])
        self.assertGreater(result.image.width, 2 * single.width - 10)
        self.assertEqual(result.details["Pixel scaling"], "0_1 | -1_1")
        self.assertEqual(result.details["Weights"], "missing | found")


if __name__ == "__main__":
    unittest.main()
