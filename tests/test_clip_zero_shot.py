"""Unit tests for the Set 10 CLIP zero-shot tool and shared code (no download)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools import clip_common as cc
from forensics_app.tools import clip_zero_shot as cz

try:
    from clip_fakes import (
        BLUE,
        RED,
        FakeClip,
        FakeProcessor,
        fake_clip,
        solid,
    )
except ImportError:  # run from the project folder as tests.test_clip_zero_shot
    from tests.clip_fakes import (
        BLUE,
        RED,
        FakeClip,
        FakeProcessor,
        fake_clip,
        solid,
    )


def document_with(image: Image.Image) -> ImageDocument:
    document = ImageDocument()
    document.apply(image)
    return document


class ParsingTests(unittest.TestCase):
    def test_labels_one_per_line_drop_blanks_and_repeats(self) -> None:
        text = "  two cats \n\nthree cats\nTwo Cats\n one cat, maybe \n"
        self.assertEqual(
            cc.parse_labels(text), ["two cats", "three cats", "one cat, maybe"]
        )

    def test_single_line_is_split_on_commas(self) -> None:
        self.assertEqual(
            cc.parse_labels("coffee, tea,, soup"), ["coffee", "tea", "soup"]
        )

    def test_empty_text_gives_no_labels(self) -> None:
        self.assertEqual(cc.parse_labels(" \n "), [])

    def test_template_needs_exactly_one_placeholder(self) -> None:
        self.assertEqual(cc.validate_template(" a photo of {}. "), "a photo of {}.")
        for bad in ("a photo", "{} and {}"):
            with self.assertRaises(ValueError):
                cc.validate_template(bad)

    def test_templates_one_per_line(self) -> None:
        self.assertEqual(cc.parse_templates("{}\n\n a {}. \n{}"), ["{}", "a {}."])
        with self.assertRaises(ValueError):
            cc.parse_templates("  \n")
        with self.assertRaises(ValueError):
            cc.parse_templates("{}\nno placeholder")

    def test_prompts_keep_braces_inside_labels(self) -> None:
        self.assertEqual(
            cc.build_prompts(["a {b}"], ["x {}.", "y {}"]), [["x a {b}.", "y a {b}"]]
        )

    def test_resolve_device(self) -> None:
        self.assertEqual(cc.resolve_device("cpu"), "cpu")
        with patch.object(cc, "_available_accelerator", return_value="mps"):
            self.assertEqual(cc.resolve_device("auto"), "mps")
            self.assertEqual(cc.resolve_device("bogus"), "mps")

    def test_examples_and_presets_are_valid(self) -> None:
        for templates, labels in cz.LABEL_EXAMPLES.values():
            for template in templates:
                cc.validate_template(template)
            self.assertGreaterEqual(len(labels), 2)
        for template in cz.ENSEMBLE_TEMPLATES + cz.COMPARE_TEMPLATES:
            cc.validate_template(template)


class CommonTests(unittest.TestCase):
    def test_softmax_is_stable_and_normalised(self) -> None:
        probs = cc.softmax_probabilities([[1.0, 0.0], [100.0, 99.0]], 100.0)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])
        self.assertTrue(np.isfinite(probs).all())

    def test_list_images_filters_and_recurses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            solid(RED).save(root / "a.png")
            solid(RED).save(root / "sub" / "b.JPG", format="JPEG")
            (root / "notes.txt").write_text("x")
            solid(RED).save(root / ".hidden.png")
            self.assertEqual([p.name for p in cc.list_images(root)], ["a.png"])
            self.assertEqual(
                [p.name for p in cc.list_images(root, recursive=True)],
                ["a.png", "b.JPG"],
            )

    def test_source_image_skips_a_chart_on_the_canvas(self) -> None:
        document = document_with(solid(RED))
        self.assertEqual(cc.source_image(document), (document.current, False))
        document.apply(cc.mark_chart(solid(BLUE)))
        image, used_original = cc.source_image(document)
        self.assertTrue(used_original)
        self.assertEqual(image.getpixel((0, 0)), RED)
        with self.assertRaises(ValueError):
            cc.source_image(ImageDocument())


class LoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        cc._MODEL_CACHE.clear()

    def tearDown(self) -> None:
        cc._MODEL_CACHE.clear()

    def test_load_is_cached_and_keeps_two_models(self) -> None:
        loads: list[str] = []

        class Model:
            def to(self, _device):
                return self

            def eval(self):
                return self

        class Loader:
            @staticmethod
            def from_pretrained(name, cache_dir=None):
                loads.append(name)
                return Model()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cc, "CLIP_CACHE_DIR", Path(tmp)),
            patch.object(cc, "_import_clip", return_value=(Loader, Loader)),
        ):
            self.assertIn("Downloading", cc.loading_message(["m"], "cpu"))
            first = cc.load_clip("m", "cpu")
            self.assertIs(cc.load_clip("m", "cpu"), first)
            self.assertEqual(cc.loading_message(["m"], "cpu"), "")
            cc.load_clip("b", "cpu")
            self.assertTrue(cc.is_model_cached("m", "cpu"))
            cc.load_clip("c", "cpu")
            self.assertFalse(cc.is_model_cached("m", "cpu"))
        self.assertEqual(loads, ["m", "m", "b", "b", "c", "c"])

    def test_load_uses_lru_eviction_and_local_files_only_when_downloaded(self) -> None:
        calls: list[tuple[str, bool]] = []

        class Model:
            def to(self, _device):
                return self

            def eval(self):
                return self

        class Loader:
            @staticmethod
            def from_pretrained(name, cache_dir=None, local_files_only=False, **_kw):
                calls.append((name, local_files_only))
                return Model()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(cc, "CLIP_CACHE_DIR", Path(tmp)),
            patch.object(cc, "_import_clip", return_value=(Loader, Loader)),
        ):
            snap = Path(tmp) / "models--org--a" / "snapshots" / "rev1"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")

            cc.load_clip("org/a", "cpu")
            cc.load_clip("org/b", "cpu")
            # Touch org/a so org/b becomes least recently used.
            cc.load_clip("org/a", "cpu")
            cc.load_clip("org/c", "cpu")

            self.assertTrue(cc.is_model_cached("org/a", "cpu"))
            self.assertFalse(cc.is_model_cached("org/b", "cpu"))
            self.assertTrue(cc.is_model_cached("org/c", "cpu"))
        self.assertEqual(calls[0], ("org/a", True))
        self.assertEqual(calls[2], ("org/b", False))


class ClassifyTests(unittest.TestCase):
    image = solid(RED)

    def test_probabilities_follow_scaled_cosine_softmax(self) -> None:
        model = FakeClip()
        result = cz.classify(
            model, FakeProcessor(), self.image, ["cat", "dog", "owl"], ["a {}."]
        )

        np.testing.assert_allclose(result.similarities, [1.0, 0.0, 0.0], atol=1e-4)
        expected = np.exp(10.0 * result.similarities)
        np.testing.assert_allclose(result.probabilities, expected / expected.sum())
        self.assertAlmostEqual(result.logit_scale, 10.0, places=4)
        self.assertEqual(model.texts, ["a cat.", "a dog.", "a owl."])
        self.assertEqual(result.top[0], "cat")

    def test_output_objects_from_newer_transformers_are_accepted(self) -> None:
        model, processor = fake_clip(wrap_outputs=True)
        result = cz.classify(model, processor, self.image, ["dog", "cat"], ["{}"])
        self.assertEqual([r[0] for r in result.ranked()], ["cat", "dog"])

    def test_ensemble_averages_normalised_text_embeddings(self) -> None:
        templates = ["a photo of a {}.", "a close-up photo of a {}."]
        model = FakeClip()
        result = cz.classify(
            model, FakeProcessor(), self.image, ["cat", "dog"], templates
        )

        self.assertEqual(len(model.texts), 4)
        plain = np.array([1.0, 0.0, 0.0])
        close = np.array([1.0, 0.0, 0.25]) / np.linalg.norm([1.0, 0.0, 0.25])
        mean = (plain + close) / 2
        self.assertAlmostEqual(
            float(result.similarities[0]), mean[0] / np.linalg.norm(mean), places=4
        )

    def test_needs_two_classes(self) -> None:
        with self.assertRaises(ValueError):
            cz.classify(FakeClip(), FakeProcessor(), self.image, ["cat"], ["{}"])

    def test_compare_templates_adds_the_ensemble(self) -> None:
        model, processor = fake_clip()
        templates = ["a photo of a {}.", "a close-up photo of a {}."]
        panels = cz.compare_templates(
            model, processor, self.image, ["cat", "owl"], templates
        )
        self.assertEqual(
            [c for c, _ in panels],
            [
                '"a photo of a {}."',
                '"a close-up photo of a {}."',
                "ensemble of 2 templates",
            ],
        )
        self.assertEqual(model.images_embedded, 1)  # the image is embedded once
        # Single-pass text encoding: 2 classes x 2 templates = 4 prompts (not 8).
        self.assertEqual(len(model.texts), 4)
        single = cz.compare_templates(
            model, processor, self.image, ["cat", "owl"], ["{}"]
        )
        self.assertEqual(len(single), 1)

    def test_embed_texts_chunks_and_requests_truncation(self) -> None:
        seen_kwargs: list[dict] = []

        class RecordingProcessor(FakeProcessor):
            def __call__(self, images=None, text=None, **kwargs):
                seen_kwargs.append(kwargs)
                return super().__call__(images=images, text=text, **kwargs)

        model = FakeClip()
        emb = cc.embed_texts(
            model,
            RecordingProcessor(),
            ["cat", "dog", "owl", "cat", "dog"],
            batch_size=2,
        )
        self.assertEqual(tuple(emb.shape), (5, 3))
        self.assertEqual(len(seen_kwargs), 3)
        self.assertTrue(all(kw.get("truncation") is True for kw in seen_kwargs))

    def test_compare_models_loads_each_model(self) -> None:
        loaded: list[str] = []

        def loader(name: str, _device: str):
            loaded.append(name)
            return fake_clip(scale=5.0 if name.endswith("b") else 50.0)

        panels = cz.compare_models(
            loader, ["org/a", "org/b"], self.image, ["cat", "dog"], ["{}"]
        )
        self.assertEqual(loaded, ["org/a", "org/b"])
        self.assertEqual([c for c, _ in panels], ["a", "b"])
        self.assertGreater(panels[0][1].top[1], panels[1][1].top[1])
        with self.assertRaises(ValueError):
            cz.compare_models(loader, [], self.image, ["cat", "dog"], ["{}"])


class ToolTests(unittest.TestCase):
    def base_parameters(self, **extra) -> dict:
        return {
            "clip": fake_clip(),
            "device": "cpu",
            "labels": ["dog", "cat", "owl"],
            "templates": ["a photo of a {}."],
            "title": "test",
            **extra,
        }

    def test_tool_is_registered(self) -> None:
        ids = [tool.tool_id for tool in build_tool_registry().all()]
        self.assertIn(cz.ClipZeroShotTool.tool_id, ids)

    def test_classify_returns_chart_and_ranked_details(self) -> None:
        document = document_with(solid(RED, (40, 30)))
        result = cz.ClipZeroShotTool().run(
            None, document, parameters=self.base_parameters()
        )

        self.assertIn("cat", result.message)
        self.assertIn("3 classes", result.message)
        self.assertTrue(cc.is_chart(result.image))
        self.assertEqual(result.details["Device"], "cpu")
        self.assertEqual(result.details["Model"], cc.DEFAULT_MODEL)
        self.assertTrue(result.details["#1 cat"].startswith("p = 1.000"))

    def test_rerun_on_the_chart_classifies_the_original(self) -> None:
        document = document_with(solid(BLUE, (40, 30)))
        tool = cz.ClipZeroShotTool()
        first = tool.run(None, document, parameters=self.base_parameters())
        document.apply(first.image)
        second = tool.run(None, document, parameters=self.base_parameters())
        self.assertIn("owl", second.message)
        self.assertIn("original", second.details["Image"])

    def test_every_mode_produces_a_chart(self) -> None:
        document = document_with(solid(RED, (60, 40)))
        tool = cz.ClipZeroShotTool()
        for mode in cz.MODES:
            with self.subTest(mode=mode):
                parameters = self.base_parameters(
                    mode=mode,
                    models=["org/one", "org/two"],
                    loader=lambda _n, _d: fake_clip(),
                    templates=["a photo of a {}.", "a close-up photo of a {}."],
                )
                result = tool.run(None, document, parameters=parameters)
                self.assertTrue(cc.is_chart(result.image))
                self.assertIn("cat", result.message + str(result.details))

    def test_run_without_image_fails(self) -> None:
        with self.assertRaises(ValueError):
            cz.ClipZeroShotTool().run(
                None,
                ImageDocument(),
                parameters={"clip": fake_clip(), "labels": ["a", "b"]},
            )


if __name__ == "__main__":
    unittest.main()
