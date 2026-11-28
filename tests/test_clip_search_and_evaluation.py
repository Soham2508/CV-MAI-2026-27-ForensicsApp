"""Unit tests for the Set 10 CLIP image search and folder evaluation tools."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools import clip_common as cc
from forensics_app.tools import clip_folder_evaluation as ce
from forensics_app.tools import clip_image_search as cs
from forensics_app.tools import clip_zero_shot as cz

try:
    from clip_fakes import BLUE, GREEN, RED, fake_clip, solid
except ImportError:  # run from the project folder
    from tests.clip_fakes import BLUE, GREEN, RED, fake_clip, solid


class SearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "nested").mkdir()
        solid(RED).save(self.root / "red.png")
        solid((200, 30, 30)).save(self.root / "nested" / "dark_red.png")
        solid(GREEN).save(self.root / "green.png")
        solid(BLUE).save(self.root / "blue.png")
        cs._INDEX_CACHE.clear()

    def tearDown(self) -> None:
        cs._INDEX_CACHE.clear()
        self._tmp.cleanup()

    def paths(self) -> list[Path]:
        return cc.list_images(self.root, recursive=True)

    def test_text_query_ranks_matching_images_first(self) -> None:
        model, processor = fake_clip()
        hits, info = cs.search(model, processor, self.paths(), "a dog", model_name="m")
        self.assertEqual(hits[0].path.name, "green.png")
        self.assertEqual(len(hits), cs.DEFAULT_TOP_K)
        self.assertEqual(info["indexed"], 4)
        self.assertFalse(info["from_cache"])

    def test_folder_embeddings_are_reused_until_the_model_changes(self) -> None:
        model, processor = fake_clip()
        cs.search(model, processor, self.paths(), "cat", model_name="m")
        embedded = model.images_embedded
        _, info = cs.search(model, processor, self.paths(), "owl", model_name="m")
        self.assertTrue(info["from_cache"])
        self.assertEqual(model.images_embedded, embedded)
        _, info = cs.search(model, processor, self.paths(), "owl", model_name="x")
        self.assertFalse(info["from_cache"])

    def test_query_and_images_are_required(self) -> None:
        model, processor = fake_clip()
        with self.assertRaises(ValueError):
            cs.search(model, processor, self.paths(), "  ", model_name="m")
        with self.assertRaises(ValueError):
            cs.search(model, processor, [], "cat", model_name="m")

    def test_tool_returns_ranked_matches(self) -> None:
        params = {
            "task": cz.TASK_SEARCH,
            "clip": fake_clip(),
            "device": "cpu",
            "folder": str(self.root),
            "recursive": True,
            "text": "an owl",
            "top_k": 2,
        }
        result = cz.ClipZeroShotTool().run(None, ImageDocument(), parameters=params)
        self.assertIn("blue.png", result.message)
        self.assertTrue(cc.is_chart(result.image))
        self.assertEqual(result.details["Query"], '"an owl"')
        self.assertIn("#1 blue.png", result.details)
        with self.assertRaises(FileNotFoundError):
            cz.ClipZeroShotTool().run(
                None,
                ImageDocument(),
                parameters={**params, "folder": str(self.root / "missing")},
            )


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "photos"
        for folder, colors in {
            "cat": [RED, RED, BLUE],  # the blue "cat" is a mistake (owl)
            "dog": [GREEN, GREEN],
            "baby_owl": [BLUE],
            "empty": [],
        }.items():
            (self.root / folder).mkdir(parents=True)
            for i, color in enumerate(colors):
                solid(color).save(self.root / folder / f"{i}.png")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_class_text_from_folder_names(self) -> None:
        self.assertEqual(ce.class_text("hot_chocolate"), "hot chocolate")
        self.assertEqual(ce.class_text("sea-lion__2"), "sea lion 2")

    def test_collect_dataset_skips_empty_folders_and_limits(self) -> None:
        folders, labels, paths, targets = ce.collect_dataset(self.root)
        self.assertEqual(folders, ["baby_owl", "cat", "dog"])
        self.assertEqual(labels, ["baby owl", "cat", "dog"])
        self.assertEqual(len(paths), 6)
        self.assertEqual(targets.tolist(), [0, 1, 1, 1, 2, 2])
        _, _, paths, _ = ce.collect_dataset(self.root, max_per_class=1)
        self.assertEqual(len(paths), 3)
        with self.assertRaises(ValueError):
            ce.collect_dataset(self.root / "dog")

    def test_metrics(self) -> None:
        matrix = ce.confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
        self.assertEqual(matrix.tolist(), [[1, 1], [0, 2]])
        (p0, r0, f0, n0), (p1, r1, f1, n1) = ce.per_class_metrics(matrix)
        self.assertEqual((p0, r0, n0), (1.0, 0.5, 2))
        self.assertAlmostEqual(p1, 2 / 3)
        self.assertEqual(r1, 1.0)
        self.assertAlmostEqual(f0, 2 / 3)

    def test_evaluate_scores_every_template_and_the_ensemble(self) -> None:
        model, processor = fake_clip()
        evaluation = ce.evaluate_folder(
            model,
            processor,
            self.root,
            ["a photo of a {}.", "a close-up photo of a {}."],
        )
        self.assertEqual(len(evaluation.schemes), 3)
        self.assertAlmostEqual(evaluation.best.accuracy, 5 / 6)
        self.assertEqual(evaluation.k, 2)
        self.assertEqual(evaluation.matrix[1, 0], 1)  # cat predicted as owl
        errors = ce.mistakes(evaluation)
        self.assertEqual(len(errors), 1)
        self.assertEqual(evaluation.paths[errors[0][0]].parent.name, "cat")

    def test_tool_saves_csv(self) -> None:
        out = Path(self._tmp.name) / "out"
        result = cz.ClipZeroShotTool().run(
            None,
            ImageDocument(),
            parameters={
                "task": cz.TASK_EVALUATE,
                "clip": fake_clip(),
                "device": "cpu",
                "folder": str(self.root),
                "templates": ["a photo of a {}."],
                "save_csv": True,
                "output_dir": str(out),
            },
        )
        self.assertIn("83.3%", result.message)
        self.assertTrue(cc.is_chart(result.image))
        csv_folder = Path(result.details["CSV"])
        with open(csv_folder / "predictions.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(int(r["correct"]) for r in rows), 5)
        self.assertTrue((csv_folder / "templates.csv").is_file())

    def test_only_one_clip_tool_is_registered(self) -> None:
        clip = [t for t in build_tool_registry().all() if "clip" in t.tool_id]
        self.assertEqual([t.tool_id for t in clip], [cz.ClipZeroShotTool.tool_id])
        self.assertFalse(cz.ClipZeroShotTool.requires_image)


if __name__ == "__main__":
    unittest.main()
