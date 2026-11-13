"""Unit tests for the Set 8.3 CNN Feature Projection tool (synthetic features)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools import feature_projection as fp

CLASSES = ["cat", "dog", "frog"]
PER_CLASS = 30
DIM = 16


def make_features(seed: int = 0) -> dict[str, np.ndarray]:
    """Return three well-separated Gaussian blobs, one per class."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(CLASSES)), PER_CLASS)
    centres = np.eye(len(CLASSES), DIM) * 20.0
    features = centres[labels] + rng.normal(size=(labels.size, DIM))
    predictions = labels.copy()
    predictions[::10] = (labels[::10] + 1) % len(CLASSES)
    return {
        "features": features.astype(np.float32),
        "labels": labels.astype(np.int64),
        "predictions": predictions.astype(np.int64),
        "class_names": np.array(CLASSES),
        "layer_name": np.array("features"),
        "model_name": np.array("fake"),
        "split": np.array("test"),
    }


def write_model_folder(
    root: Path, name: str, *, features: dict[str, np.ndarray] | None = None
) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    passport = {
        "model_name": name,
        "display_name": f"Fake {name}",
        "class_names": list(CLASSES),
        "preprocessing": {"input_size": [8, 8], "scaling": "0_1"},
        "evaluation": {"test_accuracy": 0.9},
        "feature_extraction": {
            "layer": "features",
            "dimension": DIM,
            "split": "test",
            "num_samples": PER_CLASS * len(CLASSES),
            "file": "features_test.npz",
        },
        "files": {"model": "model.keras", "features": "features_test.npz"},
    }
    (folder / "passport.json").write_text(json.dumps(passport), encoding="utf-8")
    if features is not None:
        np.savez(folder / "features_test.npz", **features)
    return folder


class FakeUMAP:
    """Stand-in for ``umap.UMAP`` that records its options."""

    last_options: dict = {}

    def __init__(
        self,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=None,
        n_jobs=-1,
        compatibility_layout=None,
    ) -> None:
        FakeUMAP.last_options = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": random_state,
            "compatibility_layout": compatibility_layout,
        }

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)[:, :2]


def fake_umap_module() -> types.ModuleType:
    module = types.ModuleType("umap")
    module.UMAP = FakeUMAP
    return module


class FeatureProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def feature_set(self) -> fp.FeatureSet:
        folder = write_model_folder(self.root, "m", features=make_features())
        return fp.load_feature_set(fp.load_passport(folder))

    # -- feature files -----------------------------------------------------

    def test_list_feature_models_skips_folders_without_features(self) -> None:
        write_model_folder(self.root, "with", features=make_features())
        write_model_folder(self.root, "without")
        (self.root / "sample_images").mkdir()

        names = [m.name for m in fp.list_feature_models(self.root)]
        self.assertEqual(names, ["with"])

    def test_load_feature_set_reads_arrays_and_metadata(self) -> None:
        data = self.feature_set()
        self.assertEqual(data.num_samples, PER_CLASS * len(CLASSES))
        self.assertEqual(data.dimension, DIM)
        self.assertEqual(data.layer_name, "features")
        self.assertEqual(data.class_names, CLASSES)
        self.assertAlmostEqual(data.accuracy, 0.9)
        np.testing.assert_array_equal(data.indices, np.arange(data.num_samples))

    def test_load_feature_set_rejects_broken_files(self) -> None:
        missing = make_features()
        del missing["predictions"]
        folder = write_model_folder(self.root, "a", features=missing)
        with self.assertRaises(ValueError):
            fp.load_feature_set(fp.load_passport(folder))

        short = make_features()
        short["labels"] = short["labels"][:-1]
        folder = write_model_folder(self.root, "b", features=short)
        with self.assertRaises(ValueError):
            fp.load_feature_set(fp.load_passport(folder))

    def test_subsample_per_class_is_balanced_seeded_and_sorted(self) -> None:
        labels = np.repeat(np.arange(3), [10, 4, 10])
        rows = fp.subsample_per_class(labels, 5, seed=1)

        self.assertEqual(np.bincount(labels[rows]).tolist(), [5, 4, 5])
        self.assertTrue(np.all(np.diff(rows) > 0))
        np.testing.assert_array_equal(rows, fp.subsample_per_class(labels, 5, 1))
        self.assertEqual(fp.subsample_per_class(labels, None, 1).size, labels.size)
        self.assertEqual(fp.subsample_per_class(labels, 0, 1).size, labels.size)

    def test_subset_keeps_original_indices(self) -> None:
        data = self.feature_set()
        sub = data.subset(np.array([3, 40, 70]))
        np.testing.assert_array_equal(sub.indices, [3, 40, 70])
        np.testing.assert_array_equal(sub.labels, data.labels[[3, 40, 70]])

    # -- reduction and quality ---------------------------------------------

    def test_pca_reduces_and_reports_variance(self) -> None:
        x = make_features()["features"]
        reduced, kept = fp.reduce_with_pca(x, 4, seed=0)
        self.assertEqual(reduced.shape, (x.shape[0], 4))
        self.assertGreater(kept, 0.5)
        self.assertLessEqual(kept, 1.0)

        same, kept = fp.reduce_with_pca(x, 64, seed=0)
        self.assertIs(same, x)
        self.assertEqual(kept, 1.0)

    def test_knn_accuracy_separates_blobs_from_shuffled_labels(self) -> None:
        data = make_features()
        x, labels = data["features"], data["labels"]
        self.assertEqual(fp.knn_accuracy(x, labels), 1.0)
        shuffled = np.random.default_rng(0).permutation(labels)
        self.assertLess(fp.knn_accuracy(x, shuffled), 0.7)

    def test_knn_correct_excludes_the_point_itself(self) -> None:
        x = np.array([[0.0], [0.1], [0.2], [10.0]])
        labels = np.array([0, 0, 0, 1])
        correct = fp.knn_correct(x, labels, k=1)
        self.assertEqual(correct.tolist(), [True, True, True, False])

    def test_silhouette_is_high_for_separated_blobs(self) -> None:
        data = make_features()
        self.assertGreater(fp.silhouette(data["features"], data["labels"], 0), 0.8)

    # -- pipeline ----------------------------------------------------------

    def test_compute_projections_with_tsne_only(self) -> None:
        settings = fp.ProjectionSettings(
            samples_per_class=20, pca_components=8, perplexity=10, methods=("t-SNE",)
        )
        result = fp.compute_projections(self.feature_set(), settings)

        self.assertEqual(result.feature_set.num_samples, 60)
        self.assertEqual(result.embeddings["t-SNE"].shape, (60, 2))
        self.assertEqual(set(result.quality), {"Original", "PCA", "t-SNE"})
        self.assertGreater(result.quality["t-SNE"]["knn_accuracy"], 0.9)
        self.assertIsNotNone(result.pca_variance)

    def test_umap_is_skipped_when_not_installed(self) -> None:
        settings = fp.ProjectionSettings(pca_components=None, methods=("UMAP",))
        with patch.object(fp, "umap_available", return_value=False):
            result = fp.compute_projections(self.feature_set(), settings)
        self.assertEqual(result.embeddings, {})
        self.assertIn("UMAP", result.skipped)
        self.assertIn("UMAP skipped", fp.projection_summary(result))

    def test_umap_receives_settings_and_keeps_legacy_layout(self) -> None:
        settings = fp.ProjectionSettings(
            pca_components=None,
            n_neighbors=500,
            min_dist=0.3,
            seed=7,
            methods=("UMAP",),
        )
        with (
            patch.object(fp, "umap_available", return_value=True),
            patch.object(fp, "_import_umap", return_value=fake_umap_module()),
        ):
            result = fp.compute_projections(self.feature_set(), settings)

        self.assertEqual(result.embeddings["UMAP"].shape, (90, 2))
        options = FakeUMAP.last_options
        self.assertEqual(options["n_neighbors"], 89)  # capped at n - 1
        self.assertEqual(options["min_dist"], 0.3)
        self.assertEqual(options["random_state"], 7)
        self.assertTrue(options["compatibility_layout"])

    def test_unknown_method_is_rejected(self) -> None:
        settings = fp.ProjectionSettings(methods=("PCA-2D",))
        with self.assertRaises(ValueError):
            fp.compute_projections(self.feature_set(), settings)

    # -- figures -----------------------------------------------------------

    def projected(self) -> fp.ProjectionResult:
        settings = fp.ProjectionSettings(
            samples_per_class=None, pca_components=None, methods=("t-SNE", "UMAP")
        )
        with (
            patch.object(fp, "umap_available", return_value=True),
            patch.object(fp, "_import_umap", return_value=fake_umap_module()),
        ):
            return fp.compute_projections(self.feature_set(), settings)

    def test_projection_figure_for_every_colour_mode(self) -> None:
        result = self.projected()
        for mode in fp.COLOR_MODES:
            image = fp.render_projection_figure(result, mode)
            self.assertIsInstance(image, Image.Image)
            self.assertGreater(image.width, image.height)
        with self.assertRaises(ValueError):
            fp.render_projection_figure(result, "rainbow")

    def test_class_panels_figure(self) -> None:
        result = self.projected()
        self.assertIsInstance(fp.render_class_panels(result, "UMAP"), Image.Image)
        with self.assertRaises(ValueError):
            fp.render_class_panels(result, "missing")

    def test_projection_comparison_has_one_row_per_model(self) -> None:
        result = self.projected()
        single = fp.render_projection_figure(result, "true")
        for mode in fp.COLOR_MODES:
            grid = fp.render_projection_comparison([result, result], mode)
            self.assertGreater(grid.height, 1.6 * single.height)
        with self.assertRaises(ValueError):
            fp.render_projection_comparison([result], "rainbow")

    def test_projection_comparison_needs_the_same_methods(self) -> None:
        both = self.projected()
        tsne_only = fp.ProjectionResult(
            feature_set=both.feature_set,
            settings=both.settings,
            embeddings={"t-SNE": both.embeddings["t-SNE"]},
        )
        with self.assertRaises(ValueError):
            fp.render_projection_comparison([both, tsne_only])

    def test_export_comparison_outputs(self) -> None:
        result = self.projected()
        paths = fp.export_comparison_outputs([result, result], self.root / "cmp")
        self.assertEqual({p.parent for p in paths}, {self.root / "cmp" / "comparison"})
        self.assertEqual(
            sorted(p.name for p in paths),
            [
                "comparison_errors.png",
                "comparison_panels_tsne.png",
                "comparison_panels_umap.png",
                "comparison_predicted.png",
                "comparison_true.png",
            ],
        )

    def test_class_panel_rows_has_one_row_per_model(self) -> None:
        result = self.projected()
        one = fp.render_class_panel_rows([result], "t-SNE")
        two = fp.render_class_panel_rows([result, result], "t-SNE")
        self.assertGreater(two.height, 1.6 * one.height)
        with self.assertRaises(ValueError):
            fp.render_class_panel_rows([result], "missing")

    def test_spread_labels_removes_overlaps_and_stays_inside(self) -> None:
        anchors = np.full((5, 2), 0.5)
        widths = np.full(5, 0.1)
        placed = fp._spread_labels(anchors, widths, height=0.05)
        for i in range(5):
            for j in range(i + 1, 5):
                d = np.abs(placed[i] - placed[j])
                self.assertTrue(d[0] >= 0.1 or d[1] >= 0.05 - 1e-9)
        self.assertTrue(np.all((placed >= 0.04) & (placed <= 0.96)))

    def test_class_colors_use_names_then_index(self) -> None:
        self.assertEqual(fp.class_colors(["cat"]), [fp.CLASS_COLORS["cat"]])
        self.assertEqual(fp.class_colors(["a", "b"]), fp.FALLBACK_COLORS[:2])
        many = fp.class_colors([f"c{i}" for i in range(12)])
        self.assertEqual(len(set(many[:10])), 10)

    def test_stack_images_centres_on_the_widest(self) -> None:
        stacked = fp.stack_images([Image.new("RGB", (10, 5)), Image.new("RGB", (4, 3))])
        self.assertEqual(stacked.size, (10, 8))

    # -- tool --------------------------------------------------------------

    def test_settings_from_parameters_validates_input(self) -> None:
        settings = fp.settings_from_parameters(
            {"samples_per_class": 0, "pca_components": None, "methods": ["t-SNE"]}
        )
        self.assertIsNone(settings.samples_per_class)
        self.assertIsNone(settings.pca_components)
        for bad in (
            {"methods": []},
            {"samples_per_class": -1},
            {"pca_components": 1},
            {"perplexity": 0},
            {"n_neighbors": 1},
            {"min_dist": 1.0},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fp.settings_from_parameters(bad)

    def test_tool_is_registered_under_deep_learning(self) -> None:
        tools = [
            t for t in build_tool_registry().all() if t.tool_id == "feature_projection"
        ]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].category, "Deep Learning")
        self.assertFalse(tools[0].requires_image)

    def test_tool_projects_one_model(self) -> None:
        folder = write_model_folder(self.root, "m", features=make_features())
        result = fp.FeatureProjectionTool().run(
            None,
            ImageDocument(),
            parameters={
                "model_dirs": [str(folder)],
                "samples_per_class": 0,
                "pca_components": None,
                "methods": ["t-SNE"],
                "perplexity": 10,
                "color_by": "errors",
                "class_panels": "t-SNE",
            },
        )

        self.assertIn("Fake m", result.message)
        self.assertIsInstance(result.image, Image.Image)
        self.assertEqual(result.details["Model"], "Fake m")
        self.assertEqual(result.details["Points"], "90 (test split)")
        self.assertIn("t-SNE — 5-NN acc / silhouette", result.details)

    def test_tool_compares_models_in_one_table(self) -> None:
        a = write_model_folder(self.root, "a", features=make_features(0))
        b = write_model_folder(self.root, "b", features=make_features(1))
        with (
            patch.object(fp, "umap_available", return_value=True),
            patch.object(fp, "_import_umap", return_value=fake_umap_module()),
        ):
            result = fp.FeatureProjectionTool().run(
                None,
                ImageDocument(),
                parameters={
                    "model_dirs": [str(a), str(b)],
                    "pca_components": None,
                    "methods": ["UMAP"],
                },
            )

        self.assertEqual(result.details["Model"], "Fake a | Fake b")
        single = fp.render_projection_figure(
            fp.compute_projections(
                fp.load_feature_set(fp.load_passport(a)),
                fp.ProjectionSettings(pca_components=None, methods=("t-SNE",)),
            )
        )
        self.assertGreater(result.image.height, 1.5 * single.height)

    def test_save_and_load_projection_round_trip(self) -> None:
        result = self.projected()
        path = fp.save_projection(result, self.root / "out" / "p.npz")
        loaded = fp.load_projection(path)

        np.testing.assert_array_equal(loaded["tsne"], result.embeddings["t-SNE"])
        np.testing.assert_array_equal(loaded["umap"], result.embeddings["UMAP"])
        np.testing.assert_array_equal(loaded["indices"], result.feature_set.indices)
        self.assertEqual(loaded["class_names"], CLASSES)
        self.assertEqual(loaded["settings"]["seed"], 42)
        self.assertIn("t-SNE", loaded["quality"])

    def test_export_writes_every_figure_and_the_coordinates(self) -> None:
        out = self.root / "exports"
        paths = fp.export_projection_outputs(self.projected(), out)
        names = sorted(p.name for p in paths)
        self.assertEqual(
            names,
            [
                "m_errors.png",
                "m_panels_tsne.png",
                "m_panels_umap.png",
                "m_predicted.png",
                "m_projection.npz",
                "m_true.png",
            ],
        )
        self.assertTrue(all(p.is_file() for p in paths))
        self.assertEqual({p.parent for p in paths}, {out / "m"})

    def test_tool_saves_outputs_when_asked(self) -> None:
        folder = write_model_folder(self.root, "m", features=make_features())
        out = self.root / "saved"
        result = fp.FeatureProjectionTool().run(
            None,
            ImageDocument(),
            parameters={
                "model_dirs": [str(folder)],
                "pca_components": None,
                "methods": ["t-SNE"],
                "perplexity": 10,
                "save_dir": str(out),
            },
        )
        self.assertEqual(result.details["Saved to"], str(out))
        self.assertTrue((out / "m" / "m_projection.npz").is_file())
        self.assertTrue((out / "m" / "m_true.png").is_file())

    def test_tool_explains_when_no_method_could_run(self) -> None:
        folder = write_model_folder(self.root, "m", features=make_features())
        with patch.object(fp, "umap_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "umap-learn"):
                fp.FeatureProjectionTool().run(
                    None,
                    ImageDocument(),
                    parameters={"model_dirs": [str(folder)], "methods": ["UMAP"]},
                )


if __name__ == "__main__":
    unittest.main()
