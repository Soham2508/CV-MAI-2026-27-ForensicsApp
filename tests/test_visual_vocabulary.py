"""Unit tests for the Bag of Visual Words (Visual Vocabulary) pipeline tools:
Codebook Construction, Histogram Construction, and Image Classification.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
from PIL import Image, ImageDraw

from forensics_app.core import ImageDocument
from forensics_app.tools import (
    CodebookConstructionTool,
    HistogramConstructionTool,
    ImageClassificationTool,
    build_tool_registry,
)
from forensics_app.tools.codebook_construction import (
    build_codebook,
    default_descriptor_config,
    extract_local_descriptors_from_image,
    load_codebook,
    load_visual_words_dataset,
    save_codebook,
)
from forensics_app.tools.histogram_construction import (
    assign_descriptors_to_words,
    build_and_save_histogram_datasets,
    build_histogram,
    load_histogram_dataset,
    normalize_histogram,
)
from forensics_app.tools.image_classification import (
    classify_image,
    load_classifier,
    train_classifier,
)


def _make_synthetic_class_image(
    class_kind: str, seed: int, size: int = 80
) -> Image.Image:
    """Create a synthetic textured image with distinct local patterns per class."""
    rng = np.random.default_rng(seed)
    base = rng.integers(20, 60, size=(size, size), dtype=np.uint8)
    img = Image.fromarray(base, mode="L")
    draw = ImageDraw.Draw(img)

    if class_kind == "butterfly":
        # Symmetric diagonal crosses and bright spots
        for r in range(12, size - 12, 16):
            for c in range(12, size - 12, 16):
                draw.line((c - 5, r - 5, c + 5, r + 5), fill=240, width=2)
                draw.line((c - 5, r + 5, c + 5, r - 5), fill=240, width=2)
    elif class_kind == "dalmatian":
        # High-contrast circular spots on bright background
        img = Image.fromarray(230 - base, mode="L")
        draw = ImageDraw.Draw(img)
        for r in range(10, size - 10, 14):
            for c in range(10, size - 10, 14):
                jx = int(rng.integers(-2, 3))
                jy = int(rng.integers(-2, 3))
                draw.ellipse((c + jx - 4, r + jy - 4, c + jx + 4, r + jy + 4), fill=15)
    else:
        # Kangaroo-like rectangular blocks and step edges
        for r in range(10, size - 14, 18):
            for c in range(10, size - 14, 18):
                draw.rectangle((c, r, c + 9, r + 6), outline=250, fill=190)
                draw.point((c + 4, r + 3), fill=10)

    return img.convert("RGB")


class VisualVocabularyTests(unittest.TestCase):
    """Tests for Codebook, Histogram Construction, and Image Classification."""

    def _populate_synthetic_dataset(
        self, root: Path, images_per_class: int = 4, pre_split: bool = False
    ) -> Path:
        classes = ("butterfly", "dalmatian", "kangaroo")
        seed = 100
        if pre_split:
            for split_name, count in (
                ("train", max(2, images_per_class - 1)),
                ("test", 1),
            ):
                for cname in classes:
                    cdir = root / split_name / cname
                    cdir.mkdir(parents=True, exist_ok=True)
                    for i in range(count):
                        seed += 1
                        img = _make_synthetic_class_image(cname, seed)
                        img.save(cdir / f"img_{i:02d}.png")
        else:
            for cname in classes:
                cdir = root / cname
                cdir.mkdir(parents=True, exist_ok=True)
                # Also create a hidden macOS resource file that must be ignored
                (cdir / "._hidden.png").write_bytes(b"not_an_image")
                for i in range(images_per_class):
                    seed += 1
                    img = _make_synthetic_class_image(cname, seed)
                    img.save(cdir / f"img_{i:02d}.png")
        return root

    def test_visual_vocabulary_tools_registration_and_metadata(self) -> None:
        """Verify all three Visual Vocabulary tools are registered with metadata."""
        registry = build_tool_registry()
        by_id = {t.tool_id: t for t in registry.all()}

        expected = {
            "codebook_construction": "Codebook Construction",
            "histogram_construction": "Histogram Construction",
            "image_classification": "Image Classification",
        }
        for tid, title in expected.items():
            self.assertIn(tid, by_id)
            tool = by_id[tid]
            self.assertEqual(tool.title, title)
            self.assertEqual(tool.category, "Visual Vocabulary")
            self.assertFalse(tool.requires_image)

    def test_dataset_loading_stratified_split_and_presplit_handling(self) -> None:
        """Verify flat class-folder stratified split and pre-split loading."""
        with tempfile.TemporaryDirectory() as tmp:
            flat_dir = Path(tmp) / "flat_ds"
            self._populate_synthetic_dataset(
                flat_dir, images_per_class=4, pre_split=False
            )

            ds_flat = load_visual_words_dataset(
                flat_dir, test_size=0.25, random_state=0
            )
            self.assertEqual(
                ds_flat["class_names"], ["butterfly", "dalmatian", "kangaroo"]
            )
            self.assertEqual(ds_flat["num_classes"], 3)
            self.assertEqual(len(ds_flat["train_records"]), 9)
            self.assertEqual(len(ds_flat["test_records"]), 3)

            # Verify no overlap between train and test paths (no data leakage)
            train_paths = {r["path"] for r in ds_flat["train_records"]}
            test_paths = {r["path"] for r in ds_flat["test_records"]}
            self.assertTrue(train_paths.isdisjoint(test_paths))

            # Verify pre-split layout
            presplit_dir = Path(tmp) / "presplit_ds"
            self._populate_synthetic_dataset(
                presplit_dir, images_per_class=3, pre_split=True
            )
            ds_pre = load_visual_words_dataset(presplit_dir)
            self.assertEqual(ds_pre["split_mode"], "pre_split")
            self.assertEqual(len(ds_pre["train_records"]), 6)
            self.assertEqual(len(ds_pre["test_records"]), 3)

    def test_local_descriptor_extraction_sift_and_orb_reuse(self) -> None:
        """Verify SIFT and ORB local descriptor extraction and keypoint capping."""
        img = _make_synthetic_class_image("butterfly", seed=7, size=96)

        sift_cfg = default_descriptor_config(
            method="SIFT", max_keypoints=25, n_octaves=3
        )
        kps_sift, desc_sift, gray_sift = extract_local_descriptors_from_image(
            img, sift_cfg
        )
        self.assertEqual(kps_sift.ndim, 2)
        self.assertEqual(desc_sift.ndim, 2)
        self.assertEqual(desc_sift.shape[1], 128)
        self.assertEqual(desc_sift.dtype, np.float32)
        self.assertLessEqual(len(desc_sift), 25)
        self.assertGreater(len(desc_sift), 0)
        self.assertEqual(gray_sift.mode, "L")

        orb_cfg = default_descriptor_config(method="ORB", max_keypoints=20)
        kps_orb, desc_orb, _ = extract_local_descriptors_from_image(img, orb_cfg)
        self.assertEqual(desc_orb.shape[1], 256)
        self.assertEqual(desc_orb.dtype, np.float32)
        self.assertLessEqual(len(desc_orb), 20)
        self.assertEqual(len(kps_orb), len(desc_orb))

    def test_codebook_construction_kmeans_clustering_and_persistence(self) -> None:
        """Verify K = 10 * num_classes codebook construction, gallery, and saving."""
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = self._populate_synthetic_dataset(
                Path(tmp) / "ds", images_per_class=3
            )
            model_dir = Path(tmp) / "codebook_out"

            ds = load_visual_words_dataset(ds_dir, test_size=0.33, random_state=0)
            cfg = default_descriptor_config(
                method="SIFT", max_keypoints=60, n_octaves=3
            )

            bundle = build_codebook(
                ds["train_records"],
                class_names=ds["class_names"],
                descriptor_config=cfg,
                clusters_per_class=10,
                clustering_algorithm="MiniBatchKMeans",
                random_state=0,
                test_records=ds["test_records"],
                dataset_dir=ds_dir,
            )

            # K = 10 * 3 classes = 30
            self.assertEqual(bundle["codebook"].shape, (30, 128))
            self.assertEqual(bundle["visual_word_patches"].shape[0], 30)
            self.assertEqual(len(bundle["cluster_counts"]), 30)
            self.assertEqual(
                int(np.sum(bundle["cluster_counts"])),
                bundle["metadata"]["total_descriptors"],
            )
            self.assertIsInstance(bundle["visualization_image"], Image.Image)

            saved = save_codebook(bundle, output_dir=model_dir)
            self.assertTrue(saved["codebook_path"].exists())
            self.assertTrue(saved["metadata_path"].exists())
            self.assertTrue(saved["descriptor_config_path"].exists())
            self.assertTrue(saved["visualization_path"].exists())

            reloaded = load_codebook(model_dir)
            self.assertEqual(reloaded["codebook_id"], bundle["codebook_id"])
            np.testing.assert_allclose(
                reloaded["codebook"], bundle["codebook"], atol=1e-6
            )
            self.assertEqual(reloaded["metadata"]["n_clusters"], 30)

    def test_nearest_centroid_word_assignment_and_histogram_normalization(self) -> None:
        """Verify nearest-centroid Euclidean assignment and L1/L2/raw normalization."""
        codebook = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        descriptors = np.array(
            [
                [0.1, -0.1, 0.0, 0.0],  # closest to word 0
                [9.8, 0.2, 0.0, 0.0],  # closest to word 1
                [10.2, -0.1, 0.0, 0.0],  # closest to word 1
                [0.0, 9.5, 0.0, 0.0],  # closest to word 2
            ],
            dtype=np.float32,
        )
        words, dists = assign_descriptors_to_words(descriptors, codebook)
        np.testing.assert_array_equal(words, np.array([0, 1, 1, 2]))
        self.assertEqual(len(dists), 4)
        self.assertTrue(np.all(dists < 1.0))

        counts = np.bincount(words, minlength=3)
        h_l1 = normalize_histogram(counts, "l1")
        np.testing.assert_allclose(h_l1, np.array([0.25, 0.50, 0.25], dtype=np.float32))
        self.assertAlmostEqual(float(np.sum(h_l1)), 1.0, places=6)

        h_l2 = normalize_histogram(counts, "l2")
        self.assertAlmostEqual(float(np.linalg.norm(h_l2)), 1.0, places=6)

        h_raw = normalize_histogram(counts, "none")
        np.testing.assert_allclose(h_raw, np.array([1.0, 2.0, 1.0], dtype=np.float32))

    def test_histogram_dataset_batch_generation_and_codebook_compatibility(
        self,
    ) -> None:
        """Verify batch histogram matrix generation, saving, and codebook ID check."""
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = self._populate_synthetic_dataset(
                Path(tmp) / "ds", images_per_class=3
            )
            cb_dir = Path(tmp) / "cb"
            ds = load_visual_words_dataset(ds_dir, test_size=0.33, random_state=0)
            cfg = default_descriptor_config(
                method="SIFT", max_keypoints=40, n_octaves=3
            )

            bundle = build_codebook(
                ds["train_records"],
                class_names=ds["class_names"],
                descriptor_config=cfg,
                clusters_per_class=4,
                test_records=ds["test_records"],
            )
            save_codebook(bundle, output_dir=cb_dir)
            loaded_cb = load_codebook(cb_dir)

            hist_out = build_and_save_histogram_datasets(loaded_cb, normalization="l1")
            self.assertEqual(hist_out["X_train"].shape, (6, 12))
            self.assertEqual(hist_out["y_train"].shape, (6,))
            self.assertEqual(hist_out["X_test"].shape, (3, 12))
            self.assertEqual(hist_out["y_test"].shape, (3,))

            reloaded_hist = load_histogram_dataset(
                cb_dir, expected_codebook_id=loaded_cb["codebook_id"]
            )
            np.testing.assert_allclose(reloaded_hist["X_train"], hist_out["X_train"])

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_histogram_dataset(cb_dir, expected_codebook_id="incompatible_id")

    def test_classifier_training_probability_calibration_and_image_classification(
        self,
    ) -> None:
        """Verify Linear SVM, RBF SVM, K-NN, and Random Forest classifier training."""
        rng = np.random.default_rng(42)
        class_names = ["butterfly", "dalmatian", "kangaroo"]
        # Construct separable synthetic 15-bin histograms for 3 classes (6 per class)
        X_train = np.zeros((18, 15), dtype=np.float32)
        y_train = np.repeat([0, 1, 2], 6).astype(np.int64)
        for i, cls_id in enumerate(y_train):
            X_train[i, cls_id * 5 : (cls_id + 1) * 5] = rng.uniform(0.5, 1.0, size=5)
            X_train[i] += rng.uniform(0.0, 0.05, size=15)
            X_train[i] /= np.sum(X_train[i])

        for alg in ("linear_svm", "rbf_svm", "knn", "random_forest"):
            clf_bundle = train_classifier(
                X_train,
                y_train,
                class_names=class_names,
                codebook_id="cb_test_123",
                algorithm=alg,
                X_test=X_train[:3],
                y_test=y_train[:3],
            )
            self.assertGreaterEqual(clf_bundle["config"]["train_accuracy"], 0.9)
            probs = clf_bundle["classifier"].predict_proba(X_train[:1])[0]
            self.assertEqual(len(probs), 3)
            self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
            self.assertEqual(int(np.argmax(probs)), 0)

    def test_end_to_end_visual_vocabulary_tools_pipeline(self) -> None:
        """Verify end-to-end execution of CodebookConstructionTool,
        HistogramConstructionTool, and ImageClassificationTool.
        """
        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = self._populate_synthetic_dataset(
                Path(tmp) / "ds", images_per_class=4
            )
            model_dir = Path(tmp) / "model_001"

            # Stage 1: Codebook Construction Tool (with no image initially loaded)
            doc_cb = ImageDocument()
            self.assertIsNone(doc_cb.current)
            cb_tool = CodebookConstructionTool()
            cb_res = cb_tool.run(
                MagicMock(),
                doc_cb,
                parameters={
                    "dataset_dir": str(ds_dir),
                    "output_dir": str(model_dir),
                    "test_size": 0.25,
                    "clusters_per_class": 4,
                    "descriptor_config": default_descriptor_config(
                        method="SIFT", max_keypoints=50, n_octaves=3
                    ),
                },
            )
            self.assertIsNotNone(cb_res)
            assert cb_res is not None
            self.assertIsInstance(cb_res.image, Image.Image)
            assert cb_res.image is not None
            doc_cb.apply(cb_res.image)
            self.assertIsNotNone(doc_cb.current)
            self.assertEqual(cb_res.details["Visual words"], 12)

            # Stage 2: Histogram Construction Tool (with no image initially loaded)
            doc_hist = ImageDocument()
            self.assertIsNone(doc_hist.current)
            hist_tool = HistogramConstructionTool()
            hist_res = hist_tool.run(
                MagicMock(),
                doc_hist,
                parameters={
                    "codebook_dir": str(model_dir),
                    "normalization": "l1",
                    "batch_process": True,
                },
            )
            self.assertIsNotNone(hist_res)
            assert hist_res is not None
            self.assertIsInstance(hist_res.image, Image.Image)
            assert hist_res.image is not None
            doc_hist.apply(hist_res.image)
            self.assertIsNotNone(doc_hist.current)
            self.assertIn("X_train shape", hist_res.details)

            # Stage 3: Image Classification Tool (with no image initially loaded)
            doc_clf = ImageDocument()
            self.assertIsNone(doc_clf.current)
            query_img = _make_synthetic_class_image("kangaroo", seed=999)
            clf_tool = ImageClassificationTool()
            clf_res = clf_tool.run(
                MagicMock(),
                doc_clf,
                parameters={
                    "codebook_dir": str(model_dir),
                    "train_classifier": True,
                    "algorithm": "Linear SVM",
                    "test_image": query_img,
                },
            )
            self.assertIsNotNone(clf_res)
            assert clf_res is not None
            self.assertIsInstance(clf_res.image, Image.Image)
            assert clf_res.image is not None
            doc_clf.apply(clf_res.image)
            self.assertIsNotNone(doc_clf.current)
            self.assertIn("Predicted class", clf_res.details)
            self.assertIn("Top probability", clf_res.details)
            self.assertTrue(str(clf_res.details["Top probability"]).startswith("p("))

            # Verify persisted classifier and train_raw_counts can be reloaded
            loaded_cb = load_codebook(model_dir)
            self.assertIsNotNone(loaded_cb.get("train_raw_counts"))
            loaded_clf = load_classifier(
                model_dir, expected_codebook_id=loaded_cb["codebook_id"]
            )
            direct_cls = classify_image(query_img, loaded_cb, loaded_clf)
            self.assertIn(
                direct_cls["predicted_class"], ["butterfly", "dalmatian", "kangaroo"]
            )
            self.assertGreater(direct_cls["top_probability"], 0.0)
            self.assertLessEqual(direct_cls["top_probability"], 1.0)
            self.assertIsInstance(
                build_histogram(query_img, loaded_cb)["histogram"], np.ndarray
            )


if __name__ == "__main__":
    unittest.main()
