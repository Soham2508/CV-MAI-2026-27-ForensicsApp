"""Tests for face analysis: Viola-Jones, YuNet, PCA eigenfaces, and recognition."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
from skimage import data as skimage_data

from forensics_app.core import ImageDocument
from forensics_app.tools import (
    EigenfaceTrainerTool,
    FaceDetectionTool,
    FaceRecognitionTool,
    ViolaJonesTrainerTool,
    WebcamFaceDetectionTool,
    build_tool_registry,
)
from forensics_app.tools.eigenface_trainer import (
    load_eigenface_model,
    load_knn_classifier,
)
from forensics_app.tools.face_detection import (
    LANDMARK_NAMES,
    ViolaJonesDetector,
    YuNetDetector,
    detect_faces_in_image,
)
from forensics_app.tools.viola_jones_trainer import (
    FeatureType,
    HaarLikeFeature,
    ViolaJonesCascadeModel,
    _normalize_patch,
    compute_integral_image,
    create_synthetic_face_nonface_dataset,
    ensemble_vote,
    ensemble_vote_all,
    ensemble_vote_all_t,
    ensemble_vote_t,
    evaluate_cascade,
    learn,
    load_integral_images,
    sum_region,
)
from forensics_app.tools.webcam_detection import (
    parse_resolution,
)


class FaceAnalysisTests(unittest.TestCase):
    """Test suite covering face detection, cascade learning, PCA, and recognition."""

    def test_face_analysis_tools_registration(self) -> None:
        """Verify all face analysis tools are registered with valid metadata."""
        registry = build_tool_registry()
        tools_by_id = {tool.tool_id: tool for tool in registry.all()}

        expected = (
            ("viola_jones_trainer", ViolaJonesTrainerTool, False),
            ("face_detection", FaceDetectionTool, True),
            ("webcam_face_detection", WebcamFaceDetectionTool, False),
            ("eigenface_trainer", EigenfaceTrainerTool, False),
            ("face_recognition", FaceRecognitionTool, False),
        )
        for tool_id, cls, req_img in expected:
            with self.subTest(tool_id=tool_id):
                self.assertIn(tool_id, tools_by_id)
                tool = tools_by_id[tool_id]
                self.assertIsInstance(tool, cls)
                self.assertEqual(tool.category, "Face Analysis")
                self.assertEqual(tool.requires_image, req_img)

    # ------------------------------------------------------------------
    # Viola-Jones: Integral Image, Haar Features, AdaBoost, XML Cascade
    # ------------------------------------------------------------------
    def test_integral_image_and_region_sum_calculation(self) -> None:
        """Verify integral image generation and rectangular region sum calculation."""
        img_array = np.array(
            [
                [1, 2, 2, 2, 1],
                [1, 2, 2, 2, 1],
                [1, 2, 2, 2, 1],
                [1, 2, 2, 2, 1],
            ]
        )
        ii = compute_integral_image(img_array)
        self.assertEqual(ii.shape, (5, 6))
        self.assertTrue(np.all(ii[0, :] == 0))
        self.assertTrue(np.all(ii[:, 0] == 0))

        # Validate sum of inner sub-rectangle and total image area
        self.assertEqual(sum_region(ii, [1, 1], [3, 4]), 12.0)
        self.assertEqual(sum_region(ii, [0, 0], [-1, -1]), 32.0)

    def test_patch_normalization_and_haar_stump_batch_scoring(self) -> None:
        """Verify patch normalization validation and Haar feature batch evaluation."""
        with self.assertRaises(ValueError):
            _normalize_patch(np.zeros((0, 0), dtype=np.float32))
        with self.assertRaises(ValueError):
            _normalize_patch(np.zeros((10, 10, 3), dtype=np.uint8))

        uint8_arr = np.full((10, 10), 255, dtype=np.uint8)
        norm_arr = _normalize_patch(uint8_arr)
        self.assertEqual(norm_arr.shape, (10, 10))
        self.assertAlmostEqual(float(norm_arr.max()), 1.0)
        self.assertEqual(norm_arr.dtype, np.float32)

        # Test Haar feature batch score vs single score
        img1 = np.ones((12, 12), dtype=np.float32) * 0.2
        img2 = np.ones((12, 12), dtype=np.float32) * 0.8
        ii1 = compute_integral_image(img1)
        ii2 = compute_integral_image(img2)
        ii_stack = np.stack([ii1, ii2])

        feature = HaarLikeFeature(
            FeatureType.TWO_HORIZONTAL,
            (0, 0),
            width=6,
            height=6,
            threshold=0.0,
            polarity=1,
        )
        single1 = feature.get_score(ii1)
        single2 = feature.get_score(ii2)
        batch = feature.get_score_batch(ii_stack)
        self.assertAlmostEqual(batch[0], single1, places=5)
        self.assertAlmostEqual(batch[1], single2, places=5)

        # Serialization roundtrip
        f_dict = feature.to_dict()
        reconstructed = HaarLikeFeature.from_dict(f_dict)
        self.assertEqual(reconstructed.type, feature.type)
        self.assertEqual(reconstructed.top_left, feature.top_left)
        self.assertEqual(reconstructed.width, feature.width)
        self.assertEqual(reconstructed.height, feature.height)

    def test_haar_like_feature_evaluation_and_adaboost_voting(self) -> None:
        """Verify Haar-like feature evaluation and AdaBoost ensemble voting."""
        rng = np.random.default_rng(7)
        pos_iis = []
        neg_iis = []
        for _ in range(5):
            # Top-left darker than bottom-right: score = first - second < 0 -> +weight
            pos_patch = np.ones((12, 12), dtype=np.float64) * 0.9
            pos_patch[:6, :6] = 0.1
            pos_patch += rng.uniform(0, 0.02, (12, 12))
            ii_pos = compute_integral_image(pos_patch)
            pos_iis.append(ii_pos / np.max(ii_pos))

            neg_patch = np.ones((12, 12), dtype=np.float64) * 0.1
            neg_patch[:6, :6] = 0.9
            neg_patch += rng.uniform(0, 0.02, (12, 12))
            ii_neg = compute_integral_image(neg_patch)
            neg_iis.append(ii_neg / np.max(ii_neg))

        classifiers = learn(
            pos_iis,
            neg_iis,
            num_classifiers=2,
            min_feature_width=4,
            max_feature_width=6,
            min_feature_height=4,
            max_feature_height=6,
        )
        # When a single weak learner separates perfectly, AdaBoost early-stops at 1
        self.assertGreaterEqual(len(classifiers), 1)
        self.assertLessEqual(len(classifiers), 2)
        self.assertIsInstance(classifiers[0], HaarLikeFeature)

        pos_votes = ensemble_vote_all(pos_iis, classifiers)
        neg_votes = ensemble_vote_all(neg_iis, classifiers)
        self.assertEqual(sum(pos_votes), len(pos_iis))
        self.assertEqual(sum(neg_votes), 0)

        pos_votes_t = ensemble_vote_all_t(pos_iis, classifiers, -0.1)
        self.assertEqual(len(pos_votes_t), len(pos_iis))
        self.assertEqual(ensemble_vote_t(pos_iis[0], classifiers, -0.1), 1)
        self.assertEqual(ensemble_vote(neg_iis[0], classifiers), 0)

        # Multi-classifier AdaBoost learning when error > 0 across rounds
        rng2 = np.random.default_rng(42)
        multi_pos = [
            compute_integral_image(rng2.uniform(0.3, 0.7, (16, 16))) for _ in range(10)
        ]
        multi_neg = [
            compute_integral_image(rng2.uniform(0.3, 0.7, (16, 16))) for _ in range(10)
        ]
        for p in multi_pos:
            p[:8, :8] += 0.2
        multi_clfs = learn(
            multi_pos,
            multi_neg,
            num_classifiers=3,
            min_feature_width=4,
            max_feature_width=8,
            min_feature_height=4,
            max_feature_height=8,
        )
        self.assertEqual(len(multi_clfs), 3)

    def test_viola_jones_cascade_training_serialization_and_loading(self) -> None:
        """Verify Viola-Jones cascade training, XML persistence, and model loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ds_paths = create_synthetic_face_nonface_dataset(
                root / "data", window_size=(16, 16), n_pos=8, n_neg=8
            )
            loaded_iis = load_integral_images(
                ds_paths["pos_train"], window_size=(16, 16)
            )
            self.assertGreater(len(loaded_iis), 0)

            xml_out = root / "models" / "cascade.xml"
            params = {
                "positive_dir": str(ds_paths["pos_train"]),
                "negative_dir": str(ds_paths["neg_train"]),
                "pos_test_dir": str(ds_paths["pos_test"]),
                "neg_test_dir": str(ds_paths["neg_test"]),
                "output_xml_path": str(xml_out),
                "window_width": 16,
                "window_height": 16,
                "num_pos": 6,
                "num_neg": 6,
                "num_stages": 2,
                "max_weak_count": 2,
                "min_feature_width": 6,
                "max_feature_width": 8,
                "min_feature_height": 6,
                "max_feature_height": 8,
            }

            tool = ViolaJonesTrainerTool()
            result = tool.run(MagicMock(), ImageDocument(), parameters=params)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertIsNotNone(result.image)
            self.assertTrue(xml_out.exists())
            self.assertTrue((xml_out.parent / "training_config.json").exists())
            self.assertTrue((xml_out.parent / "training_log.txt").exists())

            reloaded = ViolaJonesCascadeModel.load_xml(xml_out)
            self.assertEqual(reloaded.window_width, 16)
            self.assertEqual(reloaded.window_height, 16)
            self.assertGreaterEqual(len(reloaded.stages), 1)
            self.assertLessEqual(len(reloaded.stages), 2)
            self.assertGreaterEqual(len(reloaded.all_classifiers), 1)

            # Test evaluate_window and evaluate_cascade on loaded model
            passes_face, _ = reloaded.evaluate_window(loaded_iis[0])
            self.assertTrue(passes_face)

            cascade_eval = evaluate_cascade(reloaded, loaded_iis[:4], loaded_iis[:4])
            self.assertIn("accuracy_pct", cascade_eval)
            self.assertIn("face_recall_pct", cascade_eval)

            # Verify ViolaJonesDetector can load the custom-trained XML
            vj_det = ViolaJonesDetector(model_path=xml_out, min_neighbors=1)
            vj_det.load_model()
            test_img = Image.new("L", (32, 32), 120)
            detections = vj_det.detect(test_img)
            self.assertIsInstance(detections, list)
            test_img = Image.new("L", (32, 32), 120)
            detections = vj_det.detect(test_img)
            self.assertIsInstance(detections, list)

    # ------------------------------------------------------------------
    # Face Detection (YuNet & Viola-Jones) & Webcam Stream
    # ------------------------------------------------------------------
    def test_yunet_face_detection_and_landmarks_extraction(self) -> None:
        """Verify YuNet ONNX face detection and facial landmarks extraction."""
        astronaut_pil = Image.fromarray(skimage_data.astronaut())
        det = YuNetDetector(confidence_threshold=0.5, nms_threshold=0.3, top_k=5000)
        detections = det.detect(astronaut_pil)

        self.assertGreaterEqual(len(detections), 1)
        first = detections[0]
        self.assertIn("bbox", first)
        self.assertIn("confidence", first)
        self.assertIn("landmarks", first)
        self.assertEqual(len(first["bbox"]), 4)
        self.assertIsNotNone(first["confidence"])
        self.assertGreater(first["confidence"], 0.5)
        self.assertEqual(len(first["landmarks"]), 5)

        # Run through FaceDetectionTool
        doc = ImageDocument()
        doc.current = astronaut_pil
        tool = FaceDetectionTool()
        result = tool.run(
            MagicMock(),
            doc,
            parameters={
                "detector_type": "YuNet",
                "confidence_threshold": 0.5,
                "nms_threshold": 0.3,
                "top_k": 5000,
                "show_landmarks": True,
                "show_confidence": True,
            },
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIsNotNone(result.image)
        self.assertEqual(result.details["Detector"], "YuNet")
        self.assertGreaterEqual(result.details["Faces detected"], 1)
        for lm_name in LANDMARK_NAMES:
            self.assertIn(lm_name, result.details)

    def test_viola_jones_detector_inference_on_static_image(self) -> None:
        """Verify Viola-Jones detector interface and bounding box detection on image."""
        astronaut_pil = Image.fromarray(skimage_data.astronaut())
        out = detect_faces_in_image(
            astronaut_pil,
            detector_type="Viola–Jones",
            scale_factor=1.15,
            min_neighbors=2,
            min_size=(30, 30),
        )
        self.assertEqual(out["detector_type"], "Viola–Jones")
        self.assertIsInstance(out["detections"], list)
        self.assertIsInstance(out["annotated_image"], Image.Image)
        for d in out["detections"]:
            self.assertIsNone(d["confidence"])
            self.assertIsNone(d["landmarks"])

    def test_webcam_face_detection_frame_processing_and_resolution_parsing(
        self,
    ) -> None:
        """Verify webcam frame processing, overlay annotation, and resolution."""
        self.assertEqual(parse_resolution("1280x720"), (1280, 720))
        self.assertEqual(parse_resolution("invalid"), (640, 480))

        astronaut_pil = Image.fromarray(skimage_data.astronaut())
        tool = WebcamFaceDetectionTool()
        res = tool.run(
            MagicMock(),
            ImageDocument(),
            test_frame=astronaut_pil,
            parameters={"detector_type": "YuNet", "confidence_threshold": 0.5},
        )
        self.assertIsNotNone(res)
        assert res is not None
        self.assertIsNotNone(res.image)
        self.assertEqual(res.details["Detector"], "YuNet")
        self.assertGreaterEqual(res.details["Faces detected"], 1)

    # ------------------------------------------------------------------
    # Eigenface / PCA Trainer, K-NN Classification & Face Recognition
    # ------------------------------------------------------------------
    def _create_synthetic_identity_dataset(self, root: Path) -> Path:
        """Create a local dataset folder with 3 distinct synthetic identities."""
        rng = np.random.default_rng(123)
        h, w = 32, 24
        identities = ["Alice_Smith", "Bob_Jones", "Carol_White"]
        for person_idx, name in enumerate(identities):
            pdir = root / name
            pdir.mkdir(parents=True, exist_ok=True)
            base_pattern = np.zeros((h, w), dtype=np.float32)
            # Give each identity a distinct spatial intensity signature
            base_pattern[
                (person_idx * 8) : (person_idx * 8 + 12),
                (person_idx * 4) : (person_idx * 4 + 14),
            ] = 200.0
            for sample_idx in range(12):
                noisy = np.clip(
                    base_pattern + rng.normal(50.0, 8.0, (h, w)), 0, 255
                ).astype(np.uint8)
                Image.fromarray(noisy, mode="L").save(
                    pdir / f"{name}_{sample_idx:02d}.png"
                )
        return root

    def test_eigenface_pca_training_knn_classification_and_face_recognition(
        self,
    ) -> None:
        """Verify PCA eigenface training, K-NN fitting, and recognition pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_dir = self._create_synthetic_identity_dataset(tmp / "identities")
            export_dir = tmp / "exported_split"
            model_dir = tmp / "eigenface_model"

            # 1. Train PCA & K-NN via EigenfaceTrainerTool
            trainer_tool = EigenfaceTrainerTool()
            train_res = trainer_tool.run(
                MagicMock(),
                ImageDocument(),
                parameters={
                    "source": "Local",
                    "folder_path": str(ds_dir),
                    "export_dir": str(export_dir),
                    "model_dir": str(model_dir),
                    "n_components": 10,
                    "svd_solver": "full",
                    "whiten": True,
                    "train_knn": True,
                    "use_grid_search": True,
                    "k_candidates": [1, 3, 5],
                    "cv_folds": 3,
                },
            )
            self.assertIsNotNone(train_res)
            assert train_res is not None
            self.assertIsNotNone(train_res.image)
            self.assertEqual(train_res.details["PCA components"], 10)
            self.assertIn("Test accuracy", train_res.details)

            # Verify required model files exist on disk
            self.assertTrue((model_dir / "pca_model.npz").exists())
            self.assertTrue((model_dir / "metadata.json").exists())
            self.assertTrue((model_dir / "training_embeddings.npy").exists())
            self.assertTrue((model_dir / "training_labels.npy").exists())
            self.assertTrue((model_dir / "training_image_paths.json").exists())
            self.assertTrue((model_dir / "eigenfaces_gallery.png").exists())
            self.assertTrue((model_dir / "knn_classifier.joblib").exists())
            self.assertTrue((export_dir / "correspondence.json").exists())

            # 2. Reload persisted models and run FaceRecognitionTool
            loaded_pca = load_eigenface_model(model_dir)
            loaded_knn = load_knn_classifier(
                model_dir / "knn_classifier.joblib",
                expected_pca_model_id=loaded_pca["model_id"],
            )
            self.assertEqual(loaded_pca["n_components"], 10)
            self.assertEqual(loaded_knn["pca_model_id"], loaded_pca["model_id"])

            # Verify incompatible PCA model_id raises ValueError
            with self.assertRaises(ValueError):
                load_knn_classifier(
                    model_dir / "knn_classifier.joblib",
                    expected_pca_model_id="wrong_model_id",
                )

            test_rec = loaded_pca["test_records"][0]
            test_img_path = Path(test_rec["path"])
            true_person_name = test_rec["person_name"]

            rec_tool = FaceRecognitionTool()
            rec_res = rec_tool.run(
                MagicMock(),
                ImageDocument(),
                parameters={
                    "pca_model_path": str(model_dir),
                    "knn_model_path": str(model_dir / "knn_classifier.joblib"),
                    "test_image_path": str(test_img_path),
                    "n_nearest": 3,
                },
            )
            self.assertIsNotNone(rec_res)
            assert rec_res is not None
            self.assertIsNotNone(rec_res.image)
            self.assertEqual(rec_res.details["Predicted identity"], true_person_name)
            self.assertEqual(rec_res.details["Ground-truth identity"], true_person_name)
            self.assertIn("Match 1", rec_res.details)
            self.assertIn("Match 2", rec_res.details)
            self.assertIn("Match 3", rec_res.details)


if __name__ == "__main__":
    unittest.main()
