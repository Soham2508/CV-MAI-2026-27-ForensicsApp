"""Unit tests for Set 8 CNN Classification tools:
CNNTrainerTool (8.1), CNNClassificationTool (8.2), and CNNFeatureProjectionTool (8.3).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

from forensics_app.core import ImageDocument
from forensics_app.tools import (
    CNNClassificationTool,
    CNNFeatureProjectionTool,
    CNNTrainerTool,
    build_tool_registry,
)
from forensics_app.tools.cnn_common import (
    build_cnn_model,
    build_preprocessing_spec,
    format_passport_text,
    load_cnn_bundle,
    preprocess_pil_image,
)
from forensics_app.tools.cnn_feature_projection import (
    extract_and_save_test_features,
    load_extracted_features,
    project_features_2d,
)
from forensics_app.tools.cnn_trainer import train_cnn_model


def _make_synthetic_rgb(kind: str, seed: int, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(15, 55, size=(size, size, 3), dtype=np.uint8)
    if kind == "circles":
        arr[:, :, 0] = np.clip(arr[:, :, 0] + 150, 0, 255)
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        draw.ellipse((14, 14, size - 14, size - 14), fill=(240, 220, 60))
    elif kind == "stripes":
        arr[:, :, 1] = np.clip(arr[:, :, 1] + 140, 0, 255)
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        for y in range(4, size, 10):
            draw.line((0, y, size, y), fill=(20, 240, 220), width=3)
    else:
        arr[:, :, 2] = np.clip(arr[:, :, 2] + 150, 0, 255)
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle((12, 12, size - 12, size - 12), outline=(255, 255, 255), width=4)
    return img


def _populate_dataset(root: Path, images_per_class: int = 8) -> Path:
    seed = 10
    for cname in ("circles", "stripes", "squares"):
        cdir = root / cname
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            seed += 1
            _make_synthetic_rgb(cname, seed).save(cdir / f"{cname}_{i:02d}.png")
    return root


class CNNToolsTests(unittest.TestCase):
    """Tests for Set 8 CNN training, passport/inference, and t-SNE/UMAP projection."""

    def test_cnn_tools_registered_with_expected_metadata(self) -> None:
        registry = build_tool_registry()
        by_id = {t.tool_id: t for t in registry.all()}
        expected = {
            "cnn_trainer": "Train Classification CNN",
            "cnn_classifier": "CNN Image Classification",
            "cnn_feature_projection": "CNN Feature Projection (t-SNE / UMAP)",
        }
        for tid, title in expected.items():
            self.assertIn(tid, by_id)
            self.assertEqual(by_id[tid].title, title)
            self.assertEqual(by_id[tid].category, "CNN Classification")
            self.assertFalse(by_id[tid].requires_image)

    def test_architectures_forward_and_feature_shapes(self) -> None:
        specs = [
            ("shallow_cnn", 32, 100),
            ("deeper_vgg", 32, 128),
            ("complete_vgg", 64, 128),
            ("inception_v3", 128, 1024),
            ("resnet50", 64, 2048),
        ]
        for arch_key, size, feat_dim in specs:
            with self.subTest(arch_key=arch_key):
                model = build_cnn_model(
                    arch_key,
                    num_classes=4,
                    input_size=size,
                    pretrained=False,
                    freeze_backbone=False,
                )
                model.eval()
                x = torch.randn(2, 3, size, size)
                with torch.no_grad():
                    logits = model(x)
                    feats = model.extract_features(x)
                self.assertEqual(tuple(logits.shape), (2, 4))
                self.assertEqual(tuple(feats.shape), (2, feat_dim))

    def test_preprocessing_spec_and_image_tensor_shape(self) -> None:
        img = Image.new("L", (40, 30), 128)
        prep_unit = build_preprocessing_spec(input_size=32, normalization="unit_0_1")
        t_unit = preprocess_pil_image(img, prep_unit)
        self.assertEqual(tuple(t_unit.shape), (3, 32, 32))
        self.assertAlmostEqual(float(t_unit.mean()), 128.0 / 255.0, places=3)

        prep_im = build_preprocessing_spec(input_size=64, normalization="imagenet")
        t_im = preprocess_pil_image(img, prep_im)
        self.assertEqual(tuple(t_im.shape), (3, 64, 64))
        self.assertTrue(torch.isfinite(t_im).all())

    def test_end_to_end_cnn_train_classify_and_feature_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_dir = _populate_dataset(tmp / "dataset", images_per_class=8)
            model_dir = tmp / "models" / "cnn_test_001"

            # Set 8.1: Train & Save weights + passport
            doc_train = ImageDocument()
            train_res = CNNTrainerTool().run(
                None,
                doc_train,
                parameters={
                    "arch_key": "deeper_vgg",
                    "input_size": 32,
                    "normalization": "unit_0_1",
                    "source_type": "image_folder",
                    "dataset_dir": str(ds_dir),
                    "test_size": 0.25,
                    "val_split": 0.15,
                    "epochs": 4,
                    "batch_size": 8,
                    "optimizer": "Adam",
                    "learning_rate": 0.005,
                    "random_state": 42,
                    "device": "auto",
                    "model_dir": str(model_dir),
                },
            )
            self.assertIsNotNone(train_res)
            self.assertIsNotNone(train_res.image)
            self.assertTrue((model_dir / "weights.pt").is_file())
            self.assertTrue((model_dir / "passport.json").is_file())
            self.assertTrue((model_dir / "training_curves.png").is_file())

            bundle = load_cnn_bundle(model_dir, device_preference="auto")
            passport = bundle["passport"]
            self.assertEqual(passport["class_names"], ["circles", "squares", "stripes"])
            self.assertEqual(passport["dataset"]["sample_size"]["total"], 24)
            self.assertEqual(len(passport["training_history"]["epoch"]), 4)
            passport_txt = format_passport_text(passport)
            self.assertIn("Preprocessing Instructions", passport_txt)
            self.assertIn("Training History", passport_txt)

            # Set 8.2: Individual held-out test image
            doc_ind = ImageDocument()
            ind_res = CNNClassificationTool().run(
                None,
                doc_ind,
                parameters={
                    "model_source": "saved",
                    "model_dir": str(model_dir),
                    "device": "auto",
                    "classify_mode": "held_out_individual",
                    "test_record_index": 0,
                },
            )
            self.assertIsNotNone(ind_res)
            self.assertIn("Predicted class", ind_res.details)
            self.assertIn("True class", ind_res.details)

            # Set 8.2: Group of held-out test images
            doc_grp = ImageDocument()
            grp_res = CNNClassificationTool().run(
                None,
                doc_grp,
                parameters={
                    "model_source": "saved",
                    "model_dir": str(model_dir),
                    "device": "auto",
                    "classify_mode": "held_out_group",
                    "class_filter": "All",
                    "group_max_images": 0,
                },
            )
            self.assertIsNotNone(grp_res)
            self.assertEqual(grp_res.details["Images classified"], 6)
            self.assertIn("Group accuracy", grp_res.details)

            # Set 8.2: Completely new image (on canvas and as file)
            new_img = _make_synthetic_rgb("stripes", seed=999)
            new_path = tmp / "brand_new.png"
            new_img.save(new_path)

            doc_new = ImageDocument()
            doc_new.current = new_img
            new_res = CNNClassificationTool().run(
                None,
                doc_new,
                parameters={
                    "model_source": "saved",
                    "model_dir": str(model_dir),
                    "device": "auto",
                    "classify_mode": "new_file",
                    "custom_image_path": str(new_path),
                },
            )
            self.assertIsNotNone(new_res)
            self.assertNotIn("True class", new_res.details)

            # Set 8.3: Extract & save test features, reload, and project (t-SNE & UMAP)
            feat_out = extract_and_save_test_features(
                model_source="saved",
                model_dir=model_dir,
                device_preference="auto",
            )
            self.assertTrue((model_dir / "features_test.npz").is_file())
            self.assertEqual(feat_out["features"].shape, (6, 128))

            reloaded = load_extracted_features(model_dir / "features_test.npz")
            self.assertEqual(reloaded["features"].shape, (6, 128))

            doc_proj = ImageDocument()
            proj_res = CNNFeatureProjectionTool().run(
                None,
                doc_proj,
                parameters={
                    "feature_source": "load_saved",
                    "features_path": str(model_dir / "features_test.npz"),
                    "method": "both",
                    "perplexity": 30.0,
                    "tsne_max_iter": 250,
                    "umap_n_neighbors": 15,
                    "umap_min_dist": 0.1,
                    "umap_metric": "euclidean",
                    "random_state": 42,
                },
            )
            self.assertIsNotNone(proj_res)
            self.assertIsNotNone(proj_res.image)
            self.assertTrue((model_dir / "projection_2d.png").is_file())
            self.assertIn("t-SNE", proj_res.details["Methods"])
            self.assertIn("UMAP", proj_res.details["Methods"])

    def test_transfer_learning_two_stage_inception_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds_dir = _populate_dataset(tmp / "ds_inc", images_per_class=4)
            model_dir = tmp / "inc_model"
            out = train_cnn_model(
                {
                    "arch_key": "inception_v3",
                    "input_size": 128,
                    "normalization": "imagenet",
                    "pretrained": False,
                    "freeze_backbone": True,
                    "source_type": "image_folder",
                    "dataset_dir": str(ds_dir),
                    "test_size": 0.25,
                    "val_split": 0.2,
                    "epochs": 1,
                    "fine_tune_epochs": 1,
                    "batch_size": 4,
                    "optimizer": "Adam",
                    "learning_rate": 0.001,
                    "fine_tune_lr": 0.0001,
                    "random_state": 7,
                    "device": "auto",
                    "model_dir": str(model_dir),
                }
            )
            hist = out["passport"]["training_history"]
            self.assertEqual(hist["phase"], ["train", "fine_tune"])
            self.assertTrue((model_dir / "weights.pt").is_file())

    def test_projection_rejects_fewer_than_three_samples(self) -> None:
        with self.assertRaises(ValueError):
            project_features_2d(
                np.zeros((2, 16), dtype=np.float32),
                np.array([0, 1]),
                method="both",
            )


if __name__ == "__main__":
    unittest.main()
