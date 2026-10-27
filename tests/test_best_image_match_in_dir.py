import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry
from forensics_app.tools.best_image_match_in_dir import (
    BestImageMatchInDirTool,
    compare_features,
    find_best_image_match,
)
from forensics_app.tools.image_matching import extract_descriptors


class BestImageMatchInDirTests(unittest.TestCase):
    def _create_textured_image(
        self, size: tuple[int, int] = (150, 150), seed: int = 42
    ) -> Image.Image:
        """Create a uniquely textured RGB image with high-contrast distinct features."""
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_tool_registered_in_registry(self) -> None:
        registry = build_tool_registry()
        tools = [
            t for t in registry.all() if isinstance(t, BestImageMatchInDirTool)
        ]
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool.tool_id, "best_image_match_in_dir")
        self.assertEqual(tool.category, "Matching & Descriptors")
        self.assertTrue(tool.requires_image)
        self.assertFalse(hasattr(tool, "menu_options"))

    def test_compare_features_insufficient_matches_returns_none(self) -> None:
        kp1 = np.array([[10, 10], [20, 20]], dtype=float)
        desc1 = np.ones((2, 256), dtype=bool)
        kp2 = np.array([[10, 10], [20, 20]], dtype=float)
        desc2 = np.ones((2, 256), dtype=bool)

        result = compare_features(kp1, desc1, kp2, desc2, method="ORB")
        self.assertIsNone(result)

    def test_compare_features_empty_or_mismatched_descriptors(self) -> None:
        kp = np.zeros((0, 2), dtype=float)
        empty_desc = np.zeros((0, 256), dtype=bool)
        valid_desc = np.ones((5, 256), dtype=bool)
        sift_desc = np.ones((5, 128), dtype=np.uint8)

        self.assertIsNone(compare_features(kp, empty_desc, kp, valid_desc))
        self.assertIsNone(compare_features(kp, valid_desc, kp, empty_desc))
        self.assertIsNone(compare_features(kp, valid_desc, kp, sift_desc))

    def test_compare_features_valid_homography(self) -> None:
        img1 = self._create_textured_image(seed=100)
        img2 = self._create_textured_image(seed=100)

        kp1, desc1 = extract_descriptors(img1, method="ORB", n_keypoints=150)
        kp2, desc2 = extract_descriptors(img2, method="ORB", n_keypoints=150)

        result = compare_features(kp1, desc1, kp2, desc2, method="ORB")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result["n_inliers"], 4)
        self.assertGreater(result["inlier_ratio"], 0.0)
        self.assertIn("model", result)
        self.assertIn("inliers", result)

    def test_find_best_image_match_excludes_query_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            query_path = folder / "query.png"
            other1_path = folder / "other1.png"
            other2_path = folder / "other2.png"

            query_img = self._create_textured_image(seed=200)
            other1_img = self._create_textured_image(seed=200)  # matching image
            other2_img = self._create_textured_image(seed=300)  # different image

            query_img.save(query_path)
            other1_img.save(other1_path)
            other2_img.save(other2_path)

            best = find_best_image_match(
                query_image=query_img,
                query_path=query_path,
                folder_path=folder,
                method="ORB",
                n_keypoints=150,
            )

            self.assertIn("path", best)
            # The query itself must NOT be selected as the match
            self.assertEqual(Path(best["path"]).resolve(), other1_path.resolve())
            self.assertNotEqual(Path(best["path"]).resolve(), query_path.resolve())
            self.assertGreaterEqual(best["n_inliers"], 4)

    def test_find_best_image_match_textureless_query_reports_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            cand_path = folder / "candidate.png"
            self._create_textured_image().save(cand_path)

            blank_query = Image.new("RGB", (64, 64), (100, 100, 100))
            result = find_best_image_match(
                query_image=blank_query,
                query_path=None,
                folder_path=folder,
                method="ORB",
            )
            self.assertEqual(result["status"], "no_match")
            self.assertIn("detectable features", result["reason"])

    def test_find_best_image_match_empty_dir_reports_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            query_img = self._create_textured_image()
            result = find_best_image_match(
                query_image=query_img,
                query_path=None,
                folder_path=tmpdir,
                method="ORB",
            )
            self.assertEqual(result["status"], "no_match")

    def test_tool_run_successful_search_and_result_details(self) -> None:
        tool = BestImageMatchInDirTool()
        document = ImageDocument()

        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            query_path = folder / "query.png"
            match_path = folder / "match.png"

            query_img = self._create_textured_image(seed=500)
            match_img = self._create_textured_image(seed=500)

            query_img.save(query_path)
            match_img.save(match_path)

            document.load(query_path)

            params = {
                "method": "ORB",
                "desc_settings": {"n_keypoints": 150},
                "cross_check": True,
                "max_ratio": 0.8,
                "residual_threshold": 2.0,
                "max_trials": 500,
                "min_inliers_for_confident_match": 4,
            }

            result = tool.run(
                MagicMock(),
                document,
                folder_path=folder,
                parameters=params,
                show_plot=False,
            )

            self.assertIsNotNone(result)
            assert result is not None
            self.assertIsNotNone(result.image)
            self.assertEqual(result.image.mode, "RGB")
            # Result details MUST include the path of the matched image
            self.assertIn("Best match path", result.details)
            self.assertEqual(
                Path(result.details["Best match path"]).resolve(),
                match_path.resolve(),
            )
            self.assertEqual(result.details["Best match filename"], match_path.name)
            self.assertIn("RANSAC inliers", result.details)
            self.assertIn("Inlier ratio", result.details)
            self.assertIn("Best match found", result.message)

    def test_tool_run_cancelled_folder_selection(self) -> None:
        tool = BestImageMatchInDirTool()
        document = ImageDocument()
        document.current = self._create_textured_image()

        with patch(
            "forensics_app.tools.best_image_match_in_dir.filedialog.askdirectory",
            return_value="",
        ):
            result = tool.run(MagicMock(), document)
            self.assertIsNone(result)

    def test_tool_run_cancelled_parameters(self) -> None:
        tool = BestImageMatchInDirTool()
        document = ImageDocument()
        document.current = self._create_textured_image()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "forensics_app.tools.best_image_match_in_dir.ask_directory_match_parameters",
                return_value=None,
            ):
                result = tool.run(
                    MagicMock(),
                    document,
                    folder_path=tmpdir,
                    show_plot=False,
                )
                self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
