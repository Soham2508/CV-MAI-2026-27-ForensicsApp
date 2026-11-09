"""Tool 1 — Viola-Jones (Haar + AdaBoost) Cascade Trainer.

Implements integral images, correctly threshold-optimised Haar decision stumps,
discrete AdaBoost, stage-wise cascade training, full-cascade patch evaluation,
and custom XML/JSON persistence. The XML is a ForensicsApp format, not OpenCV's
native CascadeClassifier XML format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any
import xml.etree.ElementTree as ET

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from skimage import data as skimage_data

from forensics_app.core import ImageDocument
from forensics_app.paths import DATASETS_DIR, MODELS_DIR, resolve_path
from forensics_app.utils.background import run_with_progress

from .base import ForensicsTool, ToolResult

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm"}
DEFAULT_CASCADE_DIR = MODELS_DIR / "viola_jones"
DEFAULT_CASCADE_XML = DEFAULT_CASCADE_DIR / "cascade.xml"
# Feature columns processed at a time when searching for AdaBoost stumps.
LEARN_FEATURE_CHUNK = 2048


def enum(**enums: Any) -> type:
    """Create a simple enumeration type matching the reference notebook."""
    return type("Enum", (), enums)


FeatureType = enum(
    TWO_VERTICAL=(1, 2),
    TWO_HORIZONTAL=(2, 1),
    THREE_HORIZONTAL=(3, 1),
    THREE_VERTICAL=(1, 3),
    FOUR=(2, 2),
)
FeatureTypes = [
    FeatureType.TWO_VERTICAL,
    FeatureType.TWO_HORIZONTAL,
    FeatureType.THREE_VERTICAL,
    FeatureType.THREE_HORIZONTAL,
    FeatureType.FOUR,
]

FEATURE_TYPE_TO_NAME: dict[tuple[int, int], str] = {
    FeatureType.TWO_VERTICAL: "TWO_VERTICAL",
    FeatureType.TWO_HORIZONTAL: "TWO_HORIZONTAL",
    FeatureType.THREE_VERTICAL: "THREE_VERTICAL",
    FeatureType.THREE_HORIZONTAL: "THREE_HORIZONTAL",
    FeatureType.FOUR: "FOUR",
}
NAME_TO_FEATURE_TYPE: dict[str, tuple[int, int]] = {
    v: k for k, v in FEATURE_TYPE_TO_NAME.items()
}


def compute_integral_image(img_arr: np.ndarray) -> np.ndarray:
    """Calculate the integral image based on the original 2D image data.

    Includes an additional first row and first column of zeros, matching
    Exercise #1 in the reference notebook.
    """
    arr = np.asarray(img_arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("compute_integral_image expects a 2D array.")
    integral_image_arr = np.zeros(
        (arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64
    )
    integral_image_arr[1:, 1:] = np.cumsum(arr, axis=0)
    integral_image_arr = np.cumsum(integral_image_arr, axis=1)
    return integral_image_arr


def sum_region(
    integral_img_arr: np.ndarray,
    top_left: tuple[int, int] | list[int],
    bottom_right: tuple[int, int] | list[int],
) -> float:
    """Compute the sum of pixel intensities within a rectangle using the integral image.

    Matches Exercise #2 in the reference notebook.
    """
    a = integral_img_arr[top_left[0], top_left[1]]
    b = integral_img_arr[top_left[0], bottom_right[1]]
    c = integral_img_arr[bottom_right[0], top_left[1]]
    d = integral_img_arr[bottom_right[0], bottom_right[1]]
    return float(d - c - b + a)


def _normalize_patch(image_arr: np.ndarray) -> np.ndarray:
    """Convert grayscale pixels to a stable [0, 1] floating representation.

    Do not divide each patch by its own maximum: that makes the feature scale
    depend on a single bright pixel and can amplify noise in dark patches.
    """
    source = np.asarray(image_arr)
    arr = source.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a two-dimensional grayscale image.")
    if arr.size == 0:
        raise ValueError("Cannot normalise an empty image.")
    # Decide the scale from the data type, not the values: a very dark 8-bit
    # frame whose brightest pixel is 1 must still be divided by 255.
    if np.issubdtype(source.dtype, np.integer) or float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _integral_from_patch(image_arr: np.ndarray) -> np.ndarray:
    """Build a zero-padded integral image from a grayscale patch."""
    return compute_integral_image(_normalize_patch(image_arr))


def load_integral_images(
    path: str | Path,
    *,
    window_size: tuple[int, int] | None = None,
    max_samples: int | None = None,
    random_seed: int = 42,
) -> list[np.ndarray]:
    """Load cropped face/non-face patches and return their integral images.

    ``window_size`` is (width, height). When provided, every patch is resized
    to the same detector window. Files are shuffled deterministically so that
    a sample limit does not systematically select the alphabetically first
    images.
    """
    folder = resolve_path(path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Image directory does not exist: {folder}")

    files = [
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    rng = np.random.default_rng(random_seed)
    rng.shuffle(files)
    if max_samples is not None and max_samples > 0:
        files = files[:max_samples]

    ii_images: list[np.ndarray] = []
    for file_path in files:
        try:
            with Image.open(file_path) as img:
                gray = img.convert("L")
                if window_size is not None:
                    gray = gray.resize(window_size, Image.Resampling.BILINEAR)
                arr = np.asarray(gray, dtype=np.uint8)
        except (OSError, ValueError):
            continue
        ii_images.append(_integral_from_patch(arr))
    return ii_images


class HaarLikeFeature:
    """One Haar-like feature used as an AdaBoost decision stump.

    ``polarity=+1`` predicts face for score >= threshold.
    ``polarity=-1`` predicts face for score < threshold.
    ``weight`` is the AdaBoost coefficient (alpha).
    """

    def __init__(
        self,
        feature_type: tuple[int, int],
        position: tuple[int, int],
        width: int,
        height: int,
        threshold: float,
        polarity: int,
        weight: float = 1.0,
    ) -> None:
        if polarity not in (-1, 1):
            raise ValueError("polarity must be -1 or +1")
        self.type = tuple(feature_type)
        self.top_left = (int(position[0]), int(position[1]))  # row, column
        self.width = int(width)
        self.height = int(height)
        self.bottom_right = (
            self.top_left[0] + self.height,
            self.top_left[1] + self.width,
        )
        self.threshold = float(threshold)
        self.polarity = int(polarity)
        self.weight = float(weight)

    @staticmethod
    def _rect_sum_batch(
        ii_stack: np.ndarray,
        top: int,
        left: int,
        bottom: int,
        right: int,
    ) -> np.ndarray:
        return (
            ii_stack[:, bottom, right]
            - ii_stack[:, top, right]
            - ii_stack[:, bottom, left]
            + ii_stack[:, top, left]
        )

    def get_score_batch(self, ii_stack: np.ndarray) -> np.ndarray:
        """Evaluate this feature for a stack of integral images."""
        r, c = self.top_left
        w, h = self.width, self.height
        if self.type == FeatureType.TWO_VERTICAL:
            half = h // 2
            return self._rect_sum_batch(
                ii_stack, r, c, r + half, c + w
            ) - self._rect_sum_batch(ii_stack, r + half, c, r + h, c + w)
        if self.type == FeatureType.TWO_HORIZONTAL:
            half = w // 2
            return self._rect_sum_batch(
                ii_stack, r, c, r + h, c + half
            ) - self._rect_sum_batch(ii_stack, r, c + half, r + h, c + w)
        if self.type == FeatureType.THREE_HORIZONTAL:
            third = w // 3
            return (
                self._rect_sum_batch(ii_stack, r, c, r + h, c + third)
                - self._rect_sum_batch(ii_stack, r, c + third, r + h, c + 2 * third)
                + self._rect_sum_batch(ii_stack, r, c + 2 * third, r + h, c + w)
            )
        if self.type == FeatureType.THREE_VERTICAL:
            third = h // 3
            return (
                self._rect_sum_batch(ii_stack, r, c, r + third, c + w)
                - self._rect_sum_batch(ii_stack, r + third, c, r + 2 * third, c + w)
                + self._rect_sum_batch(ii_stack, r + 2 * third, c, r + h, c + w)
            )
        if self.type == FeatureType.FOUR:
            half_h, half_w = h // 2, w // 2
            return (
                self._rect_sum_batch(ii_stack, r, c, r + half_h, c + half_w)
                - self._rect_sum_batch(ii_stack, r, c + half_w, r + half_h, c + w)
                - self._rect_sum_batch(ii_stack, r + half_h, c, r + h, c + half_w)
                + self._rect_sum_batch(ii_stack, r + half_h, c + half_w, r + h, c + w)
            )
        raise ValueError(f"Unsupported Haar feature type: {self.type}")

    def get_score(self, int_img: np.ndarray) -> float:
        """Evaluate the feature on one zero-padded integral image."""
        return float(self.get_score_batch(np.asarray(int_img)[None, ...])[0])

    def predict_from_score(self, scores: np.ndarray | float) -> np.ndarray:
        """Return +1 for face and -1 for non-face."""
        values = np.asarray(scores)
        if self.polarity == 1:
            return np.where(values >= self.threshold, 1, -1)
        return np.where(values < self.threshold, 1, -1)

    def get_vote(self, int_img: np.ndarray) -> float:
        pred = int(self.predict_from_score(self.get_score(int_img)))
        return float(self.weight * pred)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_type": FEATURE_TYPE_TO_NAME.get(self.type, "TWO_VERTICAL"),
            "position": list(self.top_left),
            "width": self.width,
            "height": self.height,
            "threshold": self.threshold,
            "polarity": self.polarity,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HaarLikeFeature":
        ftype_raw = data.get("feature_type", "TWO_VERTICAL")
        ftype = (
            NAME_TO_FEATURE_TYPE.get(ftype_raw, FeatureType.TWO_VERTICAL)
            if isinstance(ftype_raw, str)
            else tuple(ftype_raw)
        )
        pos = tuple(data["position"])
        return cls(
            feature_type=ftype,
            position=(int(pos[0]), int(pos[1])),
            width=int(data["width"]),
            height=int(data["height"]),
            threshold=float(data.get("threshold", 0.0)),
            polarity=int(data.get("polarity", 1)),
            weight=float(data.get("weight", 1.0)),
        )


def _create_features(
    img_height: int,
    img_width: int,
    min_feature_width: int,
    max_feature_width: int,
    min_feature_height: int,
    max_feature_height: int,
    feature_mode: str = "BASIC",
    *,
    max_features: int = 3000,
    random_seed: int = 42,
) -> list[HaarLikeFeature]:
    """Generate valid Haar rectangles and cap their number reproducibly.

    Width/height maxima are inclusive. Polarity is learned by AdaBoost, so each
    geometric feature is generated once (not duplicated for both polarities).
    """
    mode = feature_mode.upper()
    if mode not in {"BASIC", "CORE", "ALL"}:
        raise ValueError("feature_mode must be BASIC, CORE or ALL")
    active_types = FeatureTypes if mode in {"BASIC", "ALL"} else FeatureTypes[:-1]

    min_fw = max(2, int(min_feature_width))
    min_fh = max(2, int(min_feature_height))
    max_fw = min(int(max_feature_width), img_width)
    max_fh = min(int(max_feature_height), img_height)

    # Enumerate (type, width, height) groups in the same nested order as a
    # full listing; each group has one feature per top-left position, row by
    # row. Only the sampled indices are turned into objects, so the defaults
    # build 3,000 features instead of all ~48,000 candidates.
    groups: list[tuple[tuple[int, int], int, int, int]] = []
    for ftype in active_types:
        unit_w, unit_h = ftype
        for width in range(min_fw, max_fw + 1):
            if width % unit_w:
                continue
            for height in range(min_fh, max_fh + 1):
                if height % unit_h:
                    continue
                n_rows = img_height - height + 1
                n_cols = img_width - width + 1
                if n_rows > 0 and n_cols > 0:
                    groups.append((ftype, width, height, n_cols))
    counts = np.asarray(
        [(img_height - h + 1) * n_cols for _, _, h, n_cols in groups], dtype=np.int64
    )
    total = int(counts.sum()) if len(counts) else 0
    if total == 0:
        raise ValueError(
            "No Haar features fit the training window. Increase feature bounds "
            "or use a larger window."
        )
    if max_features > 0 and total > max_features:
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(total, size=max_features, replace=False)
    else:
        selected = np.arange(total)

    offsets = np.concatenate([[0], np.cumsum(counts)])
    group_of = np.searchsorted(offsets, selected, side="right") - 1
    features: list[HaarLikeFeature] = []
    for index, group in zip(selected, group_of):
        ftype, width, height, n_cols = groups[int(group)]
        row, col = divmod(int(index - offsets[group]), n_cols)
        features.append(HaarLikeFeature(ftype, (row, col), width, height, 0.0, 1))
    return features


def learn(
    positive_iis: list[np.ndarray],
    negative_iis: list[np.ndarray],
    num_classifiers: int = -1,
    min_feature_width: int = 2,
    max_feature_width: int = 12,
    min_feature_height: int = 2,
    max_feature_height: int = 12,
    *,
    feature_mode: str = "BASIC",
    max_features: int = 3000,
    weight_trimming_rate: float = 1.0,
    random_seed: int = 42,
) -> list[HaarLikeFeature]:
    """Train discrete-AdaBoost decision stumps over Haar feature responses.

    Each weak learner optimises both its threshold and polarity by weighted
    classification error. The feature responses are computed once and sorted
    once, then the sample distribution is updated between boosting rounds.
    """
    if not positive_iis or not negative_iis:
        raise ValueError("Both positive and negative integral-image sets are required.")
    if not 0.0 < weight_trimming_rate <= 1.0:
        raise ValueError("weight_trimming_rate must be in (0, 1].")

    all_ii = np.stack(positive_iis + negative_iis).astype(np.float32, copy=False)
    height, width = all_ii.shape[1] - 1, all_ii.shape[2] - 1
    if any(ii.shape != all_ii.shape[1:] for ii in positive_iis + negative_iis):
        raise ValueError("All integral images must have identical dimensions.")

    features = _create_features(
        height,
        width,
        min_feature_width,
        max_feature_width,
        min_feature_height,
        max_feature_height,
        feature_mode,
        max_features=max_features,
        random_seed=random_seed,
    )
    # Matrix shape: samples x candidate features. float32 keeps memory moderate.
    scores = np.empty((all_ii.shape[0], len(features)), dtype=np.float32)
    for j, feature in enumerate(features):
        scores[:, j] = feature.get_score_batch(all_ii)

    n_pos, n_neg = len(positive_iis), len(negative_iis)
    labels = np.concatenate(
        [np.ones(n_pos, dtype=np.int8), -np.ones(n_neg, dtype=np.int8)]
    )
    weights = np.concatenate(
        [np.full(n_pos, 0.5 / n_pos), np.full(n_neg, 0.5 / n_neg)]
    ).astype(np.float64)
    n_learners = (
        len(features)
        if num_classifiers is None or num_classifiers < 0
        else min(int(num_classifiers), len(features))
    )

    # Feature order is reusable across boosting iterations.
    order = np.argsort(scores, axis=0, kind="mergesort")
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    learners: list[HaarLikeFeature] = []
    unavailable = np.zeros(len(features), dtype=bool)
    eps = 1e-10
    n_features = len(features)
    # Error tables are (samples + 1) x features float64 arrays; building them
    # for all features at once needs several GB at 50,000 features. Columns
    # are independent, so they are processed in chunks with the same result.
    # Each chunk's sample order, sorted labels and valid split points never
    # change between boosting rounds, so they are prepared once, contiguously.
    chunks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, n_features, LEARN_FEATURE_CHUNK):
        cols = slice(start, min(start + LEARN_FEATURE_CHUNK, n_features))
        order_c = np.ascontiguousarray(order[:, cols])
        sorted_c = sorted_scores[:, cols]
        invalid_c = np.zeros((labels.size + 1, order_c.shape[1]), dtype=bool)
        if labels.size > 1:
            invalid_c[1:-1, :] = ~(sorted_c[:-1, :] < sorted_c[1:, :])
        chunks.append((start, order_c, labels[order_c], invalid_c))
    del order

    for _ in range(n_learners):
        active_weights = weights.copy()
        if weight_trimming_rate < 1.0:
            ranked = np.argsort(active_weights)[::-1]
            cumulative = np.cumsum(active_weights[ranked])
            keep_count = int(
                np.searchsorted(cumulative, weight_trimming_rate, side="left") + 1
            )
            keep = np.zeros(len(weights), dtype=bool)
            keep[ranked[:keep_count]] = True
            for cls in (-1, 1):
                class_idx = np.flatnonzero(labels == cls)
                if class_idx.size and not np.any(keep[class_idx]):
                    keep[class_idx[np.argmax(active_weights[class_idx])]] = True
            active_weights *= keep
        active_weights /= max(active_weights.sum(), eps)
        # Best (error, split, feature) for each polarity; ties go to the
        # smallest split, then the smallest feature, exactly like an argmin
        # over the full (split, feature) table in row-major order.
        best_right = (np.inf, 0, 0)
        best_left = (np.inf, 0, 0)
        for start, order_c, labels_c, invalid_c in chunks:
            cols = slice(start, start + order_c.shape[1])
            sw = active_weights[order_c]
            pos_w = np.where(labels_c == 1, sw, 0.0)
            neg_w = np.where(labels_c == -1, sw, 0.0)
            width_c = pos_w.shape[1]
            cp = np.vstack([np.zeros((1, width_c)), np.cumsum(pos_w, axis=0)])
            cn = np.vstack([np.zeros((1, width_c)), np.cumsum(neg_w, axis=0)])
            tp, tn = pos_w.sum(axis=0), neg_w.sum(axis=0)
            err_pos_right = cp + (tn[None, :] - cn)
            err_pos_left = cn + (tp[None, :] - cp)
            err_pos_right[invalid_c] = np.inf
            err_pos_left[invalid_c] = np.inf
            err_pos_right[:, unavailable[cols]] = np.inf
            err_pos_left[:, unavailable[cols]] = np.inf
            for table, polarity_key in ((err_pos_right, 1), (err_pos_left, -1)):
                split_c, col_c = np.unravel_index(np.argmin(table), table.shape)
                candidate = (
                    float(table[split_c, col_c]),
                    int(split_c),
                    start + int(col_c),
                )
                if polarity_key == 1:
                    best_right = min(best_right, candidate)
                else:
                    best_left = min(best_left, candidate)

        if best_right[0] <= best_left[0]:
            error, split, feature_idx = best_right
            polarity = 1
        else:
            error, split, feature_idx = best_left
            polarity = -1

        if not np.isfinite(error) or error >= 0.5 - 1e-12:
            break  # No remaining weak learner is better than random guessing.

        column = sorted_scores[:, feature_idx]
        if split == 0:
            threshold = float(np.nextafter(column[0], -np.inf))
        elif split == len(labels):
            threshold = float(np.nextafter(column[-1], np.inf))
        else:
            prev_v = column[split - 1]
            threshold = float(prev_v + (column[split] - prev_v) / 2.0)

        alpha = 0.5 * np.log((1.0 - max(error, eps)) / max(error, eps))
        learner = HaarLikeFeature(
            features[feature_idx].type,
            features[feature_idx].top_left,
            features[feature_idx].width,
            features[feature_idx].height,
            threshold,
            polarity,
            float(alpha),
        )
        learners.append(learner)
        predictions = learner.predict_from_score(scores[:, feature_idx]).astype(np.int8)
        weights *= np.exp(-alpha * labels * predictions)
        weights /= max(weights.sum(), eps)
        unavailable[feature_idx] = True

        if error <= eps:
            break  # Perfect stump on this training sample; further rounds add no value.

    if not learners:
        raise RuntimeError(
            "AdaBoost could not find a Haar weak learner better than chance. "
            "Check the positive/negative data and feature-size settings."
        )
    return learners


def ensemble_vote(int_img: np.ndarray, classifiers: list[HaarLikeFeature]) -> int:
    """Return 1 if sum of classifier votes is positive (> 0), else 0 (Exercise #6)."""
    return 1 if sum(clf.get_vote(int_img) for clf in classifiers) > 0 else 0


def ensemble_vote_all(
    int_imgs: list[np.ndarray], classifiers: list[HaarLikeFeature]
) -> list[int]:
    """Compute ensemble_vote for each integral image in int_imgs (Exercise #6)."""
    return [ensemble_vote(int_img, classifiers) for int_img in int_imgs]


def ensemble_vote_t(
    int_img: np.ndarray, classifiers: list[HaarLikeFeature], th: float
) -> int:
    """Return 1 if sum of classifier votes exceeds th, else 0 (Exercise #8)."""
    return 1 if sum(clf.get_vote(int_img) for clf in classifiers) > th else 0


def ensemble_vote_all_t(
    int_images: list[np.ndarray], classifiers: list[HaarLikeFeature], th: float
) -> list[int]:
    """Compute ensemble_vote_t for each integral image in int_images (Exercise #8)."""
    return [ensemble_vote_t(int_img, classifiers, th) for int_img in int_images]


def _vote_matrix(
    classifiers: list[HaarLikeFeature], ii_stack: np.ndarray
) -> np.ndarray:
    """Weighted votes (alpha * ±1) with one row per sample, one column per stump.

    Each entry equals ``clf.get_vote(ii)``; a running sum along a row gives
    the same totals as summing ``get_vote`` one classifier at a time.
    """
    votes = np.empty((ii_stack.shape[0], len(classifiers)), dtype=np.float64)
    for j, clf in enumerate(classifiers):
        votes[:, j] = clf.weight * clf.predict_from_score(clf.get_score_batch(ii_stack))
    return votes


def _strong_scores(
    int_imgs: list[np.ndarray] | None, classifiers: list[HaarLikeFeature]
) -> np.ndarray:
    """Sum of all classifiers' votes for each integral image."""
    if not int_imgs:
        return np.zeros(0, dtype=np.float64)
    if not classifiers:
        return np.zeros(len(int_imgs), dtype=np.float64)
    return np.cumsum(_vote_matrix(classifiers, np.stack(int_imgs)), axis=1)[:, -1]


def _cascade_passes(
    model: "ViolaJonesCascadeModel", int_imgs: list[np.ndarray]
) -> np.ndarray:
    """Boolean per integral image: does it pass every cascade stage?

    Same decisions as ``model.evaluate_window`` for each image, computed for
    all images at once stage by stage (rejected images skip later stages).
    """
    passed = np.zeros(len(int_imgs), dtype=bool)
    if not int_imgs or not model.stages:
        return passed
    stack = np.stack(int_imgs)
    alive = np.arange(len(int_imgs))
    for stage in model.stages:
        stage_score = np.zeros(len(alive), dtype=np.float64)
        for clf in stage.classifiers:
            stage_score += clf.weight * clf.predict_from_score(
                clf.get_score_batch(stack[alive])
            )
        alive = alive[stage_score >= stage.stage_threshold]
        if alive.size == 0:
            return passed
    passed[alive] = True
    return passed


def evaluate_thresholds(
    ii_faces_training: list[np.ndarray],
    ii_non_faces_training: list[np.ndarray],
    ii_faces_testing: list[np.ndarray] | None,
    ii_non_faces_testing: list[np.ndarray] | None,
    classifiers: list[HaarLikeFeature],
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate face and non-face accuracy across decision thresholds (Exercise #8)."""
    if thresholds is None:
        thresholds = np.array([x for x in range(-5, 5, 1)], dtype=np.float64) / 10.0
    else:
        thresholds = np.asarray(thresholds, dtype=np.float64)

    pcnt_faces_train: list[float] = []
    pcnt_nofaces_train: list[float] = []
    pcnt_faces_test: list[float] = []
    pcnt_nofaces_test: list[float] = []

    total_pos_train = max(1, len(ii_faces_training))
    total_neg_train = max(1, len(ii_non_faces_training))
    total_pos_test = max(1, len(ii_faces_testing)) if ii_faces_testing else 0
    total_neg_test = max(1, len(ii_non_faces_testing)) if ii_non_faces_testing else 0

    # Each image's strong-classifier score is computed once; a threshold only
    # changes the comparison (same decisions as ensemble_vote_t).
    pos_train = _strong_scores(ii_faces_training, classifiers)
    neg_train = _strong_scores(ii_non_faces_training, classifiers)
    pos_test = _strong_scores(ii_faces_testing, classifiers)
    neg_test = _strong_scores(ii_non_faces_testing, classifiers)

    for th in thresholds:
        correct_pos_train = int(np.count_nonzero(pos_train > float(th)))
        incorrect_neg_train = int(np.count_nonzero(neg_train > float(th)))
        correct_neg_train = len(ii_non_faces_training) - incorrect_neg_train

        pcnt_faces_train.append(100.0 * correct_pos_train / total_pos_train)
        pcnt_nofaces_train.append(100.0 * correct_neg_train / total_neg_train)

        if ii_faces_testing:
            correct_pos_test = int(np.count_nonzero(pos_test > float(th)))
            pcnt_faces_test.append(100.0 * correct_pos_test / total_pos_test)
        if ii_non_faces_testing:
            incorrect_neg_test = int(np.count_nonzero(neg_test > float(th)))
            correct_neg_test = len(ii_non_faces_testing) - incorrect_neg_test
            pcnt_nofaces_test.append(100.0 * correct_neg_test / total_neg_test)

    return {
        "thresholds": thresholds.tolist(),
        "pcnt_faces_train": pcnt_faces_train,
        "pcnt_nofaces_train": pcnt_nofaces_train,
        "pcnt_faces_test": pcnt_faces_test,
        "pcnt_nofaces_test": pcnt_nofaces_test,
    }


@dataclass
class CascadeStage:
    """A single boosted stage in a Viola-Jones cascade."""

    stage_index: int
    classifiers: list[HaarLikeFeature] = field(default_factory=list)
    stage_threshold: float = 0.0
    hit_rate: float = 1.0
    false_alarm_rate: float = 0.0

    def score(self, int_img: np.ndarray) -> float:
        return float(sum(clf.get_vote(int_img) for clf in self.classifiers))

    def passes(self, int_img: np.ndarray, threshold_offset: float = 0.0) -> bool:
        return self.score(int_img) >= (self.stage_threshold + threshold_offset)


@dataclass
class ViolaJonesCascadeModel:
    """Complete multi-stage Viola-Jones Haar + AdaBoost cascade model."""

    window_width: int = 24
    window_height: int = 24
    feature_type: str = "HAAR"
    stage_type: str = "BOOST"
    boost_type: str = "DISCRETE"
    feature_mode: str = "BASIC"
    stages: list[CascadeStage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_classifiers(self) -> list[HaarLikeFeature]:
        return [clf for stage in self.stages for clf in stage.classifiers]

    def evaluate_window(
        self, int_img: np.ndarray, threshold_offset: float = 0.0
    ) -> tuple[bool, float]:
        """Evaluate an integral image window through all cascade stages."""
        if not self.stages:
            return False, 0.0
        total_score = 0.0
        for stage in self.stages:
            s = stage.score(int_img)
            if s < stage.stage_threshold + threshold_offset:
                return False, total_score
            total_score += s
        return True, total_score

    def save_xml(self, xml_path: str | Path) -> Path:
        """Persist the trained cascade model to an XML file."""
        out_path = resolve_path(xml_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        root = ET.Element("opencv_storage")
        cascade_el = ET.SubElement(
            root, "cascade", attrib={"type_id": "forensics-viola-jones-cascade"}
        )
        ET.SubElement(cascade_el, "stageType").text = self.stage_type
        ET.SubElement(cascade_el, "featureType").text = self.feature_type
        ET.SubElement(cascade_el, "boostType").text = self.boost_type
        ET.SubElement(cascade_el, "featureMode").text = self.feature_mode
        ET.SubElement(cascade_el, "height").text = str(self.window_height)
        ET.SubElement(cascade_el, "width").text = str(self.window_width)
        ET.SubElement(cascade_el, "stageNum").text = str(len(self.stages))

        stages_el = ET.SubElement(cascade_el, "stages")
        for stage in self.stages:
            st_el = ET.SubElement(
                stages_el,
                "_",
                attrib={
                    "index": str(stage.stage_index),
                    "hit_rate": f"{stage.hit_rate:.6f}",
                    "false_alarm_rate": f"{stage.false_alarm_rate:.6f}",
                },
            )
            ET.SubElement(st_el, "maxWeakCount").text = str(len(stage.classifiers))
            ET.SubElement(st_el, "stageThreshold").text = f"{stage.stage_threshold:.8f}"
            weak_el = ET.SubElement(st_el, "weakClassifiers")
            for clf in stage.classifiers:
                c_el = ET.SubElement(weak_el, "_")
                ET.SubElement(c_el, "type").text = FEATURE_TYPE_TO_NAME.get(
                    clf.type, "TWO_VERTICAL"
                )
                ET.SubElement(
                    c_el, "topLeft"
                ).text = f"{clf.top_left[0]} {clf.top_left[1]}"
                ET.SubElement(c_el, "size").text = f"{clf.width} {clf.height}"
                ET.SubElement(c_el, "threshold").text = f"{clf.threshold:.8f}"
                ET.SubElement(c_el, "polarity").text = str(clf.polarity)
                ET.SubElement(c_el, "weight").text = f"{clf.weight:.8f}"

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(out_path, encoding="utf-8", xml_declaration=True)
        return out_path

    @classmethod
    def load_xml(cls, xml_path: str | Path) -> ViolaJonesCascadeModel:
        """Load a custom-trained Viola-Jones cascade XML file."""
        path = resolve_path(xml_path)
        if not path.exists():
            raise FileNotFoundError(f"Cascade XML not found: {path}")

        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise ValueError(f"{path.name} is not a valid XML file: {exc}") from exc
        root = tree.getroot()
        cascade_el = root.find("cascade")
        if cascade_el is None:
            raise ValueError(f"Invalid Viola-Jones cascade XML: {path}")

        type_id = cascade_el.attrib.get("type_id", "")
        if type_id != "forensics-viola-jones-cascade":
            raise ValueError(
                f"XML file {path.name} is not a custom Forensics Viola-Jones cascade."
            )

        width = int(cascade_el.findtext("width", "24"))
        height = int(cascade_el.findtext("height", "24"))
        stage_type = cascade_el.findtext("stageType", "BOOST")
        feature_type = cascade_el.findtext("featureType", "HAAR")
        boost_type = cascade_el.findtext("boostType", "DISCRETE")
        feature_mode = cascade_el.findtext("featureMode", "BASIC")

        stages: list[CascadeStage] = []
        stages_el = cascade_el.find("stages")
        if stages_el is not None:
            for idx, st_el in enumerate(stages_el.findall("_")):
                stage_idx = int(st_el.attrib.get("index", str(idx)))
                hit_rate = float(st_el.attrib.get("hit_rate", "1.0"))
                fa_rate = float(st_el.attrib.get("false_alarm_rate", "0.0"))
                stage_thresh = float(st_el.findtext("stageThreshold", "0.0"))
                classifiers: list[HaarLikeFeature] = []
                weak_el = st_el.find("weakClassifiers")
                if weak_el is not None:
                    for c_el in weak_el.findall("_"):
                        ftype_name = c_el.findtext("type", "TWO_VERTICAL")
                        tl_parts = c_el.findtext("topLeft", "0 0").split()
                        sz_parts = c_el.findtext("size", "4 4").split()
                        thresh = float(c_el.findtext("threshold", "0.0"))
                        polarity = int(c_el.findtext("polarity", "1"))
                        weight = float(c_el.findtext("weight", "1.0"))
                        classifiers.append(
                            HaarLikeFeature(
                                feature_type=NAME_TO_FEATURE_TYPE.get(
                                    ftype_name, FeatureType.TWO_VERTICAL
                                ),
                                position=(int(tl_parts[0]), int(tl_parts[1])),
                                width=int(sz_parts[0]),
                                height=int(sz_parts[1]),
                                threshold=thresh,
                                polarity=polarity,
                                weight=weight,
                            )
                        )
                stages.append(
                    CascadeStage(
                        stage_index=stage_idx,
                        classifiers=classifiers,
                        stage_threshold=stage_thresh,
                        hit_rate=hit_rate,
                        false_alarm_rate=fa_rate,
                    )
                )

        return cls(
            window_width=width,
            window_height=height,
            feature_type=feature_type,
            stage_type=stage_type,
            boost_type=boost_type,
            feature_mode=feature_mode,
            stages=stages,
        )


def parse_annotations_file(
    annotations_path: str | Path, base_dir: str | Path | None = None
) -> list[tuple[Path, list[tuple[int, int, int, int]]]]:
    """Parse an OpenCV-style positive annotations file.

    Supports both:
    - ``img_rel_path num_objs x1 y1 w1 h1 ...``
    - ``img_rel_path x1 y1 w1 h1``
    """
    ann_file = Path(annotations_path)
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    root_dir = Path(base_dir) if base_dir else ann_file.parent

    entries: list[tuple[Path, list[tuple[int, int, int, int]]]] = []
    for raw_line in ann_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        img_path = Path(parts[0])
        if not img_path.is_absolute():
            candidate = root_dir / img_path
            if not candidate.exists():
                candidate = ann_file.parent / img_path
            img_path = candidate

        nums = [int(float(v)) for v in parts[1:]]
        boxes: list[tuple[int, int, int, int]] = []
        if len(nums) == 4:
            boxes.append((nums[0], nums[1], nums[2], nums[3]))
        elif len(nums) >= 5 and (len(nums) - 1) % 4 == 0:
            count = nums[0]
            for i in range(count):
                b = nums[1 + i * 4 : 1 + (i + 1) * 4]
                if len(b) == 4:
                    boxes.append((b[0], b[1], b[2], b[3]))
        if boxes:
            entries.append((img_path, boxes))
    return entries


def validate_cascade_dataset(
    positive_dir: str | Path,
    negative_dir: str | Path,
    annotations_path: str | Path | None = None,
    window_size: tuple[int, int] = (24, 24),
) -> dict[str, Any]:
    """Validate folders and positive annotations before sample generation."""
    pos_folder, neg_folder = resolve_path(positive_dir), resolve_path(negative_dir)
    if not pos_folder.is_dir():
        raise ValueError(f"Positive images folder does not exist: {pos_folder}")
    if not neg_folder.is_dir():
        raise ValueError(f"Negative images folder does not exist: {neg_folder}")
    if min(window_size) < 8:
        raise ValueError("Training window dimensions must each be at least 8 pixels.")

    pos_files = [
        p
        for p in pos_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    neg_files = [
        p
        for p in neg_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not pos_files:
        raise ValueError(f"No positive images found in {pos_folder}.")
    if not neg_files:
        raise ValueError(f"No negative images found in {neg_folder}.")

    annotated_boxes = 0
    if annotations_path:
        entries = parse_annotations_file(annotations_path, base_dir=pos_folder)
        missing = [str(path) for path, _ in entries if not path.is_file()]
        annotated_boxes = sum(len(boxes) for _, boxes in entries)
        if not entries or annotated_boxes == 0:
            raise ValueError(
                "The annotation file contains no usable face bounding boxes."
            )
        if missing:
            raise ValueError(
                f"{len(missing)} annotated image path(s) could not be resolved; "
                f"first: {missing[0]}"
            )

    return {
        "positive_files": len(pos_files),
        "negative_files": len(neg_files),
        "annotated_boxes": annotated_boxes,
        "window_size": tuple(window_size),
    }


def generate_positive_samples(
    positive_dir: str | Path,
    annotations_path: str | Path | None = None,
    window_size: tuple[int, int] = (24, 24),
    max_samples: int = 100,
    vec_output_path: str | Path | None = None,
    *,
    random_seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract annotated face regions (or already-cropped face images).

    For WIDER FACE, provide its converted bounding-box annotation file.
    Without annotations, every image in ``positive_dir`` is assumed to be a
    single face crop (appropriate for an LFW-style cropped-face folder).
    """
    pos_folder = Path(positive_dir)
    rng = np.random.default_rng(random_seed)
    candidates: list[tuple[Path, tuple[int, int, int, int] | None]] = []

    if annotations_path:
        for image_path, boxes in parse_annotations_file(
            annotations_path, base_dir=pos_folder
        ):
            for box in boxes:
                candidates.append((image_path, box))
    else:
        files = [
            p
            for p in pos_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        candidates = [(p, None) for p in files]

    rng.shuffle(candidates)
    if max_samples > 0:
        candidates = candidates[:max_samples]

    patches: list[np.ndarray] = []
    integral_images: list[np.ndarray] = []
    target_w, target_h = window_size
    for image_path, box in candidates:
        try:
            with Image.open(image_path) as img:
                gray = img.convert("L")
                if box is not None:
                    x, y, w, h = box
                    if (
                        w <= 0
                        or h <= 0
                        or x < 0
                        or y < 0
                        or x + w > gray.width
                        or y + h > gray.height
                    ):
                        continue
                    gray = gray.crop((x, y, x + w, y + h))
                if gray.width == 0 or gray.height == 0:
                    continue
                # The coursework/reference notebook uses direct resizing to a
                # fixed PCA/feature window; preserve that convention here.
                gray = gray.resize((target_w, target_h), Image.Resampling.BILINEAR)
                arr = np.asarray(gray, dtype=np.uint8)
        except (OSError, ValueError):
            continue
        patches.append(_normalize_patch(arr))
        integral_images.append(compute_integral_image(_normalize_patch(arr)))

    if vec_output_path and patches:
        vec_path = Path(vec_output_path)
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        # This is a NumPy patch archive, not OpenCV's binary .vec format.
        np.save(vec_path, np.stack(patches, axis=0))
    return patches, integral_images


def generate_negative_samples(
    negative_dir: str | Path,
    window_size: tuple[int, int] = (24, 24),
    max_samples: int = 100,
    *,
    random_seed: int = 42,
    patches_per_image: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Draw diverse random non-face windows from background images.

    Negative directories should contain background-only images. The sampler
    randomises image order and crop locations; it does not assume that a
    single image's top-left non-overlapping tiles represent the whole set.
    """
    neg_folder = Path(negative_dir)
    files = [
        p
        for p in neg_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise ValueError(f"No negative/background images found in {neg_folder}.")
    rng = np.random.default_rng(random_seed)
    rng.shuffle(files)
    target_w, target_h = window_size
    patches: list[np.ndarray] = []
    integral_images: list[np.ndarray] = []
    usable_files: list[Path] = []
    for path in files:
        try:
            with Image.open(path) as img:
                img.verify()
            usable_files.append(path)
        except (OSError, ValueError):
            continue
    if not usable_files:
        raise ValueError("None of the negative images could be decoded.")

    # Round-robin allocation across background images improves diversity while
    # streaming from disk instead of keeping an entire image collection in RAM.
    per_image = patches_per_image or max(
        1, int(np.ceil(max_samples / len(usable_files)))
    )
    for _round in range(per_image):
        order = list(range(len(usable_files)))
        rng.shuffle(order)
        for idx in order:
            if len(patches) >= max_samples:
                break
            try:
                with Image.open(usable_files[idx]) as img:
                    gray = img.convert("L")
                    if gray.width >= target_w and gray.height >= target_h:
                        left = int(rng.integers(0, gray.width - target_w + 1))
                        top = int(rng.integers(0, gray.height - target_h + 1))
                        crop = gray.crop((left, top, left + target_w, top + target_h))
                    else:
                        crop = gray.resize(
                            (target_w, target_h), Image.Resampling.BILINEAR
                        )
                    arr = np.asarray(crop, dtype=np.uint8)
            except (OSError, ValueError):
                continue
            norm = _normalize_patch(arr)
            patches.append(norm)
            integral_images.append(compute_integral_image(norm))
        if len(patches) >= max_samples:
            break
    if len(patches) < min(max_samples, 2):
        raise ValueError("At least two usable negative patches are required.")
    return patches, integral_images


def create_synthetic_face_nonface_dataset(
    base_dir: str | Path,
    window_size: tuple[int, int] = (19, 19),
    n_pos: int = 16,
    n_neg: int = 16,
) -> dict[str, Path]:
    """Create a realistic face vs non-face dataset using scikit-image samples."""
    root = Path(base_dir)
    pos_train = root / "train_data" / "faces"
    neg_train = root / "train_data" / "non_faces"
    pos_test = root / "test_data" / "faces"
    neg_test = root / "test_data" / "non_faces"
    for folder in (pos_train, neg_train, pos_test, neg_test):
        folder.mkdir(parents=True, exist_ok=True)

    astro = Image.fromarray(skimage_data.astronaut()).convert("L")
    # Face region in astronaut is approximately (178, 62, 268, 175)
    face_crop = astro.crop((178, 62, 268, 175)).resize(
        window_size, Image.Resampling.BILINEAR
    )
    face_arr = np.asarray(face_crop, dtype=np.float32)

    rng = np.random.default_rng(42)
    for i in range(n_pos):
        noisy = np.clip(face_arr + rng.normal(0, 6.0, face_arr.shape), 0, 255).astype(
            np.uint8
        )
        target_dir = pos_train if i < int(n_pos * 0.75) else pos_test
        Image.fromarray(noisy, mode="L").save(target_dir / f"face_{i:03d}.png")

    camera_arr = skimage_data.camera()
    w, h = window_size
    for i in range(n_neg):
        r = int(rng.integers(0, camera_arr.shape[0] - h))
        c = int(rng.integers(0, camera_arr.shape[1] - w))
        patch = camera_arr[r : r + h, c : c + w]
        target_dir = neg_train if i < int(n_neg * 0.75) else neg_test
        Image.fromarray(patch, mode="L").save(target_dir / f"non_face_{i:03d}.png")

    return {
        "pos_train": pos_train,
        "neg_train": neg_train,
        "pos_test": pos_test,
        "neg_test": neg_test,
    }


def _stage_threshold_for_hit_rate(
    scores: np.ndarray, target_hit_rate: float
) -> tuple[float, float]:
    """Choose an inclusive threshold retaining at least target_hit_rate positives."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot set a stage threshold without positive samples.")
    required = max(1, int(np.ceil(target_hit_rate * values.size)))
    threshold = float(np.sort(values)[values.size - required])
    achieved = float(np.mean(values >= threshold))
    return threshold, achieved


def evaluate_cascade(
    model: ViolaJonesCascadeModel,
    positive_iis: list[np.ndarray],
    negative_iis: list[np.ndarray],
) -> dict[str, float | int]:
    """Evaluate full cascade decisions on fixed-size positive/negative patches."""
    pos_pred = _cascade_passes(model, positive_iis)
    neg_pred = _cascade_passes(model, negative_iis)
    tp = int(np.count_nonzero(pos_pred))
    fn = len(pos_pred) - tp
    fp = int(np.count_nonzero(neg_pred))
    tn = len(neg_pred) - fp
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "face_recall_pct": 100.0 * tp / max(1, tp + fn),
        "nonface_recall_pct": 100.0 * tn / max(1, tn + fp),
        "accuracy_pct": 100.0 * (tp + tn) / max(1, tp + tn + fp + fn),
        "false_alarm_rate": fp / max(1, fp + tn),
    }


def train_viola_jones_cascade(
    positive_dir: str | Path,
    negative_dir: str | Path,
    *,
    annotations_path: str | Path | None = None,
    pos_test_dir: str | Path | None = None,
    neg_test_dir: str | Path | None = None,
    output_xml_path: str | Path = DEFAULT_CASCADE_XML,
    window_width: int = 24,
    window_height: int = 24,
    num_pos: int = 500,
    num_neg: int = 1000,
    num_stages: int = 5,
    max_weak_count: int = 10,
    min_hit_rate: float = 0.95,
    max_false_alarm_rate: float = 0.5,
    weight_trimming_rate: float = 1.0,
    max_weak_tree_depth: int = 1,
    boost_type: str = "DISCRETE",
    feature_mode: str = "BASIC",
    min_feature_width: int = 4,
    max_feature_width: int = 12,
    min_feature_height: int = 4,
    max_feature_height: int = 12,
    max_features: int = 3000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train a custom multi-stage Haar + discrete-AdaBoost cascade.

    Stages are trained sequentially on positive and negative training windows
    that survive all previous stages. The requested hit rate is used to set each
    stage threshold; the trainer chooses the shortest weak-classifier prefix
    meeting the false-alarm target, or reports when the weak-learner budget is
    insufficient. Negative patches are randomly sampled from the supplied
    background images; full-image hard-negative mining is not performed here.

    This is a ForensicsApp-specific model format (XML + JSON), not an OpenCV
    ``CascadeClassifier`` XML or ONNX model. Inference must use the matching
    custom detector implementation.
    """
    t_start = time.perf_counter()
    logs: list[str] = []
    # Resolve every user-supplied path against the project folder, so the
    # result does not depend on where the app was started.
    positive_dir = resolve_path(positive_dir)
    negative_dir = resolve_path(negative_dir)
    annotations_path = resolve_path(annotations_path) if annotations_path else None
    pos_test_dir = resolve_path(pos_test_dir) if pos_test_dir else None
    neg_test_dir = resolve_path(neg_test_dir) if neg_test_dir else None
    output_xml_path = resolve_path(output_xml_path)
    window_size = (int(window_width), int(window_height))
    if window_width < 8 or window_height < 8:
        raise ValueError("Window width and height must be at least 8 pixels.")
    if num_stages < 1 or max_weak_count < 1 or num_pos < 2 or num_neg < 2:
        raise ValueError(
            "Use at least 2 positive/negative samples, 1 stage and 1 weak learner."
        )
    if not 0.0 < min_hit_rate <= 1.0:
        raise ValueError("min_hit_rate must be in (0, 1].")
    if not 0.0 < max_false_alarm_rate < 1.0:
        raise ValueError("max_false_alarm_rate must be in (0, 1).")
    if not 0.0 < weight_trimming_rate <= 1.0:
        raise ValueError("weight_trimming_rate must be in (0, 1].")
    if max_weak_tree_depth != 1:
        raise ValueError(
            "This implementation uses AdaBoost decision stumps; "
            "max_weak_tree_depth must be 1."
        )
    if boost_type.upper() not in {"DISCRETE", "DAB"}:
        raise ValueError(
            "This implementation supports discrete AdaBoost only "
            "(boost_type='DISCRETE')."
        )
    boost_type = "DISCRETE"

    validation = validate_cascade_dataset(
        positive_dir,
        negative_dir,
        annotations_path=annotations_path,
        window_size=window_size,
    )
    logs.append(
        f"[Dataset] {validation['positive_files']} positive files, "
        f"{validation['negative_files']} negative files, "
        f"{validation['annotated_boxes']} annotated faces; "
        f"window={window_width}x{window_height}."
    )

    out_xml = Path(output_xml_path)
    out_dir = out_xml.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_archive = out_dir / "positive_patches.npy"

    # Prepare positive and negative windows with reproducible seed.
    pos_patches, pos_iis = generate_positive_samples(
        positive_dir,
        annotations_path,
        window_size,
        num_pos,
        patch_archive,
        random_seed=random_seed,
    )
    neg_patches, neg_iis = generate_negative_samples(
        negative_dir,
        window_size,
        num_neg,
        random_seed=random_seed + 1,
    )
    if len(pos_iis) < 2 or len(neg_iis) < 2:
        raise ValueError(
            f"Not enough usable patches: {len(pos_iis)} positive, "
            f"{len(neg_iis)} negative. Use more images or fix the annotation file."
        )
    logs.append(
        f"[Samples] Prepared {len(pos_iis)} positive and "
        f"{len(neg_iis)} negative patches."
    )
    if len(pos_iis) < num_pos:
        logs.append(
            f"[Dataset] WARNING: requested {num_pos} positive patches "
            f"but only {len(pos_iis)} were available."
        )
    if len(neg_iis) < num_neg:
        logs.append(
            f"[Dataset] WARNING: requested {num_neg} negative patches "
            f"but only {len(neg_iis)} could be sampled."
        )

    # The maximum feature dimensions must fit inside the fixed training window.
    min_fw = max(2, min(int(min_feature_width), window_width - 1))
    min_fh = max(2, min(int(min_feature_height), window_height - 1))
    max_fw = max(min_fw, min(int(max_feature_width), window_width))
    max_fh = max(min_fh, min(int(max_feature_height), window_height))

    stages: list[CascadeStage] = []
    active_pos = list(pos_iis)
    active_neg = list(neg_iis)
    stage_records: list[dict[str, Any]] = []

    for stage_idx in range(int(num_stages)):
        if len(active_pos) < 2:
            logs.append(
                f"[Stage {stage_idx + 1}] Stopped: fewer than two positive windows "
                "survive previous stages."
            )
            break
        if len(active_neg) < 2:
            logs.append(
                f"[Stage {stage_idx + 1}] Stopped: fewer than two negative windows "
                "survive previous stages."
            )
            break

        stage_start = time.perf_counter()
        learners = learn(
            active_pos,
            active_neg,
            num_classifiers=max_weak_count,
            min_feature_width=min_fw,
            max_feature_width=max_fw,
            min_feature_height=min_fh,
            max_feature_height=max_fh,
            feature_mode=feature_mode,
            max_features=max_features,
            weight_trimming_rate=weight_trimming_rate,
            random_seed=random_seed + stage_idx,
        )

        # Select the shortest prefix meeting the requested false-alarm target,
        # while setting its threshold to retain at least min_hit_rate positives.
        chosen: tuple[list[HaarLikeFeature], float, float, float] | None = None
        prefix_candidates: list[
            tuple[float, int, list[HaarLikeFeature], float, float]
        ] = []
        # Running sums of the vote matrix give every prefix's stage score at
        # once (column k-1 = score with the first k weak classifiers).
        pos_prefix_scores = np.cumsum(
            _vote_matrix(learners, np.stack(active_pos)), axis=1
        )
        neg_prefix_scores = np.cumsum(
            _vote_matrix(learners, np.stack(active_neg)), axis=1
        )
        for count in range(1, len(learners) + 1):
            prefix = learners[:count]
            pos_scores = pos_prefix_scores[:, count - 1]
            neg_scores = neg_prefix_scores[:, count - 1]
            threshold, achieved_hit = _stage_threshold_for_hit_rate(
                pos_scores, min_hit_rate
            )
            fa_rate = float(np.mean(neg_scores >= threshold))
            prefix_candidates.append((fa_rate, count, prefix, threshold, achieved_hit))
            if fa_rate <= max_false_alarm_rate:
                chosen = (prefix, threshold, achieved_hit, fa_rate)
                break
        if chosen is None:
            # The max weak-learner budget was insufficient. Retain the prefix
            # with the lowest observed false-alarm rate, not a bogus threshold.
            best = min(prefix_candidates, key=lambda item: (item[0], item[1]))
            chosen = (best[2], best[3], best[4], best[0])
            logs.append(
                f"[Stage {stage_idx + 1}] WARNING: false-alarm target "
                f"{max_false_alarm_rate:.3f} not reached; best observed {best[0]:.3f} "
                f"with {best[1]} weak learner(s)."
            )

        stage_classifiers, stage_threshold, hit_rate, fa_rate = chosen
        stage = CascadeStage(
            stage_index=stage_idx,
            classifiers=list(stage_classifiers),
            stage_threshold=float(stage_threshold),
            hit_rate=float(hit_rate),
            false_alarm_rate=float(fa_rate),
        )
        stages.append(stage)

        # Carry forward only windows that pass this stage. This is essential
        # for training later cascade stages on the hard examples that survive.
        pos_scores = pos_prefix_scores[:, len(stage_classifiers) - 1]
        neg_scores = neg_prefix_scores[:, len(stage_classifiers) - 1]
        next_pos = [
            ii for ii, score in zip(active_pos, pos_scores) if score >= stage_threshold
        ]
        next_neg = [
            ii for ii, score in zip(active_neg, neg_scores) if score >= stage_threshold
        ]
        stage_time = time.perf_counter() - stage_start
        stage_records.append(
            {
                "stage": stage_idx + 1,
                "weak_classifiers": len(stage_classifiers),
                "conditional_hit_rate": hit_rate,
                "conditional_false_alarm_rate": fa_rate,
                "positive_windows_before": len(active_pos),
                "positive_windows_after": len(next_pos),
                "negative_windows_before": len(active_neg),
                "negative_windows_after": len(next_neg),
                "duration_sec": round(stage_time, 4),
            }
        )
        logs.append(
            f"[Stage {stage_idx + 1}/{num_stages}] weak={len(stage_classifiers)}, "
            f"hit={hit_rate:.3f}, false_alarm={fa_rate:.3f}, "
            f"positives {len(active_pos)}->{len(next_pos)}, "
            f"negatives {len(active_neg)}->{len(next_neg)}, time={stage_time:.2f}s."
        )

        active_pos, active_neg = next_pos, next_neg
        if not active_neg:
            logs.append(
                "[Cascade] All sampled training negatives are rejected; stopping early."
            )
            break

    if not stages:
        raise RuntimeError(
            "No cascade stage could be trained. "
            "Check the dataset and feature configuration."
        )

    model = ViolaJonesCascadeModel(
        window_width=window_width,
        window_height=window_height,
        feature_type="HAAR",
        stage_type="BOOST",
        boost_type=boost_type,
        feature_mode=feature_mode,
        stages=stages,
        metadata={
            "format": "forensicsapp_custom_viola_jones_v2",
            "training_algorithm": (
                "Haar features + discrete AdaBoost decision stumps + cascade"
            ),
            "random_seed": random_seed,
        },
    )

    # Full cascade evaluation, not a flat vote across all weak classifiers.
    train_metrics = evaluate_cascade(model, pos_iis, neg_iis)
    pos_test_iis = (
        load_integral_images(
            pos_test_dir,
            window_size=window_size,
            max_samples=num_pos,
            random_seed=random_seed + 2,
        )
        if pos_test_dir and Path(pos_test_dir).is_dir()
        else []
    )
    neg_test_iis = (
        load_integral_images(
            neg_test_dir,
            window_size=window_size,
            max_samples=num_neg,
            random_seed=random_seed + 3,
        )
        if neg_test_dir and Path(neg_test_dir).is_dir()
        else []
    )
    test_metrics = (
        evaluate_cascade(model, pos_test_iis, neg_test_iis)
        if pos_test_iis and neg_test_iis
        else None
    )

    # Keep the reference-notebook threshold sweep as a diagnostic for the
    # flattened strong classifier, but report cascade metrics separately.
    threshold_eval = evaluate_thresholds(
        pos_iis,
        neg_iis,
        pos_test_iis or None,
        neg_test_iis or None,
        model.all_classifiers,
    )

    saved_xml = model.save_xml(out_xml)
    duration = time.perf_counter() - t_start
    logs.append(
        f"[Evaluation] Train face recall={train_metrics['face_recall_pct']:.1f}%, "
        f"train non-face recall={train_metrics['nonface_recall_pct']:.1f}%, "
        f"train accuracy={train_metrics['accuracy_pct']:.1f}%."
    )
    if test_metrics:
        logs.append(
            f"[Validation] Face recall={test_metrics['face_recall_pct']:.1f}%, "
            f"non-face recall={test_metrics['nonface_recall_pct']:.1f}%, "
            f"accuracy={test_metrics['accuracy_pct']:.1f}%."
        )
    logs.append(f"[Export] Saved custom cascade XML: {saved_xml}")

    config_data = {
        "model_format": "ForensicsApp custom XML (not OpenCV CascadeClassifier XML)",
        "positive_dir": str(positive_dir),
        "negative_dir": str(negative_dir),
        "annotations_path": str(annotations_path) if annotations_path else None,
        "window_width": window_width,
        "window_height": window_height,
        "num_pos": len(pos_iis),
        "num_neg": len(neg_iis),
        "num_stages": len(stages),
        "num_stages_requested": num_stages,
        "num_stages_trained": len(stages),
        "max_weak_count_per_stage": max_weak_count,
        "total_weak_classifiers": len(model.all_classifiers),
        "feature_type": "HAAR",
        "stage_type": "BOOST",
        "boost_type": boost_type,
        "feature_mode": feature_mode,
        "min_feature_width": min_fw,
        "max_feature_width": max_fw,
        "min_feature_height": min_fh,
        "max_feature_height": max_fh,
        "max_features_considered_per_stage": max_features,
        "min_hit_rate_target": min_hit_rate,
        "max_false_alarm_rate_target": max_false_alarm_rate,
        "weight_trimming_rate": weight_trimming_rate,
        "weak_learner": "threshold-optimised decision stump",
        "max_weak_tree_depth": 1,
        "random_seed": random_seed,
        "stage_metrics": stage_records,
        "train_face_accuracy_pct": round(float(train_metrics["face_recall_pct"]), 2),
        "train_nonface_accuracy_pct": round(
            float(train_metrics["nonface_recall_pct"]), 2
        ),
        "train_accuracy_pct": round(float(train_metrics["accuracy_pct"]), 2),
        "train_false_alarm_rate": round(float(train_metrics["false_alarm_rate"]), 6),
        "test_face_accuracy_pct": round(float(test_metrics["face_recall_pct"]), 2)
        if test_metrics
        else None,
        "test_nonface_accuracy_pct": round(float(test_metrics["nonface_recall_pct"]), 2)
        if test_metrics
        else None,
        "test_accuracy_pct": round(float(test_metrics["accuracy_pct"]), 2)
        if test_metrics
        else None,
        "test_false_alarm_rate": round(float(test_metrics["false_alarm_rate"]), 6)
        if test_metrics
        else None,
        "training_duration_sec": round(duration, 4),
    }
    config_path = out_dir / "training_config.json"
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    log_path = out_dir / "training_log.txt"
    log_path.write_text("\n".join(logs) + "\n", encoding="utf-8")

    summary_image = render_viola_jones_training_summary(
        pos_patches=pos_patches,
        neg_patches=neg_patches,
        pos_iis=pos_iis,
        neg_iis=neg_iis,
        model=model,
        threshold_eval=threshold_eval,
    )
    return {
        "model": model,
        "xml_path": saved_xml,
        "config_path": config_path,
        "log_path": log_path,
        "logs": logs,
        "config": config_data,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "threshold_eval": threshold_eval,
        "summary_image": summary_image,
    }


def render_viola_jones_training_summary(
    *,
    pos_patches: list[np.ndarray],
    neg_patches: list[np.ndarray],
    pos_iis: list[np.ndarray],
    neg_iis: list[np.ndarray],
    model: ViolaJonesCascadeModel,
    threshold_eval: dict[str, Any],
) -> Image.Image:
    """Render training summary figure (Integral images + Haar features + chart)."""
    # An off-screen Figure (not pyplot), so this can run in a worker thread.
    fig = Figure(figsize=(12, 7.5), dpi=100)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.38, wspace=0.3)

    # Top row: Sample Face, Integral Image, Non-Face, Top Haar Feature Overlay
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_pos.imshow(pos_patches[0], cmap="gray")
    ax_pos.set_title("Sample Face Patch", fontsize=10)
    ax_pos.axis("off")

    ax_pos_ii = fig.add_subplot(gs[0, 1])
    ax_pos_ii.imshow(pos_iis[0], cmap="gray")
    ax_pos_ii.set_title("Face Integral Image", fontsize=10)
    ax_pos_ii.axis("off")

    ax_neg = fig.add_subplot(gs[0, 2])
    ax_neg.imshow(neg_patches[0], cmap="gray")
    ax_neg.set_title("Sample Non-Face", fontsize=10)
    ax_neg.axis("off")

    ax_haar = fig.add_subplot(gs[0, 3])
    mean_pos = np.mean(np.stack(pos_patches, axis=0), axis=0)
    ax_haar.imshow(mean_pos, cmap="gray")
    all_clfs = model.all_classifiers
    colors = ["#00ff66", "#ff3366", "#33ccff"]
    for idx, clf in enumerate(all_clfs[:3]):
        r0, c0 = clf.top_left
        rect = Rectangle(
            (c0, r0),
            clf.width,
            clf.height,
            fill=False,
            edgecolor=colors[idx % len(colors)],
            linewidth=2.0,
            label=f"H{idx + 1} (w={clf.weight:.2f})",
        )
        ax_haar.add_patch(rect)
    if all_clfs:
        ax_haar.legend(loc="lower right", fontsize=7, framealpha=0.75)
    ax_haar.set_title("Top Selected Haar Features", fontsize=10)
    ax_haar.axis("off")

    # Bottom row: Threshold evaluation bar plot (matching Exercise #8 in notebook)
    ax_bar = fig.add_subplot(gs[1, :])
    thresholds = np.asarray(threshold_eval["thresholds"], dtype=np.float64)
    width = 0.02
    ax_bar.bar(
        thresholds - 1.5 * width,
        threshold_eval["pcnt_faces_train"],
        width,
        label="Train faces",
        color="#2b8cbe",
    )
    ax_bar.bar(
        thresholds - 0.5 * width,
        threshold_eval["pcnt_nofaces_train"],
        width,
        label="Train non-faces",
        color="#7bccc4",
    )
    if threshold_eval.get("pcnt_faces_test"):
        ax_bar.bar(
            thresholds + 0.5 * width,
            threshold_eval["pcnt_faces_test"],
            width,
            label="Test faces",
            color="#e34a33",
        )
    if threshold_eval.get("pcnt_nofaces_test"):
        ax_bar.bar(
            thresholds + 1.5 * width,
            threshold_eval["pcnt_nofaces_test"],
            width,
            label="Test non-faces",
            color="#fdbb84",
        )
    ax_bar.set_title(
        f"Flattened AdaBoost threshold diagnostic ({len(model.stages)} stage(s), "
        f"{len(all_clfs)} weak clfs) — cascade metrics reported separately",
        fontsize=11,
    )
    ax_bar.set_xlabel("Decision Threshold")
    ax_bar.set_ylabel("Correctly Identified (%)")
    ax_bar.set_ylim(0, 108)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bar.legend(loc="lower right", ncol=2, fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def ask_viola_jones_training_parameters(
    parent: tk.Misc,
) -> dict[str, Any] | None:
    """Display configuration dialog for Viola-Jones cascade training."""
    dialog = tk.Toplevel(parent)
    dialog.title("Viola-Jones Cascade Trainer")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    pos_dir_var = tk.StringVar(value="")
    ann_file_var = tk.StringVar(value="")
    neg_dir_var = tk.StringVar(value="")
    pos_test_dir_var = tk.StringVar(value="")
    neg_test_dir_var = tk.StringVar(value="")
    output_xml_var = tk.StringVar(value=str(DEFAULT_CASCADE_XML))

    win_w_var = tk.StringVar(value="24")
    win_h_var = tk.StringVar(value="24")
    num_pos_var = tk.StringVar(value="500")
    num_neg_var = tk.StringVar(value="1000")

    # Advanced Cascade Parameters (Plan Section 2.B)
    num_stages_var = tk.StringVar(value="5")
    boost_type_var = tk.StringVar(value="DISCRETE")
    min_hit_rate_var = tk.StringVar(value="0.95")
    max_fa_rate_var = tk.StringVar(value="0.5")
    max_weak_var = tk.StringVar(value="10")
    weight_trim_var = tk.StringVar(value="1.0")
    tree_depth_var = tk.StringVar(value="1")
    feature_mode_var = tk.StringVar(value="BASIC")
    max_features_var = tk.StringVar(value="3000")
    random_seed_var = tk.StringVar(value="42")
    min_feature_width_var = tk.StringVar(value="4")
    max_feature_width_var = tk.StringVar(value="12")
    min_feature_height_var = tk.StringVar(value="4")
    max_feature_height_var = tk.StringVar(value="12")

    main_frame = ttk.Frame(dialog, padding=12)
    main_frame.pack(fill="both", expand=True)

    # A. Inputs and dataset configuration
    ds_frame = ttk.LabelFrame(
        main_frame, text="A. Inputs & Dataset Configuration", padding=10
    )
    ds_frame.pack(fill="x", pady=(0, 8))

    def _browse_dir(var: tk.StringVar, title: str) -> None:
        selected = filedialog.askdirectory(parent=dialog, title=title)
        if selected:
            var.set(selected)

    def _browse_ann() -> None:
        selected = filedialog.askopenfilename(
            parent=dialog,
            title="Select positive annotations file",
            filetypes=[
                ("Text / info files", "*.txt *.info *.dat"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            ann_file_var.set(selected)

    def _browse_xml() -> None:
        selected = filedialog.asksaveasfilename(
            parent=dialog,
            title="Save trained cascade XML as",
            defaultextension=".xml",
            initialfile="cascade.xml",
            filetypes=[("XML model", "*.xml"), ("All files", "*.*")],
        )
        if selected:
            output_xml_var.set(selected)

    def _generate_demo_data() -> None:
        demo_root = DATASETS_DIR / "viola_jones_demo"
        paths = create_synthetic_face_nonface_dataset(
            demo_root, window_size=(24, 24), n_pos=40, n_neg=80
        )
        pos_dir_var.set(str(paths["pos_train"]))
        neg_dir_var.set(str(paths["neg_train"]))

    ttk.Label(ds_frame, text="Positive images folder:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Entry(ds_frame, textvariable=pos_dir_var, width=38).grid(
        row=0, column=1, padx=6, pady=3
    )
    ttk.Button(
        ds_frame,
        text="Browse…",
        command=lambda: _browse_dir(pos_dir_var, "Select positive images folder"),
    ).grid(row=0, column=2, pady=3)

    ttk.Label(ds_frame, text="Positive annotations (optional):").grid(
        row=1, column=0, sticky="w", pady=3
    )
    ttk.Entry(ds_frame, textvariable=ann_file_var, width=38).grid(
        row=1, column=1, padx=6, pady=3
    )
    ttk.Button(ds_frame, text="Browse…", command=_browse_ann).grid(
        row=1, column=2, pady=3
    )

    ttk.Label(ds_frame, text="Negative images folder:").grid(
        row=2, column=0, sticky="w", pady=3
    )
    ttk.Entry(ds_frame, textvariable=neg_dir_var, width=38).grid(
        row=2, column=1, padx=6, pady=3
    )
    ttk.Button(
        ds_frame,
        text="Browse…",
        command=lambda: _browse_dir(neg_dir_var, "Select negative images folder"),
    ).grid(row=2, column=2, pady=3)

    ttk.Label(ds_frame, text="Validation face patches (optional):").grid(
        row=3, column=0, sticky="w", pady=3
    )
    ttk.Entry(ds_frame, textvariable=pos_test_dir_var, width=38).grid(
        row=3, column=1, padx=6, pady=3
    )
    ttk.Button(
        ds_frame,
        text="Browse…",
        command=lambda: _browse_dir(pos_test_dir_var, "Select validation face patches"),
    ).grid(row=3, column=2, pady=3)

    ttk.Label(ds_frame, text="Validation non-face patches (optional):").grid(
        row=4, column=0, sticky="w", pady=3
    )
    ttk.Entry(ds_frame, textvariable=neg_test_dir_var, width=38).grid(
        row=4, column=1, padx=6, pady=3
    )
    ttk.Button(
        ds_frame,
        text="Browse…",
        command=lambda: _browse_dir(
            neg_test_dir_var, "Select validation non-face patches"
        ),
    ).grid(row=4, column=2, pady=3)

    ttk.Label(ds_frame, text="Output custom cascade XML:").grid(
        row=5, column=0, sticky="w", pady=3
    )
    ttk.Entry(ds_frame, textvariable=output_xml_var, width=38).grid(
        row=5, column=1, padx=6, pady=3
    )
    ttk.Button(ds_frame, text="Browse…", command=_browse_xml).grid(
        row=5, column=2, pady=3
    )

    ttk.Button(
        ds_frame,
        text="Generate tiny demo dataset (smoke test only)",
        command=_generate_demo_data,
    ).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(6, 2))
    ttk.Label(
        ds_frame,
        text=(
            "Use cropped face images or supply bounding-box annotations. "
            "The XML is ForensicsApp-specific."
        ),
        foreground="#555555",
    ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(3, 0))

    dims_frame = ttk.Frame(ds_frame)
    dims_frame.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(6, 0))
    for col, (lbl, var) in enumerate(
        (
            ("Window width:", win_w_var),
            ("Window height:", win_h_var),
            ("Pos samples:", num_pos_var),
            ("Neg samples:", num_neg_var),
        )
    ):
        ttk.Label(dims_frame, text=lbl).grid(
            row=0, column=col * 2, sticky="w", padx=(0, 4)
        )
        ttk.Spinbox(dims_frame, textvariable=var, from_=8, to=100000, width=8).grid(
            row=0, column=col * 2 + 1, padx=(0, 10)
        )

    # B. Cascade training parameters
    adv_frame = ttk.LabelFrame(
        main_frame, text="B. Advanced Cascade Training Parameters", padding=10
    )
    adv_frame.pack(fill="x", pady=(0, 8))

    ttk.Label(adv_frame, text="Number of stages:").grid(
        row=0, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(adv_frame, textvariable=num_stages_var, from_=1, to=25, width=8).grid(
        row=0, column=1, sticky="w", padx=6, pady=3
    )

    ttk.Label(adv_frame, text="Feature / Stage type:").grid(
        row=0, column=2, sticky="w", padx=(12, 0), pady=3
    )
    ttk.Label(adv_frame, text="HAAR / BOOST (fixed)").grid(
        row=0, column=3, sticky="w", padx=6, pady=3
    )

    ttk.Label(adv_frame, text="Boosting type:").grid(
        row=1, column=0, sticky="w", pady=3
    )
    ttk.Combobox(
        adv_frame,
        textvariable=boost_type_var,
        values=("DISCRETE",),
        state="readonly",
        width=8,
    ).grid(row=1, column=1, sticky="w", padx=6, pady=3)

    ttk.Label(adv_frame, text="Haar feature mode:").grid(
        row=1, column=2, sticky="w", padx=(12, 0), pady=3
    )
    ttk.Combobox(
        adv_frame,
        textvariable=feature_mode_var,
        values=("BASIC", "CORE", "ALL"),
        state="readonly",
        width=8,
    ).grid(row=1, column=3, sticky="w", padx=6, pady=3)

    ttk.Label(adv_frame, text="Min hit rate / stage:").grid(
        row=2, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(
        adv_frame,
        textvariable=min_hit_rate_var,
        from_=0.5,
        to=1.0,
        increment=0.005,
        width=8,
    ).grid(row=2, column=1, sticky="w", padx=6, pady=3)

    ttk.Label(adv_frame, text="Max false alarm / stage:").grid(
        row=2, column=2, sticky="w", padx=(12, 0), pady=3
    )
    ttk.Spinbox(
        adv_frame,
        textvariable=max_fa_rate_var,
        from_=0.05,
        to=0.95,
        increment=0.05,
        width=8,
    ).grid(row=2, column=3, sticky="w", padx=6, pady=3)

    ttk.Label(adv_frame, text="Max weak classifiers/stage:").grid(
        row=3, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(adv_frame, textvariable=max_weak_var, from_=1, to=200, width=8).grid(
        row=3, column=1, sticky="w", padx=6, pady=3
    )

    ttk.Label(adv_frame, text="Weight trimming / stump depth:").grid(
        row=3, column=2, sticky="w", padx=(12, 0), pady=3
    )
    trim_sub = ttk.Frame(adv_frame)
    trim_sub.grid(row=3, column=3, sticky="w", padx=6, pady=3)
    ttk.Spinbox(
        trim_sub,
        textvariable=weight_trim_var,
        from_=0.5,
        to=1.0,
        increment=0.05,
        width=5,
    ).pack(side="left")
    ttk.Spinbox(
        trim_sub,
        textvariable=tree_depth_var,
        from_=1,
        to=1,
        width=3,
        state="disabled",
    ).pack(side="left", padx=(4, 0))

    ttk.Label(adv_frame, text="Max candidate Haar features:").grid(
        row=4, column=0, sticky="w", pady=3
    )
    ttk.Spinbox(
        adv_frame,
        textvariable=max_features_var,
        from_=100,
        to=50000,
        increment=100,
        width=8,
    ).grid(row=4, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(adv_frame, text="Random seed:").grid(
        row=4, column=2, sticky="w", padx=(12, 0), pady=3
    )
    ttk.Entry(adv_frame, textvariable=random_seed_var, width=10).grid(
        row=4, column=3, sticky="w", padx=6, pady=3
    )

    ttk.Label(adv_frame, text="Haar feature width (min / max):").grid(
        row=5, column=0, sticky="w", pady=3
    )
    fw_frame = ttk.Frame(adv_frame)
    fw_frame.grid(row=5, column=1, sticky="w", padx=6, pady=3)
    ttk.Spinbox(
        fw_frame, textvariable=min_feature_width_var, from_=2, to=100, width=5
    ).pack(side="left")
    ttk.Spinbox(
        fw_frame, textvariable=max_feature_width_var, from_=2, to=100, width=5
    ).pack(side="left", padx=(4, 0))

    ttk.Label(adv_frame, text="Haar feature height (min / max):").grid(
        row=5, column=2, sticky="w", padx=(12, 0), pady=3
    )
    fh_frame = ttk.Frame(adv_frame)
    fh_frame.grid(row=5, column=3, sticky="w", padx=6, pady=3)
    ttk.Spinbox(
        fh_frame, textvariable=min_feature_height_var, from_=2, to=100, width=5
    ).pack(side="left")
    ttk.Spinbox(
        fh_frame, textvariable=max_feature_height_var, from_=2, to=100, width=5
    ).pack(side="left", padx=(4, 0))

    result: dict[str, Any] | None = None

    def apply() -> None:
        nonlocal result
        pos_dir = pos_dir_var.get().strip()
        neg_dir = neg_dir_var.get().strip()
        if not pos_dir or not neg_dir:
            messagebox.showerror(
                "Missing Dataset",
                (
                    "Select both positive and negative image folders "
                    "(or click 'Prepare Built-in Demo')."
                ),
                parent=dialog,
            )
            return
        try:
            win_w = int(win_w_var.get())
            win_h = int(win_h_var.get())
            n_pos = int(num_pos_var.get())
            n_neg = int(num_neg_var.get())
            n_stages = int(num_stages_var.get())
            max_weak = int(max_weak_var.get())
            min_hr = float(min_hit_rate_var.get())
            max_fa = float(max_fa_rate_var.get())
            w_trim = float(weight_trim_var.get())
            t_depth = int(tree_depth_var.get())
            max_features_value = int(max_features_var.get())
            seed_value = int(random_seed_var.get())
            min_fw_value = int(min_feature_width_var.get())
            max_fw_value = int(max_feature_width_var.get())
            min_fh_value = int(min_feature_height_var.get())
            max_fh_value = int(max_feature_height_var.get())
            if (
                win_w < 8
                or win_h < 8
                or n_pos < 2
                or n_neg < 2
                or n_stages < 1
                or max_weak < 1
                or max_features_value < 100
                or min_fw_value < 2
                or max_fw_value < min_fw_value
                or min_fh_value < 2
                or max_fh_value < min_fh_value
                or not (0.0 < min_hr <= 1.0)
                or not (0.0 < max_fa < 1.0)
            ):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid Parameters",
                (
                    "Please enter valid values. At least two positive and two "
                    "negative samples are required; false-alarm rate must be below 1.0."
                ),
                parent=dialog,
            )
            return

        result = {
            "positive_dir": pos_dir,
            "negative_dir": neg_dir,
            "annotations_path": ann_file_var.get().strip() or None,
            "pos_test_dir": pos_test_dir_var.get().strip() or None,
            "neg_test_dir": neg_test_dir_var.get().strip() or None,
            "output_xml_path": output_xml_var.get().strip() or str(DEFAULT_CASCADE_XML),
            "window_width": win_w,
            "window_height": win_h,
            "num_pos": n_pos,
            "num_neg": n_neg,
            "num_stages": n_stages,
            "max_weak_count": max_weak,
            "min_hit_rate": min_hr,
            "max_false_alarm_rate": max_fa,
            "weight_trimming_rate": w_trim,
            "max_weak_tree_depth": t_depth,
            "boost_type": boost_type_var.get(),
            "feature_mode": feature_mode_var.get(),
            "max_features": max_features_value,
            "random_seed": seed_value,
            "min_feature_width": min_fw_value,
            "max_feature_width": max_fw_value,
            "min_feature_height": min_fh_value,
            "max_feature_height": max_fh_value,
        }
        dialog.destroy()

    btn_bar = ttk.Frame(main_frame)
    btn_bar.pack(fill="x", pady=(6, 0))
    ttk.Button(btn_bar, text="Cancel", command=dialog.destroy).pack(side="right")
    ttk.Button(btn_bar, text="Train Cascade", command=apply).pack(
        side="right", padx=(0, 8)
    )

    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    parent.wait_window(dialog)
    return result


class ViolaJonesTrainerTool(ForensicsTool):
    """Tool 1 — Train a custom Haar + AdaBoost cascade and persist the XML model."""

    tool_id = "viola_jones_trainer"
    title = "Viola-Jones Trainer"
    category = "Face Analysis"
    description = "Train a custom Haar + AdaBoost cascade and persist the XML model."
    requires_image = False

    def run(
        self,
        parent: tk.Misc,
        document: ImageDocument,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> ToolResult | None:
        if parameters is None:
            parameters = ask_viola_jones_training_parameters(parent)
            if parameters is None:
                return None

        # Training can take minutes: run it off the UI thread with a progress
        # window (it runs directly when there is no real window, e.g. in tests).
        training_out = run_with_progress(
            parent,
            "Viola-Jones Trainer",
            "Training the cascade. This can take a few minutes for large datasets.",
            train_viola_jones_cascade,
            **parameters,
        )
        cfg = training_out["config"]
        xml_path = Path(training_out["xml_path"])

        details: dict[str, Any] = {
            "Operation": "Viola-Jones Cascade Training",
            "Cascade XML": xml_path.name,
            "XML path": str(xml_path),
            "Config JSON": Path(training_out["config_path"]).name,
            "Training log": Path(training_out["log_path"]).name,
            "Window size": f"{cfg['window_width']} × {cfg['window_height']}",
            "Positive / Negative": f"{cfg['num_pos']} / {cfg['num_neg']}",
            "Stages trained": cfg.get("num_stages_trained", cfg.get("num_stages", 0)),
            "Weak classifiers": cfg["total_weak_classifiers"],
            "Boosting / Mode": f"{cfg['boost_type']} / {cfg['feature_mode']}",
            "Train face accuracy": f"{cfg['train_face_accuracy_pct']:.1f}%",
            "Train non-face accuracy": f"{cfg['train_nonface_accuracy_pct']:.1f}%",
            "Train overall accuracy": f"{cfg['train_accuracy_pct']:.1f}%",
            "Training duration": f"{cfg['training_duration_sec']:.2f} s",
        }
        if cfg.get("test_face_accuracy_pct") is not None:
            details["Test face accuracy"] = f"{cfg['test_face_accuracy_pct']:.1f}%"
        if cfg.get("test_nonface_accuracy_pct") is not None:
            details["Test non-face accuracy"] = (
                f"{cfg['test_nonface_accuracy_pct']:.1f}%"
            )
        if cfg.get("test_accuracy_pct") is not None:
            details["Test overall accuracy"] = f"{cfg['test_accuracy_pct']:.1f}%"

        stages_count = cfg.get("num_stages_trained", cfg.get("num_stages", 0))
        return ToolResult(
            image=training_out["summary_image"],
            message=(
                f"Trained Viola-Jones cascade ({stages_count} stage(s), "
                f"{cfg['total_weak_classifiers']} weak classifiers) -> {xml_path.name}."
            ),
            details=details,
        )
