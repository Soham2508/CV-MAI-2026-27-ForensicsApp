"""Tool 1 — Viola–Jones (Haar + AdaBoost) Cascade Trainer.

Implements integral image computation, Haar-like feature evaluation, AdaBoost
weak-classifier selection from the reference notebook, and multi-stage cascade
training with XML/JSON persistence and validation plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import shutil
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data as skimage_data

from forensics_app.core import ImageDocument

from .base import ForensicsTool, ToolResult

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm"}
DEFAULT_CASCADE_DIR = Path("models/viola_jones")
DEFAULT_CASCADE_XML = DEFAULT_CASCADE_DIR / "cascade.xml"


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
    arr = np.asarray(image_arr, dtype=np.float64)
    max_val = float(np.max(arr))
    if max_val > 0:
        arr = arr / max_val
    return arr


def load_integral_images(
    path: str | Path,
    *,
    window_size: tuple[int, int] | None = None,
    max_samples: int | None = None,
) -> list[np.ndarray]:
    """Read images inside a folder and compute the normalized integral image of each.

    Matches Exercise #3 in the reference notebook while supporting optional
    window resizing and sample count limits.
    """
    folder = Path(path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Image directory does not exist: {folder}")

    ii_images: list[np.ndarray] = []
    files = sorted(
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_samples is not None and max_samples > 0:
        files = files[:max_samples]

    for file_path in files:
        try:
            with Image.open(file_path) as img:
                gray = img.convert("L")
                if window_size is not None and gray.size != window_size:
                    gray = gray.resize(window_size, Image.Resampling.BILINEAR)
                arr = np.asarray(gray, dtype=np.float64)
        except OSError:
            continue

        arr = _normalize_patch(arr)
        integral_image = compute_integral_image(arr)
        max_ii = float(np.max(integral_image))
        if max_ii > 0:
            integral_image = integral_image / max_ii
        ii_images.append(integral_image)

    return ii_images


class HaarLikeFeature:
    """Class representing a Haar-like feature (Exercise #4 in the notebook)."""

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
        self.type = tuple(feature_type)
        self.top_left = (int(position[0]), int(position[1]))
        self.bottom_right = (int(position[0] + width), int(position[1] + height))
        self.width = int(width)
        self.height = int(height)
        self.threshold = float(threshold)
        self.polarity = int(polarity)
        self.weight = float(weight)

    def get_score(self, int_img: np.ndarray) -> float:
        """Get feature score for a given integral image array."""
        score = 0.0
        tl0, tl1 = self.top_left
        br0, br1 = self.bottom_right
        w, h = self.width, self.height

        if self.type == FeatureType.TWO_VERTICAL:
            first = sum_region(int_img, self.top_left, (tl0 + w, int(tl1 + h / 2)))
            second = sum_region(int_img, (tl0, int(tl1 + h / 2)), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = sum_region(int_img, self.top_left, (int(tl0 + w / 2), tl1 + h))
            second = sum_region(int_img, (int(tl0 + w / 2), tl1), self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = sum_region(int_img, self.top_left, (int(tl0 + w / 3), tl1 + h))
            second = sum_region(
                int_img,
                (int(tl0 + w / 3), tl1),
                (int(tl0 + 2 * w / 3), tl1 + h),
            )
            third = sum_region(int_img, (int(tl0 + 2 * w / 3), tl1), self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = sum_region(int_img, self.top_left, (br0, int(tl1 + h / 3)))
            second = sum_region(
                int_img,
                (tl0, int(tl1 + h / 3)),
                (br0, int(tl1 + 2 * h / 3)),
            )
            third = sum_region(int_img, (tl0, int(tl1 + 2 * h / 3)), self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            first = sum_region(
                int_img, self.top_left, (int(tl0 + w / 2), int(tl1 + h / 2))
            )
            second = sum_region(
                int_img, (int(tl0 + w / 2), tl1), (br0, int(tl1 + h / 2))
            )
            third = sum_region(
                int_img, (tl0, int(tl1 + h / 2)), (int(tl0 + w / 2), br1)
            )
            fourth = sum_region(
                int_img, (int(tl0 + w / 2), int(tl1 + h / 2)), self.bottom_right
            )
            score = first - second - third + fourth
        return float(score)

    def get_vote(self, int_img: np.ndarray) -> float:
        """Return weighted vote (+weight or -weight) for a given integral image."""
        score = self.get_score(int_img)
        return self.weight * (1 if score < self.polarity * self.threshold else -1)

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
    def from_dict(cls, data: dict[str, Any]) -> HaarLikeFeature:
        ftype_raw = data.get("feature_type", "TWO_VERTICAL")
        if isinstance(ftype_raw, str):
            ftype = NAME_TO_FEATURE_TYPE.get(ftype_raw, FeatureType.TWO_VERTICAL)
        else:
            ftype = tuple(ftype_raw)
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
) -> list[HaarLikeFeature]:
    """Create Haar-like features for all valid sizes and locations."""
    mode = feature_mode.upper()
    if mode == "CORE":
        active_types = [
            FeatureType.TWO_VERTICAL,
            FeatureType.TWO_HORIZONTAL,
            FeatureType.THREE_HORIZONTAL,
            FeatureType.THREE_VERTICAL,
        ]
    else:
        active_types = FeatureTypes

    features: list[HaarLikeFeature] = []
    for feature in active_types:
        feature_start_width = max(min_feature_width, feature[0])
        for feature_width in range(feature_start_width, max_feature_width, feature[0]):
            feature_start_height = max(min_feature_height, feature[1])
            for feature_height in range(
                feature_start_height, max_feature_height, feature[1]
            ):
                for x in range(img_height - max_feature_height):
                    for y in range(img_width - max_feature_width):
                        features.append(
                            HaarLikeFeature(
                                feature, (x, y), feature_width, feature_height, 0, 1
                            )
                        )
                        features.append(
                            HaarLikeFeature(
                                feature, (x, y), feature_width, feature_height, 0, -1
                            )
                        )
    return features


def learn(
    positive_iis: list[np.ndarray],
    negative_iis: list[np.ndarray],
    num_classifiers: int = -1,
    min_feature_width: int = 1,
    max_feature_width: int = -1,
    min_feature_height: int = 1,
    max_feature_height: int = -1,
    *,
    feature_mode: str = "BASIC",
) -> list[HaarLikeFeature]:
    """Select a set of weak classifiers iteratively using AdaBoost (Exercise #4/#5).

    Vectorized inner loop over identical candidate features for fast execution
    while preserving the exact mathematical results of the reference notebook.
    """
    num_pos = len(positive_iis)
    num_neg = len(negative_iis)
    if num_pos == 0 or num_neg == 0:
        raise ValueError(
            "Both positive and negative integral image lists must be non-empty."
        )

    num_imgs = num_pos + num_neg
    img_height, img_width = positive_iis[0].shape

    max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    max_feature_width = img_width if max_feature_width == -1 else max_feature_width

    pos_weights = np.ones(num_pos, dtype=np.float64) / (2.0 * num_pos)
    neg_weights = np.ones(num_neg, dtype=np.float64) / (2.0 * num_neg)
    weights = np.hstack((pos_weights, neg_weights))
    labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1.0))

    images = positive_iis + negative_iis

    features = _create_features(
        img_height,
        img_width,
        min_feature_width,
        max_feature_width,
        min_feature_height,
        max_feature_height,
        feature_mode=feature_mode,
    )
    if not features:
        # Fallback to wider feature bounds if window is small
        features = _create_features(
            img_height,
            img_width,
            1,
            max(2, img_width - 1),
            1,
            max(2, img_height - 1),
            feature_mode=feature_mode,
        )
    num_features = len(features)
    feature_indexes = list(range(num_features))

    num_classifiers = (
        num_features
        if num_classifiers == -1
        else min(int(num_classifiers), num_features)
    )

    votes = np.zeros((num_imgs, num_features), dtype=np.float64)
    for i in range(num_imgs):
        votes[i, :] = np.array(
            [feature.get_vote(images[i]) for feature in features], dtype=np.float64
        )

    classifiers: list[HaarLikeFeature] = []
    eps = 1e-12

    for _ in range(num_classifiers):
        if not feature_indexes:
            break

        weights *= 1.0 / np.sum(weights)

        active_votes = votes[:, feature_indexes]
        mismatches = (active_votes != labels[:, None]).astype(np.float64)
        classification_errors = weights @ mismatches

        min_error_idx = int(np.argmin(classification_errors))
        best_error = float(
            np.clip(classification_errors[min_error_idx], eps, 1.0 - eps)
        )
        best_feature_idx = feature_indexes[min_error_idx]

        best_feature = features[best_feature_idx]
        feature_weight = 0.5 * float(np.log((1.0 - best_error) / best_error))
        best_feature.weight = feature_weight

        classifiers.append(best_feature)

        wrong_mask = labels != votes[:, best_feature_idx]
        factor_wrong = np.sqrt((1.0 - best_error) / best_error)
        factor_right = np.sqrt(best_error / (1.0 - best_error))
        weights = np.where(wrong_mask, weights * factor_wrong, weights * factor_right)

        feature_indexes.pop(min_error_idx)

    return classifiers


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

    for th in thresholds:
        correct_pos_train = sum(
            ensemble_vote_all_t(ii_faces_training, classifiers, float(th))
        )
        incorrect_neg_train = sum(
            ensemble_vote_all_t(ii_non_faces_training, classifiers, float(th))
        )
        correct_neg_train = len(ii_non_faces_training) - incorrect_neg_train

        pcnt_faces_train.append(100.0 * correct_pos_train / total_pos_train)
        pcnt_nofaces_train.append(100.0 * correct_neg_train / total_neg_train)

        if ii_faces_testing:
            correct_pos_test = sum(
                ensemble_vote_all_t(ii_faces_testing, classifiers, float(th))
            )
            pcnt_faces_test.append(100.0 * correct_pos_test / total_pos_test)
        if ii_non_faces_testing:
            incorrect_neg_test = sum(
                ensemble_vote_all_t(ii_non_faces_testing, classifiers, float(th))
            )
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
    """A single boosted stage in a Viola–Jones cascade."""

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
    """Complete multi-stage Viola–Jones Haar + AdaBoost cascade model."""

    window_width: int = 24
    window_height: int = 24
    feature_type: str = "HAAR"
    stage_type: str = "BOOST"
    boost_type: str = "GAB"
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
        out_path = Path(xml_path)
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
        """Load a custom-trained Viola–Jones cascade XML file."""
        path = Path(xml_path)
        if not path.exists():
            raise FileNotFoundError(f"Cascade XML not found: {path}")

        tree = ET.parse(path)
        root = tree.getroot()
        cascade_el = root.find("cascade")
        if cascade_el is None:
            raise ValueError(f"Invalid Viola–Jones cascade XML: {path}")

        type_id = cascade_el.attrib.get("type_id", "")
        if type_id != "forensics-viola-jones-cascade":
            raise ValueError(
                f"XML file {path.name} is not a custom Forensics Viola–Jones cascade."
            )

        width = int(cascade_el.findtext("width", "24"))
        height = int(cascade_el.findtext("height", "24"))
        stage_type = cascade_el.findtext("stageType", "BOOST")
        feature_type = cascade_el.findtext("featureType", "HAAR")
        boost_type = cascade_el.findtext("boostType", "GAB")
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
    """Step 1: Validate positive/negative folders, annotations, and dimensions."""
    pos_folder = Path(positive_dir)
    neg_folder = Path(negative_dir)
    if not pos_folder.exists() or not pos_folder.is_dir():
        raise ValueError(f"Positive images folder does not exist: {pos_folder}")
    if not neg_folder.exists() or not neg_folder.is_dir():
        raise ValueError(f"Negative images folder does not exist: {neg_folder}")

    if window_size[0] < 8 or window_size[1] < 8:
        raise ValueError("Training window dimensions must be at least 8 × 8 pixels.")

    pos_files = sorted(
        f
        for f in pos_folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    neg_files = sorted(
        f
        for f in neg_folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    annotated_boxes = 0
    if annotations_path:
        entries = parse_annotations_file(annotations_path, base_dir=pos_folder)
        annotated_boxes = sum(len(boxes) for _, boxes in entries)
        if annotated_boxes == 0:
            raise ValueError("Annotation file contains no valid bounding boxes.")
    elif not pos_files:
        raise ValueError(f"No positive image files found in {pos_folder}.")

    if not neg_files:
        raise ValueError(f"No negative image files found in {neg_folder}.")

    return {
        "positive_files": len(pos_files),
        "negative_files": len(neg_files),
        "annotated_boxes": annotated_boxes,
        "window_size": window_size,
    }


def generate_positive_samples(
    positive_dir: str | Path,
    annotations_path: str | Path | None = None,
    window_size: tuple[int, int] = (24, 24),
    max_samples: int = 100,
    vec_output_path: str | Path | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Step 2: Generate normalized positive patches and integral images."""
    pos_folder = Path(positive_dir)
    patches: list[np.ndarray] = []
    integral_images: list[np.ndarray] = []

    if annotations_path:
        entries = parse_annotations_file(annotations_path, base_dir=pos_folder)
        for img_path, boxes in entries:
            if not img_path.exists():
                continue
            try:
                with Image.open(img_path) as img:
                    gray = img.convert("L")
                    for x, y, w, h in boxes:
                        if w <= 0 or h <= 0:
                            continue
                        crop = gray.crop((x, y, x + w, y + h)).resize(
                            window_size, Image.Resampling.BILINEAR
                        )
                        arr = _normalize_patch(np.asarray(crop, dtype=np.float64))
                        ii = compute_integral_image(arr)
                        max_ii = float(np.max(ii))
                        if max_ii > 0:
                            ii = ii / max_ii
                        patches.append(arr)
                        integral_images.append(ii)
                        if len(integral_images) >= max_samples:
                            break
            except OSError:
                continue
            if len(integral_images) >= max_samples:
                break
    else:
        files = sorted(
            f
            for f in pos_folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )[:max_samples]
        for file_path in files:
            try:
                with Image.open(file_path) as img:
                    gray = img.convert("L").resize(
                        window_size, Image.Resampling.BILINEAR
                    )
                    arr = _normalize_patch(np.asarray(gray, dtype=np.float64))
                    ii = compute_integral_image(arr)
                    max_ii = float(np.max(ii))
                    if max_ii > 0:
                        ii = ii / max_ii
                    patches.append(arr)
                    integral_images.append(ii)
            except OSError:
                continue

    if vec_output_path and patches:
        vec_path = Path(vec_output_path)
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vec_path, np.stack(patches, axis=0))

    return patches, integral_images


def generate_negative_samples(
    negative_dir: str | Path,
    window_size: tuple[int, int] = (24, 24),
    max_samples: int = 100,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract normalized negative patches and integral images from backgrounds."""
    neg_folder = Path(negative_dir)
    files = sorted(
        f
        for f in neg_folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    patches: list[np.ndarray] = []
    integral_images: list[np.ndarray] = []
    w, h = window_size

    for file_path in files:
        try:
            with Image.open(file_path) as img:
                gray = img.convert("L")
                if gray.width > w * 2 and gray.height > h * 2:
                    # Extract non-overlapping patches from larger background images
                    for top in range(0, gray.height - h + 1, h):
                        for left in range(0, gray.width - w + 1, w):
                            crop = gray.crop((left, top, left + w, top + h))
                            arr = _normalize_patch(np.asarray(crop, dtype=np.float64))
                            ii = compute_integral_image(arr)
                            max_ii = float(np.max(ii))
                            if max_ii > 0:
                                ii = ii / max_ii
                            patches.append(arr)
                            integral_images.append(ii)
                            if len(integral_images) >= max_samples:
                                break
                        if len(integral_images) >= max_samples:
                            break
                else:
                    resized = gray.resize(window_size, Image.Resampling.BILINEAR)
                    arr = _normalize_patch(np.asarray(resized, dtype=np.float64))
                    ii = compute_integral_image(arr)
                    max_ii = float(np.max(ii))
                    if max_ii > 0:
                        ii = ii / max_ii
                    patches.append(arr)
                    integral_images.append(ii)
        except OSError:
            continue
        if len(integral_images) >= max_samples:
            break

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
    num_pos: int = 50,
    num_neg: int = 50,
    num_stages: int = 3,
    max_weak_count: int = 5,
    min_hit_rate: float = 0.995,
    max_false_alarm_rate: float = 0.5,
    weight_trimming_rate: float = 0.95,
    max_weak_tree_depth: int = 1,
    boost_type: str = "GAB",
    feature_mode: str = "BASIC",
    min_feature_width: int = 8,
    max_feature_width: int = 10,
    min_feature_height: int = 8,
    max_feature_height: int = 10,
) -> dict[str, Any]:
    """Execute the 5-step Viola–Jones cascade training workflow and export models."""
    t_start = time.perf_counter()
    logs: list[str] = []
    window_size = (int(window_width), int(window_height))

    # Step 1: Dataset validation
    validation = validate_cascade_dataset(
        positive_dir,
        negative_dir,
        annotations_path=annotations_path,
        window_size=window_size,
    )
    logs.append(
        f"[Step 1] Validated dataset: {validation['positive_files']} pos files, "
        f"{validation['negative_files']} neg files, "
        f"window={window_width}x{window_height}"
    )

    # Check if external OpenCV 3.4 cascade training utilities exist
    createsamples_bin = shutil.which("opencv_createsamples")
    traincascade_bin = shutil.which("opencv_traincascade")
    logs.append(
        f"[Env] External OpenCV 3.4 binaries: createsamples={bool(createsamples_bin)}, "
        f"traincascade={bool(traincascade_bin)} (using built-in Haar+AdaBoost engine)"
    )

    out_xml = Path(output_xml_path)
    out_dir = out_xml.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "samples.vec.npy"

    # Step 2: Positive & negative sample generation
    pos_patches, pos_iis = generate_positive_samples(
        positive_dir,
        annotations_path=annotations_path,
        window_size=window_size,
        max_samples=num_pos,
        vec_output_path=vec_path,
    )
    neg_patches, neg_iis = generate_negative_samples(
        negative_dir,
        window_size=window_size,
        max_samples=num_neg,
    )
    if not pos_iis or not neg_iis:
        raise ValueError("Insufficient valid positive or negative samples loaded.")

    logs.append(
        f"[Step 2] Generated {len(pos_iis)} positive and "
        f"{len(neg_iis)} negative integral images."
    )

    # Adjust feature dimensions safely to fit within the training window
    eff_min_fw = min(min_feature_width, max(2, window_width - 4))
    eff_max_fw = min(max_feature_width, max(eff_min_fw + 2, window_width - 1))
    eff_min_fh = min(min_feature_height, max(2, window_height - 4))
    eff_max_fh = min(max_feature_height, max(eff_min_fh + 2, window_height - 1))

    # Step 3 & 4: Cascade training & stage monitoring
    stages: list[CascadeStage] = []
    active_neg_iis = list(neg_iis)

    for stage_idx in range(int(num_stages)):
        stage_t0 = time.perf_counter()
        if not active_neg_iis:
            active_neg_iis = list(neg_iis)

        stage_classifiers = learn(
            pos_iis,
            active_neg_iis,
            num_classifiers=max_weak_count,
            min_feature_width=eff_min_fw,
            max_feature_width=eff_max_fw,
            min_feature_height=eff_min_fh,
            max_feature_height=eff_max_fh,
            feature_mode=feature_mode,
        )

        pos_scores = np.array(
            [sum(c.get_vote(ii) for c in stage_classifiers) for ii in pos_iis],
            dtype=np.float64,
        )
        neg_scores = np.array(
            [sum(c.get_vote(ii) for c in stage_classifiers) for ii in active_neg_iis],
            dtype=np.float64,
        )

        # Choose stage threshold to satisfy min_hit_rate on positive training samples
        sorted_pos = np.sort(pos_scores)
        cutoff_idx = max(
            0,
            min(
                len(sorted_pos) - 1,
                int(np.floor((1.0 - min_hit_rate) * len(sorted_pos))),
            ),
        )
        stage_threshold = min(0.0, float(sorted_pos[cutoff_idx]) - 1e-6)

        hit_rate = float(np.mean(pos_scores >= stage_threshold))
        fa_rate = float(np.mean(neg_scores >= stage_threshold))

        stage = CascadeStage(
            stage_index=stage_idx,
            classifiers=stage_classifiers,
            stage_threshold=stage_threshold,
            hit_rate=hit_rate,
            false_alarm_rate=fa_rate,
        )
        stages.append(stage)

        stage_dt = time.perf_counter() - stage_t0
        logs.append(
            f"[Stage {stage_idx + 1}/{num_stages}] "
            f"weak_classifiers={len(stage_classifiers)}, "
            f"hit_rate={hit_rate:.3f}, false_alarm={fa_rate:.3f}, time={stage_dt:.2f}s"
        )

        # Filter surviving false positives for subsequent stages if enough remain
        surviving_negs = [
            ii for ii, s in zip(active_neg_iis, neg_scores) if s >= stage_threshold
        ]
        if len(surviving_negs) >= 2:
            active_neg_iis = surviving_negs

        if fa_rate <= max_false_alarm_rate**num_stages and stage_idx >= 0:
            pass

    model = ViolaJonesCascadeModel(
        window_width=window_width,
        window_height=window_height,
        feature_type="HAAR",
        stage_type="BOOST",
        boost_type=boost_type,
        feature_mode=feature_mode,
        stages=stages,
    )

    # Step 5: Model export (XML, JSON config, training log)
    saved_xml = model.save_xml(out_xml)

    all_clfs = model.all_classifiers
    pos_test_iis = (
        load_integral_images(pos_test_dir, window_size=window_size, max_samples=num_pos)
        if pos_test_dir and Path(pos_test_dir).exists()
        else None
    )
    neg_test_iis = (
        load_integral_images(neg_test_dir, window_size=window_size, max_samples=num_neg)
        if neg_test_dir and Path(neg_test_dir).exists()
        else None
    )

    threshold_eval = evaluate_thresholds(
        pos_iis, neg_iis, pos_test_iis, neg_test_iis, all_clfs
    )

    train_pos_votes = ensemble_vote_all(pos_iis, all_clfs)
    train_neg_votes = ensemble_vote_all(neg_iis, all_clfs)
    train_face_acc = 100.0 * sum(train_pos_votes) / len(pos_iis)
    train_nonface_acc = 100.0 * (len(neg_iis) - sum(train_neg_votes)) / len(neg_iis)

    test_face_acc: float | None = None
    test_nonface_acc: float | None = None
    if pos_test_iis:
        test_pos_votes = ensemble_vote_all(pos_test_iis, all_clfs)
        test_face_acc = 100.0 * sum(test_pos_votes) / len(pos_test_iis)
    if neg_test_iis:
        test_neg_votes = ensemble_vote_all(neg_test_iis, all_clfs)
        test_nonface_acc = (
            100.0 * (len(neg_test_iis) - sum(test_neg_votes)) / len(neg_test_iis)
        )

    duration = time.perf_counter() - t_start
    logs.append(
        f"[Step 5] Exported cascade XML to {saved_xml} in {duration:.2f}s. "
        f"Train Faces Acc: {train_face_acc:.1f}%, "
        f"Train Non-Faces Acc: {train_nonface_acc:.1f}%"
    )

    config_data = {
        "positive_dir": str(positive_dir),
        "negative_dir": str(negative_dir),
        "annotations_path": str(annotations_path) if annotations_path else None,
        "window_width": window_width,
        "window_height": window_height,
        "num_pos": len(pos_iis),
        "num_neg": len(neg_iis),
        "num_stages": len(stages),
        "max_weak_count": max_weak_count,
        "total_weak_classifiers": len(all_clfs),
        "feature_type": "HAAR",
        "stage_type": "BOOST",
        "boost_type": boost_type,
        "feature_mode": feature_mode,
        "min_hit_rate": min_hit_rate,
        "max_false_alarm_rate": max_false_alarm_rate,
        "weight_trimming_rate": weight_trimming_rate,
        "max_weak_tree_depth": max_weak_tree_depth,
        "training_duration_sec": round(duration, 4),
        "train_face_accuracy_pct": round(train_face_acc, 2),
        "train_nonface_accuracy_pct": round(train_nonface_acc, 2),
        "test_face_accuracy_pct": (
            round(test_face_acc, 2) if test_face_acc is not None else None
        ),
        "test_nonface_accuracy_pct": (
            round(test_nonface_acc, 2) if test_nonface_acc is not None else None
        ),
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
    fig = plt.figure(figsize=(12, 7.5), dpi=100)
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
        rect = plt.Rectangle(
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
        f"Viola–Jones AdaBoost Evaluation ({len(model.stages)} stage(s), "
        f"{len(all_clfs)} weak classifiers)",
        fontsize=11,
    )
    ax_bar.set_xlabel("Decision Threshold")
    ax_bar.set_ylabel("Correctly Identified (%)")
    ax_bar.set_ylim(0, 108)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    ax_bar.legend(loc="lower right", ncol=2, fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def ask_viola_jones_training_parameters(
    parent: tk.Misc,
) -> dict[str, Any] | None:
    """Display configuration dialog for Viola–Jones cascade training."""
    dialog = tk.Toplevel(parent)
    dialog.title("Viola–Jones Cascade Trainer")
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    pos_dir_var = tk.StringVar(value="")
    ann_file_var = tk.StringVar(value="")
    neg_dir_var = tk.StringVar(value="")
    output_xml_var = tk.StringVar(value=str(DEFAULT_CASCADE_XML))

    win_w_var = tk.StringVar(value="24")
    win_h_var = tk.StringVar(value="24")
    num_pos_var = tk.StringVar(value="20")
    num_neg_var = tk.StringVar(value="20")

    # Advanced Cascade Parameters (Plan Section 2.B)
    num_stages_var = tk.StringVar(value="3")
    boost_type_var = tk.StringVar(value="GAB")
    min_hit_rate_var = tk.StringVar(value="0.995")
    max_fa_rate_var = tk.StringVar(value="0.5")
    max_weak_var = tk.StringVar(value="5")
    weight_trim_var = tk.StringVar(value="0.95")
    tree_depth_var = tk.StringVar(value="1")
    feature_mode_var = tk.StringVar(value="BASIC")

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
        demo_root = Path("datasets/viola_jones_demo")
        paths = create_synthetic_face_nonface_dataset(
            demo_root, window_size=(24, 24), n_pos=20, n_neg=20
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

    ttk.Label(ds_frame, text="Output cascade XML:").grid(
        row=3, column=0, sticky="w", pady=3
    )
    ttk.Entry(ds_frame, textvariable=output_xml_var, width=38).grid(
        row=3, column=1, padx=6, pady=3
    )
    ttk.Button(ds_frame, text="Browse…", command=_browse_xml).grid(
        row=3, column=2, pady=3
    )

    ttk.Button(
        ds_frame,
        text="Prepare Built-in Demo Face/Non-Face Dataset",
        command=_generate_demo_data,
    ).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 2))

    dims_frame = ttk.Frame(ds_frame)
    dims_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(6, 0))
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
        ttk.Spinbox(dims_frame, textvariable=var, from_=8, to=500, width=6).grid(
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
        values=("GAB", "DAB", "RAB", "LB"),
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

    ttk.Label(adv_frame, text="Min hit rate:").grid(row=2, column=0, sticky="w", pady=3)
    ttk.Spinbox(
        adv_frame,
        textvariable=min_hit_rate_var,
        from_=0.5,
        to=1.0,
        increment=0.005,
        width=8,
    ).grid(row=2, column=1, sticky="w", padx=6, pady=3)

    ttk.Label(adv_frame, text="Max false alarm rate:").grid(
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

    ttk.Label(adv_frame, text="Weight trimming / Depth:").grid(
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
    ttk.Spinbox(trim_sub, textvariable=tree_depth_var, from_=1, to=4, width=3).pack(
        side="left", padx=(4, 0)
    )

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
            if (
                win_w < 8
                or win_h < 8
                or n_pos < 1
                or n_neg < 1
                or n_stages < 1
                or max_weak < 1
                or not (0.0 < min_hr <= 1.0)
                or not (0.0 < max_fa <= 1.0)
            ):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid Parameters",
                "Please enter valid positive numbers for all training parameters.",
                parent=dialog,
            )
            return

        result = {
            "positive_dir": pos_dir,
            "negative_dir": neg_dir,
            "annotations_path": ann_file_var.get().strip() or None,
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
    title = "Viola–Jones Trainer"
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

        training_out = train_viola_jones_cascade(**parameters)
        cfg = training_out["config"]
        xml_path = Path(training_out["xml_path"])

        details: dict[str, Any] = {
            "Operation": "Viola–Jones Cascade Training",
            "Cascade XML": xml_path.name,
            "XML path": str(xml_path),
            "Config JSON": Path(training_out["config_path"]).name,
            "Training log": Path(training_out["log_path"]).name,
            "Window size": f"{cfg['window_width']} × {cfg['window_height']}",
            "Positive / Negative": f"{cfg['num_pos']} / {cfg['num_neg']}",
            "Stages trained": cfg["num_stages"],
            "Weak classifiers": cfg["total_weak_classifiers"],
            "Boosting / Mode": f"{cfg['boost_type']} / {cfg['feature_mode']}",
            "Train face accuracy": f"{cfg['train_face_accuracy_pct']:.1f}%",
            "Train non-face accuracy": f"{cfg['train_nonface_accuracy_pct']:.1f}%",
            "Training duration": f"{cfg['training_duration_sec']:.2f} s",
        }
        if cfg.get("test_face_accuracy_pct") is not None:
            details["Test face accuracy"] = f"{cfg['test_face_accuracy_pct']:.1f}%"
        if cfg.get("test_nonface_accuracy_pct") is not None:
            details["Test non-face accuracy"] = (
                f"{cfg['test_nonface_accuracy_pct']:.1f}%"
            )

        return ToolResult(
            image=training_out["summary_image"],
            message=(
                f"Trained Viola–Jones cascade ({cfg['num_stages']} stage(s), "
                f"{cfg['total_weak_classifiers']} weak classifiers) -> {xml_path.name}."
            ),
            details=details,
        )
