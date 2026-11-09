"""The image currently being examined and its edit history."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Each undo step stores a full image copy (36 MB for a 12-megapixel RGB photo),
# so the history is capped to keep memory bounded.
MAX_HISTORY = 20

# Modes that hold more than 8 bits per sample. Pillow's convert("L") clips
# these instead of scaling them, so every tool would see an almost white image.
HIGH_BIT_DEPTH_MODES = {"I;16", "I;16L", "I;16B", "I;16N", "I", "F"}


def to_display_depth(image: Image.Image) -> Image.Image:
    """Rescale 16-bit, 32-bit integer and float images to 8-bit grayscale.

    16-bit modes map 0–65535 linearly to 0–255. ``I`` and ``F`` images keep
    their values when they already fit 0–255, use the 16-bit mapping when they
    fit 0–65535, and are otherwise stretched from their minimum to maximum.
    """
    if image.mode not in HIGH_BIT_DEPTH_MODES:
        return image
    values = np.asarray(image, dtype=np.float64)
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    if image.mode.startswith("I;16") or (low >= 0.0 and 255.0 < high <= 65535.0):
        scaled = values / 257.0
    elif low >= 0.0 and high <= 255.0:
        scaled = values
    elif high > low:
        scaled = (values - low) / (high - low) * 255.0
    else:
        scaled = np.zeros_like(values)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0)
    return Image.fromarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8), mode="L")


class ImageDocument:
    """Own the original/current images without depending on Tkinter.

    Tools receive this object, but should not mutate ``current`` directly. They
    return a new PIL image and the main window calls :meth:`apply`, preserving
    undo/redo history automatically.
    """

    def __init__(self) -> None:
        self.path: Path | None = None
        self.original: Image.Image | None = None
        self.current: Image.Image | None = None
        self._undo: list[Image.Image] = []
        self._redo: list[Image.Image] = []

    @property
    def is_loaded(self) -> bool:
        return self.current is not None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def is_modified(self) -> bool:
        return bool(self._undo)

    def load(self, path: str | Path) -> None:
        """Load an image and detach it from the underlying file handle."""
        source = Path(path)
        with Image.open(source) as image:
            # Phone cameras store rotation in EXIF; Pillow does not apply it.
            loaded = to_display_depth(ImageOps.exif_transpose(image) or image).copy()
        self.path = source
        self.original = loaded.copy()
        self.current = loaded
        self._undo.clear()
        self._redo.clear()

    def _push_undo(self, image: Image.Image) -> None:
        self._undo.append(image)
        if len(self._undo) > MAX_HISTORY:
            del self._undo[0]

    def apply(self, image: Image.Image) -> None:
        if self.current is None:
            # Tools that run without an open image (trainers, webcam) still
            # return a picture: show it as a new, unsaved document.
            self.path = None
            self.original = image.copy()
            self.current = image.copy()
            self._undo.clear()
            self._redo.clear()
            return
        self._push_undo(self.current.copy())
        self.current = image.copy()
        self._redo.clear()

    def undo(self) -> bool:
        if self.current is None or not self._undo:
            return False
        self._redo.append(self.current.copy())
        self.current = self._undo.pop()
        return True

    def redo(self) -> bool:
        if self.current is None or not self._redo:
            return False
        self._push_undo(self.current.copy())
        self.current = self._redo.pop()
        return True

    def reset(self) -> bool:
        if self.original is None or self.current is None:
            return False
        self._push_undo(self.current.copy())
        self.current = self.original.copy()
        self._redo.clear()
        return True

    def save(self, path: str | Path) -> None:
        if self.current is None:
            raise RuntimeError("There is no image to save.")
        self.current.save(path)
