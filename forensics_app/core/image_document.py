"""The image currently being examined and its edit history."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


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
            loaded = image.copy()
        self.path = source
        self.original = loaded.copy()
        self.current = loaded
        self._undo.clear()
        self._redo.clear()

    def apply(self, image: Image.Image) -> None:
        if self.current is None:
            self.original = image.copy()
            self.current = image.copy()
            self._undo.clear()
            self._redo.clear()
            return
        self._undo.append(self.current.copy())
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
        self._undo.append(self.current.copy())
        self.current = self._redo.pop()
        return True

    def reset(self) -> bool:
        if self.original is None or self.current is None:
            return False
        self._undo.append(self.current.copy())
        self.current = self.original.copy()
        self._redo.clear()
        return True

    def save(self, path: str | Path) -> None:
        if self.current is None:
            raise RuntimeError("There is no image to save.")
        self.current.save(path)
