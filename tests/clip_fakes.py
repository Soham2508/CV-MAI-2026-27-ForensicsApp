"""A tiny stand-in for Hugging Face CLIP used by the Set 10 tests.

The image embedding is the image's mean RGB colour and each text points along
the colour of the animal it names, so red images are "cat", green "dog" and
blue "owl".
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch

TEXT_DIRECTIONS = {
    "cat": [1.0, 0.0, 0.0],
    "dog": [0.0, 1.0, 0.0],
    "owl": [0.0, 0.0, 1.0],
}
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


class FakeBatch(dict):
    def to(self, _device: str) -> FakeBatch:
        return self


class FakeProcessor:
    def __call__(self, images=None, text=None, **_kwargs) -> FakeBatch:
        if images is not None:
            means = [
                np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0)
                for im in images
            ]
            return FakeBatch(pixel_values=torch.tensor(np.stack(means)) / 255.0)
        return FakeBatch(texts=list(text))


class FakeOutput:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.pooler_output = tensor


class FakeClip:
    """Returns unnormalised embeddings so normalisation is exercised."""

    def __init__(self, scale: float = 10.0, wrap_outputs: bool = False) -> None:
        self.logit_scale = torch.tensor(float(np.log(scale)))
        self.wrap_outputs = wrap_outputs
        self.texts: list[str] = []
        self.images_embedded = 0

    def _out(self, tensor: torch.Tensor):
        return FakeOutput(tensor) if self.wrap_outputs else tensor

    def get_image_features(self, pixel_values):
        self.images_embedded += len(pixel_values)
        return self._out(3.0 * pixel_values + 1e-4)

    def get_text_features(self, texts):
        self.texts.extend(texts)
        rows = []
        for text in texts:
            word = next(k for k in TEXT_DIRECTIONS if k in text)
            vec = torch.tensor(TEXT_DIRECTIONS[word]) * 2.0
            if "close-up" in text:  # a template that moves the embedding a little
                vec = vec + torch.tensor([0.0, 0.0, 0.5])
            rows.append(vec)
        return self._out(torch.stack(rows))


def fake_clip(**kwargs) -> tuple[FakeClip, FakeProcessor]:
    return FakeClip(**kwargs), FakeProcessor()


def solid(color: tuple[int, int, int], size: tuple[int, int] = (24, 24)) -> Image.Image:
    return Image.new("RGB", size, color)
