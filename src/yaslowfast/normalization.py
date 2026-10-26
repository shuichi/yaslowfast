"""Shared RGB pixel statistics for preparation and standalone inspection."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


class RGBStatistics:
    """Accumulate decoded JPEG pixels without weighting overlapping windows."""

    def __init__(self) -> None:
        self.unique_frames = 0
        self.pixels = 0
        self.total = np.zeros(3, dtype=np.float64)
        self.total_squared = np.zeros(3, dtype=np.float64)

    def add_jpeg(self, source: bytes | Path) -> None:
        with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3)
        self.unique_frames += 1
        self.pixels += pixels.shape[0]
        # Integer-valued moments avoid rounding each pixel through /255 first.
        self.total += pixels.sum(axis=0)
        self.total_squared += np.square(pixels).sum(axis=0)

    def merge(self, other: RGBStatistics) -> None:
        self.unique_frames += other.unique_frames
        self.pixels += other.pixels
        self.total += other.total
        self.total_squared += other.total_squared

    def report(self) -> dict:
        if self.pixels == 0:
            raise ValueError("No training pixels found")
        mean = self.total / self.pixels
        std = np.sqrt(np.maximum(self.total_squared / self.pixels - mean**2, 0)) / 255.0
        if np.any(std <= 0):
            raise ValueError("At least one RGB channel has zero variance; cannot normalize")
        return {
            "mean": (mean / 255.0).tolist(),
            "std": std.tolist(),
            "unique_frames": self.unique_frames,
            "pixels": self.pixels,
            "split": "train",
            "note": "RGB in [0,1], all pixels of unique saved training JPEG frames before spatial crop",
        }
