"""Shared raw-video window batches for prediction and labelled evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterator

import torch


def iter_video_batches(
    video_path: str | Path,
    *,
    config: Any,
    preprocessing: dict[str, Any],
    batch_size: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield CPU tensors and window metadata, keeping only one video's frames.

    Callers that stop iteration early must close the generator to release its
    temporary directory immediately. The stored extraction and evaluation
    transforms are shared with prepared-dataset evaluation.
    """
    from .data import transform_frames
    from .prepare import extract_frames, window_frame_indices

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    required = {"fps", "short_side", "window_frames", "window_stride", "jpeg_quality"}
    missing = required.difference(preprocessing)
    if missing:
        raise ValueError(f"Checkpoint is missing data_preprocessing fields: {sorted(missing)}")
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    with tempfile.TemporaryDirectory(prefix="yaslowfast-video-") as directory:
        frames = extract_frames(
            video_path,
            Path(directory) / "frames",
            fps=preprocessing["fps"],
            short_side=preprocessing["short_side"],
            jpeg_quality=preprocessing["jpeg_quality"],
        )
        if not frames:
            raise ValueError(f"Video produced no frames: {video_path}")
        indices = window_frame_indices(
            len(frames), preprocessing["window_frames"], preprocessing["window_stride"]
        )
        for offset in range(0, len(indices), batch_size):
            index_batch = indices[offset : offset + batch_size]
            pairs = [
                transform_frames(
                    [frames[index] for index in frame_indices[:valid]],
                    config=config.data,
                    model_config=config.model,
                    training=False,
                )
                for _, frame_indices, valid in index_batch
            ]
            yield {
                "slow": torch.stack([pair[0] for pair in pairs]),
                "fast": torch.stack([pair[1] for pair in pairs]),
                "windows": [
                    {
                        "start_seconds": start / preprocessing["fps"],
                        "valid_frames": valid,
                    }
                    for start, _, valid in index_batch
                ],
            }
