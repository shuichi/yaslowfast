"""Deterministic JPEG/WebDataset input pipelines for SlowFast.

Training uses an infinite, independently seeded resampled shard stream per
rank/worker. The training loop sets the common number of microsteps. Evaluation
assigns finite shards without repetition or padding and permits empty ranks.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from webdataset.tariterators import tarfile_samples

from .labels import NUM_CLASSES, validate_class_to_idx

if TYPE_CHECKING:
    from .config import DataConfig, ModelConfig


def read_dataset(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if (path.parent / ".incomplete").exists():
        raise ValueError(f"Preprocessing is incomplete: {path.parent}")
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema_version (expected 1)")
    classes = dataset.get("class_to_idx")
    validate_class_to_idx(classes, context="dataset class_to_idx")
    if dataset.get("task", "binary_classification") != "binary_classification":
        raise ValueError("Dataset task must be binary_classification")
    if "num_classes" in dataset and (
        type(dataset["num_classes"]) is not int or dataset["num_classes"] != NUM_CLASSES
    ):
        raise ValueError("Dataset num_classes must be 2 for binary classification")
    preprocessing = dataset.get("preprocessing", {})
    if preprocessing.get("fps", 0) <= 0 or preprocessing.get("window_frames", 0) < 1:
        raise ValueError("Invalid dataset preprocessing specification")
    videos = dataset.get("videos")
    if not isinstance(videos, dict) or not videos:
        raise ValueError("Dataset has no videos")
    group_splits: dict[str, str] = {}
    for video_id, video in videos.items():
        if video.get("class_id") not in classes.values() or video.get("num_windows", 0) < 1:
            raise ValueError(f"Invalid video metadata: {video_id}")
        group_id = video.get("group_id")
        if group_id is not None:
            previous = group_splits.setdefault(group_id, video["split"])
            if previous != video["split"]:
                raise ValueError(f"Group {group_id!r} crosses dataset splits")
    seen_shards: set[str] = set()
    for name, split in dataset.get("splits", {}).items():
        selected = [video for video in videos.values() if video["split"] == name]
        if split.get("num_videos") != len(selected) or split.get("num_samples") != sum(
            video["num_windows"] for video in selected
        ):
            raise ValueError(f"Dataset counts disagree for split {name}")
        count = 0
        for shard in split["shards"]:
            shard_path = Path(shard["path"])
            if (
                shard_path.is_absolute()
                or ".." in shard_path.parts
                or str(shard_path) in seen_shards
            ):
                raise ValueError(f"Shard paths must be unique relative paths: {shard_path}")
            seen_shards.add(str(shard_path))
            if not (path.parent / shard_path).is_file() or shard["num_samples"] < 1:
                raise ValueError(f"Missing or empty shard: {shard_path}")
            count += shard["num_samples"]
        if count != split["num_samples"]:
            raise ValueError(f"Shard counts disagree for split {name}")
    if any(video["split"] not in dataset.get("splits", {}) for video in videos.values()):
        raise ValueError("Video refers to an absent split")
    return dataset


def dataset_fingerprint(path: str | Path) -> str:
    """Hash the schema including the stored shard hashes, independent of JSON spacing."""
    dataset = read_dataset(path)
    return hashlib.sha256(
        json.dumps(dataset, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _seed(base: int, epoch: int, rank: int, worker: int, purpose: str) -> int:
    encoded = f"{base}:{epoch}:{rank}:{worker}:{purpose}".encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _decode_jpeg(value: bytes | Path) -> np.ndarray:
    with Image.open(io.BytesIO(value) if isinstance(value, bytes) else value) as image:
        return np.asarray(image.convert("RGB")).copy()


def transform_frames(
    jpeg_frames: Sequence[bytes | Path],
    *,
    config: DataConfig,
    model_config: ModelConfig,
    training: bool = False,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select temporal frames, apply clip-consistent transforms, and pack pathways.

    Inputs are a complete stored window. Only the selected Fast frames are
    decoded. Short input sequences repeat their last frame, matching preparation.
    """
    from .models import pack_pathways

    if not jpeg_frames:
        raise ValueError("Cannot transform an empty video window")
    rng = rng if rng is not None else random.Random(0)
    num_frames, crop_size = model_config.num_frames, model_config.crop_size
    max_start = max(0, len(jpeg_frames) - num_frames)
    start = rng.randint(0, max_start) if training else max_start // 2
    indices = [min(start + i, len(jpeg_frames) - 1) for i in range(num_frames)]
    # A local cache prevents decoding repeated padding frames more than once.
    decoded = {index: _decode_jpeg(jpeg_frames[index]) for index in set(indices)}
    try:
        frames = (
            torch.from_numpy(np.stack([decoded[index] for index in indices]))
            .permute(0, 3, 1, 2)
            .float()
            .div_(255)
        )
    except ValueError as error:
        raise ValueError("All frames of a window must have the same spatial dimensions") from error
    height, width = frames.shape[-2:]
    target_short = (
        rng.randint(config.train_min_scale, config.train_max_scale)
        if training
        else config.train_min_scale
    )
    target_short = max(target_short, crop_size)
    new_height, new_width = (
        round(height * target_short / min(height, width)),
        round(width * target_short / min(height, width)),
    )
    frames = F.interpolate(
        frames, size=(new_height, new_width), mode="bilinear", align_corners=False, antialias=True
    )
    if training:
        top, left = rng.randint(0, new_height - crop_size), rng.randint(0, new_width - crop_size)
    else:
        top, left = (new_height - crop_size) // 2, (new_width - crop_size) // 2
    frames = frames[:, :, top : top + crop_size, left : left + crop_size]
    if training and rng.random() < config.horizontal_flip:
        frames = frames.flip(-1)
    if training and config.color_jitter > 0:
        jitter = config.color_jitter
        operations = ["brightness", "contrast", "saturation"]
        rng.shuffle(operations)
        for operation in operations:
            factor = rng.uniform(max(0.0, 1 - jitter), 1 + jitter)
            if operation == "brightness":
                frames = frames * factor
            else:
                grayscale = (
                    frames[:, :1] * 0.2989 + frames[:, 1:2] * 0.5870 + frames[:, 2:3] * 0.1140
                )
                baseline = (
                    grayscale.mean(dim=(-2, -1), keepdim=True)
                    if operation == "contrast"
                    else grayscale
                )
                frames = factor * frames + (1 - factor) * baseline
            frames = frames.clamp_(0, 1)
    frames = frames.permute(1, 0, 2, 3).contiguous()
    mean = frames.new_tensor(config.mean).view(3, 1, 1, 1)
    std = frames.new_tensor(config.std).view(3, 1, 1, 1)
    frames.sub_(mean).div_(std)
    return pack_pathways(frames, model_config.alpha)


def _shuffle_encoded(
    samples: Iterator[dict[str, Any]], buffer_size: int, rng: random.Random
) -> Iterator[dict[str, Any]]:
    if buffer_size <= 1:
        yield from samples
        return
    buffer: list[dict[str, Any]] = []
    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = sample
    rng.shuffle(buffer)
    yield from buffer


class VideoWebDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        *,
        config: DataConfig,
        model_config: ModelConfig,
        seed: int,
        epoch: int = 0,
        rank: int = 0,
        world_size: int = 1,
        training: bool = False,
    ):
        super().__init__()
        if not 0 <= rank < world_size:
            raise ValueError("Require 0 <= rank < world_size")
        self.path = Path(dataset_path).resolve()
        self.metadata = read_dataset(self.path)
        if split not in self.metadata["splits"]:
            raise ValueError(f"Unknown split: {split}")
        self.split, self.config, self.model_config = split, config, model_config
        self.seed, self.epoch, self.rank, self.world_size, self.training = (
            seed,
            epoch,
            rank,
            world_size,
            training,
        )
        self.info = self.metadata["splits"][split]
        self.urls = [str(self.path.parent / shard["path"]) for shard in self.info["shards"]]
        if training and not self.urls:
            raise ValueError(f"Training split {split} has no samples")
        if self.metadata["preprocessing"]["window_frames"] < model_config.num_frames:
            raise ValueError(
                "Stored window_frames must be >= model.num_frames; regenerate the dataset"
            )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        shard_rng = random.Random(_seed(self.seed, self.epoch, self.rank, worker, "shards"))
        shuffle_rng = random.Random(_seed(self.seed, self.epoch, self.rank, worker, "shuffle"))
        augment_rng = random.Random(_seed(self.seed, self.epoch, self.rank, worker, "augment"))
        if self.training:

            def shards() -> Iterator[dict[str, str]]:
                while True:
                    yield {"url": shard_rng.choice(self.urls)}

            sources = shards()
        else:
            assigned = self.urls[self.rank :: self.world_size][worker::workers]
            sources = iter({"url": url} for url in assigned)
        samples = tarfile_samples(sources)
        if self.training:
            samples = _shuffle_encoded(samples, self.config.shuffle_buffer, shuffle_rng)
        for sample in samples:
            try:
                metadata = json.loads(sample["json"])
                video = self.metadata["videos"][metadata["video_id"]]
                if video["split"] != self.split or metadata["class_id"] != video["class_id"]:
                    raise ValueError("Sample label/split disagrees with dataset manifest")
                names = sorted(name for name in sample if name.endswith(".jpg"))
                expected = self.metadata["preprocessing"]["window_frames"]
                if names != [f"{i:06d}.jpg" for i in range(expected)]:
                    raise ValueError("Sample JPEG frames are incomplete or not contiguous")
                valid_frames = metadata.get("valid_frames")
                if not isinstance(valid_frames, int) or not 1 <= valid_frames <= expected:
                    raise ValueError("Sample valid_frames must be in [1, window_frames]")
                # Temporal crop must use real frames, not the padded suffix.
                # Otherwise a short action can turn into an all-static clip.
                slow, fast = transform_frames(
                    [sample[name] for name in names[:valid_frames]],
                    config=self.config,
                    model_config=self.model_config,
                    training=self.training,
                    rng=augment_rng,
                )
                weight = (
                    self.info["num_samples"] / (self.info["num_videos"] * video["num_windows"])
                    if self.config.video_balanced
                    else 1.0
                )
                yield {
                    "slow": slow,
                    "fast": fast,
                    "target": metadata["class_id"],
                    "weight": torch.tensor(weight, dtype=torch.float32),
                    "video_id": metadata["video_id"],
                    "key": sample["__key__"],
                }
            except Exception as error:
                raise RuntimeError(
                    f"Invalid WebDataset sample {sample.get('__key__')} in {sample.get('__url__')}: {error}"
                ) from error


def build_loader(
    dataset_path: str | Path,
    split: str,
    *,
    config: DataConfig,
    model_config: ModelConfig,
    batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int = 0,
    rank: int = 0,
    world_size: int = 1,
    training: bool = False,
) -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    dataset = VideoWebDataset(
        dataset_path,
        split,
        config=config,
        model_config=model_config,
        seed=seed,
        epoch=epoch,
        rank=rank,
        world_size=world_size,
        training=training,
    )
    generator = torch.Generator().manual_seed(_seed(seed, epoch, rank, 0, "loader"))
    kwargs: dict[str, Any] = {}
    if num_workers:
        kwargs.update(
            prefetch_factor=config.prefetch_factor,
            persistent_workers=False,
            multiprocessing_context="spawn",
        )
    # A fresh loader per epoch makes the epoch part of every worker's seed.
    # Persistent workers would otherwise retain the previous epoch's dataset.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=training,
        pin_memory=config.pin_memory,
        generator=generator,
        **kwargs,
    )
