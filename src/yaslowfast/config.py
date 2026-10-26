"""Strict, serializable configuration shared by training and inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    num_frames: int = 32
    alpha: int = 4
    crop_size: int = 224
    dropout: float = 0.5
    pooling: str = "adaptive"
    width_per_group: int = 64
    stage_depths: tuple[int, ...] = (3, 4, 6, 3)


@dataclass
class DataConfig:
    dataset: str = "data/dataset.json"
    train_split: str = "train"
    val_split: str = "val"
    batch_size: int = 8
    num_workers: int = 4
    shuffle_buffer: int = 256
    normalization: str = "auto"
    mean: tuple[float, ...] = (0.45, 0.45, 0.45)
    std: tuple[float, ...] = (0.225, 0.225, 0.225)
    train_min_scale: int = 256
    train_max_scale: int = 320
    horizontal_flip: float = 0.0
    color_jitter: float = 0.0
    video_balanced: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class TrainConfig:
    output_dir: str = "runs/slowfast_r50"
    device: str = "auto"
    seed: int = 42
    epochs: int = 196
    steps_per_epoch: int | None = None
    accumulation_steps: int = 1
    learning_rate: float = 0.1
    reference_batch_size: int = 64
    scale_learning_rate: bool = True
    momentum: float = 0.9
    weight_decay: float = 0.0001
    norm_weight_decay: float = 0.0
    warmup_epochs: float = 34.0
    warmup_start_factor: float = 0.1
    min_lr_ratio: float = 0.0
    precision: str = "auto"
    sync_batchnorm: bool = False
    grad_clip_norm: float | None = None
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    compile: bool = False
    channels_last_3d: bool = False
    deterministic: bool = False
    allow_tf32: bool = True
    eval_every: int = 1
    log_every: int = 10
    best_metric: str = "accuracy"
    distributed_timeout_seconds: int = 300


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(cls: type, raw: Any) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    unknown = set(raw) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**raw)


def config_from_dict(raw: dict[str, Any]) -> Config:
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    unknown = set(raw) - {"model", "data", "train"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    config = Config(
        _section(ModelConfig, raw.get("model", {})),
        _section(DataConfig, raw.get("data", {})),
        _section(TrainConfig, raw.get("train", {})),
    )
    config.model.stage_depths = tuple(config.model.stage_depths)
    config.data.mean = tuple(config.data.mean)
    config.data.std = tuple(config.data.std)
    validate_config(config)
    return config


def _validate_rgb_statistics(mean: Any, std: Any, *, context: str) -> None:
    if (
        not isinstance(mean, (list, tuple))
        or not isinstance(std, (list, tuple))
        or len(mean) != 3
        or len(std) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (*mean, *std)
        )
        or any(value <= 0 for value in std)
    ):
        raise ValueError(f"{context}: mean/std must be finite RGB triples and std must be positive")


def apply_dataset_normalization(config: DataConfig, dataset: dict[str, Any]) -> bool:
    """Resolve training statistics before building loaders or saving checkpoint config.

    Evaluation and inference use the resolved values from their checkpoint and
    must not call this helper on the dataset being evaluated. Legacy datasets
    without statistics retain their configured mean/std.
    """
    if config.normalization not in ("auto", "config"):
        raise ValueError("data.normalization must be auto or config")
    if config.normalization == "config" or "normalization" not in dataset:
        return False
    statistics = dataset["normalization"]
    if not isinstance(statistics, dict):
        raise ValueError("Dataset normalization must be a mapping")
    if statistics.get("split") != config.train_split:
        raise ValueError("Dataset normalization split does not match data.train_split")
    mean, std = statistics.get("mean"), statistics.get("std")
    _validate_rgb_statistics(mean, std, context="Dataset normalization")
    config.mean = tuple(float(value) for value in mean)
    config.std = tuple(float(value) for value in std)
    return True


def validate_config(config: Config) -> None:
    m, d, t = config.model, config.data, config.train
    if m.alpha != 4 or m.num_frames < 4 or m.num_frames % m.alpha:
        raise ValueError("SlowFast requires alpha=4 and num_frames divisible by 4")
    if m.crop_size < 32 or m.pooling not in {"adaptive", "reference"}:
        raise ValueError("crop_size must be >=32; pooling must be adaptive or reference")
    if m.pooling == "reference" and (m.num_frames != 32 or m.crop_size < 224):
        raise ValueError("Reference pooling requires num_frames=32 and crop_size>=224")
    if m.width_per_group < 8 or m.width_per_group % 8:
        raise ValueError("width_per_group must be a positive multiple of 8")
    if len(m.stage_depths) != 4 or any(x < 1 for x in m.stage_depths):
        raise ValueError("stage_depths must contain four positive integers")
    if not 0 <= m.dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    if d.batch_size < 1 or d.num_workers < 0 or d.shuffle_buffer < 1:
        raise ValueError("Invalid batch_size, num_workers, or shuffle_buffer")
    if d.normalization not in ("auto", "config"):
        raise ValueError("data.normalization must be auto or config")
    _validate_rgb_statistics(d.mean, d.std, context="RGB normalization")
    if d.train_min_scale < m.crop_size or d.train_max_scale < d.train_min_scale:
        raise ValueError("Require crop_size <= train_min_scale <= train_max_scale")
    if not 0 <= d.horizontal_flip <= 1 or not 0 <= d.color_jitter <= 1:
        raise ValueError("horizontal_flip and color_jitter must be in [0,1]")
    if d.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if t.epochs < 1 or t.accumulation_steps < 1 or t.reference_batch_size < 1:
        raise ValueError("epochs, accumulation_steps, reference_batch_size must be positive")
    if t.steps_per_epoch is not None and t.steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive or null")
    if (
        t.learning_rate <= 0
        or t.weight_decay < 0
        or t.norm_weight_decay < 0
        or not 0 <= t.momentum < 1
    ):
        raise ValueError("Invalid optimizer settings")
    if t.warmup_epochs < 0 or not 0 <= t.warmup_start_factor <= 1:
        raise ValueError("Invalid warmup settings")
    if not 0 <= t.min_lr_ratio <= 1 or not 0 <= t.label_smoothing < 1 or t.mixup_alpha < 0:
        raise ValueError("Invalid min_lr_ratio, label_smoothing, or mixup_alpha")
    if t.grad_clip_norm is not None and t.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive or null")
    if t.precision not in {"auto", "fp32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, fp32, bf16, or fp16")
    if min(t.eval_every, t.log_every, t.distributed_timeout_seconds) < 1:
        raise ValueError("eval_every, log_every, distributed_timeout_seconds must be positive")
    if t.best_metric not in {"accuracy", "macro_f1"}:
        raise ValueError("best_metric must be accuracy or macro_f1")


def load_config(path: str | Path) -> Config:
    """Paths in YAML are relative to the command's working directory."""
    with Path(path).open(encoding="utf-8") as stream:
        return config_from_dict(yaml.safe_load(stream) or {})
