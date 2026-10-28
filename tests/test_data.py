from __future__ import annotations

import io
import itertools
import json
import random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import default_collate

from yaslowfast.checkpoint import capture_rng_state
from yaslowfast.config import DataConfig, ModelConfig
from yaslowfast.data import (
    VideoWebDataset,
    build_loader,
    dataset_fingerprint,
    finish_training_epoch,
    read_dataset,
    transform_frames,
)
from yaslowfast.prepare import _ShardWriter


def _jpeg(value=127, gradient=False):
    if gradient:
        pixels = np.broadcast_to(
            np.arange(64, dtype=np.uint8)[None, :, None] * 3, (48, 64, 3)
        ).copy()
    else:
        pixels = np.full((48, 64, 3), value, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture
def dataset_path(tmp_path):
    videos = {
        "a": {"class_id": 0, "split": "train", "num_windows": 2},
        "b": {"class_id": 1, "split": "train", "num_windows": 1},
        "c": {"class_id": 0, "split": "val", "num_windows": 2},
        "d": {"class_id": 1, "split": "val", "num_windows": 1},
    }
    splits = {}
    for split in ("train", "val"):
        writer = _ShardWriter(tmp_path, split, 1)  # One complete sample per shard.
        selected = {key: value for key, value in videos.items() if value["split"] == split}
        for video_id, video in selected.items():
            for index in range(video["num_windows"]):
                writer.write(
                    f"{video_id}_{index}",
                    {
                        "video_id": video_id,
                        "class_id": video["class_id"],
                        "window_index": index,
                        "valid_frames": 8,
                        "frame_times": [i / 15 for i in range(8)],
                    },
                    [_jpeg(i * 20) for i in range(8)],
                )
        writer.close()
        splits[split] = {"shards": writer.shards, "num_samples": 3, "num_videos": 2}
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "class_to_idx": {"zero": 0, "one": 1},
                "videos": videos,
                "preprocessing": {"fps": 15, "window_frames": 8},
                "splits": splits,
            }
        )
    )
    return path


def _configs():
    return DataConfig(
        train_min_scale=32, train_max_scale=48, pin_memory=False, shuffle_buffer=3
    ), ModelConfig(num_frames=4, crop_size=32)


def _loader(path, split="val", **kwargs):
    data, model = _configs()
    return build_loader(
        path,
        split,
        config=data,
        model_config=model,
        batch_size=2,
        num_workers=kwargs.pop("num_workers", 0),
        seed=11,
        **kwargs,
    )


def test_eval_shards_are_exhaustive_disjoint_and_allow_empty_ranks(dataset_path):
    keys = []
    for rank in range(5):
        batches = list(_loader(dataset_path, rank=rank, world_size=5))
        keys.extend(key for batch in batches for key in batch["key"])
        if rank >= 3:
            assert batches == []
    assert Counter(keys) == Counter({"c_0": 1, "c_1": 1, "d_0": 1})


def test_eval_workers_do_not_duplicate_samples(dataset_path):
    batches = list(_loader(dataset_path, num_workers=2))
    assert Counter(key for batch in batches for key in batch["key"]) == Counter(
        {"c_0": 1, "c_1": 1, "d_0": 1}
    )


def test_infinite_train_stream_is_seeded_by_epoch_and_rank(dataset_path):
    def sample(epoch=0, rank=0):
        return list(
            itertools.islice(
                _loader(dataset_path, "train", training=True, epoch=epoch, rank=rank, world_size=2),
                8,
            )
        )

    first, repeated = sample(), sample()
    assert [b["key"] for b in first] == [b["key"] for b in repeated]
    assert all(torch.equal(a["fast"], b["fast"]) for a, b in zip(first, repeated))
    assert [b["key"] for b in first] != [b["key"] for b in sample(epoch=1)]
    assert [b["key"] for b in first] != [b["key"] for b in sample(rank=1)]
    for batch in first:
        assert batch["fast"].shape == (2, 3, 4, 32, 32)
        assert batch["slow"].shape == (2, 3, 1, 32, 32)
        for video_id, weight in zip(batch["video_id"], batch["weight"]):
            assert weight.item() == pytest.approx(0.75 if video_id == "a" else 1.5)


def _legacy_worker_batches(path, data, model, workers, count, epoch, monkeypatch):
    """Reference: unbounded worker streams, collated in the original round-robin order."""
    streams = []
    for worker in range(max(1, workers)):
        with monkeypatch.context() as patch:
            info = SimpleNamespace(id=worker, num_workers=workers) if workers else None
            patch.setattr("yaslowfast.data.get_worker_info", lambda: info)
            dataset = VideoWebDataset(
                path,
                "train",
                config=data,
                model_config=model,
                seed=11,
                epoch=epoch,
                rank=1,
                world_size=2,
                training=True,
            )
            # No stop event, matching the original infinite dataset. Materialize
            # each worker independently so scheduling cannot affect the reference.
            stream = iter(dataset)
            batches = [default_collate([next(stream), next(stream)]) for _ in range(count)]
            streams.append(iter(batches))
    return [next(streams[index % len(streams)]) for index in range(count)]


@pytest.mark.parametrize("workers,batches", [(0, 5), (2, 1), (2, 5), (3, 7)])
def test_training_epoch_keeps_sampling_and_workers_exit_normally(
    dataset_path, monkeypatch, capfd, workers, batches
):
    data, model = _configs()
    data.horizontal_flip, data.color_jitter = 0.5, 0.3
    for epoch in (2, 3):
        expected = _legacy_worker_batches(
            dataset_path, data, model, workers, batches, epoch, monkeypatch
        )
        loader = build_loader(
            dataset_path,
            "train",
            config=data,
            model_config=model,
            batch_size=2,
            num_workers=workers,
            seed=11,
            epoch=epoch,
            rank=1,
            world_size=2,
            training=True,
        )
        iterator = iter(loader)
        processes = list(iterator._workers) if workers else []
        for reference in expected:
            actual = next(iterator)
            for key in reference:
                if isinstance(reference[key], torch.Tensor):
                    torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
                else:
                    assert actual[key] == reference[key]
        before = capture_rng_state()
        finish_training_epoch(loader, iterator)
        after = capture_rng_state()
        assert before["python"] == after["python"]
        assert before["numpy"] == after["numpy"]
        torch.testing.assert_close(before["torch"], after["torch"], rtol=0, atol=0)
        with pytest.raises(StopIteration):
            next(iterator)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
    errors = capfd.readouterr().err
    assert "Exception ignored in" not in errors
    assert "terminate called without an active exception" not in errors


def test_training_epoch_can_stop_before_reading_any_samples(dataset_path, monkeypatch):
    loader = _loader(dataset_path, "train", training=True)
    iterator = iter(loader)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Stopping must not initialize the shuffle buffer")

    monkeypatch.setattr("yaslowfast.data.tarfile_samples", unexpected_read)
    # Generator construction is lazy; stop even before constructing the input pipeline.
    finish_training_epoch(loader, iterator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinned-memory regression")
def test_training_epoch_drains_large_pinned_batches(dataset_path, capfd):
    data, _ = _configs()
    data.pin_memory = True
    data.train_min_scale = data.train_max_scale = 224
    model = ModelConfig(num_frames=8, crop_size=224)
    for epoch in range(3):
        loader = build_loader(
            dataset_path,
            "train",
            config=data,
            model_config=model,
            batch_size=8,
            num_workers=2,
            seed=11,
            epoch=epoch,
            training=True,
        )
        iterator = iter(loader)
        processes = list(iterator._workers)
        batch = next(iterator)
        assert batch["fast"].is_pinned() and batch["slow"].is_pinned()
        finish_training_epoch(loader, iterator)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
    errors = capfd.readouterr().err
    assert "Exception ignored in" not in errors
    assert "terminate called without an active exception" not in errors


def test_clip_augmentation_is_temporally_consistent():
    data, model = _configs()
    data.horizontal_flip = 0.5
    data.color_jitter = 0.3
    slow, fast = transform_frames(
        [_jpeg(gradient=True)] * 8,
        config=data,
        model_config=model,
        training=True,
        rng=random.Random(2),
    )
    assert all(torch.equal(fast[:, 0], fast[:, index]) for index in range(4))
    assert torch.equal(slow[:, 0], fast[:, 0])


def test_eval_center_selection_and_normalization():
    data, model = _configs()
    data.mean, data.std = (0, 0, 0), (1, 1, 1)
    _, fast = transform_frames([_jpeg(i * 20) for i in range(8)], config=data, model_config=model)
    assert torch.allclose(
        fast.mean(dim=(0, 2, 3)), torch.tensor([40, 60, 80, 100]) / 255, atol=1 / 255
    )


def test_short_windows_preserve_motion_before_padding(tmp_path):
    data, model = _configs()
    data.mean, data.std = (0, 0, 0), (1, 1, 1)
    writer = _ShardWriter(tmp_path, "train", 100000)
    writer.write(
        "short",
        {"video_id": "short", "class_id": 0, "valid_frames": 2},
        [_jpeg(20), _jpeg(100)] + [_jpeg(100)] * 6,
    )
    writer.close()
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "class_to_idx": {"class": 0, "other": 1},
                "preprocessing": {"fps": 15, "window_frames": 8},
                "videos": {"short": {"split": "train", "class_id": 0, "num_windows": 1}},
                "splits": {"train": {"num_samples": 1, "num_videos": 1, "shards": writer.shards}},
            }
        )
    )
    for training in (False, True):
        loader = build_loader(
            path,
            "train",
            config=data,
            model_config=model,
            batch_size=1,
            num_workers=0,
            seed=11,
            training=training,
        )
        batch = next(iter(loader))
        assert torch.allclose(
            batch["fast"].mean(dim=(0, 1, 3, 4)),
            torch.tensor([20, 100, 100, 100]) / 255,
            atol=1 / 255,
        )


def test_dataset_schema_and_fingerprint(dataset_path):
    original = dataset_fingerprint(dataset_path)
    dataset = read_dataset(dataset_path)
    dataset_path.write_text(json.dumps(dataset, indent=2))
    assert dataset_fingerprint(dataset_path) == original
    dataset["splits"]["train"]["num_samples"] = 20
    dataset_path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match="counts disagree"):
        read_dataset(dataset_path)


@pytest.mark.parametrize(
    "mapping",
    [{"one": 0}, {"one": 0, "two": 1, "three": 2}, {"zero": 0, "one": 2}],
)
def test_dataset_rejects_nonbinary_class_mapping(dataset_path, mapping):
    dataset = json.loads(dataset_path.read_text())
    dataset["class_to_idx"] = mapping
    dataset_path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError):
        read_dataset(dataset_path)


@pytest.mark.parametrize(
    "field,value", [("task", "multiclass_classification"), ("num_classes", 3), ("num_classes", 2.0)]
)
def test_dataset_rejects_conflicting_binary_task_metadata(dataset_path, field, value):
    dataset = json.loads(dataset_path.read_text())
    dataset[field] = value
    dataset_path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError, match=field):
        read_dataset(dataset_path)


def test_existing_binary_dataset_without_task_metadata_remains_compatible(dataset_path):
    legacy = read_dataset(dataset_path)
    assert "task" not in legacy and "num_classes" not in legacy
    legacy["task"] = "binary_classification"
    legacy["num_classes"] = 2
    dataset_path.write_text(json.dumps(legacy))
    assert read_dataset(dataset_path)["class_to_idx"] == {"zero": 0, "one": 1}
