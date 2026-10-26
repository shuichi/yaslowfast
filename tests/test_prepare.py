from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yaslowfast.prepare import (
    VideoRecord,
    assign_splits,
    prepare_dataset,
    read_manifest,
    verify_dataset,
    window_frame_indices,
)
from yaslowfast.stats import compute_statistics


def test_manifest_formats_and_relative_paths(tmp_path):
    records = [{"video": "a space.mp4", "label": "歩く", "split": "train", "group_id": "person1"}]
    for suffix, contents in (
        (".json", json.dumps(records)),
        (".jsonl", json.dumps(records[0]) + "\n"),
        (".txt", "a space.mp4\t歩く\ttrain\tperson1\n"),
        (".tsv", "video\tlabel\tsplit\tgroup_id\na space.mp4\t歩く\ttrain\tperson1\n"),
    ):
        path = tmp_path / ("manifest" + suffix)
        path.write_text(contents, encoding="utf-8")
        assert read_manifest(path) == [
            VideoRecord(tmp_path / "a space.mp4", "歩く", "train", "person1")
        ]


def test_manifest_rejects_duplicate_video_and_cross_split_group(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps([{"video": "a.mp4", "label": "a"}, {"video": "a.mp4", "label": "a"}])
    )
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)
    path.write_text(
        json.dumps(
            [
                {"video": "a.mp4", "label": "a", "split": "train", "group_id": "same"},
                {"video": "b.mp4", "label": "a", "split": "val", "group_id": "same"},
            ]
        )
    )
    with pytest.raises(ValueError, match="multiple splits"):
        read_manifest(path)


def test_group_split_is_deterministic_and_keeps_groups_together():
    records = [
        VideoRecord(Path(f"/videos/{i}-{j}.mp4"), str(i % 2), group_id=f"g{i}")
        for i in range(12)
        for j in range(2)
    ]
    first = assign_splits(records, seed=3)
    assert first == assign_splits(records, seed=3)
    assert first != assign_splits(records, seed=5)
    for group in {record.group_id for record in first}:
        assert len({record.split for record in first if record.group_id == group}) == 1
    assert {record.split for record in first} == {"train", "val", "test"}
    explicit = assign_splits(
        [VideoRecord(Path("a"), "a", "val", "g"), VideoRecord(Path("b"), "b", group_id="g")]
    )
    assert [record.split for record in explicit] == ["val", "val"]


def test_windows_cover_tail_without_duplicate_last_window():
    assert window_frame_indices(3, 4, 2) == [(0, [0, 1, 2, 2], 3)]
    assert [start for start, _, _ in window_frame_indices(8, 4, 2)] == [0, 2, 4]
    assert [start for start, _, _ in window_frame_indices(9, 4, 2)] == [0, 2, 4, 5]
    with pytest.raises(ValueError):
        window_frame_indices(9, 4, 5)


def _make_video(path: Path, color: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=64x48:r=10:d=1,drawbox=x=0:y=0:w=16:h=16:color=white:t=fill",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
@pytest.mark.parametrize(
    "class_names,expected_mapping",
    [(None, {"blue": 0, "red": 1}), (("red", "blue"), {"red": 0, "blue": 1})],
)
def test_prepare_real_videos_verify_and_prevent_overwrite(tmp_path, class_names, expected_mapping):
    _make_video(tmp_path / "red.mp4", "red")
    _make_video(tmp_path / "blue.mp4", "blue")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"video": "red.mp4", "label": "red", "split": "train"},
                {"video": "blue.mp4", "label": "blue", "split": "val"},
            ]
        )
    )
    path = prepare_dataset(
        manifest,
        tmp_path / "dataset",
        fps=4,
        short_side=48,
        window_frames=4,
        window_stride=2,
        shard_size_mb=0.02,
        class_names=class_names,
    )
    dataset = json.loads(path.read_text())
    assert dataset["class_to_idx"] == expected_mapping
    assert dataset["task"] == "binary_classification"
    assert dataset["num_classes"] == 2
    assert dataset["splits"]["train"]["num_videos"] == 1
    assert dataset["splits"]["val"]["num_videos"] == 1
    assert dataset["normalization"]["split"] == "train"
    assert dataset["normalization"]["unique_frames"] == 4
    assert not (path.parent / ".incomplete").exists()
    assert verify_dataset(path) == {"shards": 2, "samples": 2, "images": 8, "videos": 2}
    with pytest.raises(FileExistsError):
        prepare_dataset(manifest, path.parent)
    shard = path.parent / dataset["splits"]["train"]["shards"][0]["path"]
    with shard.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum/size"):
        verify_dataset(path)


def test_bad_video_leaves_failure_report_and_unpublished_dataset(tmp_path):
    (tmp_path / "bad.mp4").write_bytes(b"invalid video")
    (tmp_path / "other.mp4").write_bytes(b"other invalid video")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"video": "bad.mp4", "label": "bad", "split": "train"},
                {"video": "other.mp4", "label": "other", "split": "train"},
            ]
        )
    )
    with pytest.raises(RuntimeError):
        prepare_dataset(manifest, tmp_path / "dataset")
    assert not (tmp_path / "dataset/dataset.json").exists()
    assert (tmp_path / "dataset/.incomplete").exists()
    assert len(json.loads((tmp_path / "dataset/errors.json").read_text())) == 1


def _prepare_jpeg_fixture(tmp_path, monkeypatch, *, train=True, constant=False):
    records = [
        {"video": "short.mp4", "label": "a", "split": "train" if train else "val"},
        {"video": "long.mp4", "label": "b", "split": "train" if train else "val"},
        {"video": "held_out.mp4", "label": "a", "split": "val"},
    ]
    saved_training_pixels = []
    for record in records:
        (tmp_path / record["video"]).write_bytes(record["video"].encode())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records))

    def fake_extract(video, directory, **kwargs):
        frames = []
        count = 2 if video.name == "short.mp4" else 7
        for index in range(count):
            path = Path(directory) / f"{index:06d}.jpg"
            value = 40 if constant else 20 + index * 20
            if video.name == "held_out.mp4":
                value = 250
            Image.new("RGB", (4, 3), (value, value, value)).save(path)
            if video.name != "held_out.mp4" and train:
                with Image.open(path) as image:
                    saved_training_pixels.append(np.asarray(image).reshape(-1, 3) / 255.0)
            frames.append(path)
        return frames

    monkeypatch.setattr("yaslowfast.prepare.extract_frames", fake_extract)
    return manifest, saved_training_pixels


def test_preparation_saves_training_statistics_without_overlap_or_padding(tmp_path, monkeypatch):
    manifest, pixels = _prepare_jpeg_fixture(tmp_path, monkeypatch)
    path = prepare_dataset(manifest, tmp_path / "dataset", window_frames=4, window_stride=2)
    normalization = json.loads(path.read_text())["normalization"]
    expected = np.concatenate(pixels)
    np.testing.assert_allclose(normalization["mean"], expected.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(normalization["std"], expected.std(axis=0), atol=1e-12)
    assert normalization["pixels"] == 9 * 4 * 3
    assert normalization["unique_frames"] == 9
    assert "dataset_fingerprint" not in normalization
    inspected = compute_statistics(path)
    for key, value in normalization.items():
        assert inspected[key] == value
    assert compute_statistics(path, max_frames=3)["unique_frames"] == 3


def test_preparation_without_training_split_omits_statistics(tmp_path, monkeypatch):
    manifest, _ = _prepare_jpeg_fixture(tmp_path, monkeypatch, train=False)
    path = prepare_dataset(manifest, tmp_path / "dataset", window_frames=4, window_stride=2)
    assert "normalization" not in json.loads(path.read_text())
    assert not (path.parent / ".incomplete").exists()


def test_skipped_video_does_not_contribute_partial_statistics(tmp_path, monkeypatch):
    from yaslowfast import prepare

    manifest, pixels = _prepare_jpeg_fixture(tmp_path, monkeypatch)
    records = json.loads(manifest.read_text())
    records.append({"video": "broken.mp4", "label": "a", "split": "train"})
    manifest.write_text(json.dumps(records))
    (tmp_path / "broken.mp4").write_bytes(b"broken video")
    extract = prepare.extract_frames

    def partly_broken_extract(video, directory, **kwargs):
        if video.name != "broken.mp4":
            return extract(video, directory, **kwargs)
        first, second = Path(directory) / "first.jpg", Path(directory) / "second.jpg"
        Image.new("RGB", (4, 3), (250, 250, 250)).save(first)
        second.write_bytes(b"not a JPEG")
        return [first, second]

    monkeypatch.setattr(prepare, "extract_frames", partly_broken_extract)
    path = prepare_dataset(
        manifest, tmp_path / "dataset", window_frames=4, window_stride=2, skip_errors=True
    )
    dataset = json.loads(path.read_text())
    assert dataset["num_failed_videos"] == 1
    assert dataset["normalization"]["unique_frames"] == 9
    expected = np.concatenate(pixels)
    np.testing.assert_allclose(dataset["normalization"]["mean"], expected.mean(axis=0))
    np.testing.assert_allclose(dataset["normalization"]["std"], expected.std(axis=0))


def test_zero_variance_statistics_prevent_dataset_publication(tmp_path, monkeypatch):
    manifest, _ = _prepare_jpeg_fixture(tmp_path, monkeypatch, constant=True)
    with pytest.raises(ValueError, match="zero variance"):
        prepare_dataset(manifest, tmp_path / "dataset", window_frames=4, window_stride=2)
    assert (tmp_path / "dataset/.incomplete").exists()
    assert not (tmp_path / "dataset/dataset.json").exists()


@pytest.mark.parametrize("labels", [("one",), ("one", "two", "three")])
def test_preparation_rejects_nonbinary_manifest_before_creating_output(tmp_path, labels):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [{"video": f"{index}.mp4", "label": label} for index, label in enumerate(labels)]
        )
    )
    with pytest.raises(ValueError, match="two"):
        prepare_dataset(manifest, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("class_names", [("one", "one"), ("one", "other"), ("one",), "ab"])
def test_preparation_rejects_invalid_explicit_class_names(tmp_path, class_names):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"video": "one.mp4", "label": "one"},
                {"video": "two.mp4", "label": "two"},
            ]
        )
    )
    with pytest.raises(ValueError, match="two"):
        prepare_dataset(manifest, tmp_path / "dataset", class_names=class_names)
    assert not (tmp_path / "dataset").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_skip_errors_cannot_publish_a_single_class_dataset(tmp_path):
    _make_video(tmp_path / "good.mp4", "red")
    (tmp_path / "bad.mp4").write_bytes(b"invalid video")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"video": "good.mp4", "label": "good", "split": "train"},
                {"video": "bad.mp4", "label": "bad", "split": "train"},
            ]
        )
    )
    with pytest.raises(ValueError, match="both binary classes"):
        prepare_dataset(manifest, tmp_path / "dataset", skip_errors=True)
    assert not (tmp_path / "dataset/dataset.json").exists()
    assert (tmp_path / "dataset/.incomplete").exists()
    assert len(json.loads((tmp_path / "dataset/errors.json").read_text())) == 1
