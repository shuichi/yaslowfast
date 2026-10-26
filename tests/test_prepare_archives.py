from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from yaslowfast import prepare
from yaslowfast.config import DataConfig, ModelConfig
from yaslowfast.data import build_loader, read_dataset
from yaslowfast.stats import compute_statistics


def _archive(path, entries):
    with tarfile.open(path, "w") as archive:
        for name, content in entries:
            member = tarfile.TarInfo(name) if isinstance(name, str) else name
            if member.isfile():
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            else:
                archive.addfile(member)
    return path


def _inputs(root, count=3, reverse=False):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for label in ("positive", "negative"):
        entries = [(f"nested/{index}.mp4", f"{label}:{index}".encode()) for index in range(count)]
        paths.append(_archive(root / f"{label}.tar", entries[::-1] if reverse else entries))
    return paths


@pytest.fixture
def fake_decode(monkeypatch):
    decoded = []

    def extract(video, directory, **kwargs):
        content = Path(video).read_bytes()
        decoded.append(content)
        if content == b"broken":
            raise RuntimeError("deliberately broken video")
        frames = []
        for index in range(6):
            path = Path(directory) / f"{index:09d}.jpg"
            Image.new("RGB", (12, 8), (20 + index * 30,) * 3).save(path)
            frames.append(path)
        return frames

    monkeypatch.setattr(prepare, "extract_frames", extract)
    return decoded


def _prepare(positive, negative, output, **kwargs):
    return prepare.prepare_archives(
        positive,
        negative,
        output,
        fps=4,
        short_side=8,
        window_frames=4,
        window_stride=2,
        **kwargs,
    )


def test_archive_manifest_provenance_and_window_splits(tmp_path, fake_decode):
    positive, negative = _inputs(tmp_path / "input")
    path = _prepare(positive, negative, tmp_path / "dataset")
    metadata = read_dataset(path)
    manifest = [
        json.loads(line) for line in (path.parent / "manifest.jsonl").read_text().splitlines()
    ]
    assert metadata["class_to_idx"] == {"positive": 0, "negative": 1}
    assert len(manifest) == len(fake_decode) == 6
    assert {row["source_archive"] for row in manifest} == {str(positive), str(negative)}
    assert {row["source_member"] for row in manifest} == {f"nested/{i}.mp4" for i in range(3)}
    assert Counter((row["label"], row["split"]) for row in manifest) == Counter(
        (label, split) for label in ("positive", "negative") for split in ("train", "val", "test")
    )
    for row in manifest:
        digest = hashlib.sha256(
            f"{row['label']}:{Path(row['source_member']).stem}".encode()
        ).hexdigest()
        assert row["sha256"] == digest
        assert row["video_id"] == digest[:24]
        assert row["class_id"] == metadata["class_to_idx"][row["label"]]
        assert "group_id" not in row
        video = metadata["videos"][row["video_id"]]
        for key in ("sha256", "label", "class_id", "split", "source_archive", "source_member"):
            assert video[key] == row[key]
        assert "group_id" not in video
    for split, info in metadata["splits"].items():
        assert info["num_videos"] == 2
        for shard in info["shards"]:
            with tarfile.open(path.parent / shard["path"]) as archive:
                for member in archive:
                    if member.name.endswith(".json"):
                        sample = json.load(archive.extractfile(member))
                        assert "group_id" not in sample
                        assert metadata["videos"][sample["video_id"]]["split"] == split
    assert prepare.verify_dataset(path) == {"shards": 3, "samples": 12, "images": 48, "videos": 6}
    assert (
        compute_statistics(path)["unique_frames"]
        == metadata["normalization"]["unique_frames"]
        == 12
    )
    assert sorted(item.name for item in path.parent.iterdir()) == [
        "dataset.json",
        "errors.json",
        "manifest.jsonl",
        "test-000000.tar",
        "train-000000.tar",
        "val-000000.tar",
    ]


@pytest.mark.parametrize(
    "count,expected", [(1, (1, 0, 0)), (2, (1, 1, 0)), (3, (1, 1, 1)), (10, (8, 1, 1))]
)
def test_small_archive_classes_keep_training_examples(tmp_path, fake_decode, count, expected):
    positive, negative = _inputs(tmp_path / "input", count=count)
    path = _prepare(positive, negative, tmp_path / "dataset")
    metadata = read_dataset(path)
    for split, count_per_class in zip(("train", "val", "test"), expected):
        info = metadata["splits"][split]
        assert info["num_videos"] == count_per_class * 2
        if count_per_class:
            assert info["class_video_counts"] == {
                "positive": count_per_class,
                "negative": count_per_class,
            }
        else:
            assert info["shards"] == []
            assert not list(path.parent.glob(f"{split}-*.tar"))


def test_splits_stable_across_archive_order_location_and_seed(tmp_path, fake_decode):
    paths = []
    for name, reverse, seed in (("first", False, 42), ("moved", True, 42), ("other", False, 9)):
        positive, negative = _inputs(tmp_path / name, count=12, reverse=reverse)
        paths.append(_prepare(positive, negative, tmp_path / f"output-{name}", seed=seed))
    assignments = [
        {key: row["split"] for key, row in read_dataset(path)["videos"].items()} for path in paths
    ]
    assert assignments[0] == assignments[1]
    assert assignments[0] != assignments[2]
    first, moved = map(read_dataset, paths[:2])
    assert first["splits"] == moved["splits"]


def test_train_only_warning_and_cli_output(tmp_path, fake_decode, capsys, caplog):
    positive, negative = _inputs(tmp_path / "input", count=1)
    output = tmp_path / "dataset"
    prepare.main(
        ["--positive", str(positive), "--negative", str(negative), "--output", str(output)]
    )
    captured = capsys.readouterr()
    assert str(output / "dataset.json") in captured.out
    warning = captured.err + caplog.text
    assert "val" in warning.lower()
    assert "train" in warning.lower()
    assert read_dataset(output / "dataset.json")["splits"]["train"]["num_videos"] == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--positive", "positive.tar"],
        ["--negative", "negative.tar"],
        ["--positive", "positive.tar", "--negative", "negative.tar", "--manifest", "manifest.json"],
        ["--positive", "positive.tar", "--negative", "negative.tar", "--video-root", "videos"],
        [
            "--positive",
            "positive.tar",
            "--negative",
            "negative.tar",
            "--class-names",
            "positive",
            "negative",
        ],
    ],
)
def test_cli_rejects_ambiguous_or_incomplete_archive_inputs(tmp_path, extra):
    output = tmp_path / "dataset"
    with pytest.raises(SystemExit) as error:
        prepare.main([*extra, "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_nested_videos_ignore_metadata_and_keep_same_names_in_both_archives(tmp_path, fake_decode):
    directory = tarfile.TarInfo("nested/")
    directory.type = tarfile.DIRTYPE
    positive = _archive(
        tmp_path / "positive.tar",
        [
            (directory, b""),
            ("nested/same.MP4", b"positive"),
            ("README.txt", b"notes"),
            ("__MACOSX/nested/._same.MP4", b"resource fork"),
            ("nested/._same.MP4", b"resource fork"),
        ],
    )
    negative = _archive(tmp_path / "negative.tar", [("nested/same.MP4", b"negative")])
    path = _prepare(positive, negative, tmp_path / "dataset")
    assert read_dataset(path)["splits"]["train"]["num_videos"] == 2
    assert set(fake_decode) == {b"positive", b"negative"}


@pytest.mark.parametrize(
    "name", ["../outside.mp4", "/absolute.mp4", "nested/../../outside.mp4", "..\\outside.mp4"]
)
def test_unsafe_archive_paths_are_rejected(tmp_path, fake_decode, name):
    positive = _archive(tmp_path / "positive.tar", [(name, b"positive")])
    negative = _archive(tmp_path / "negative.tar", [("safe.mp4", b"negative")])
    with pytest.raises(ValueError):
        _prepare(positive, negative, tmp_path / "dataset")
    assert fake_decode == []
    assert not (tmp_path / "outside.mp4").exists()
    assert not (tmp_path / "dataset/dataset.json").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_archive_links_and_special_members_are_rejected(tmp_path, fake_decode, kind):
    member = tarfile.TarInfo("linked.mp4")
    member.type, member.linkname = kind, "safe.mp4"
    positive = _archive(tmp_path / "positive.tar", [("safe.mp4", b"positive"), (member, b"")])
    negative = _archive(tmp_path / "negative.tar", [("safe.mp4", b"negative")])
    with pytest.raises(ValueError):
        _prepare(positive, negative, tmp_path / "dataset")
    assert fake_decode == []


@pytest.mark.parametrize(
    "entries", [[], [("README.txt", b"notes")], [("same.mp4", b"a"), ("same.mp4", b"b")]]
)
def test_empty_video_archives_and_repeated_member_names_are_rejected(
    tmp_path, fake_decode, entries
):
    positive = _archive(tmp_path / "positive.tar", entries)
    negative = _archive(tmp_path / "negative.tar", [("negative.mp4", b"negative")])
    with pytest.raises(ValueError):
        _prepare(positive, negative, tmp_path / "dataset")
    assert fake_decode == []


@pytest.mark.parametrize("directory_first", [False, True])
@pytest.mark.parametrize("skip_errors", [False, True])
def test_file_directory_member_collisions_are_rejected_before_decode(
    tmp_path, fake_decode, directory_first, skip_errors
):
    directory = tarfile.TarInfo("clip.mp4/")
    directory.type = tarfile.DIRTYPE
    collision = [("clip.mp4", b"clip"), (directory, b"")]
    if directory_first:
        collision.reverse()
    positive = _archive(tmp_path / "positive.tar", [("good.mp4", b"positive"), *collision])
    negative = _archive(tmp_path / "negative.tar", [("good.mp4", b"negative")])
    with pytest.raises(ValueError, match="Duplicate archive member"):
        _prepare(positive, negative, tmp_path / "dataset", skip_errors=skip_errors)
    assert fake_decode == []
    assert not (tmp_path / "dataset/dataset.json").exists()


@pytest.mark.parametrize("across_classes", [False, True])
@pytest.mark.parametrize("skip_errors", [False, True])
def test_duplicate_contents_never_enter_different_splits(
    tmp_path, fake_decode, across_classes, skip_errors
):
    positive_entries = [("one.mp4", b"same")]
    if not across_classes:
        positive_entries.append(("two.mp4", b"same"))
    positive = _archive(tmp_path / "positive.tar", positive_entries)
    negative = _archive(
        tmp_path / "negative.tar", [("one.mp4", b"same" if across_classes else b"other")]
    )
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _prepare(positive, negative, tmp_path / "dataset", skip_errors=skip_errors)
    assert fake_decode == []
    assert not (tmp_path / "dataset/dataset.json").exists()


def test_archive_output_cannot_overwrite_existing_files(tmp_path, fake_decode):
    positive, negative = _inputs(tmp_path / "input")
    output = tmp_path / "dataset"
    output.mkdir()
    existing = output / "manifest.jsonl"
    existing.write_text("keep this")
    with pytest.raises(FileExistsError):
        _prepare(positive, negative, output)
    assert existing.read_text() == "keep this"
    assert fake_decode == []


@pytest.mark.parametrize("skip_errors", [False, True])
def test_decode_failure_keeps_source_provenance_and_cleans_temporary_files(
    tmp_path, fake_decode, skip_errors
):
    positive = _archive(
        tmp_path / "positive.tar", [("good.mp4", b"positive"), ("bad.mp4", b"broken")]
    )
    negative = _archive(tmp_path / "negative.tar", [("good.mp4", b"negative")])
    output = tmp_path / "dataset"
    kwargs = {"skip_errors": skip_errors, "split_ratios": (1, 0, 0)}
    if skip_errors:
        path = _prepare(positive, negative, output, **kwargs)
        metadata = read_dataset(path)
        assert metadata["num_failed_videos"] == 1
        assert metadata["splits"]["train"]["num_videos"] == 2
        assert metadata["normalization"]["unique_frames"] == 12
        assert not (output / ".incomplete").exists()
    else:
        with pytest.raises(RuntimeError, match="broken"):
            _prepare(positive, negative, output, **kwargs)
        assert (output / ".incomplete").exists()
        assert not (output / "dataset.json").exists()
    errors = json.loads((output / "errors.json").read_text())
    assert len(errors) == 1
    assert "bad.mp4" in json.dumps(errors)
    assert "positive.tar" in json.dumps(errors)
    assert not any(item.is_dir() for item in output.iterdir())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_real_tar_videos_verify_statistics_and_training_loader(tmp_path):
    archives = []
    colors = (("red", "green", "blue"), ("yellow", "magenta", "cyan"))
    for label, class_colors in zip(("positive", "negative"), colors):
        entries = []
        for index, color in enumerate(class_colors):
            video = tmp_path / f"{label}-{index}.mp4"
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
                    f"color=c={color}:s=64x48:r=4:d=1,"
                    "drawbox=x=0:y=0:w=16:h=16:color=white:t=fill,"
                    "drawbox=x=48:y=32:w=16:h=16:color=black:t=fill",
                    "-c:v",
                    "mpeg4",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "1",
                    str(video),
                ],
                check=True,
                capture_output=True,
            )
            entries.append((f"clips/{index}.mp4", video.read_bytes()))
        archives.append(_archive(tmp_path / f"{label}.tar", entries))
    path = prepare.prepare_archives(
        *archives,
        tmp_path / "dataset",
        fps=4,
        short_side=48,
        window_frames=4,
        window_stride=2,
    )
    metadata = read_dataset(path)
    assert prepare.verify_dataset(path) == {"shards": 3, "samples": 6, "images": 24, "videos": 6}
    assert compute_statistics(path)["mean"] == metadata["normalization"]["mean"]
    loader = build_loader(
        path,
        "train",
        config=DataConfig(train_min_scale=32, train_max_scale=48, pin_memory=False),
        model_config=ModelConfig(num_frames=4, crop_size=32),
        batch_size=2,
        num_workers=0,
        seed=42,
        training=False,
    )
    batch = next(iter(loader))
    assert sorted(batch["target"].tolist()) == [0, 1]
    assert tuple(batch["fast"].shape) == (2, 3, 4, 32, 32)
