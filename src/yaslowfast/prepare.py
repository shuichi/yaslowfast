"""Convert binary-labelled videos into reproducible, bounded WebDataset samples.

FFmpeg is used only during preprocessing. Training reads JPEGs directly from tar
shards; it never invokes a video decoder. All paths passed to FFmpeg are argv
elements, not shell commands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .labels import NUM_CLASSES, validate_class_to_idx
from .normalization import RGBStatistics


@dataclass(frozen=True)
class VideoRecord:
    video: Path
    label: str
    split: str | None = None
    group_id: str | None = None


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path, video_root: str | Path | None = None) -> list[VideoRecord]:
    """Read JSON arrays, JSONL, or tab-separated video/label[/split/group_id].

    TSV accepts either a header or headerless rows; paths containing spaces are
    preserved. Relative video paths resolve against ``video_root`` when supplied,
    otherwise the manifest's parent directory.
    """
    path = Path(path).resolve()
    root = Path(video_root).resolve() if video_root is not None else path.parent
    contents = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        raw = json.loads(contents)
        if not isinstance(raw, list):
            raise ValueError("A JSON manifest must be an array of video records")
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        raw = [json.loads(line) for line in contents.splitlines() if line.strip()]
    else:
        rows = list(csv.reader(io.StringIO(contents), delimiter="\t"))
        rows = [row for row in rows if row and any(value.strip() for value in row)]
        if rows and rows[0][:2] == ["video", "label"]:
            header, rows = rows[0], rows[1:]
            if len(set(header)) != len(header) or set(header) - {
                "video",
                "label",
                "split",
                "group_id",
            }:
                raise ValueError(
                    "TSV header must contain video, label, and optionally split/group_id"
                )
        else:
            header = ["video", "label", "split", "group_id"]
        raw = []
        for number, row in enumerate(rows, 1):
            if len(row) < 2 or len(row) > len(header):
                raise ValueError(f"Invalid TSV row {number}: expected 2 to {len(header)} columns")
            raw.append(dict(zip(header, row)))
    if not raw:
        raise ValueError("Manifest contains no videos")
    records: list[VideoRecord] = []
    seen: set[Path] = set()
    group_splits: dict[str, str] = {}
    for index, row in enumerate(raw):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("video"), str)
            or not isinstance(row.get("label"), str)
        ):
            raise ValueError(f"Record {index}: video and label must be strings")
        if not row["video"].strip() or not row["label"].strip():
            raise ValueError(f"Record {index}: video and label cannot be empty")
        video = (root / row["video"]).resolve()
        if video in seen:
            raise ValueError(f"Duplicate source video: {video}")
        seen.add(video)
        split = row.get("split") or None
        group_id = row.get("group_id") or None
        if split is not None and (
            not isinstance(split, str) or split not in {"train", "val", "test"}
        ):
            raise ValueError(f"Record {index}: split must be train, val, or test")
        if group_id is not None and not isinstance(group_id, str):
            raise ValueError(f"Record {index}: group_id must be a string")
        if group_id is not None and split is not None:
            previous = group_splits.setdefault(group_id, split)
            if previous != split:
                raise ValueError(f"Group {group_id!r} appears in multiple splits")
        records.append(VideoRecord(video, row["label"], split, group_id))
    return records


def assign_splits(
    records: Sequence[VideoRecord],
    *,
    seed: int = 42,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> list[VideoRecord]:
    """Split complete groups, stratifying by each group's set of labels.

    Explicit splits take precedence, including for unassigned members of an
    explicitly assigned group. Small strata retain training examples; validation
    and test receive a group when there are enough groups to populate them.
    """
    if (
        len(split_ratios) != 3
        or any(not math.isfinite(x) or x < 0 for x in split_ratios)
        or not math.isclose(sum(split_ratios), 1.0)
    ):
        raise ValueError("split_ratios must be three non-negative fractions summing to one")
    groups: dict[str, list[VideoRecord]] = defaultdict(list)
    for record in records:
        groups[
            "group:" + record.group_id
            if record.group_id is not None
            else "video:" + str(record.video)
        ].append(record)
    assigned: dict[str, str] = {}
    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for key, group in groups.items():
        explicit = {record.split for record in group if record.split is not None}
        if len(explicit) > 1:
            raise ValueError(f"Group {key} appears in multiple splits")
        if explicit:
            assigned[key] = next(iter(explicit))
        else:
            strata[tuple(sorted({record.label for record in group}))].append(key)
    rng = random.Random(seed)
    split_names = ("train", "val", "test")
    for labels in sorted(strata):
        keys = sorted(strata[labels])
        rng.shuffle(keys)
        raw_counts = [len(keys) * ratio for ratio in split_ratios]
        counts = [int(value) for value in raw_counts]
        for index in sorted(range(3), key=lambda i: (raw_counts[i] - counts[i], -i), reverse=True)[
            : len(keys) - sum(counts)
        ]:
            counts[index] += 1
        # Keep a train group first, then fill requested held-out splits when the
        # stratum has enough groups. This is preferable to an empty val set for
        # e.g. three videos with an 80/10/10 requested split.
        positive = [i for i, ratio in enumerate(split_ratios) if ratio > 0]
        for index in positive[: len(keys)]:
            if counts[index] == 0:
                donors = [i for i in positive if counts[i] > 1]
                if donors:
                    donor = max(donors, key=lambda i: counts[i])
                    counts[donor] -= 1
                    counts[index] += 1
        offset = 0
        for split, count in zip(split_names, counts):
            for key in keys[offset : offset + count]:
                assigned[key] = split
            offset += count
    return [
        VideoRecord(record.video, record.label, assigned[key], record.group_id)
        for key, group in groups.items()
        for record in group
    ]


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 15.0,
    short_side: int = 320,
    jpeg_quality: int = 95,
) -> list[Path]:
    """Decode a video to numbered JPEGs with timestamps ``index / fps``.

    PTS is normalized to zero before FFmpeg's fps filter, so VFR inputs are
    sampled in seconds rather than by source-frame index. JPEG quality 1..100
    maps to FFmpeg's MJPEG qscale 31..2 (it is not a JPEG percentage setting).
    The directory must not already contain frames.
    """
    if not math.isfinite(fps) or fps <= 0 or short_side < 1 or not 1 <= jpeg_quality <= 100:
        raise ValueError("fps/short_side must be positive and jpeg_quality must be in [1, 100]")
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            "FFmpeg is required for preprocessing; install ffmpeg and put it on PATH"
        )
    video_path, output_dir = Path(video_path).resolve(), Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"Frame output directory must be empty: {output_dir}")
    qscale = min(31, max(2, round(31 - (jpeg_quality - 1) * 29 / 99)))
    video_filter = f"setpts=PTS-STARTPTS,fps={fps:.12g},scale={short_side}:{short_side}:force_original_aspect_ratio=increase"
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        video_filter,
        "-q:v",
        str(qscale),
        "-threads",
        "1",
        "-start_number",
        "0",
        str(output_dir / "%09d.jpg"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed for {video_path}: {result.stderr.strip()[-4000:]}")
    frames = sorted(output_dir.glob("*.jpg"))
    if not frames:
        raise ValueError(f"Video produced no frames at {fps:g} fps: {video_path}")
    # Detect an incomplete/corrupt image before it can enter a shard.
    for frame in frames:
        with Image.open(frame) as image:
            image.load()
    return frames


def window_frame_indices(
    num_frames: int, window_frames: int = 64, window_stride: int = 32
) -> list[tuple[int, list[int], int]]:
    """Return (start index, padded frame indices, valid count) per window.

    The final full window is anchored at the last frame, covering the tail once
    without generating a second copy when the stride already lands there.
    """
    if num_frames < 1 or window_frames < 1 or not 1 <= window_stride <= window_frames:
        raise ValueError("Frame counts must be positive and 1 <= window_stride <= window_frames")
    if num_frames <= window_frames:
        return [
            (
                0,
                list(range(num_frames)) + [num_frames - 1] * (window_frames - num_frames),
                num_frames,
            )
        ]
    starts = list(range(0, num_frames - window_frames + 1, window_stride))
    last_start = num_frames - window_frames
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, list(range(start, start + window_frames)), window_frames) for start in starts]


class _ShardWriter:
    def __init__(self, directory: Path, split: str, max_bytes: int):
        self.directory, self.split, self.max_bytes = directory, split, max_bytes
        self.shards: list[dict[str, Any]] = []
        self.archive: tarfile.TarFile | None = None
        self.temporary: Path | None = None
        self.count = 0
        self.bytes_written = 0

    def _open(self) -> None:
        self.temporary = self.directory / f"{self.split}-{len(self.shards):06d}.tar.tmp"
        self.archive = tarfile.open(self.temporary, mode="w", format=tarfile.USTAR_FORMAT)
        self.count, self.bytes_written = 0, 0

    def write(self, key: str, metadata: dict[str, Any], images: Sequence[bytes]) -> None:
        fields = [
            ("json", json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        ]
        fields.extend((f"{index:06d}.jpg", value) for index, value in enumerate(images))
        size = sum(512 + 512 * math.ceil(len(value) / 512) for _, value in fields)
        if self.archive is not None and self.count and self.bytes_written + size > self.max_bytes:
            self.close()
        if self.archive is None:
            self._open()
        assert self.archive is not None
        for suffix, content in fields:
            info = tarfile.TarInfo(f"{key}.{suffix}")
            info.size, info.mtime, info.mode = len(content), 0, 0o644
            self.archive.addfile(info, io.BytesIO(content))
        self.count += 1
        self.bytes_written += size

    def close(self) -> None:
        if self.archive is None:
            return
        self.archive.close()
        self.archive = None
        assert self.temporary is not None
        with tarfile.open(self.temporary, "r:") as archive:
            sample_count = sum(member.name.endswith(".json") for member in archive)
        if sample_count != self.count:
            raise RuntimeError(f"Shard verification failed: {self.temporary}")
        final = self.temporary.with_suffix("")
        self.temporary.replace(final)
        self.shards.append(
            {
                "path": final.name,
                "num_samples": self.count,
                "size_bytes": final.stat().st_size,
                "sha256": _sha256(final),
            }
        )

    def abort(self) -> None:
        if self.archive is not None:
            self.archive.close()
            self.archive = None


def prepare_dataset(
    manifest: str | Path,
    output: str | Path,
    *,
    video_root: str | Path | None = None,
    fps: float = 15.0,
    short_side: int = 320,
    window_frames: int = 64,
    window_stride: int = 32,
    jpeg_quality: int = 95,
    shard_size_mb: float = 512.0,
    seed: int = 42,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    skip_errors: bool = False,
    class_names: Sequence[str] | None = None,
) -> Path:
    """Create a two-class dataset; never overwrite an existing dataset.

    ``class_names`` explicitly orders class IDs 0 and 1. Otherwise the two
    manifest labels are sorted. Individual held-out splits may contain one class.
    Unique training JPEG pixels supply the saved RGB normalization statistics.
    """
    if not math.isfinite(fps) or fps <= 0 or short_side < 1 or not 1 <= jpeg_quality <= 100:
        raise ValueError("Invalid frame extraction parameters")
    window_frame_indices(1, window_frames, window_stride)
    if not math.isfinite(shard_size_mb) or shard_size_mb <= 0:
        raise ValueError("shard_size_mb must be positive")
    records = assign_splits(
        read_manifest(manifest, video_root), seed=seed, split_ratios=split_ratios
    )
    labels = {record.label for record in records}
    if class_names is not None and (
        isinstance(class_names, str)
        or len(class_names) != NUM_CLASSES
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
    ):
        raise ValueError("class_names must contain exactly two non-empty class names")
    ordered_names = sorted(labels) if class_names is None else list(class_names)
    class_to_idx = {label: index for index, label in enumerate(ordered_names)}
    validate_class_to_idx(class_to_idx, context="manifest class_to_idx")
    if set(class_to_idx) != labels:
        raise ValueError("class_names must exactly match the two labels in the manifest")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty; existing data will not be overwritten: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    incomplete = output / ".incomplete"
    incomplete.write_text(
        "Preprocessing is incomplete. Do not train on this directory.\n", encoding="utf-8"
    )
    writers = {
        split: _ShardWriter(output, split, int(shard_size_mb * 1024 * 1024))
        for split in ("train", "val", "test")
    }
    videos: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    hashes: dict[str, VideoRecord] = {}
    statistics = RGBStatistics()
    rng = random.Random(seed)
    rng.shuffle(records)
    try:
        for record in records:
            assert record.split is not None
            writing = False
            try:
                content_hash = _sha256(record.video)
                if content_hash in hashes:
                    previous = hashes[content_hash]
                    raise ValueError(
                        f"Duplicate video contents: {record.video} and {previous.video}; refusing duplicate/leaked sources"
                    )
                with tempfile.TemporaryDirectory(prefix="frames-", dir=output) as temporary:
                    frames = extract_frames(
                        record.video,
                        temporary,
                        fps=fps,
                        short_side=short_side,
                        jpeg_quality=jpeg_quality,
                    )
                    windows = window_frame_indices(len(frames), window_frames, window_stride)
                    video_statistics = RGBStatistics()
                    if record.split == "train":
                        # Windows cover every real frame. Count each JPEG once,
                        # before overlap/padding, and commit only a complete video.
                        for frame in frames:
                            video_statistics.add_jpeg(frame)
                    video_id = content_hash[:24]
                    # Source hashes and IDs are stable across directory changes.
                    video_info = {
                        "video": str(record.video),
                        "sha256": content_hash,
                        "class_id": class_to_idx[record.label],
                        "label": record.label,
                        "split": record.split,
                        "group_id": record.group_id,
                        "num_windows": len(windows),
                        "num_frames": len(frames),
                    }
                    order = list(enumerate(windows))
                    rng.shuffle(order)
                    # Once any member is written, an I/O error must abort the
                    # entire dataset rather than publish an orphaned partial
                    # video when --skip-errors was requested.
                    writing = True
                    for window_index, (start, indices, valid_frames) in order:
                        key = f"{video_id}_{window_index:08d}"
                        metadata = {
                            "video_id": video_id,
                            "class_id": class_to_idx[record.label],
                            "group_id": record.group_id,
                            "window_index": window_index,
                            "window_start": start / fps,
                            "fps": fps,
                            "valid_frames": valid_frames,
                            "frame_times": [min(index, len(frames) - 1) / fps for index in indices],
                        }
                        images = [frames[index].read_bytes() for index in indices]
                        writers[record.split].write(key, metadata, images)
                    videos[video_id] = video_info
                    hashes[content_hash] = record
                    statistics.merge(video_statistics)
            except (OSError, ValueError, RuntimeError) as exception:
                errors.append({"video": str(record.video), "error": str(exception)})
                _atomic_json(output / "errors.json", errors)
                if writing or not skip_errors:
                    raise
        if not videos:
            raise ValueError("No usable videos were produced")
        if {video["class_id"] for video in videos.values()} != set(range(NUM_CLASSES)):
            raise ValueError(
                "Usable videos must include both binary classes; dataset was not published"
            )
        for writer in writers.values():
            writer.close()
        splits: dict[str, dict[str, Any]] = {}
        for split, writer in writers.items():
            selected = [video for video in videos.values() if video["split"] == split]
            window_counts: Counter[str] = Counter()
            for video in selected:
                window_counts[video["label"]] += video["num_windows"]
            splits[split] = {
                "shards": writer.shards,
                "num_samples": sum(video["num_windows"] for video in selected),
                "num_videos": len(selected),
                "class_video_counts": dict(Counter(video["label"] for video in selected)),
                "class_window_counts": dict(window_counts),
            }
        dataset = {
            "schema_version": 1,
            "task": "binary_classification",
            "num_classes": NUM_CLASSES,
            "class_to_idx": class_to_idx,
            "preprocessing": {
                "fps": fps,
                "short_side": short_side,
                "window_frames": window_frames,
                "window_stride": window_stride,
                "jpeg_quality": jpeg_quality,
            },
            "split_seed": seed,
            "split_ratios": list(split_ratios),
            "splits": splits,
            "videos": videos,
            "num_failed_videos": len(errors),
        }
        if splits["train"]["num_samples"]:
            dataset["normalization"] = statistics.report()
        # Publish only after shards and training normalization are complete.
        _atomic_json(output / "dataset.json", dataset)
        _atomic_json(output / "errors.json", errors)
        incomplete.unlink()
        return output / "dataset.json"
    finally:
        for writer in writers.values():
            writer.abort()


def verify_dataset(path: str | Path, *, verify_images: bool = True) -> dict[str, int]:
    """Check hashes, sample grouping, labels, frame counts, and JPEG contents."""
    from webdataset.tariterators import tarfile_samples

    from .data import read_dataset

    path = Path(path).resolve()
    dataset = read_dataset(path)
    seen: set[str] = set()
    observed_windows: Counter[str] = Counter()
    result = {"shards": 0, "samples": 0, "images": 0, "videos": len(dataset["videos"])}
    for split_name, split in dataset["splits"].items():
        count = 0
        for shard in split["shards"]:
            shard_path = path.parent / shard["path"]
            if (
                shard_path.stat().st_size != shard["size_bytes"]
                or _sha256(shard_path) != shard["sha256"]
            ):
                raise ValueError(f"Shard checksum/size mismatch: {shard_path}")
            shard_count = 0
            for sample in tarfile_samples(iter([{"url": str(shard_path)}])):
                key = sample["__key__"]
                if key in seen:
                    raise ValueError(f"Duplicate sample key: {key}")
                seen.add(key)
                metadata = json.loads(sample["json"])
                video = dataset["videos"][metadata["video_id"]]
                if video["split"] != split_name or metadata["class_id"] != video["class_id"]:
                    raise ValueError(f"Sample split/label mismatch: {key}")
                frames = sorted(name for name in sample if name.endswith(".jpg"))
                expected = dataset["preprocessing"]["window_frames"]
                if frames != [f"{i:06d}.jpg" for i in range(expected)]:
                    raise ValueError(f"Missing/unexpected frames in {key}")
                if (
                    len(metadata["frame_times"]) != expected
                    or not 1 <= metadata["valid_frames"] <= expected
                ):
                    raise ValueError(f"Invalid frame metadata: {key}")
                if verify_images:
                    for frame in frames:
                        with Image.open(io.BytesIO(sample[frame])) as image:
                            image.load()
                observed_windows[metadata["video_id"]] += 1
                shard_count += 1
                result["images"] += len(frames)
            if shard_count != shard["num_samples"]:
                raise ValueError(f"Shard sample count mismatch: {shard_path}")
            count += shard_count
            result["shards"] += 1
        if count != split["num_samples"]:
            raise ValueError(f"Split sample count mismatch: {split_name}")
        result["samples"] += count
    for video_id, video in dataset["videos"].items():
        if observed_windows[video_id] != video["num_windows"]:
            raise ValueError(f"Video window count mismatch: {video_id}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument(
        "--class-names",
        nargs=NUM_CLASSES,
        metavar=("CLASS_0", "CLASS_1"),
        help="Explicitly assign the two manifest labels to class IDs 0 and 1; default: sorted labels",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--short-side", type=int, default=320)
    parser.add_argument("--window-frames", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--shard-size-mb", type=float, default=512.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Explicitly exclude unreadable/duplicate videos and record errors.json",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="DATASET_JSON",
        help="Verify an existing dataset, including all JPEGs",
    )
    args = parser.parse_args(argv)
    if args.verify is not None:
        print(json.dumps(verify_dataset(args.verify), indent=2))
        return
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required unless --verify is specified")
    values = vars(args)
    values.pop("verify")
    print(prepare_dataset(**values))


if __name__ == "__main__":
    main()
