"""Compute RGB normalization statistics using only unique training frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from webdataset.tariterators import tarfile_samples

from .data import dataset_fingerprint, read_dataset
from .evaluate import write_json
from .normalization import RGBStatistics


def compute_statistics(dataset_path: str | Path, max_frames: int | None = None):
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    path = Path(dataset_path).resolve()
    dataset = read_dataset(path)
    if "train" not in dataset["splits"] or not dataset["splits"]["train"]["num_samples"]:
        raise ValueError("Statistics require a nonempty train split")
    sources = iter(
        {"url": str(path.parent / shard["path"])} for shard in dataset["splits"]["train"]["shards"]
    )
    seen = set()
    statistics = RGBStatistics()
    done = False
    for sample in tarfile_samples(sources):
        metadata = json.loads(sample["json"])
        for index, timestamp in enumerate(metadata["frame_times"][: metadata["valid_frames"]]):
            # Overlapping windows and short-video padding must not count a frame twice.
            identity = (metadata["video_id"], timestamp)
            if identity in seen:
                continue
            seen.add(identity)
            statistics.add_jpeg(sample[f"{index:06d}.jpg"])
            if max_frames is not None and len(seen) >= max_frames:
                done = True
                break
        if done:
            break
    report = statistics.report()
    report["dataset_fingerprint"] = dataset_fingerprint(path)
    if max_frames is not None:
        report["note"] += "; max_frames is a deterministic prefix"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("normalization.json"))
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    report = compute_statistics(args.dataset, args.max_frames)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
