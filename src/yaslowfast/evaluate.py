"""Evaluate labelled video archives or prepared datasets, with confusion-matrix reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .labels import NUM_CLASSES, validate_class_to_idx
from .metrics import VideoClassificationAccumulator
from .precision import autocast_context, resolve_precision


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Remove training wrappers so uneven eval does not invoke DDP collectives."""
    while True:
        if isinstance(model, DistributedDataParallel):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device | str,
    num_classes: int = NUM_CLASSES,
    *,
    precision: str = "fp32",
    distributed: bool = False,
) -> dict[str, Any]:
    """Evaluate all supplied windows once and aggregate probabilities per video.

    No model-forward collective is performed. Ranks may have unequal numbers of
    batches, including zero. State is collected once after exhausting each rank's
    loader. Every rank receives summary metrics; only rank zero receives the
    potentially large ``predictions`` list. Any aggregation errors are broadcast
    before raising so that peers are not left waiting on a collective.
    """
    device = torch.device(device)
    precision = resolve_precision(precision, device)
    if distributed and (not dist.is_available() or not dist.is_initialized()):
        raise RuntimeError("Distributed evaluation requires an initialized process group")
    accumulator = VideoClassificationAccumulator(num_classes)
    module = unwrap_model(model)
    previous_training = module.training
    module.eval()
    local_error = None
    try:
        with torch.inference_mode():
            if distributed:
                # SyncBatchNorm does not synchronize running statistics in eval.
                # All ranks must evaluate exactly the same buffers and weights.
                for tensor in list(module.parameters()) + list(module.buffers()):
                    dist.broadcast(tensor, src=0)
            try:
                for batch in loader:
                    slow = batch["slow"].to(device, non_blocking=True)
                    fast = batch["fast"].to(device, non_blocking=True)
                    target = batch["target"].to(device, non_blocking=True)
                    with autocast_context(device, precision):
                        logits = module(slow, fast)
                    if logits.shape != (target.numel(), num_classes):
                        raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
                    logits = logits.float()
                    loss_sum = F.cross_entropy(logits, target, reduction="sum")
                    accumulator.update(
                        logits.softmax(dim=1).cpu().numpy(),
                        target.cpu().numpy(),
                        batch["video_id"],
                        batch["key"],
                        loss_sum=float(loss_sum.item()),
                    )
            except Exception as error:
                if not distributed:
                    raise
                local_error = f"{type(error).__name__}: {error}"

        if not distributed:
            return accumulator.compute()

        rank = dist.get_rank()
        gathered: list[Any] | None = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(
            {"state": accumulator.state_dict(), "error": local_error}, gathered, dst=0
        )
        report = None
        status: list[Any] = [None]
        if rank == 0:
            try:
                combined = VideoClassificationAccumulator(num_classes)
                assert gathered is not None
                for source_rank, payload in enumerate(gathered):
                    if payload["error"]:
                        raise RuntimeError(
                            f"Evaluation failed on rank {source_rank}: {payload['error']}"
                        )
                    combined.merge(payload["state"])
                report = combined.compute()
                status[0] = {
                    "report": {key: value for key, value in report.items() if key != "predictions"}
                }
            except Exception as error:
                status[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(status, src=0)
        if "error" in status[0]:
            raise RuntimeError(status[0]["error"])
        return report if rank == 0 else status[0]["report"]
    finally:
        module.train(previous_training)


def evaluate_archives(
    model: torch.nn.Module,
    positive: str | Path,
    negative: str | Path,
    *,
    config: Any,
    preprocessing: dict[str, Any],
    class_to_idx: dict[str, int],
    device: torch.device | str,
    precision: str = "fp32",
    batch_size: int = 1,
    progress: bool = False,
) -> dict[str, Any]:
    """Evaluate all labelled raw videos, with the same window metrics as datasets."""
    from .prepare import _read_archives, _source_video
    from .video import iter_video_batches

    validate_class_to_idx(class_to_idx, context="Archive evaluation class_to_idx")
    if set(class_to_idx) != {"positive", "negative"}:
        raise ValueError("Archive evaluation requires checkpoint classes positive and negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records = _read_archives(positive, negative)
    active_source = None

    def batches():
        nonlocal active_source
        archives: dict[Path, tarfile.TarFile] = {}
        try:
            with tempfile.TemporaryDirectory(prefix="yaslowfast-evaluate-") as directory:
                for number, record in enumerate(records, 1):
                    active_source = record.source
                    if progress:
                        print(
                            f"[{number}/{len(records)}] {record.label}: {record.source}",
                            file=sys.stderr,
                            flush=True,
                        )
                    with (
                        _source_video(record, Path(directory), archives) as video,
                        closing(
                            iter_video_batches(
                                video,
                                config=config,
                                preprocessing=preprocessing,
                                batch_size=batch_size,
                            )
                        ) as video_batches,
                    ):
                        offset = 0
                        for batch in video_batches:
                            count = len(batch["windows"])
                            yield {
                                "slow": batch["slow"],
                                "fast": batch["fast"],
                                "target": torch.full(
                                    (count,), class_to_idx[record.label], dtype=torch.long
                                ),
                                "video_id": [record.sha256] * count,
                                "key": [
                                    f"{record.sha256}_{i}" for i in range(offset, offset + count)
                                ],
                            }
                            offset += count
                        if offset == 0:
                            raise ValueError("Video produced no evaluation windows")
        finally:
            for archive in archives.values():
                archive.close()

    with closing(batches()) as loader:
        try:
            report = evaluate_model(model, loader, device, precision=precision)
        except Exception as error:
            if active_source is None:
                raise
            raise RuntimeError(f"Evaluation failed for {active_source}: {error}") from error
    sources = {record.sha256: record for record in records}
    if {row["video_id"] for row in report["predictions"]} != set(sources):
        raise ValueError("Archive evaluation coverage mismatch: not all videos were evaluated")
    for row in report["predictions"]:
        record = sources[row["video_id"]]
        row.update(
            source_archive=str(record.source_archive),
            source_member=record.source_member,
            sha256=record.sha256,
        )
    report["class_to_idx"] = class_to_idx
    report["inputs"] = {
        "positive": str(Path(positive).resolve()),
        "negative": str(Path(negative).resolve()),
    }
    return report


def main(argv: list[str] | None = None) -> None:
    from .checkpoint import load_checkpoint
    from .config import config_from_dict
    from .data import build_loader, read_dataset
    from .distributed import DistributedContext
    from .models import build_model
    from .report import format_console_report, write_html_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Override the saved dataset.json path")
    parser.add_argument("--split", help="Prepared dataset split (default: val)")
    parser.add_argument("--positive", type=Path, help="Tar containing positive raw videos")
    parser.add_argument("--negative", type=Path, help="Tar containing negative raw videos")
    parser.add_argument(
        "--report", type=Path, help="Write index.html, metrics.json and predictions.csv"
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--allow-preprocessing-mismatch",
        action="store_true",
        help="Explicitly allow evaluation data prepared with different extraction settings",
    )
    args = parser.parse_args(argv)
    archive_mode = args.positive is not None or args.negative is not None
    if archive_mode:
        if args.positive is None or args.negative is None:
            parser.error("--positive and --negative must be supplied together")
        if (
            args.dataset is not None
            or args.split is not None
            or args.num_workers is not None
            or args.allow_preprocessing_mismatch
        ):
            parser.error(
                "--dataset, --split, --num-workers and --allow-preprocessing-mismatch "
                "are only supported for prepared datasets"
            )
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            parser.error("Archive evaluation is single-process; run without torchrun")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    started = time.perf_counter()
    evaluated_at = datetime.now(timezone.utc).isoformat()
    ctx = DistributedContext.from_environment(args.device)
    try:
        checkpoint = load_checkpoint(args.checkpoint)
        config = config_from_dict(checkpoint["config"])
        precision = resolve_precision(args.precision, ctx.device)
        batch_size = args.batch_size if args.batch_size is not None else config.data.batch_size
        split = args.split or "val"
        dataset_path = None
        metadata = None
        mismatch = {}
        if not archive_mode:
            dataset_path = args.dataset or Path(config.data.dataset)
            metadata = read_dataset(dataset_path)
            if metadata["class_to_idx"] != checkpoint["class_to_idx"]:
                raise ValueError(
                    "Dataset class_to_idx does not match the checkpoint; class IDs must be identical"
                )
            mismatch = {
                key: {
                    "checkpoint": checkpoint["data_preprocessing"].get(key),
                    "dataset": metadata["preprocessing"].get(key),
                }
                for key in ("fps", "short_side", "window_frames", "window_stride", "jpeg_quality")
                if checkpoint["data_preprocessing"].get(key) != metadata["preprocessing"].get(key)
            }
            if mismatch and not args.allow_preprocessing_mismatch:
                raise ValueError(
                    f"Dataset preprocessing differs from checkpoint: {mismatch}. "
                    "Use --allow-preprocessing-mismatch only for an intentional evaluation change."
                )
        inputs = [args.checkpoint, args.positive, args.negative, dataset_path]
        input_paths = {path.resolve() for path in inputs if path is not None}
        output_paths = [args.output]
        if args.report is not None:
            output_paths.extend(
                args.report / name for name in ("index.html", "metrics.json", "predictions.csv")
            )
        if any(path.resolve() in input_paths for path in output_paths):
            raise ValueError("Evaluation output must not overwrite an input file")
        if args.report is not None and args.output.resolve() in {
            (args.report / name).resolve() for name in ("index.html", "predictions.csv")
        }:
            raise ValueError("--output must not collide with an HTML or CSV report file")
        model = build_model(config.model, len(checkpoint["class_to_idx"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(ctx.device)
        if archive_mode:
            report = evaluate_archives(
                model,
                args.positive,
                args.negative,
                config=config,
                preprocessing=checkpoint["data_preprocessing"],
                class_to_idx=checkpoint["class_to_idx"],
                device=ctx.device,
                precision=precision,
                batch_size=batch_size,
                progress=ctx.is_main,
            )
        else:
            loader = build_loader(
                dataset_path,
                split,
                config=config.data,
                model_config=config.model,
                batch_size=batch_size,
                num_workers=args.num_workers
                if args.num_workers is not None
                else config.data.num_workers,
                seed=config.train.seed,
                rank=ctx.rank,
                world_size=ctx.world_size,
                training=False,
            )
            report = evaluate_model(
                model,
                loader,
                ctx.device,
                len(checkpoint["class_to_idx"]),
                precision=precision,
                distributed=ctx.distributed,
            )
            expected = metadata["splits"][split]
            if (
                report["num_windows"] != expected["num_samples"]
                or report["num_videos"] != expected["num_videos"]
            ):
                raise ValueError(
                    f"Evaluation coverage mismatch: read {report['num_windows']} windows / "
                    f"{report['num_videos']} videos; expected {expected['num_samples']} windows / "
                    f"{expected['num_videos']} videos"
                )
        if ctx.is_main:
            if metadata is not None:
                for row in report["predictions"]:
                    source = metadata.get("videos", {}).get(row["video_id"], {})
                    for key in ("source_archive", "source_member", "sha256"):
                        if key in source:
                            row[key] = source[key]
                    if "video" in source:
                        row["source"] = source["video"]
                report["dataset"] = str(dataset_path.resolve())
                report["split"] = split
            report.update(
                class_to_idx=checkpoint["class_to_idx"],
                checkpoint=str(args.checkpoint.resolve()),
                mode="archives" if archive_mode else "dataset",
                preprocessing_mismatch=mismatch,
                preprocessing=checkpoint["data_preprocessing"]
                if archive_mode
                else metadata["preprocessing"],
                normalization={"mean": list(config.data.mean), "std": list(config.data.std)},
                transforms={
                    "num_frames": config.model.num_frames,
                    "alpha": config.model.alpha,
                    "crop_size": config.model.crop_size,
                    "eval_short_side": max(config.data.train_min_scale, config.model.crop_size),
                },
                evaluated_at=evaluated_at,
                elapsed_seconds=time.perf_counter() - started,
                device=str(ctx.device),
                precision=precision,
                batch_size=batch_size,
            )
            write_json(args.output, report)
            # Keep the existing summary JSON as the first stdout line.
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "accuracy",
                            "macro_f1",
                            "loss",
                            "video_loss",
                            "num_videos",
                            "num_windows",
                        )
                    },
                    ensure_ascii=False,
                )
            )
            print(format_console_report(report))
            if args.report is not None:
                index = write_html_report(report, args.report)
                print(f"HTML report: {index.resolve()}")
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
