"""Finite, duplicate-free video evaluation, including uneven DDP partitions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .labels import NUM_CLASSES
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


def main(argv: list[str] | None = None) -> None:
    from .checkpoint import load_checkpoint
    from .config import config_from_dict
    from .data import build_loader, read_dataset
    from .distributed import DistributedContext
    from .models import build_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Override the saved dataset.json path")
    parser.add_argument("--split", default="val")
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
    ctx = DistributedContext.from_environment(args.device)
    try:
        checkpoint = load_checkpoint(args.checkpoint)
        config = config_from_dict(checkpoint["config"])
        dataset_path = args.dataset or Path(config.data.dataset)
        metadata = read_dataset(dataset_path)
        if metadata["class_to_idx"] != checkpoint["class_to_idx"]:
            raise ValueError(
                "Dataset class_to_idx does not match the checkpoint; class IDs must be identical"
            )
        if "data_preprocessing" not in checkpoint:
            raise ValueError(
                "Checkpoint lacks data_preprocessing; extraction settings cannot be checked"
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
        model = build_model(config.model, len(checkpoint["class_to_idx"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(ctx.device)
        loader = build_loader(
            dataset_path,
            args.split,
            config=config.data,
            model_config=config.model,
            batch_size=args.batch_size if args.batch_size is not None else config.data.batch_size,
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
            precision=args.precision,
            distributed=ctx.distributed,
        )
        expected = metadata["splits"][args.split]
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
            report["class_to_idx"] = checkpoint["class_to_idx"]
            report["checkpoint"] = str(args.checkpoint.resolve())
            report["dataset"] = str(dataset_path.resolve())
            report["split"] = args.split
            report["preprocessing_mismatch"] = mismatch
            write_json(args.output, report)
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
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
