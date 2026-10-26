"""Scratch training with equal optimizer-update budgets on every DDP rank."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import atomic_save, capture_rng_state, load_checkpoint, restore_rng_state
from .config import Config, apply_dataset_normalization, load_config, validate_config
from .data import build_loader, dataset_fingerprint, read_dataset
from .distributed import DistributedContext, all_ranks_finite
from .evaluate import evaluate_model
from .labels import NUM_CLASSES, validate_class_to_idx
from .models import build_model
from .precision import autocast_context, resolve_precision
from .schedule import WarmupCosine

LOGGER = logging.getLogger("yaslowfast.train")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_backend(config: Config) -> None:
    if config.train.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(config.train.deterministic)
    torch.backends.cudnn.benchmark = not config.train.deterministic
    if torch.cuda.is_available():
        mode = "tf32" if config.train.allow_tf32 else "ieee"
        torch.backends.cuda.matmul.fp32_precision = mode
        torch.backends.cudnn.conv.fp32_precision = mode


def weighted_classification_loss(
    logits,
    target,
    weight,
    *,
    label_smoothing=0.0,
    paired_target=None,
    paired_weight=None,
    mixing=1.0,
):
    # Normalize by count, NOT sum(weight). Dataset-wide weights have mean one.
    first = F.cross_entropy(
        logits.float(), target, reduction="none", label_smoothing=label_smoothing
    )
    loss = mixing * first * weight
    if paired_target is not None:
        second = F.cross_entropy(
            logits.float(), paired_target, reduction="none", label_smoothing=label_smoothing
        )
        loss = loss + (1 - mixing) * second * paired_weight
    return loss.mean()


def _to_device(batch, device, channels_last):
    slow = batch["slow"].to(device, non_blocking=True)
    fast = batch["fast"].to(device, non_blocking=True)
    if channels_last:
        slow = slow.contiguous(memory_format=torch.channels_last_3d)
        fast = fast.contiguous(memory_format=torch.channels_last_3d)
    return (
        slow,
        fast,
        batch["target"].to(device, non_blocking=True),
        batch["weight"].to(device, dtype=torch.float32, non_blocking=True),
    )


def backoff_scaler(scaler) -> None:
    """Synchronously skip an overflow and reset the public scaler state."""
    state = scaler.state_dict()
    state["scale"] *= scaler.get_backoff_factor()
    state["_growth_tracker"] = 0
    # update resets the per-optimizer stage; load restores the reset growth counter.
    scaler.update(new_scale=state["scale"])
    scaler.load_state_dict(state)


def optimizer_parameter_groups(model, weight_decay: float, norm_weight_decay: float):
    """Keep BN scale/shift out of convolution weight decay, as in the reference recipe."""
    norms = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, (torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))
        for parameter in module.parameters(recurse=False)
    }
    return [
        {
            "params": [p for p in model.parameters() if p.requires_grad and id(p) not in norms],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in model.parameters() if p.requires_grad and id(p) in norms],
            "weight_decay": norm_weight_decay,
        },
    ]


def train_epoch(
    model,
    raw_model,
    loader,
    optimizer,
    scheduler,
    scaler,
    config,
    context,
    precision,
    epoch,
    steps_per_epoch,
    global_step,
):
    model.train()
    iterator = iter(loader)
    updates = attempts = skipped = 0
    metric_dtype = torch.float32 if context.device.type == "mps" else torch.float64
    totals = torch.zeros(3, dtype=metric_dtype, device=context.device)
    start = time.perf_counter()
    while updates < steps_per_epoch:
        attempts += 1
        if attempts > steps_per_epoch + 100:
            raise RuntimeError("Too many FP16 overflow skips; try precision=bf16 or fp32")
        optimizer.zero_grad(set_to_none=True)
        update_totals = torch.zeros_like(totals)
        for microstep in range(config.train.accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "Training stream ended before the common DDP step budget"
                ) from exc
            slow, fast, target, weight = _to_device(
                batch, context.device, config.train.channels_last_3d
            )
            paired_target = paired_weight = None
            mixing = 1.0
            if config.train.mixup_alpha > 0:
                mixing = float(np.random.beta(config.train.mixup_alpha, config.train.mixup_alpha))
                permutation = torch.randperm(target.shape[0], device=context.device)
                slow = mixing * slow + (1 - mixing) * slow[permutation]
                fast = mixing * fast + (1 - mixing) * fast[permutation]
                paired_target, paired_weight = target[permutation], weight[permutation]
            synchronize = microstep == config.train.accumulation_steps - 1
            sync_context = (
                nullcontext() if synchronize or not context.distributed else model.no_sync()
            )
            # no_sync must cover the forward as well as backward.
            with sync_context:
                with autocast_context(context.device, precision):
                    logits = model(slow, fast)
                    loss = weighted_classification_loss(
                        logits,
                        target,
                        weight,
                        label_smoothing=config.train.label_smoothing,
                        paired_target=paired_target,
                        paired_weight=paired_weight,
                        mixing=mixing,
                    )
                if not all_ranks_finite(loss, context):
                    raise FloatingPointError("Non-finite loss detected on at least one rank")
                scaler.scale(loss / config.train.accumulation_steps).backward()
            update_totals[0] += loss.detach().to(metric_dtype) * target.numel()
            update_totals[1] += (logits.detach().argmax(1) == target).sum()
            update_totals[2] += target.numel()
        scaler.unscale_(optimizer)
        # Computing the norm also detects non-finite gradients without many CPU synchronizations.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), config.train.grad_clip_norm or float("inf")
        )
        if not all_ranks_finite(grad_norm, context):
            if not scaler.is_enabled():
                raise FloatingPointError("Non-finite gradients detected on at least one rank")
            # All ranks skip together, including a rank whose local scaler saw finite values.
            backoff_scaler(scaler)
            skipped += 1
            continue
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        updates += 1
        global_step += 1
        totals += update_totals
        if context.is_main and (
            updates % config.train.log_every == 0 or updates == steps_per_epoch
        ):
            LOGGER.info(
                "epoch=%d update=%d/%d global_step=%d loss=%.5f lr=%.6g grad_norm=%.4f",
                epoch + 1,
                updates,
                steps_per_epoch,
                global_step,
                (update_totals[0] / update_totals[2]).item(),
                optimizer.param_groups[0]["lr"],
                grad_norm.item(),
            )
    if context.distributed:
        dist.all_reduce(totals)
    duration = time.perf_counter() - start
    return {
        "loss": (totals[0] / totals[2]).item(),
        "accuracy": (totals[1] / totals[2]).item() if config.train.mixup_alpha == 0 else None,
        "sample_visits": int(totals[2].item()),
        "optimizer_steps": updates,
        "overflow_skips": skipped,
        "seconds": duration,
        "clips_per_second": totals[2].item() / duration,
    }, global_step


def _resume_contract(config: Config):
    value = asdict(config)
    # Logging/output location may change; the sampling and optimization schedule may not.
    for key in ("output_dir", "log_every", "distributed_timeout_seconds"):
        value["train"].pop(key)
    # Content identity is checked independently, so a copied dataset can be relocated.
    value["data"].pop("dataset")
    # Compare the resolved mean/std, not how they were selected. Omitting this
    # new selector also keeps pre-automatic-normalization checkpoints compatible.
    value["data"].pop("normalization")
    return value


def _json_write(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_training(config: Config, resume: str | None = None, stop_after_epoch: int | None = None):
    """stop_after_epoch stops after a saved boundary without changing the LR schedule."""
    validate_config(config)
    if config.train.warmup_epochs >= config.train.epochs:
        raise ValueError("warmup_epochs must be smaller than epochs")
    configure_backend(config)
    context = DistributedContext.from_environment(
        config.train.device, config.train.distributed_timeout_seconds
    )
    try:
        precision = resolve_precision(config.train.precision, context.device)
        dataset = read_dataset(config.data.dataset)
        fingerprint = dataset_fingerprint(config.data.dataset)
        classes = dataset["class_to_idx"]
        validate_class_to_idx(classes, context="Training class_to_idx")
        split = dataset["splits"].get(config.data.train_split)
        if not split or split["num_samples"] < 1:
            raise ValueError("Training split is missing or empty")
        train_classes = {
            video["class_id"]
            for video in dataset["videos"].values()
            if video["split"] == config.data.train_split
        }
        missing_classes = [name for name, index in classes.items() if index not in train_classes]
        if missing_classes:
            raise ValueError(f"Training split has no examples for classes: {missing_classes}")
        if (
            config.data.val_split not in dataset["splits"]
            or not dataset["splits"][config.data.val_split]["num_samples"]
        ):
            raise ValueError("Validation split is missing or empty")
        dataset_normalization = apply_dataset_normalization(config.data, dataset)
        if context.is_main:
            if config.data.normalization == "auto" and not dataset_normalization:
                LOGGER.warning(
                    "Dataset has no normalization statistics; using configured mean/std. "
                    "Prepare the dataset again to compute training statistics automatically."
                )
            LOGGER.info(
                "normalization=%s mean=%s std=%s",
                "dataset" if dataset_normalization else "config",
                config.data.mean,
                config.data.std,
            )
        effective_batch = (
            config.data.batch_size * context.world_size * config.train.accumulation_steps
        )
        steps_per_epoch = config.train.steps_per_epoch or math.ceil(
            split["num_samples"] / effective_batch
        )
        total_steps = config.train.epochs * steps_per_epoch
        lr = config.train.learning_rate
        if config.train.scale_learning_rate:
            lr *= effective_batch / config.train.reference_batch_size
        output = Path(config.train.output_dir)
        if context.is_main:
            output.mkdir(parents=True, exist_ok=True)
            if (output / "latest.pt").exists() and resume is None:
                raise FileExistsError(
                    f"{output}/latest.pt exists; use --resume or another output_dir"
                )
        context.barrier()
        seed_everything(config.train.seed)
        raw_model = build_model(config.model, len(classes))
        if config.train.sync_batchnorm:
            if not context.distributed or context.device.type != "cuda":
                raise ValueError("sync_batchnorm requires multi-process CUDA DDP")
            raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
        raw_model.to(context.device)
        if config.train.channels_last_3d:
            raw_model.to(memory_format=torch.channels_last_3d)
        optimizer = torch.optim.SGD(
            optimizer_parameter_groups(
                raw_model, config.train.weight_decay, config.train.norm_weight_decay
            ),
            lr=lr,
            momentum=config.train.momentum,
        )
        scheduler = WarmupCosine(
            optimizer,
            total_steps,
            int(config.train.warmup_epochs * steps_per_epoch),
            config.train.warmup_start_factor,
            config.train.min_lr_ratio,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=context.device.type == "cuda" and precision == "fp16"
        )
        start_epoch = global_step = 0
        best_metric = -float("inf")
        checkpoint = load_checkpoint(resume) if resume else None
        if checkpoint:
            if (
                checkpoint["dataset_fingerprint"] != fingerprint
                or checkpoint["class_to_idx"] != classes
            ):
                raise ValueError("Checkpoint dataset identity or class mapping does not match")
            if checkpoint["resume_contract"] != _resume_contract(config):
                raise ValueError(
                    "Resume configuration changed. Keep sampling/model/optimizer settings identical"
                )
            if (
                checkpoint["world_size"] != context.world_size
                or checkpoint["precision"] != precision
            ):
                raise ValueError(
                    "Exact epoch-boundary resume requires the same world size and precision"
                )
            if checkpoint["torch_version"] != str(torch.__version__):
                raise ValueError("Exact epoch-boundary resume requires the same PyTorch version")
            raw_model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch, global_step, best_metric = (
                checkpoint["epoch"],
                checkpoint["global_step"],
                checkpoint["best_metric"],
            )
        train_model = raw_model
        if context.distributed:
            train_model = DistributedDataParallel(
                raw_model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )
        if config.train.compile:
            train_model = torch.compile(train_model)
        # DDP initialization broadcasts the same parameters; stochastic training differs per rank.
        seed_everything(config.train.seed + context.rank)
        if checkpoint:
            restore_rng_state(checkpoint["rng_states"][context.rank])
        if context.is_main:
            _json_write(output / "config.json", asdict(config))
            _json_write(
                output / "environment.json",
                {
                    "torch": str(torch.__version__),
                    "device": str(context.device),
                    "precision": precision,
                    "world_size": context.world_size,
                    "effective_batch_size": effective_batch,
                    "steps_per_epoch": steps_per_epoch,
                    "learning_rate": lr,
                    "dataset_fingerprint": fingerprint,
                    "epoch_semantics": "resampled fixed optimizer updates",
                    "parameters": sum(p.numel() for p in raw_model.parameters()),
                },
            )
            LOGGER.info(
                "device=%s precision=%s ranks=%d effective_batch=%d updates/epoch=%d parameters=%d",
                context.device,
                precision,
                context.world_size,
                effective_batch,
                steps_per_epoch,
                sum(p.numel() for p in raw_model.parameters()),
            )
        for epoch in range(start_epoch, config.train.epochs):
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                break
            loader = build_loader(
                config.data.dataset,
                config.data.train_split,
                config=config.data,
                model_config=config.model,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                seed=config.train.seed,
                epoch=epoch,
                rank=context.rank,
                world_size=context.world_size,
                training=True,
            )
            training_metrics, global_step = train_epoch(
                train_model,
                raw_model,
                loader,
                optimizer,
                scheduler,
                scaler,
                config,
                context,
                precision,
                epoch,
                steps_per_epoch,
                global_step,
            )
            del loader  # Epoch seeding and worker recreation are part of the resume contract.
            validation = None
            improved = False
            if (epoch + 1) % config.train.eval_every == 0 or epoch + 1 == config.train.epochs:
                val_loader = build_loader(
                    config.data.dataset,
                    config.data.val_split,
                    config=config.data,
                    model_config=config.model,
                    batch_size=config.data.batch_size,
                    num_workers=config.data.num_workers,
                    seed=config.train.seed,
                    epoch=0,
                    rank=context.rank,
                    world_size=context.world_size,
                    training=False,
                )
                validation = evaluate_model(
                    raw_model,
                    val_loader,
                    context.device,
                    len(classes),
                    precision=precision,
                    distributed=context.distributed,
                )
                del val_loader
                expected = dataset["splits"][config.data.val_split]
                if (
                    validation["num_videos"] != expected["num_videos"]
                    or validation["num_windows"] != expected["num_samples"]
                ):
                    raise ValueError("Evaluation counts do not match the dataset manifest")
                metric = validation[config.train.best_metric]
                improved = metric > best_metric
                best_metric = max(best_metric, metric)
            rng = capture_rng_state()
            rng_states = [None] * context.world_size
            if context.distributed:
                dist.all_gather_object(rng_states, rng)
            else:
                rng_states[0] = rng
            if context.is_main:
                state = {
                    "format_version": 1,
                    "task": "binary_classification",
                    "num_classes": NUM_CLASSES,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_metric": best_metric,
                    "config": asdict(config),
                    "class_to_idx": classes,
                    "data_preprocessing": dataset["preprocessing"],
                    "dataset_fingerprint": fingerprint,
                    "rng_states": rng_states,
                    "resume_contract": _resume_contract(config),
                    "world_size": context.world_size,
                    "precision": precision,
                    "torch_version": str(torch.__version__),
                }
                atomic_save(state, output / "latest.pt")
                if improved:
                    atomic_save(state, output / "best.pt")
                report = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train": training_metrics,
                    "validation": {
                        k: v for k, v in (validation or {}).items() if k != "predictions"
                    },
                }
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
                if validation is not None:
                    _json_write(output / "validation.json", validation)
                    LOGGER.info(
                        "epoch=%d validation accuracy=%.4f macro_f1=%.4f",
                        epoch + 1,
                        validation["accuracy"],
                        validation["macro_f1"],
                    )
            context.barrier()
        return {"global_step": global_step, "best_metric": best_metric}
    finally:
        context.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Resume a complete epoch boundary from latest.pt")
    parser.add_argument("--dataset", help="Override data.dataset")
    parser.add_argument("--output-dir", help="Override train.output_dir")
    parser.add_argument("--device", help="Override train.device")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Stop after this completed epoch, preserving the full schedule",
    )
    parser.add_argument(
        "--num-threads", type=int, default=None, help="CPU intra-op threads per rank"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if args.dataset:
        config.data.dataset = args.dataset
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.device:
        config.train.device = args.device
    if args.num_threads is not None:
        if args.num_threads < 1:
            parser.error("--num-threads must be positive")
        torch.set_num_threads(args.num_threads)
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("--stop-after-epoch must be positive")
    run_training(config, args.resume, args.stop_after_epoch)


if __name__ == "__main__":
    main()
