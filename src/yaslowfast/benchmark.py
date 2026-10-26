"""Measure model-only latency on the current machine with fixed-size inputs."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluate import write_json
from .labels import NUM_CLASSES
from .precision import autocast_context, resolve_precision


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_model(
    model: torch.nn.Module,
    slow: torch.Tensor,
    fast: torch.Tensor,
    *,
    device: torch.device | str,
    precision: str = "fp32",
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, Any]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    device = torch.device(device)
    precision = resolve_precision(precision, device)
    if slow.device != device or fast.device != device:
        # torch.device('cuda') and cuda:0 designate the same active device but
        # compare unequal. Moving is a no-op in that case, and outside timing.
        slow, fast = slow.to(device), fast.to(device)
    previous_training = model.training
    model.eval()
    try:
        with torch.inference_mode(), autocast_context(device, precision):
            for _ in range(warmup):
                model(slow, fast)
            synchronize(device)
            baseline_memory = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                baseline_memory = torch.cuda.memory_allocated(device)
            durations = []
            for _ in range(iterations):
                synchronize(device)
                start = time.perf_counter()
                output = model(slow, fast)
                synchronize(device)
                durations.append(time.perf_counter() - start)
            if output.shape != (fast.shape[0], NUM_CLASSES):
                raise ValueError("Binary model output must have shape (batch_size, 2)")
            if not torch.isfinite(output).all():
                raise ValueError("Benchmark output contains non-finite values")
        milliseconds = np.asarray(durations) * 1000
        return {
            "task": "binary_classification",
            "num_classes": NUM_CLASSES,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.machine(),
            "torch_version": torch.__version__,
            "precision": precision,
            "batch_size": fast.shape[0],
            "input_shapes": {"slow": list(slow.shape), "fast": list(fast.shape)},
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "latency_ms_mean": float(milliseconds.mean()),
            "latency_ms_p50": float(np.percentile(milliseconds, 50)),
            "latency_ms_p95": float(np.percentile(milliseconds, 95)),
            "clips_per_second": float(fast.shape[0] * iterations / sum(durations)),
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "incremental_peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
            - baseline_memory
            if device.type == "cuda"
            else None,
            "measurement": "Model forward only; synthetic resident tensors; synchronized per batch; excludes video decoding, JPEG, transforms and transfers",
        }
    finally:
        model.train(previous_training)


def main(argv: list[str] | None = None) -> None:
    from .checkpoint import load_checkpoint
    from .config import config_from_dict, load_config
    from .distributed import DistributedContext
    from .models import build_model

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--config", type=Path, help="Benchmark a randomly initialized architecture")
    parser.add_argument(
        "--num-classes",
        type=int,
        choices=[NUM_CLASSES],
        default=NUM_CLASSES,
        help="Binary classification is fixed to two classes",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--channels-last-3d", action="store_true")
    parser.add_argument("--threads", type=int, help="Number of CPU intra-op threads")
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.threads is not None:
        if args.threads <= 0:
            parser.error("threads must be positive")
        torch.set_num_threads(args.threads)
    ctx = DistributedContext.from_environment(args.device)
    try:
        if ctx.distributed:
            raise ValueError("Inference benchmarking is single-process; run without torchrun")
        checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
        config = config_from_dict(checkpoint["config"]) if checkpoint else load_config(args.config)
        num_classes = len(checkpoint["class_to_idx"]) if checkpoint else args.num_classes
        model = build_model(config.model, num_classes)
        if checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
        model.to(ctx.device).eval()
        precision = resolve_precision(args.precision, ctx.device)
        fast = torch.randn(
            args.batch_size,
            3,
            config.model.num_frames,
            config.model.crop_size,
            config.model.crop_size,
            device=ctx.device,
        )
        slow = torch.randn(
            args.batch_size,
            3,
            config.model.num_frames // config.model.alpha,
            config.model.crop_size,
            config.model.crop_size,
            device=ctx.device,
        )
        with torch.inference_mode(), autocast_context(ctx.device, precision):
            expected = model(slow, fast).float()
        if args.channels_last_3d:
            model.to(memory_format=torch.channels_last_3d)
            slow = slow.contiguous(memory_format=torch.channels_last_3d)
            fast = fast.contiguous(memory_format=torch.channels_last_3d)
        if args.compile:
            model = torch.compile(model)
        with torch.inference_mode(), autocast_context(ctx.device, precision):
            actual = model(slow, fast).float()
        tolerance = 2e-2 if precision in {"bf16", "fp16"} else 1e-4
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        report = benchmark_model(
            model,
            slow,
            fast,
            device=ctx.device,
            precision=precision,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        report.update(
            {
                "compiled": args.compile,
                "channels_last_3d": args.channels_last_3d,
                "max_output_difference": float((expected - actual).abs().max().item()),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "cpu_threads": torch.get_num_threads(),
            }
        )
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
