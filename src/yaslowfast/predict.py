"""Classify videos using the checkpoint's frame extraction and transforms."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .evaluate import write_json
from .labels import NUM_CLASSES, validate_class_to_idx
from .precision import autocast_context, resolve_precision


def predict_video(
    model: torch.nn.Module,
    video_path: str | Path,
    *,
    config: Any,
    preprocessing: dict[str, Any],
    class_to_idx: dict[str, int],
    device: torch.device | str,
    precision: str = "fp32",
    batch_size: int = 1,
    return_windows: bool = False,
) -> dict[str, Any]:
    """Apply the exact stored window policy, then average window probabilities."""
    from .data import transform_frames
    from .prepare import extract_frames, window_frame_indices

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    validate_class_to_idx(class_to_idx, context="Prediction class_to_idx")
    required = {"fps", "short_side", "window_frames", "window_stride", "jpeg_quality"}
    missing = required.difference(preprocessing)
    if missing:
        raise ValueError(f"Checkpoint is missing data_preprocessing fields: {sorted(missing)}")
    device = torch.device(device)
    precision = resolve_precision(precision, device)
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    started = time.perf_counter()
    probability_sum = torch.zeros(len(classes), dtype=torch.float64)
    count = 0
    windows = []
    previous_training = model.training
    model.eval()
    try:
        with tempfile.TemporaryDirectory(prefix="yaslowfast-predict-") as directory:
            frames = extract_frames(
                video_path,
                Path(directory) / "frames",
                fps=preprocessing["fps"],
                short_side=preprocessing["short_side"],
                jpeg_quality=preprocessing["jpeg_quality"],
            )
            if not frames:
                raise ValueError(f"Video produced no frames: {video_path}")
            indices = window_frame_indices(
                len(frames), preprocessing["window_frames"], preprocessing["window_stride"]
            )
            with torch.inference_mode():
                for offset in range(0, len(indices), batch_size):
                    index_batch = indices[offset : offset + batch_size]
                    pairs = [
                        transform_frames(
                            [frames[index] for index in frame_indices[:valid]],
                            config=config.data,
                            model_config=config.model,
                            training=False,
                        )
                        for _, frame_indices, valid in index_batch
                    ]
                    slow = torch.stack([pair[0] for pair in pairs]).to(device)
                    fast = torch.stack([pair[1] for pair in pairs]).to(device)
                    with autocast_context(device, precision):
                        logits = model(slow, fast)
                    if logits.shape != (len(index_batch), NUM_CLASSES):
                        raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
                    probabilities = logits.float().softmax(dim=1).cpu().double()
                    if not torch.isfinite(probabilities).all():
                        raise ValueError("Model produced non-finite probabilities")
                    probability_sum += probabilities.sum(dim=0)
                    count += len(index_batch)
                    if return_windows:
                        for (start, _, valid), probability in zip(
                            index_batch, probabilities, strict=True
                        ):
                            windows.append(
                                {
                                    "start_seconds": start / preprocessing["fps"],
                                    "valid_frames": valid,
                                    "probabilities": probability.tolist(),
                                }
                            )
        if count == 0:
            raise ValueError("Video produced no evaluation windows")
        probabilities = probability_sum / count
        prediction = int(probabilities.argmax())
        result = {
            "task": "binary_classification",
            "num_classes": NUM_CLASSES,
            "video": str(video_path),
            "prediction": prediction,
            "label": classes[prediction],
            "confidence": float(probabilities[prediction]),
            "probabilities": {
                name: float(probabilities[index]) for index, name in enumerate(classes)
            },
            "num_windows": count,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if return_windows:
            result["windows"] = windows
        return result
    finally:
        model.train(previous_training)


def main(argv: list[str] | None = None) -> None:
    from .checkpoint import load_checkpoint
    from .config import config_from_dict
    from .distributed import DistributedContext
    from .models import build_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--include-windows", action="store_true")
    args = parser.parse_args(argv)
    ctx = DistributedContext.from_environment(args.device)
    try:
        if ctx.distributed:
            raise ValueError("Video prediction is single-process; run without torchrun")
        checkpoint = load_checkpoint(args.checkpoint)
        if "data_preprocessing" not in checkpoint:
            raise ValueError(
                "Checkpoint lacks data_preprocessing; recreate it with the training command"
            )
        config = config_from_dict(checkpoint["config"])
        model = build_model(config.model, len(checkpoint["class_to_idx"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(ctx.device)
        results = [
            predict_video(
                model,
                video,
                config=config,
                preprocessing=checkpoint["data_preprocessing"],
                class_to_idx=checkpoint["class_to_idx"],
                device=ctx.device,
                precision=args.precision,
                batch_size=args.batch_size,
                return_windows=args.include_windows,
            )
            for video in args.video
        ]
        write_json(
            args.output,
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "class_to_idx": checkpoint["class_to_idx"],
                "predictions": results,
            },
        )
        for result in results:
            print(
                json.dumps(
                    {key: result[key] for key in ("video", "label", "confidence", "num_windows")},
                    ensure_ascii=False,
                )
            )
    finally:
        ctx.close()


if __name__ == "__main__":
    main()
