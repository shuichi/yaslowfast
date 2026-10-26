"""Export SlowFast with native inputs or the optional SlowFast compatibility interface."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .config import config_from_dict
from .labels import NUM_CLASSES, validate_class_to_idx
from .models import build_model
from .onnx_export import DEFAULT_OPSET, MAX_OPSET, export_onnx, validate_opset


def _validate_output(value: Tensor, batch_size: int) -> None:
    if not isinstance(value, Tensor) or value.shape != (batch_size, NUM_CLASSES):
        raise ValueError("Binary model output must have shape (batch_size, 2)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Export requires finite model outputs")


def export_model(
    model: nn.Module,
    inputs: tuple[Tensor, Tensor],
    output: str | Path,
    *,
    format: str = "torch",
    metadata: dict[str, Any] | None = None,
    compat_mode: str | None = None,
    opset_version: int = DEFAULT_OPSET,
) -> dict[str, Any]:
    """Export on CPU, reload the artifact, and check output before publishing.

    ONNX requires the optional ``export`` dependencies. Both formats produce
    two logits per clip; probabilities are softmax(logits, dim=1). Input tensors
    must already use the saved sampling and normalization in native mode.
    Compatibility mode requires a CompatSlowFast wrapper and raw portrait clips.
    This moves the supplied model into float32 CPU evaluation mode.
    """
    if format not in {"torch", "onnx"}:
        raise ValueError("format must be torch or onnx")
    _validate_options(format, compat_mode, opset_version)
    description = dict(metadata or {})
    input_names, output_name = ("slow", "fast"), "logits"
    finalize = None
    if compat_mode == "slowfast":
        from .compat_slowfast import (
            INPUT_NAMES,
            INPUT_SHAPES,
            OUTPUT_NAME,
            CompatSlowFast,
            finalize_compat_graph,
        )

        if not isinstance(model, CompatSlowFast):
            raise ValueError("Compatibility export_model requires a CompatSlowFast wrapper")
        if (
            len(inputs) != 2
            or any(
                value.ndim != 5 or tuple(value.shape[1:]) != shape
                for value, shape in zip(inputs, INPUT_SHAPES)
            )
            or inputs[0].shape[0] != inputs[1].shape[0]
            or inputs[0].shape[0] < 1
        ):
            raise ValueError("Compatibility inputs must match the two-input portrait clip contract")
        description.update(model.metadata())
        input_names, output_name = INPUT_NAMES, OUTPUT_NAME
        finalize = finalize_compat_graph
    if "class_to_idx" in description:
        validate_class_to_idx(description["class_to_idx"], context="Export class_to_idx")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().float().eval()
    inputs = tuple(x.detach().cpu().float() for x in inputs)
    with torch.inference_mode():
        expected = model(*inputs).cpu()
    _validate_output(expected, inputs[0].shape[0])
    description.update(
        {
            "format_version": 1,
            "task": "binary_classification",
            "num_classes": NUM_CLASSES,
            "format": format,
            "torch_version": str(torch.__version__),
            "inputs": [
                {
                    "name": name,
                    "shape": (
                        ["batch_size", *value.shape[1:]] if compat_mode else list(value.shape)
                    ),
                    "dtype": "float32",
                    "layout": "NCTHW",
                }
                for name, value in zip(input_names, inputs)
            ],
            "output": {
                "name": output_name,
                "shape": ["batch_size", NUM_CLASSES] if compat_mode else list(expected.shape),
                "activation": "none",
                "class_ids": [0, 1],
                "probability_activation": "softmax(dim=1)",
            },
        }
    )
    # Validate serializability before creating the exported artifact.
    json.dumps(description, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".pt2" if format == "torch" else ".onnx", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        onnx_details: dict[str, Any] = {}
        batch_errors: dict[str, float] = {}
        if format == "torch":
            program = torch.export.export(model, inputs, strict=True)
            torch.export.save(program, temporary)
            restored = torch.export.load(temporary).module()
            with torch.inference_mode():
                observed = restored(*inputs).cpu()
        else:
            # Symbolic export specializes dimensions of size one. Trace with at
            # least two examples, then separately verify real batch-one inference.
            trace_inputs = inputs
            if compat_mode and inputs[0].shape[0] == 1:
                trace_inputs = tuple(x.expand(2, *x.shape[1:]).contiguous() for x in inputs)
            session, onnx_details = export_onnx(
                model,
                trace_inputs,
                temporary,
                input_names=input_names,
                output_name=output_name,
                opset_version=opset_version,
                dynamic_batch=bool(compat_mode),
                finalize=finalize,
            )
            del trace_inputs
            feeds = {name: value.numpy() for name, value in zip(input_names, inputs)}
            observed = torch.from_numpy(np.asarray(session.run([output_name], feeds)[0]))
            if compat_mode:
                from .compat_slowfast import make_compat_inputs

                generator = torch.Generator().manual_seed(1)
                for batch_size in (1, 2, 4):
                    if batch_size == inputs[0].shape[0]:
                        reference, actual = expected, observed
                    else:
                        batch_inputs = make_compat_inputs(batch_size, generator=generator)
                        with torch.inference_mode():
                            reference = model(*batch_inputs).cpu()
                        batch_feeds = {
                            name: value.numpy() for name, value in zip(input_names, batch_inputs)
                        }
                        actual = torch.from_numpy(
                            np.asarray(session.run([output_name], batch_feeds)[0])
                        )
                        del batch_inputs, batch_feeds
                    _validate_output(reference, batch_size)
                    _validate_output(actual, batch_size)
                    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
                    batch_errors[str(batch_size)] = float((actual - reference).abs().max())
        _validate_output(observed, inputs[0].shape[0])
        torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
        description["verification"] = {
            "backend": "torch.export" if format == "torch" else "onnxruntime-cpu",
            "max_absolute_error": max(
                float((observed - expected).abs().max()), *batch_errors.values()
            )
            if batch_errors
            else float((observed - expected).abs().max()),
            "rtol": 1e-4,
            "atol": 1e-5,
            **onnx_details,
        }
        if batch_errors:
            description["verification"]["batch_max_absolute_errors"] = batch_errors
        json.dumps(description, allow_nan=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, suffix=".json", delete=False
    ) as stream:
        json.dump(description, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        metadata_temporary = Path(stream.name)
    try:
        os.replace(metadata_temporary, metadata_path)
    finally:
        metadata_temporary.unlink(missing_ok=True)
    return {
        "artifact": str(destination),
        "metadata": str(metadata_path),
        "verification": description["verification"],
    }


def _validate_options(
    format: str,
    compat_mode: str | None,
    opset_version: int,
) -> None:
    if compat_mode not in {None, "slowfast"}:
        raise ValueError("compat_mode must be None or slowfast")
    if compat_mode and format != "onnx":
        raise ValueError("Compatibility mode only supports ONNX export")
    if format == "onnx":
        validate_opset(opset_version)


def export_checkpoint(
    checkpoint_path: str | Path,
    output: str | Path,
    *,
    format: str = "torch",
    batch_size: int = 1,
    compat_mode: str | None = None,
    opset_version: int = DEFAULT_OPSET,
) -> dict[str, Any]:
    from .checkpoint import load_checkpoint

    _validate_options(format, compat_mode, opset_version)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint = load_checkpoint(checkpoint_path)
    config = config_from_dict(checkpoint["config"])
    model = build_model(config.model, len(checkpoint["class_to_idx"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    generator = torch.Generator().manual_seed(0)
    metadata = {
        "class_to_idx": checkpoint["class_to_idx"],
        "model": asdict(config.model),
        "preprocessing": {
            "video_extraction": checkpoint.get("data_preprocessing", {}),
            "color_space": "RGB",
            "pixel_range_before_normalization": [0.0, 1.0],
            "mean": list(config.data.mean),
            "std": list(config.data.std),
            "resize_short_side": config.data.train_min_scale,
            "resize_long_side_rounding": "round(original_long_side * short_side / original_short_side)",
            "resize_interpolation": "bilinear, align_corners=False, antialias=True",
            "spatial_crop": "center",
            "crop_offset": "floor((resized_dimension - crop_size) / 2)",
            "crop_size": config.model.crop_size,
            "fast_num_frames": config.model.num_frames,
            "slow_num_frames": config.model.num_frames // config.model.alpha,
            "slow_indices": torch.linspace(
                0, config.model.num_frames - 1, config.model.num_frames // config.model.alpha
            )
            .long()
            .tolist(),
            "temporal_crop": "center of each window's valid_frames prefix; repeat the last frame if short",
            "video_aggregation": "mean of clip probabilities",
        },
    }
    if compat_mode == "slowfast":
        from .compat_slowfast import CompatSlowFast, make_compat_inputs

        model = CompatSlowFast(model, config.model, config.data)
        inputs = make_compat_inputs(batch_size, generator=generator)
        metadata.update(model.metadata())
        metadata["training_video_extraction"] = checkpoint.get("data_preprocessing", {})
    else:
        common = (batch_size, 3)
        spatial = (config.model.crop_size, config.model.crop_size)
        inputs = (
            torch.randn(
                *common,
                config.model.num_frames // config.model.alpha,
                *spatial,
                generator=generator,
            ),
            torch.randn(*common, config.model.num_frames, *spatial, generator=generator),
        )
    return export_model(
        model,
        inputs,
        output,
        format=format,
        metadata=metadata,
        compat_mode=compat_mode,
        opset_version=opset_version,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--compat-mode",
        choices=("slowfast",),
        help="Export the single-file, dynamic-batch SlowFast compatibility interface (ONNX only)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        choices=range(DEFAULT_OPSET, MAX_OPSET + 1),
        default=DEFAULT_OPSET,
        help="ONNX operator set (default: 18, maximum: 22)",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="CPU export and verification threads"
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        _validate_options(args.format, args.compat_mode, args.opset_version)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)
    result = export_checkpoint(
        args.checkpoint,
        args.output,
        format=args.format,
        batch_size=args.batch_size,
        compat_mode=args.compat_mode,
        opset_version=args.opset_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
