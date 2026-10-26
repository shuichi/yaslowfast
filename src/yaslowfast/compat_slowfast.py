"""Removable adapter for the two-input SlowFast ONNX compatibility contract.

The external inputs contain RGB values in [0, 1]. Spatial preprocessing and
normalization belong to this adapter. Fast frames are authoritative: the Slow
pathway is regenerated using the checkpoint's sampling rule. The original Slow
input remains declared in ONNX solely to preserve the caller's input interface.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import DataConfig, ModelConfig

if TYPE_CHECKING:
    import onnx

INPUT_NAMES = ("input_slow", "input_fast")
OUTPUT_NAME = "output"
INPUT_SHAPES = ((3, 8, 430, 295), (3, 32, 430, 295))
BATCH_SYMBOL = "batch_size"
SLOW_INDICES = (0, 4, 8, 13, 17, 22, 26, 31)


class CompatSlowFast(nn.Module):
    """Adapt raw portrait clips to a checkpoint's evaluation preprocessing.

    Output columns zero and one preserve the model's training class IDs. The
    checkpoint must use the caller's fixed semantics for those IDs. Class names
    are neither inspected nor used to reorder the logits.
    """

    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        data_config: DataConfig,
    ) -> None:
        super().__init__()
        if model_config.num_frames != 32 or model_config.alpha != 4:
            raise ValueError("SlowFast compatibility requires num_frames=32 and alpha=4")
        if type(model_config.crop_size) is not int or model_config.crop_size < 32:
            raise ValueError("Compatibility crop_size must be an integer >=32")
        if (
            type(data_config.train_min_scale) is not int
            or data_config.train_min_scale < model_config.crop_size
        ):
            raise ValueError("Compatibility resize short side must be an integer >=crop_size")
        if (
            len(data_config.mean) != 3
            or len(data_config.std) != 3
            or not all(math.isfinite(value) for value in (*data_config.mean, *data_config.std))
            or any(value <= 0 for value in data_config.std)
        ):
            raise ValueError("Compatibility mean/std must be finite RGB triples with positive std")
        self.model = model
        self.crop_size = model_config.crop_size
        self.resize_short_side = data_config.train_min_scale
        self.resize_size = (
            round(INPUT_SHAPES[1][-2] * self.resize_short_side / INPUT_SHAPES[1][-1]),
            self.resize_short_side,
        )
        self.crop_top = (self.resize_size[0] - self.crop_size) // 2
        self.crop_left = (self.resize_size[1] - self.crop_size) // 2
        self.mean_values = tuple(data_config.mean)
        self.std_values = tuple(data_config.std)
        mean = torch.tensor(self.mean_values, dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor(self.std_values, dtype=torch.float32).view(1, 3, 1, 1, 1)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError(
                "Compatibility mean/std must remain finite with positive std in float32"
            )
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("slow_indices", torch.tensor(SLOW_INDICES, dtype=torch.int64))

    def forward(self, slow: Tensor, fast: Tensor) -> Tensor:
        """Preserve output columns [0, 1]; ``slow`` is retained for the external ABI."""
        batch = fast.shape[0]
        frames = fast.permute(0, 2, 1, 3, 4).reshape(-1, 3, 430, 295)
        frames = F.interpolate(
            frames,
            size=self.resize_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        frames = frames[
            :,
            :,
            self.crop_top : self.crop_top + self.crop_size,
            self.crop_left : self.crop_left + self.crop_size,
        ]
        frames = frames.reshape(batch, 32, 3, self.crop_size, self.crop_size).permute(0, 2, 1, 3, 4)
        frames = (frames - self.mean) / self.std
        slow_frames = torch.index_select(frames, 2, self.slow_indices)
        return self.model(slow_frames, frames)

    def metadata(self) -> dict[str, Any]:
        """Describe fixed output indices and all graph-internal transforms."""
        return {
            "compat_mode": "slowfast",
            "output_class_indices": [0, 1],
            "preprocessing": {
                "location": "inside_onnx",
                "color_space": "RGB",
                "pixel_range_before_normalization": [0.0, 1.0],
                "authoritative_input": "input_fast",
                "retained_unused_inputs": ["input_slow"],
                "external_slow_indices": list(range(0, 32, 4)),
                "mean": list(self.mean_values),
                "std": list(self.std_values),
                "resize_short_side": self.resize_short_side,
                "resize_size": list(self.resize_size),
                "resize_long_side_rounding": "round(original_long_side * short_side / original_short_side)",
                "resize_interpolation": "bilinear, align_corners=False, antialias=True",
                "spatial_crop": "center",
                "crop_size": self.crop_size,
                "crop_offset": [self.crop_top, self.crop_left],
                "fast_num_frames": 32,
                "slow_num_frames": 8,
                "slow_indices": list(SLOW_INDICES),
                "temporal_sampling": "use the caller's 32 Fast frames without fps conversion",
            },
        }


def make_compat_inputs(
    batch_size: int, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Construct raw RGB clips and the caller's stride-four Slow input."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Compatibility batch_size must be a positive integer")
    fast = torch.rand(batch_size, *INPUT_SHAPES[1], generator=generator, dtype=torch.float32)
    return fast[:, :, ::4].contiguous(), fast


def _check_descriptor(value: Any, name: str, trailing_shape: tuple[int, ...]) -> str:
    import onnx

    if value.name != name or not value.type.HasField("tensor_type"):
        raise ValueError(f"Compatibility requires tensor {name}")
    tensor = value.type.tensor_type
    if tensor.elem_type != onnx.TensorProto.FLOAT or not tensor.HasField("shape"):
        raise ValueError(f"Compatibility tensor {name} must be float32 with a declared shape")
    dimensions = tensor.shape.dim
    if len(dimensions) != len(trailing_shape) + 1:
        raise ValueError(f"Compatibility tensor {name} has an invalid rank")
    if not dimensions[0].HasField("dim_param") or not dimensions[0].dim_param:
        raise ValueError(f"Compatibility tensor {name} must have a dynamic batch dimension")
    for dimension, expected in zip(dimensions[1:], trailing_shape):
        if not dimension.HasField("dim_value") or dimension.dim_value != expected:
            raise ValueError(
                f"Compatibility tensor {name} must have trailing shape {trailing_shape}"
            )
    return dimensions[0].dim_param


def finalize_compat_graph(model: onnx.ModelProto) -> onnx.ModelProto:
    """Restore an unused Slow input and standardize the verified dynamic signature.

    Export optimizers may remove the unused Slow input. Declaring it as a graph
    input is sufficient for callers to keep feeding it; no dummy arithmetic is
    inserted. Static batch exports are rejected instead of relabelled dynamic.
    """
    import onnx

    inputs = {value.name: value for value in model.graph.input}
    if len(inputs) != len(model.graph.input) or set(inputs) not in (
        {INPUT_NAMES[1]},
        set(INPUT_NAMES),
    ):
        raise ValueError("Compatibility graph must expose input_fast and optionally input_slow")
    if len(model.graph.output) != 1:
        raise ValueError("Compatibility graph must have exactly one output")
    symbol = _check_descriptor(inputs[INPUT_NAMES[1]], INPUT_NAMES[1], INPUT_SHAPES[1])
    output = model.graph.output[0]
    if _check_descriptor(output, OUTPUT_NAME, (2,)) != symbol:
        raise ValueError("Compatibility input and output must share a dynamic batch dimension")
    if INPUT_NAMES[0] in inputs:
        if _check_descriptor(inputs[INPUT_NAMES[0]], INPUT_NAMES[0], INPUT_SHAPES[0]) != symbol:
            raise ValueError("Compatibility inputs must share a dynamic batch dimension")
    else:
        inputs[INPUT_NAMES[0]] = onnx.helper.make_tensor_value_info(
            INPUT_NAMES[0], onnx.TensorProto.FLOAT, [symbol, *INPUT_SHAPES[0]]
        )
    ordered = [inputs[name] for name in INPUT_NAMES]
    for value in (*ordered, output):
        value.type.tensor_type.shape.dim[0].dim_param = BATCH_SYMBOL
    # Keep shape annotations consistent wherever they use the exported batch symbol.
    for value in model.graph.value_info:
        if value.type.HasField("tensor_type"):
            for dimension in value.type.tensor_type.shape.dim:
                if dimension.HasField("dim_param") and dimension.dim_param == symbol:
                    dimension.dim_param = BATCH_SYMBOL
    del model.graph.input[:]
    model.graph.input.extend(ordered)
    validate_compat_graph(model)
    return model


def validate_compat_graph(model: onnx.ModelProto) -> None:
    """Require exactly the public float32, two-input, dynamic-batch signature."""
    if tuple(value.name for value in model.graph.input) != INPUT_NAMES:
        raise ValueError("Compatibility graph input names/order do not match the contract")
    if len(model.graph.output) != 1:
        raise ValueError("Compatibility graph must have exactly one output")
    for value, name, shape in zip(model.graph.input, INPUT_NAMES, INPUT_SHAPES):
        if _check_descriptor(value, name, shape) != BATCH_SYMBOL:
            raise ValueError("Compatibility inputs must use the shared batch_size dimension")
    if _check_descriptor(model.graph.output[0], OUTPUT_NAME, (2,)) != BATCH_SYMBOL:
        raise ValueError("Compatibility output must use the shared batch_size dimension")
    if any(initializer.name in INPUT_NAMES for initializer in model.graph.initializer):
        raise ValueError("Compatibility inputs must not be optional initializer inputs")
