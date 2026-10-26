"""Modern torch.amp helpers, with explicit device/precision validation."""

from contextlib import nullcontext

import torch


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        if device.type == "cuda":
            return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        return "fp32"
    if requested not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unknown precision: {requested}")
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("FP16 training/inference is supported on CUDA only")
    if requested == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support BF16")
        if device.type not in {"cuda", "cpu"}:
            raise ValueError("BF16 is supported on CUDA and CPU only")
    return requested


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)
