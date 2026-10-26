"""One process per device, launched with torchrun (or directly for one process)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(cls, device_arg: str = "auto", timeout_seconds: int = 300):
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid RANK/WORLD_SIZE")
        # CPU is the portable default on non-CUDA hosts. MPS is an explicit opt-in.
        name = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device_arg == "auto" else device_arg
        )
        device = torch.device(name)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            if world_size > 1 and device.index not in (None, local_rank):
                raise ValueError("With torchrun, use --device cuda; LOCAL_RANK selects the GPU")
            device = torch.device("cuda", local_rank if device.index is None else device.index)
            torch.cuda.set_device(device)
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available")
            if world_size > 1:
                raise ValueError("MPS does not support this DDP configuration; use CUDA or CPU")
        elif device.type != "cpu":
            raise ValueError("Supported devices: auto, cuda, cuda:N, cpu, mps")
        if world_size > 1:
            backend = "nccl" if device.type == "cuda" else "gloo"
            kwargs = {"device_id": device} if device.type == "cuda" else {}
            dist.init_process_group(backend, timeout=timedelta(seconds=timeout_seconds), **kwargs)
        return cls(device, rank, local_rank, world_size)

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def all_ranks_finite(value: torch.Tensor, context: DistributedContext) -> bool:
    flag = torch.isfinite(value.detach()).all().to(dtype=torch.int32, device=context.device)
    if context.distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())
