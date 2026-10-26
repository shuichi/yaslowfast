"""Learning rates are indexed by completed optimizer updates, not microbatches."""

import math


class WarmupCosine:
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        warmup_start_factor: float = 0.1,
        min_lr_ratio: float = 0.0,
    ):
        if total_steps < 1 or not 0 <= warmup_steps < total_steps:
            raise ValueError("Require 0 <= warmup_steps < total_steps")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_index = 0
        self._apply()

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (
                self.warmup_start_factor + (1 - self.warmup_start_factor) * step / self.warmup_steps
            )
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps - 1)
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(1, progress))
        )

    def _apply(self) -> None:
        factor = self.factor(self.step_index)
        for base, group in zip(self.base_lrs, self.optimizer.param_groups, strict=True):
            group["lr"] = base * factor

    def step(self) -> None:
        self.step_index += 1
        self._apply()

    def state_dict(self):
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_lrs": self.base_lrs,
            "warmup_start_factor": self.warmup_start_factor,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, state):
        current = self.state_dict()
        for key in current.keys() - {"step_index"}:
            if current[key] != state[key]:
                raise ValueError(f"Scheduler setting changed on resume: {key}")
        self.step_index = state["step_index"]
        self._apply()
