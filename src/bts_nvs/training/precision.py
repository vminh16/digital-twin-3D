from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Mapping

import torch


class TrainingPrecision:
    """Own FP32/AMP mechanics without changing FP32 model parameters."""

    def __init__(self, mode: str, device: torch.device) -> None:
        if mode not in {"fp32", "amp-fp16"}:
            raise ValueError(f"unsupported training precision: {mode}")
        if mode == "amp-fp16" and device.type != "cuda":
            raise ValueError("amp-fp16 requires CUDA")
        self.mode = mode
        self.device = device
        self.scaler = (
            torch.amp.GradScaler("cuda") if mode == "amp-fp16" else None
        )
        self._active_optimizers: tuple[torch.optim.Optimizer, ...] = ()

    def autocast(self) -> ContextManager:
        if self.mode == "fp32":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def backward_and_unscale(
        self,
        loss: torch.Tensor,
        optimizers: Mapping[str, torch.optim.Optimizer],
        projected_means: torch.Tensor,
    ) -> float:
        if self.scaler is None:
            loss.backward()
            return 1.0

        scale = float(self.scaler.get_scale())
        self.scaler.scale(loss).backward()
        self._active_optimizers = tuple(
            optimizer
            for optimizer in optimizers.values()
            if any(
                parameter.grad is not None
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
        )
        for optimizer in self._active_optimizers:
            self.scaler.unscale_(optimizer)

        if projected_means.grad is None:
            raise RuntimeError("AMP requires a retained projected means gradient")
        projected_means.grad.div_(scale)
        return scale

    def step(self, optimizers: Mapping[str, torch.optim.Optimizer]) -> None:
        if self.scaler is None:
            for optimizer in optimizers.values():
                optimizer.step()
            return

        for optimizer in self._active_optimizers:
            self.scaler.step(optimizer)
        self.scaler.update()
        self._active_optimizers = ()

    def state_dict(self) -> dict:
        return {} if self.scaler is None else self.scaler.state_dict()

    def load_state_dict(self, state: dict) -> None:
        if self.scaler is None:
            if state:
                raise ValueError("FP32 precision state must be empty")
            return
        if not state:
            raise ValueError("AMP checkpoint is missing precision state")
        self.scaler.load_state_dict(state)
