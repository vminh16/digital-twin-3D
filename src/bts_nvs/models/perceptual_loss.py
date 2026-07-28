from __future__ import annotations

import torch
import torch.nn.functional as F


def perceptual_sensitivity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != valid_mask.shape:
        raise ValueError("sensitivity prediction, target, and mask must match")
    if prediction.ndim != 2 or valid_mask.dtype != torch.bool:
        raise ValueError("sensitivity inputs must be 2D with a boolean mask")
    if not bool(valid_mask.any()):
        raise ValueError("sensitivity mask must contain valid pixels")
    losses = F.binary_cross_entropy(
        prediction.clamp(1e-6, 1.0 - 1e-6),
        target,
        reduction="none",
    )
    return losses[valid_mask].mean()
