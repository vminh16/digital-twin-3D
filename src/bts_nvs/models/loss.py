from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_inputs(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rgb_pred.ndim not in (3, 4) or rgb_pred.shape[-1] != 3:
        raise ValueError("prediction must have shape (H, W, 3) or (B, H, W, 3)")
    if rgb_gt.shape != rgb_pred.shape:
        raise ValueError("prediction and target must have the same RGB shape")
    if mask.shape != rgb_pred.shape[:-1]:
        raise ValueError(
            "mask shape must match image dimensions without the RGB channel"
        )
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if not rgb_pred.is_floating_point():
        raise ValueError("prediction must be floating point")
    if rgb_gt.dtype != torch.uint8 and not rgb_gt.is_floating_point():
        raise ValueError("target must be uint8 or floating point")
    if not mask.any():
        raise ValueError("mask contains no valid pixels")

    x = rgb_pred.float()
    if rgb_gt.dtype == torch.uint8:
        y = rgb_gt.to(device=x.device, dtype=torch.float32) / 255.0
    else:
        y = rgb_gt.to(device=x.device, dtype=torch.float32)
    m = mask.to(device=x.device)

    if x.ndim == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        m = m.unsqueeze(0)
    return x, y, m


def _masked_l1(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (torch.abs(x - y) * mask.unsqueeze(-1)).sum() / (3 * mask.sum())


class MaskedL1Loss(nn.Module):
    def forward(
        self,
        rgb_pred: torch.Tensor,
        rgb_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x, y, m = _prepare_inputs(rgb_pred, rgb_gt, mask)
        return _masked_l1(x, y, m)


class MaskedSSIMLoss(nn.Module):
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(kernel_size, bool)
            or not isinstance(kernel_size, int)
            or kernel_size < 3
            or kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be an odd integer of at least 3")
        for name, value in {
            "sigma": sigma,
            "k1": k1,
            "k2": k2,
            "data_range": data_range,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

        self.kernel_size = kernel_size
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-coords.square() / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        self.register_buffer(
            "kernel",
            kernel_2d.expand(3, 1, kernel_size, kernel_size).clone(),
        )
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2

    def forward(
        self,
        rgb_pred: torch.Tensor,
        rgb_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_prepared(*_prepare_inputs(rgb_pred, rgb_gt, mask))

    def _forward_prepared(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[1] < self.kernel_size or x.shape[2] < self.kernel_size:
            raise ValueError("images are smaller than the SSIM kernel")
        if self.kernel.device != x.device:
            raise ValueError("SSIM loss module and images must be on the same device")

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        kernel = self.kernel
        mu_x = F.conv2d(x, kernel, groups=3)
        mu_y = F.conv2d(y, kernel, groups=3)
        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y
        variance_x = (F.conv2d(x.square(), kernel, groups=3) - mu_x_sq).clamp_min(0.0)
        variance_y = (F.conv2d(y.square(), kernel, groups=3) - mu_y_sq).clamp_min(0.0)
        covariance = F.conv2d(x * y, kernel, groups=3) - mu_xy

        numerator = (2 * mu_xy + self.c1) * (2 * covariance + self.c2)
        denominator = (mu_x_sq + mu_y_sq + self.c1) * (
            variance_x + variance_y + self.c2
        )
        ssim_map = numerator / denominator

        invalid = (~mask).float().unsqueeze(1)
        valid_windows = (
            F.max_pool2d(
                invalid,
                kernel_size=self.kernel_size,
                stride=1,
            )
            == 0
        )
        if not valid_windows.any():
            raise ValueError("mask contains no valid SSIM windows after erosion")
        valid = valid_windows.float()
        mean_ssim = (ssim_map * valid).sum() / (3 * valid.sum())
        return 1.0 - mean_ssim


class JointLoss(nn.Module):
    def __init__(
        self,
        lambda_dssim: float = 0.2,
        ssim_kernel_size: int = 11,
        ssim_sigma: float = 1.5,
        ssim_k1: float = 0.01,
        ssim_k2: float = 0.03,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(lambda_dssim) or not 0.0 <= lambda_dssim <= 1.0:
            raise ValueError("lambda_dssim must be finite and in [0, 1]")
        self.lambda_dssim = lambda_dssim
        self.ssim_loss = MaskedSSIMLoss(
            kernel_size=ssim_kernel_size,
            sigma=ssim_sigma,
            k1=ssim_k1,
            k2=ssim_k2,
            data_range=data_range,
        )

    def forward(
        self,
        rgb_pred: torch.Tensor,
        rgb_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x, y, m = _prepare_inputs(rgb_pred, rgb_gt, mask)
        l1 = _masked_l1(x, y, m)
        dssim = self.ssim_loss._forward_prepared(x, y, m)
        return (1.0 - self.lambda_dssim) * l1 + self.lambda_dssim * dssim
