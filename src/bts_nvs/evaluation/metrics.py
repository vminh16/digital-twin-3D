from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Literal, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class MetricConfig:
    psnr_max: float
    lpips_backbone: Literal["alex", "vgg"] = "alex"
    lpips_weight_version: str = "0.1"
    ssim_kernel_size: int = 11
    ssim_sigma: float = 1.5
    ssim_k1: float = 0.01
    ssim_k2: float = 0.03
    ssim_padding: Literal["valid"] = "valid"
    crop_border: int = 0
    data_range: float = 1.0

    def __post_init__(self) -> None:
        numeric = (self.psnr_max, self.ssim_sigma, self.ssim_k1, self.ssim_k2, self.data_range)
        if not np.all(np.isfinite(numeric)) or any(value <= 0 for value in numeric):
            raise ValueError("metric scales must be positive and finite")
        if self.lpips_backbone not in {"alex", "vgg"}:
            raise ValueError("LPIPS backbone must be alex or vgg")
        if self.ssim_kernel_size < 3 or self.ssim_kernel_size % 2 == 0:
            raise ValueError("SSIM kernel size must be odd and at least 3")
        if self.ssim_padding != "valid":
            raise ValueError("only valid SSIM padding is supported")
        if isinstance(self.crop_border, bool) or not isinstance(self.crop_border, int) or self.crop_border < 0:
            raise ValueError("crop_border must be a non-negative integer")


class LpipsCallable(Protocol):
    package: str
    version: str
    device: str
    dtype: str

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float: ...


class LpipsBackend:
    package = "lpips"
    dtype = "float32"

    def __init__(self, backbone: str = "alex", device: str = "cpu") -> None:
        import lpips
        import torch

        self.version = version("lpips")
        self.device = device
        self._torch = torch
        self._model = lpips.LPIPS(net=backbone, version="0.1").to(device).eval()

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        torch = self._torch
        with torch.no_grad():
            prediction_tensor = torch.from_numpy(prediction).to(self.device)
            target_tensor = torch.from_numpy(target).to(self.device)
            return float(self._model(prediction_tensor, target_tensor).item())


def _image_array(image: np.ndarray, data_range: float) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("metric images must have shape (H, W, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("metric images must be finite")
    if np.any(array < 0.0) or np.any(array > data_range):
        raise ValueError(f"metric images must be in [0, {data_range:g}]")
    return array


def _crop(image: np.ndarray, border: int) -> np.ndarray:
    if border == 0:
        return image
    if image.shape[0] <= 2 * border or image.shape[1] <= 2 * border:
        raise ValueError("crop border removes the entire image")
    return image[border:-border, border:-border]


def _ssim(prediction: np.ndarray, target: np.ndarray, config: MetricConfig) -> float:
    size = config.ssim_kernel_size
    if prediction.shape[0] < size or prediction.shape[1] < size:
        raise ValueError("images are smaller than the SSIM kernel")
    kernel = cv2.getGaussianKernel(size, config.ssim_sigma, cv2.CV_64F)
    margin = size // 2

    def valid_filter(values: np.ndarray) -> np.ndarray:
        filtered = cv2.sepFilter2D(values, cv2.CV_64F, kernel, kernel, borderType=cv2.BORDER_CONSTANT)
        return filtered[margin:-margin, margin:-margin]

    mean_prediction = valid_filter(prediction)
    mean_target = valid_filter(target)
    variance_prediction = np.maximum(valid_filter(prediction * prediction) - mean_prediction**2, 0.0)
    variance_target = np.maximum(valid_filter(target * target) - mean_target**2, 0.0)
    covariance = valid_filter(prediction * target) - mean_prediction * mean_target
    c1 = (config.ssim_k1 * config.data_range) ** 2
    c2 = (config.ssim_k2 * config.data_range) ** 2
    numerator = (2.0 * mean_prediction * mean_target + c1) * (2.0 * covariance + c2)
    denominator = (mean_prediction**2 + mean_target**2 + c1) * (
        variance_prediction + variance_target + c2
    )
    value = float(np.mean(numerator / denominator))
    if not np.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError(f"SSIM result is invalid: {value}")
    return value


def evaluate_image(
    prediction: np.ndarray,
    target: np.ndarray,
    config: MetricConfig,
    lpips_backend: LpipsCallable,
) -> dict[str, float | bool | None]:
    prediction_array = _image_array(prediction, config.data_range)
    target_array = _image_array(target, config.data_range)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target resolutions differ")
    prediction_array = _crop(prediction_array, config.crop_border)
    target_array = _crop(target_array, config.crop_border)

    mse = float(np.mean((prediction_array - target_array) ** 2, dtype=np.float64))
    if mse == 0.0:
        psnr_db = None
        psnr_is_infinite = True
        psnr_normalized = 1.0
    else:
        raw_psnr = float(10.0 * np.log10(config.data_range**2 / mse))
        psnr_db = raw_psnr
        psnr_is_infinite = False
        psnr_normalized = float(np.clip(raw_psnr / config.psnr_max, 0.0, 1.0))

    ssim = _ssim(prediction_array, target_array, config)
    prediction_lpips = np.transpose(
        2.0 * prediction_array / config.data_range - 1.0, (2, 0, 1)
    )[None].astype(np.float32)
    target_lpips = np.transpose(
        2.0 * target_array / config.data_range - 1.0, (2, 0, 1)
    )[None].astype(np.float32)
    lpips_value = float(lpips_backend(prediction_lpips, target_lpips))
    if not np.isfinite(lpips_value) or not 0.0 <= lpips_value <= 1.0:
        raise ValueError(f"LPIPS result is invalid: {lpips_value}")
    composite = 0.4 * (1.0 - lpips_value) + 0.3 * ssim + 0.3 * psnr_normalized
    return {
        "psnr_db": psnr_db,
        "psnr_is_infinite": psnr_is_infinite,
        "psnr_normalized": psnr_normalized,
        "ssim": ssim,
        "lpips": lpips_value,
        "composite": float(composite),
    }

