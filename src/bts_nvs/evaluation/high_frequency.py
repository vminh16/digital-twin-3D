from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _rgb_image(value: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(value, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3)")
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must be finite")
    if np.any(image < 0.0) or np.any(image > 1.0):
        raise ValueError(f"{name} must be in [0, 1]")
    return image


def _luminance(image: np.ndarray) -> np.ndarray:
    return image @ np.asarray((0.299, 0.587, 0.114), dtype=np.float64)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(x, y)


def high_frequency_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    predicted_rgb = _rgb_image(prediction, "prediction")
    target_rgb = _rgb_image(target, "target")
    if predicted_rgb.shape != target_rgb.shape:
        raise ValueError("prediction and target resolutions differ")

    predicted_y = _luminance(predicted_rgb)
    target_y = _luminance(target_rgb)
    predicted_gradient = _gradient_magnitude(predicted_y)
    target_gradient = _gradient_magnitude(target_y)
    edge_mask = target_gradient >= np.quantile(target_gradient, 0.90)
    flat_mask = target_gradient <= np.quantile(target_gradient, 0.50)
    scale = max(float(np.mean(target_gradient[edge_mask])), 1e-12)

    missing_edge = float(
        np.mean(np.maximum(target_gradient - predicted_gradient, 0.0)[edge_mask])
        / scale
    )
    spurious_edge = float(
        np.mean(np.maximum(predicted_gradient - target_gradient, 0.0)[flat_mask])
        / scale
    )
    target_laplacian = cv2.Laplacian(target_y, cv2.CV_64F, ksize=3)
    predicted_laplacian = cv2.Laplacian(predicted_y, cv2.CV_64F, ksize=3)
    hf_l1 = float(np.mean(np.abs(target_laplacian - predicted_laplacian)))
    result = {
        "hf_l1": hf_l1,
        "missing_edge": missing_edge,
        "spurious_edge": spurious_edge,
    }
    if not all(np.isfinite(tuple(result.values()))):
        raise ValueError("high-frequency metrics must be finite")
    return result


def evaluate_render_directory(dataset, render_dir: Path) -> dict:
    samples = tuple(dataset[index] for index in range(len(dataset)))
    if not samples:
        raise ValueError("validation dataset is empty")
    output_names = tuple(
        Path(sample.image_name).with_suffix(".png").name for sample in samples
    )
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError("validation render names collide after PNG conversion")

    image_reports: dict[str, dict[str, float]] = {}
    root = Path(render_dir)
    for sample, output_name in zip(samples, output_names):
        path = root / output_name
        if not path.is_file():
            raise FileNotFoundError(f"missing validation render: {path}")
        with Image.open(path) as source:
            if source.mode != "RGB":
                raise ValueError(f"validation render must be RGB: {path}")
            prediction = np.asarray(source, dtype=np.float64) / 255.0
        target = sample.image.astype(np.float64) / 255.0
        if prediction.shape != target.shape:
            raise ValueError(f"validation render resolution mismatch: {path}")
        prediction = prediction.copy()
        prediction[~sample.valid_mask] = target[~sample.valid_mask]
        image_reports[sample.image_name] = high_frequency_metrics(
            prediction,
            target,
        )

    values = tuple(image_reports.values())
    return {
        "schema_version": 1,
        "scene_id": dataset.manifest.scene_id,
        "image_count": len(values),
        "hf_l1_mean": float(np.mean([item["hf_l1"] for item in values])),
        "missing_edge_mean": float(
            np.mean([item["missing_edge"] for item in values])
        ),
        "spurious_edge_mean": float(
            np.mean([item["spurious_edge"] for item in values])
        ),
        "images": image_reports,
    }
