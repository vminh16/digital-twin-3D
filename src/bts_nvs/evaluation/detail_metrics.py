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


def _mask(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError(f"valid_mask must have shape {shape}")
    if mask.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean")
    if not np.any(mask):
        raise ValueError("valid_mask must contain at least one valid pixel")
    return mask


def _luminance(image: np.ndarray) -> np.ndarray:
    return image @ np.asarray((0.299, 0.587, 0.114), dtype=np.float64)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(x, y)


def _symmetric_edge_distance(
    predicted_edges: np.ndarray,
    target_edges: np.ndarray,
) -> float:
    predicted_count = int(np.count_nonzero(predicted_edges))
    target_count = int(np.count_nonzero(target_edges))
    if predicted_count == 0 and target_count == 0:
        return 0.0
    if predicted_count == 0 or target_count == 0:
        return 1.0

    distance_to_target = cv2.distanceTransform(
        (~target_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_prediction = cv2.distanceTransform(
        (~predicted_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    diagonal = float(np.hypot(*predicted_edges.shape))
    return float(
        0.5
        * (
            np.mean(distance_to_target[predicted_edges])
            + np.mean(distance_to_prediction[target_edges])
        )
        / diagonal
    )


def detail_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    predicted_rgb = _rgb_image(prediction, "prediction").copy()
    target_rgb = _rgb_image(target, "target")
    if predicted_rgb.shape != target_rgb.shape:
        raise ValueError("prediction and target resolutions differ")
    mask = _mask(valid_mask, target_rgb.shape[:2])
    predicted_rgb[~mask] = target_rgb[~mask]

    predicted_y = _luminance(predicted_rgb)
    target_y = _luminance(target_rgb)
    predicted_gradient = _gradient_magnitude(predicted_y)
    target_gradient = _gradient_magnitude(target_y)
    valid_target_gradient = target_gradient[mask]
    edge_threshold = max(float(np.quantile(valid_target_gradient, 0.90)), 1e-12)
    flat_threshold = float(np.quantile(valid_target_gradient, 0.50))
    target_edges = (target_gradient >= edge_threshold) & mask
    predicted_edges = (predicted_gradient >= edge_threshold) & mask
    flat_mask = (target_gradient <= flat_threshold) & mask
    scale = max(
        float(np.mean(target_gradient[target_edges]))
        if np.any(target_edges)
        else 0.0,
        1e-12,
    )

    missing_edge = float(
        np.mean(
            np.maximum(target_gradient - predicted_gradient, 0.0)[target_edges]
        )
        / scale
        if np.any(target_edges)
        else 0.0
    )
    spurious_edge = float(
        np.mean(
            np.maximum(predicted_gradient - target_gradient, 0.0)[flat_mask]
        )
        / scale
        if np.any(flat_mask)
        else 0.0
    )
    target_laplacian = cv2.Laplacian(target_y, cv2.CV_64F, ksize=3)
    predicted_laplacian = cv2.Laplacian(predicted_y, cv2.CV_64F, ksize=3)
    result = {
        "hf_l1": float(
            np.mean(np.abs(target_laplacian - predicted_laplacian)[mask])
        ),
        "missing_edge": missing_edge,
        "spurious_edge": spurious_edge,
        "symmetric_edge_distance": _symmetric_edge_distance(
            predicted_edges, target_edges
        ),
    }
    if not all(np.isfinite(tuple(result.values()))):
        raise ValueError("detail metrics must be finite")
    return result


def evaluate_detail_directory(dataset, render_dir: Path) -> dict[str, object]:
    samples = tuple(dataset[index] for index in range(len(dataset)))
    if not samples:
        raise ValueError("validation dataset is empty")
    output_names = tuple(
        Path(sample.image_name).with_suffix(".png").name for sample in samples
    )
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError("validation render names collide after PNG conversion")

    root = Path(render_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"validation render directory does not exist: {root}")
    expected_names = set(output_names)
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        raise FileNotFoundError(f"missing validation render: {root / missing[0]}")
    if extra:
        raise ValueError(
            f"validation render filenames mismatch; missing={missing}, extra={extra}"
        )

    image_reports: dict[str, dict[str, float]] = {}
    for sample, output_name in zip(samples, output_names):
        path = root / output_name
        try:
            with Image.open(path) as source:
                if source.mode != "RGB":
                    raise ValueError(f"validation render must be RGB: {path}")
                prediction = np.asarray(source, dtype=np.float64).copy() / 255.0
        except OSError as error:
            raise ValueError(f"cannot decode validation render: {path}") from error
        target = np.asarray(sample.image)
        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError("validation target must have shape (H, W, 3)")
        if prediction.shape != target.shape:
            raise ValueError(f"validation render resolution mismatch: {path}")
        image_reports[sample.image_name] = detail_metrics(
            prediction,
            target.astype(np.float64) / 255.0,
            np.asarray(sample.valid_mask),
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
        "symmetric_edge_distance_mean": float(
            np.mean([item["symmetric_edge_distance"] for item in values])
        ),
        "images": image_reports,
    }
