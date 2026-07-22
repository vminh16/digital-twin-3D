from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


STRATA = ("easy", "medium", "hard")
FULL_FRAME_METRICS = ("psnr_db", "ssim", "lpips")
DETAIL_METRICS = (
    "hf_l1",
    "missing_edge",
    "spurious_edge",
    "symmetric_edge_distance",
)
RESOURCE_FIELDS = (
    "total_time_seconds",
    "max_vram_mb",
    "peak_gaussians",
    "final_num_gaussians",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _finite_number(value: object, field: str, *, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _images(report: Mapping[str, object], field: str) -> Mapping[str, object]:
    images = _mapping(report.get("images"), f"{field}.images")
    if not images or any(not isinstance(name, str) or not name for name in images):
        raise ValueError(f"{field} image names must be non-empty strings")
    return images


def _validate_image_count(
    report: Mapping[str, object], images: Mapping[str, object], field: str
) -> None:
    count = report.get("image_count")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(images):
        raise ValueError(f"{field}.image_count does not match images")


def _hash(value: str, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def local_score50(metrics: Mapping[str, object]) -> float:
    """Compute the locked local diagnostic score with PSNR capped at 50 dB."""
    psnr = _finite_number(metrics.get("psnr_db"), "psnr_db")
    ssim = _finite_number(metrics.get("ssim"), "ssim")
    lpips = _finite_number(metrics.get("lpips"), "lpips")
    return 40.0 * (1.0 - lpips) + 30.0 * ssim + 0.6 * min(max(psnr, 0.0), 50.0)


def _aggregate(
    image_reports: Mapping[str, Mapping[str, object]], names: Sequence[str]
) -> dict[str, float | int]:
    if not names:
        raise ValueError("each pose stratum must contain at least one image")
    result: dict[str, float | int] = {"image_count": len(names)}
    for metric in FULL_FRAME_METRICS + DETAIL_METRICS:
        result[metric] = sum(float(image_reports[name][metric]) for name in names) / len(
            names
        )
    result["score50"] = local_score50(result)
    return result


def build_experiment_report(
    *,
    scene_id: str,
    candidate_id: str,
    step: int,
    config_sha256: str,
    manifest_sha256: str,
    holdout_sha256: str,
    full_frame_report: Mapping[str, object],
    detail_report: Mapping[str, object],
    pose_strata_report: Mapping[str, object],
    resource_summary: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("scene_id must be a non-empty string")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    hashes = {
        "config_sha256": _hash(config_sha256, "config_sha256"),
        "manifest_sha256": _hash(manifest_sha256, "manifest_sha256"),
        "holdout_sha256": _hash(holdout_sha256, "holdout_sha256"),
    }

    full_images = _images(full_frame_report, "full_frame_report")
    detail_images = _images(detail_report, "detail_report")
    pose_images = _images(pose_strata_report, "pose_strata_report")
    names = set(full_images)
    if set(detail_images) != names or set(pose_images) != names:
        raise ValueError("input reports must contain identical image names")
    _validate_image_count(full_frame_report, full_images, "full_frame_report")
    _validate_image_count(detail_report, detail_images, "detail_report")
    _validate_image_count(pose_strata_report, pose_images, "pose_strata_report")
    if detail_report.get("scene_id") != scene_id or pose_strata_report.get("scene_id") != scene_id:
        raise ValueError("report scene_id does not match experiment scene_id")
    if pose_strata_report.get("holdout_manifest_sha256") != holdout_sha256:
        raise ValueError("pose strata holdout identity does not match holdout_sha256")

    combined: dict[str, dict[str, object]] = {}
    for name in sorted(names):
        full = _mapping(full_images[name], f"full_frame_report.images.{name}")
        detail = _mapping(detail_images[name], f"detail_report.images.{name}")
        pose = _mapping(pose_images[name], f"pose_strata_report.images.{name}")
        image: dict[str, object] = {}
        for metric in FULL_FRAME_METRICS:
            image[metric] = _finite_number(full.get(metric), f"{name}.{metric}")
        for metric in DETAIL_METRICS:
            image[metric] = _finite_number(
                detail.get(metric), f"{name}.{metric}", minimum=0.0
            )
        stratum = pose.get("stratum")
        if stratum not in STRATA:
            raise ValueError(f"{name}.stratum must be one of {STRATA}")
        nearest = pose.get("nearest_train_image_name")
        if not isinstance(nearest, str) or not nearest:
            raise ValueError(f"{name}.nearest_train_image_name must be non-empty")
        image["nearest_train_image_name"] = nearest
        for field in ("pose_distance", "center_distance", "rotation_angle_deg"):
            image[field] = _finite_number(pose.get(field), f"{name}.{field}", minimum=0.0)
        image["stratum"] = stratum
        image["score50"] = local_score50(image)
        combined[name] = image

    resources: dict[str, float | int] = {}
    for field in RESOURCE_FIELDS:
        value = _finite_number(resource_summary.get(field), field, minimum=0.0)
        resources[field] = int(value) if field.endswith("gaussians") else value

    ordered_names = tuple(sorted(combined))
    strata = {
        stratum: _aggregate(
            combined,
            tuple(name for name in ordered_names if combined[name]["stratum"] == stratum),
        )
        for stratum in STRATA
    }
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "candidate_id": candidate_id,
        "step": step,
        **hashes,
        "image_count": len(combined),
        "overall": _aggregate(combined, ordered_names),
        "strata": strata,
        "resources": resources,
        "images": combined,
    }


def save_experiment_report(report: Mapping[str, object], path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
