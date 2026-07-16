from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np

from .manifest import SceneManifest, load_scene_manifest
from .validation import DataContractError


SCHEMA_VERSION = 1
EXPECTED_SCENE_COUNT = 18
MIN_TRAIN_IMAGES = 100
CALIBRATION_SCENE_IDS = (
    "hcm0031",
    "HCM0181",
    "HCM0421",
    "HCM1439",
    "HNI0131",
    "HNI0265",
)
MAX_ESTIMATED_GAUSSIANS = 10_000_000
GAUSSIAN_GROWTH_FACTOR = 30
FULL_CHECKPOINT_BYTES_PER_GAUSSIAN = 768
COMPACT_MODEL_BYTES_PER_GAUSSIAN = 236
HOST_RAM_HEADROOM_BYTES = 4 * 1024**3
DISK_HEADROOM_BYTES = 10 * 1024**3
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SceneInventory:
    scene_id: str
    train_image_count: int
    test_pose_count: int
    sparse_point_count: int
    trajectory_nn_p90: float
    distortion_abs_max: float
    native_widths: tuple[int, ...]
    native_heights: tuple[int, ...]


@dataclass(frozen=True)
class SceneCapacityEstimate:
    scene_id: str
    cache_bytes: int
    output_raw_bytes: int
    estimated_peak_gaussians: int
    full_checkpoint_bytes: int
    compact_model_bytes: int


@dataclass(frozen=True)
class CohortAssignment:
    calibration_scene_ids: tuple[str, ...]
    confirmation_scene_ids: tuple[str, ...]
    production_scene_ids: tuple[str, ...]


@dataclass(frozen=True, order=True)
class InventoryIssue:
    scene_id: str
    code: str
    detail: str


@dataclass(frozen=True)
class Phase4InventoryReport:
    schema_version: int
    expected_scene_count: int
    status: Literal["ready", "incomplete_cohort", "invalid"]
    scenes: tuple[SceneInventory, ...]
    capacities: tuple[SceneCapacityEstimate, ...]
    cohort: CohortAssignment | None
    issues: tuple[InventoryIssue, ...]
    required_host_ram_bytes: int
    required_artifact_disk_bytes: int


def _trajectory_nn_p90(camera_to_world: np.ndarray) -> float:
    centers = np.asarray(camera_to_world, dtype=np.float64)[:, :3, 3]
    if len(centers) < 2 or not np.all(np.isfinite(centers)):
        raise ValueError("train trajectory requires at least two finite camera centers")
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    if np.any(nearest <= 1e-12):
        raise ValueError("train trajectory requires distinct camera centers")
    value = float(np.percentile(nearest, 90, method="linear"))
    if not np.isfinite(value) or value <= 1e-12:
        raise ValueError("train trajectory is degenerate")
    return value


def build_scene_inventory(manifest: SceneManifest) -> SceneInventory:
    sparse_count = len(manifest.sparse_points)
    if sparse_count == 0:
        raise ValueError("scene requires non-empty sparse points")

    distortion = max(
        (
            abs(coefficient)
            for camera in manifest.train_distortion
            for coefficient in camera.coefficients
        ),
        default=0.0,
    )
    return SceneInventory(
        scene_id=manifest.scene_id,
        train_image_count=len(manifest.train_image_names),
        test_pose_count=len(manifest.test_image_names),
        sparse_point_count=sparse_count,
        trajectory_nn_p90=_trajectory_nn_p90(manifest.train_camera_to_world),
        distortion_abs_max=float(distortion),
        native_widths=tuple(camera.width for camera in manifest.train_intrinsics),
        native_heights=tuple(camera.height for camera in manifest.train_intrinsics),
    )


def estimate_scene_capacity(
    manifest: SceneManifest,
    inventory: SceneInventory,
) -> SceneCapacityEstimate:
    if manifest.scene_id != inventory.scene_id:
        raise ValueError("manifest and inventory scene IDs differ")
    peak = min(
        MAX_ESTIMATED_GAUSSIANS,
        max(1, inventory.sparse_point_count) * GAUSSIAN_GROWTH_FACTOR,
    )
    cache_bytes = sum(
        camera.width * camera.height * 4 for camera in manifest.train_intrinsics
    )
    output_raw_bytes = sum(
        camera.width * camera.height * 3 for camera in manifest.test_intrinsics
    )
    return SceneCapacityEstimate(
        scene_id=manifest.scene_id,
        cache_bytes=cache_bytes,
        output_raw_bytes=output_raw_bytes,
        estimated_peak_gaussians=peak,
        full_checkpoint_bytes=peak * FULL_CHECKPOINT_BYTES_PER_GAUSSIAN,
        compact_model_bytes=peak * COMPACT_MODEL_BYTES_PER_GAUSSIAN,
    )


def _cohort_features(inventories: tuple[SceneInventory, ...]) -> np.ndarray:
    values = np.asarray(
        [
            [
                np.log(item.train_image_count),
                np.log(item.sparse_point_count / item.train_image_count),
                item.trajectory_nn_p90,
                item.distortion_abs_max,
            ]
            for item in inventories
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("cohort features must be finite and positive where logged")
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, (25, 75), axis=0, method="linear")
    spread = q75 - q25
    spread[spread == 0.0] = 1.0
    return (values - median) / spread


def select_scene_cohort(
    inventories: tuple[SceneInventory, ...] | list[SceneInventory],
    *,
    expected_scene_count: int = EXPECTED_SCENE_COUNT,
) -> CohortAssignment:
    ordered = tuple(sorted(inventories, key=lambda item: item.scene_id))
    if len(ordered) != expected_scene_count:
        raise ValueError(f"cohort requires exactly {expected_scene_count} scenes")
    if len({item.scene_id for item in ordered}) != len(ordered):
        raise ValueError("cohort scene IDs must be unique")
    if len(ordered) < 5:
        raise ValueError("cohort requires at least five unique scenes")

    features = _cohort_features(ordered)
    distances = np.abs(features[:, None] - features[None, :]).sum(axis=2)
    best = min(
        combinations(range(len(ordered)), 3),
        key=lambda indices: (
            float(distances[:, indices].min(axis=1).sum()),
            tuple(ordered[index].scene_id for index in indices),
        ),
    )
    calibration = tuple(ordered[index].scene_id for index in best)
    remaining = [index for index in range(len(ordered)) if index not in best]
    ranked = sorted(
        remaining,
        key=lambda index: (
            -float(distances[index, best].min()),
            ordered[index].scene_id,
        ),
    )
    confirmation = tuple(ordered[index].scene_id for index in ranked[:2])
    selected = set(calibration + confirmation)
    production = tuple(
        item.scene_id for item in ordered if item.scene_id not in selected
    )
    return CohortAssignment(calibration, confirmation, production)


def aggregate_capacity(
    capacities: tuple[SceneCapacityEstimate, ...] | list[SceneCapacityEstimate],
) -> tuple[int, int]:
    if not capacities:
        return HOST_RAM_HEADROOM_BYTES, DISK_HEADROOM_BYTES
    host = max(item.cache_bytes for item in capacities) + HOST_RAM_HEADROOM_BYTES
    disk = (
        sum(item.compact_model_bytes + item.output_raw_bytes for item in capacities)
        + 2 * max(item.full_checkpoint_bytes for item in capacities)
        + DISK_HEADROOM_BYTES
    )
    return host, disk


def _safe_error_detail(error: BaseException, roots: tuple[Path, ...]) -> str:
    detail = str(error)
    labels = ("<scenes_root>", "<manifests_root>")
    for root, label in zip(roots, labels):
        for value in (str(root), str(root.resolve())):
            detail = detail.replace(value, label)
    return detail


def audit_phase4_inventory(
    scenes_root: Path,
    manifests_root: Path,
    *,
    expected_scene_count: int = EXPECTED_SCENE_COUNT,
) -> Phase4InventoryReport:
    scenes_root = Path(scenes_root)
    manifests_root = Path(manifests_root)
    if expected_scene_count <= 0:
        raise ValueError("expected_scene_count must be positive")
    if not scenes_root.is_dir():
        raise DataContractError(f"missing scenes root: {scenes_root}")

    scene_dirs = sorted(
        (
            path
            for path in scenes_root.iterdir()
            if path.is_dir() and (path / "train" / "images").is_dir()
        ),
        key=lambda path: path.name,
    )
    issues: list[InventoryIssue] = []
    if len(scene_dirs) != expected_scene_count:
        issues.append(
            InventoryIssue(
                "",
                "scene_count",
                f"expected {expected_scene_count} scenes, found {len(scene_dirs)}",
            )
        )

    inventories: list[SceneInventory] = []
    capacities: list[SceneCapacityEstimate] = []
    roots = (scenes_root, manifests_root)
    for scene_dir in scene_dirs:
        physical_names = tuple(
            sorted(
                path.name
                for path in (scene_dir / "train" / "images").iterdir()
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
            )
        )
        if len(physical_names) < MIN_TRAIN_IMAGES:
            issues.append(
                InventoryIssue(
                    scene_dir.name,
                    "insufficient_train_images",
                    f"requires at least {MIN_TRAIN_IMAGES}, found {len(physical_names)}",
                )
            )
        manifest_path = manifests_root / scene_dir.name / "manifest.json"
        if not manifest_path.is_file():
            issues.append(
                InventoryIssue(scene_dir.name, "missing_manifest", "manifest.json is missing")
            )
            continue
        try:
            manifest = load_scene_manifest(manifest_path, scene_dir)
            if manifest.scene_id != scene_dir.name:
                raise DataContractError(
                    f"manifest scene_id {manifest.scene_id!r} does not match directory"
                )
            if set(manifest.train_image_names) != set(physical_names):
                raise DataContractError(
                    "manifest train split does not match physical train images"
                )
            inventory = build_scene_inventory(manifest)
            inventories.append(inventory)
            capacities.append(estimate_scene_capacity(manifest, inventory))
        except (DataContractError, OSError, ValueError) as error:
            issues.append(
                InventoryIssue(
                    scene_dir.name,
                    "invalid_scene",
                    _safe_error_detail(error, roots),
                )
            )

    inventories.sort(key=lambda item: item.scene_id)
    capacities.sort(key=lambda item: item.scene_id)
    issues.sort()
    cohort = None
    blocking_codes = {"insufficient_train_images", "invalid_scene"}
    if any(issue.code in blocking_codes for issue in issues):
        status: Literal["ready", "incomplete_cohort", "invalid"] = "invalid"
    elif issues:
        status = "incomplete_cohort"
    else:
        try:
            scene_ids = {item.scene_id for item in inventories}
            if (
                expected_scene_count == EXPECTED_SCENE_COUNT
                and set(CALIBRATION_SCENE_IDS) <= scene_ids
            ):
                cohort = CohortAssignment(
                    CALIBRATION_SCENE_IDS,
                    (),
                    tuple(sorted(scene_ids - set(CALIBRATION_SCENE_IDS))),
                )
            else:
                cohort = select_scene_cohort(
                    inventories, expected_scene_count=expected_scene_count
                )
            status = "ready"
        except ValueError as error:
            issues.append(InventoryIssue("", "invalid_cohort", str(error)))
            issues.sort()
            status = "invalid"

    required_ram, required_disk = aggregate_capacity(capacities)
    return Phase4InventoryReport(
        schema_version=SCHEMA_VERSION,
        expected_scene_count=expected_scene_count,
        status=status,
        scenes=tuple(inventories),
        capacities=tuple(capacities),
        cohort=cohort,
        issues=tuple(issues),
        required_host_ram_bytes=required_ram,
        required_artifact_disk_bytes=required_disk,
    )


def save_phase4_inventory_report(
    report: Phase4InventoryReport,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["capacity_assumptions"] = {
        "compact_model_bytes_per_gaussian": COMPACT_MODEL_BYTES_PER_GAUSSIAN,
        "disk_headroom_bytes": DISK_HEADROOM_BYTES,
        "full_checkpoint_bytes_per_gaussian": FULL_CHECKPOINT_BYTES_PER_GAUSSIAN,
        "gaussian_growth_factor": GAUSSIAN_GROWTH_FACTOR,
        "host_ram_headroom_bytes": HOST_RAM_HEADROOM_BYTES,
        "max_estimated_gaussians": MAX_ESTIMATED_GAUSSIANS,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
