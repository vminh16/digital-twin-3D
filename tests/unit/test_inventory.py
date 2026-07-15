from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.inventory import (
    COMPACT_MODEL_BYTES_PER_GAUSSIAN,
    DISK_HEADROOM_BYTES,
    FULL_CHECKPOINT_BYTES_PER_GAUSSIAN,
    GAUSSIAN_GROWTH_FACTOR,
    HOST_RAM_HEADROOM_BYTES,
    Phase4InventoryReport,
    SceneCapacityEstimate,
    SceneInventory,
    aggregate_capacity,
    audit_phase4_inventory,
    build_scene_inventory,
    estimate_scene_capacity,
    save_phase4_inventory_report,
    select_scene_cohort,
)
from bts_nvs.data.manifest import SceneManifest
from bts_nvs.data.run_inventory import check_local_feasibility


def _manifest(
    scene_id: str = "scene_a",
    *,
    centers: tuple[float, ...] = (0.0, 2.0),
    sparse_count: int = 4,
) -> SceneManifest:
    count = len(centers)
    intrinsics = CameraIntrinsics(100, 80, 70.0, 70.0, 50.0, 40.0)
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    c2w[:, 0, 3] = centers
    w2c = np.linalg.inv(c2w)
    return SceneManifest(
        schema_version=1,
        scene_id=scene_id,
        train_image_paths=tuple(f"train/images/{i}.JPG" for i in range(count)),
        train_image_names=tuple(f"{i}.JPG" for i in range(count)),
        train_world_to_camera=w2c,
        train_camera_to_world=c2w,
        train_intrinsics=(intrinsics,) * count,
        train_distortion=(CameraDistortion("SIMPLE_RADIAL", (-0.1,)),) * count,
        test_image_names=("test.JPG",),
        test_output_names=("test.png",),
        test_world_to_camera=np.eye(4, dtype=np.float64)[None],
        test_intrinsics=(intrinsics,),
        test_distortion=(CameraDistortion("SIMPLE_RADIAL", (-0.2,)),),
        sparse_points=np.zeros((sparse_count, 3), dtype=np.float64),
        sparse_colors=np.zeros((sparse_count, 3), dtype=np.uint8),
        normalization_transform=np.eye(4, dtype=np.float64),
        inverse_normalization_transform=np.eye(4, dtype=np.float64),
    )


def test_build_scene_inventory_uses_train_geometry_only() -> None:
    inventory = build_scene_inventory(_manifest())

    assert inventory.scene_id == "scene_a"
    assert inventory.train_image_count == 2
    assert inventory.test_pose_count == 1
    assert inventory.sparse_point_count == 4
    assert inventory.trajectory_nn_p90 == pytest.approx(2.0)
    assert inventory.distortion_abs_max == pytest.approx(0.1)
    assert inventory.native_widths == (100, 100)
    assert inventory.native_heights == (80, 80)


def test_inventory_rejects_degenerate_trajectory_and_empty_points() -> None:
    with pytest.raises(ValueError, match="trajectory"):
        build_scene_inventory(_manifest(centers=(1.0, 1.0)))
    with pytest.raises(ValueError, match="distinct"):
        build_scene_inventory(_manifest(centers=(0.0, 0.0, 2.0)))
    with pytest.raises(ValueError, match="sparse"):
        build_scene_inventory(_manifest(sparse_count=0))


def test_inventory_keeps_native_dimensions_per_train_camera() -> None:
    manifest = _manifest(centers=(0.0, 1.0, 3.0))
    wide = CameraIntrinsics(120, 90, 80.0, 80.0, 60.0, 45.0)
    manifest = replace(
        manifest,
        train_intrinsics=(manifest.train_intrinsics[0], wide, manifest.train_intrinsics[0]),
    )

    inventory = build_scene_inventory(manifest)

    assert inventory.native_widths == (100, 120, 100)
    assert inventory.native_heights == (80, 90, 80)


def test_estimate_scene_capacity_uses_declared_upper_bound() -> None:
    manifest = _manifest(sparse_count=4)
    inventory = build_scene_inventory(manifest)

    capacity = estimate_scene_capacity(manifest, inventory)
    peak = 4 * GAUSSIAN_GROWTH_FACTOR
    assert capacity.cache_bytes == 2 * 100 * 80 * 4
    assert capacity.output_raw_bytes == 100 * 80 * 3
    assert capacity.estimated_peak_gaussians == peak
    assert capacity.full_checkpoint_bytes == peak * FULL_CHECKPOINT_BYTES_PER_GAUSSIAN
    assert capacity.compact_model_bytes == peak * COMPACT_MODEL_BYTES_PER_GAUSSIAN


def _inventories() -> tuple[SceneInventory, ...]:
    return tuple(
        SceneInventory(
            scene_id=f"scene_{index:02d}",
            train_image_count=150 + index * 5,
            test_pose_count=40 + index,
            sparse_point_count=1_000 * (index + 1),
            trajectory_nn_p90=0.1 + index * 0.05,
            distortion_abs_max=(index % 4) * 0.03,
            native_widths=(100 + index,),
            native_heights=(80 + index,),
        )
        for index in range(13)
    )


def _write_train_images(scene_root: Path, names: tuple[str, ...]) -> None:
    image_dir = scene_root / "train" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (image_dir / name).touch()


def test_cohort_is_deterministic_and_ignores_test_metadata() -> None:
    inventories = _inventories()
    expected = select_scene_cohort(inventories)
    changed_test_metadata = tuple(
        SceneInventory(
            **{
                **item.__dict__,
                "test_pose_count": 10_000 - index,
                "native_widths": (9_999,),
                "native_heights": (8_888,),
            }
        )
        for index, item in enumerate(reversed(inventories))
    )

    assert select_scene_cohort(changed_test_metadata) == expected
    assert len(expected.calibration_scene_ids) == 3
    assert len(expected.confirmation_scene_ids) == 2
    assert len(expected.production_scene_ids) == 8
    assert set(
        expected.calibration_scene_ids
        + expected.confirmation_scene_ids
        + expected.production_scene_ids
    ) == {item.scene_id for item in inventories}


def test_cohort_requires_complete_unique_scene_set() -> None:
    with pytest.raises(ValueError, match="exactly 13"):
        select_scene_cohort(_inventories()[:-1])
    with pytest.raises(ValueError, match="unique"):
        select_scene_cohort(_inventories()[:-1] + (_inventories()[0],))


def test_aggregate_capacity_keeps_two_full_checkpoints() -> None:
    capacities = (
        SceneCapacityEstimate("a", 100, 10, 1, 1_000, 100),
        SceneCapacityEstimate("b", 200, 20, 1, 2_000, 200),
    )

    host, disk = aggregate_capacity(capacities)
    assert host == 200 + HOST_RAM_HEADROOM_BYTES
    assert disk == 100 + 200 + 10 + 20 + 2 * 2_000 + DISK_HEADROOM_BYTES


def test_audit_reports_missing_manifest_without_guessing_cohort(
    tmp_path, monkeypatch
) -> None:
    scenes_root = tmp_path / "scenes"
    manifests_root = tmp_path / "manifests"
    for scene_id in ("a", "b"):
        _write_train_images(
            scenes_root / scene_id,
            tuple(f"{i}.JPG" for i in range(150)),
        )
    (manifests_root / "a").mkdir(parents=True)
    (manifests_root / "a" / "manifest.json").touch()
    manifest = _manifest("a", centers=tuple(float(i) for i in range(150)))
    monkeypatch.setattr(
        "bts_nvs.data.inventory.load_scene_manifest",
        lambda path, scene_root: manifest,
    )

    report = audit_phase4_inventory(
        scenes_root, manifests_root, expected_scene_count=2
    )

    assert report.status == "incomplete_cohort"
    assert report.cohort is None
    assert [(issue.scene_id, issue.code) for issue in report.issues] == [
        ("b", "missing_manifest")
    ]


def test_ready_audit_and_json_are_deterministic(tmp_path, monkeypatch) -> None:
    scenes_root = tmp_path / "scenes"
    manifests_root = tmp_path / "manifests"
    manifests = {}
    for index in range(5):
        scene_id = f"scene_{index}"
        _write_train_images(
            scenes_root / scene_id,
            tuple(f"{i}.JPG" for i in range(150)),
        )
        manifest_path = manifests_root / scene_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.touch()
        manifests[scene_id] = _manifest(
            scene_id,
            centers=tuple(float(i) * (index + 1) for i in range(150)),
            sparse_count=index + 1,
        )

    monkeypatch.setattr(
        "bts_nvs.data.inventory.load_scene_manifest",
        lambda path, scene_root: manifests[scene_root.name],
    )
    report = audit_phase4_inventory(
        scenes_root, manifests_root, expected_scene_count=5
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    save_phase4_inventory_report(report, first)
    save_phase4_inventory_report(report, second)

    assert report.status == "ready"
    assert report.cohort is not None
    assert report.issues == ()
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()
    assert str(tmp_path).encode() not in first.read_bytes()


def test_audit_rejects_manifest_with_stale_physical_split(tmp_path, monkeypatch) -> None:
    scenes_root = tmp_path / "scenes"
    manifests_root = tmp_path / "manifests"
    scene_root = scenes_root / "scene_a"
    _write_train_images(scene_root, tuple(f"physical_{i}.JPG" for i in range(150)))
    manifest_path = manifests_root / "scene_a" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.touch()
    manifest = _manifest("scene_a", centers=tuple(float(i) for i in range(150)))
    monkeypatch.setattr(
        "bts_nvs.data.inventory.load_scene_manifest",
        lambda path, scene_root: manifest,
    )

    report = audit_phase4_inventory(
        scenes_root, manifests_root, expected_scene_count=1
    )

    assert report.status == "invalid"
    assert any(issue.code == "invalid_scene" for issue in report.issues)


def test_local_feasibility_reports_ram_and_disk_shortfall() -> None:
    inventory_report = Phase4InventoryReport(
        schema_version=1,
        expected_scene_count=13,
        status="incomplete_cohort",
        scenes=(),
        capacities=(),
        cohort=None,
        issues=(),
        required_host_ram_bytes=100,
        required_artifact_disk_bytes=200,
    )

    assert check_local_feasibility(
        inventory_report,
        available_host_ram_bytes=99,
        available_artifact_disk_bytes=199,
    ) == ("insufficient_host_ram", "insufficient_artifact_disk")
    assert check_local_feasibility(
        inventory_report,
        available_host_ram_bytes=100,
        available_artifact_disk_bytes=200,
    ) == ()


def test_inventory_cli_can_run_directly_like_training_cli() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "src/bts_nvs/data/run_inventory.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
