from __future__ import annotations

from pathlib import Path

import pytest

from bts_nvs.data.inventory import audit_phase4_inventory


@pytest.mark.real_data
def test_phase4_inventory_reads_available_public_scenes() -> None:
    root = Path(__file__).resolve().parents[2]
    scenes_root = root / "data" / "phase1" / "public_set"
    manifests_root = root / "runs" / "manifests"
    if not scenes_root.is_dir() or not manifests_root.is_dir():
        pytest.skip("local public scenes or manifests are unavailable")

    physical_count = sum(
        1
        for path in scenes_root.iterdir()
        if path.is_dir() and (path / "train" / "images").is_dir()
    )
    report = audit_phase4_inventory(scenes_root, manifests_root)

    assert len(report.scenes) <= physical_count
    assert all(scene.train_image_count >= 150 for scene in report.scenes)
    if physical_count != 13:
        assert report.status == "incomplete_cohort"
        assert report.cohort is None
