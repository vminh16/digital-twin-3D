from __future__ import annotations

from pathlib import Path

import pytest

from bts_nvs.data.holdout import build_pose_holdout
from bts_nvs.data.manifest import load_scene_manifest


@pytest.mark.real_data
def test_hcm0181_holdout_is_deterministic_and_leakage_controlled() -> None:
    root = Path(__file__).resolve().parents[2]
    scene_root = root / "data" / "phase1" / "public_set" / "HCM0181"
    manifest_json = root / "runs" / "manifests" / "HCM0181" / "manifest.json"
    if not scene_root.is_dir() or not manifest_json.is_file():
        pytest.skip("HCM0181 scene or manifest is unavailable")

    manifest = load_scene_manifest(manifest_json, scene_root)
    first = build_pose_holdout(manifest)
    second = build_pose_holdout(manifest)
    train = set(first.train_image_names)
    validation = set(first.validation_image_names)
    guard = set(first.guard_image_names)

    assert first == second
    assert len(train) >= 120
    assert len(validation) >= 8
    assert train.isdisjoint(validation)
    assert train.isdisjoint(guard)
    assert validation.isdisjoint(guard)
    assert train | validation | guard == set(manifest.train_image_names)
    assert not (train | validation | guard) & set(manifest.test_image_names)
