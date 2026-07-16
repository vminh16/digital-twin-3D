from pathlib import Path
from types import SimpleNamespace

import pytest

import bts_nvs.data.prepare_phase4_artifacts as prepare


def _scene(root: Path, name: str) -> Path:
    scene = root / name
    (scene / "train" / "images").mkdir(parents=True)
    return scene


def test_prepare_scene_builds_missing_manifest_and_holdout(tmp_path, monkeypatch):
    scene = _scene(tmp_path / "scenes", "HCM0181")
    artifact = tmp_path / "artifacts" / "HCM0181"
    manifest = SimpleNamespace(scene_id="HCM0181")
    split = SimpleNamespace(scene_id="HCM0181")
    calls: list[str] = []

    monkeypatch.setattr(prepare, "build_scene_manifest", lambda path: manifest)
    monkeypatch.setattr(
        prepare,
        "save_scene_manifest",
        lambda value, path: (path.mkdir(parents=True), (path / "manifest.json").write_text("{}")),
    )
    monkeypatch.setattr(prepare, "build_pose_holdout", lambda value: split)
    monkeypatch.setattr(
        prepare,
        "save_holdout_split",
        lambda value, path: (calls.append("holdout"), path.write_text("{}")),
    )

    manifest_path, holdout_path = prepare.prepare_scene_artifacts(scene, artifact)

    assert manifest_path == artifact / "manifest.json"
    assert holdout_path == artifact / "holdout.json"
    assert calls == ["holdout"]


def test_prepare_scene_validates_existing_artifacts(tmp_path, monkeypatch):
    scene = _scene(tmp_path / "scenes", "HCM0181")
    artifact = tmp_path / "artifacts" / "HCM0181"
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text("{}")
    (artifact / "holdout.json").write_text("{}")
    manifest = SimpleNamespace(scene_id="HCM0181")
    loaded: list[str] = []

    monkeypatch.setattr(
        prepare,
        "load_scene_manifest",
        lambda path, root: (loaded.append("manifest"), manifest)[1],
    )
    monkeypatch.setattr(
        prepare,
        "load_holdout_split",
        lambda path, value: loaded.append("holdout"),
    )
    monkeypatch.setattr(
        prepare,
        "build_scene_manifest",
        lambda path: pytest.fail("existing manifest must not be rebuilt"),
    )

    prepare.prepare_scene_artifacts(scene, artifact)

    assert loaded == ["manifest", "holdout"]


def test_batch_is_sorted_and_strict_count_fails_before_generation(tmp_path, monkeypatch):
    scenes = tmp_path / "scenes"
    _scene(scenes, "scene_b")
    _scene(scenes, "scene_a")
    prepared: list[str] = []
    monkeypatch.setattr(
        prepare,
        "prepare_scene_artifacts",
        lambda scene, artifact: prepared.append(scene.name),
    )

    with pytest.raises(ValueError, match="expected 13"):
        prepare.prepare_all_artifacts(
            scenes,
            tmp_path / "artifacts",
            expected_scenes=13,
            require_expected=True,
        )
    assert prepared == []

    result = prepare.prepare_all_artifacts(scenes, tmp_path / "artifacts")
    assert prepared == ["scene_a", "scene_b"]
    assert result == ("scene_a", "scene_b")
