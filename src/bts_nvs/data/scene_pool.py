from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


CANONICAL_SCENES = (
    "hcm0031",
    "hcm0034",
    "HCM0181",
    "HCM0193",
    "HCM0204",
    "HCM0249",
    "HCM0254",
    "HCM0276",
    "HCM0421",
    "HCM0539",
    "HCM0540",
    "HCM0644",
    "HCM0674",
    "HCM1439",
    "HNI0131",
    "HNI0265",
    "HNI0366",
    "HNI0437",
)


def validate_scene_selection(
    scene_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if scene_ids is None:
        return CANONICAL_SCENES
    selected = tuple(scene_ids)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(scene_id not in CANONICAL_SCENES for scene_id in selected)
    ):
        raise ValueError(
            "scene selection must be non-empty, unique, case-sensitive, and canonical"
        )
    return selected


def _directory_names(root: Path) -> set[str]:
    if not Path(root).is_dir():
        raise FileNotFoundError(f"required directory does not exist: {root}")
    return {path.name for path in Path(root).iterdir() if path.is_dir()}


def validate_scene_pool(
    scenes_root: Path,
    manifests_root: Path,
) -> tuple[str, ...]:
    expected = set(CANONICAL_SCENES)
    scene_names = _directory_names(Path(scenes_root))
    manifest_names = _directory_names(Path(manifests_root))
    if scene_names != expected:
        raise ValueError("scene root does not match the canonical 18-scene pool")
    if manifest_names != expected:
        raise ValueError("manifest root does not match the canonical 18-scene pool")

    for scene_id in CANONICAL_SCENES:
        manifest_path = Path(manifests_root) / scene_id / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("scene_id") != scene_id:
            raise ValueError(f"manifest scene identity mismatch: {manifest_path}")
        arrays_name = payload.get("arrays_file")
        if (
            not isinstance(arrays_name, str)
            or Path(arrays_name).is_absolute()
            or Path(arrays_name).name != arrays_name
        ):
            raise ValueError(f"invalid manifest arrays_file: {manifest_path}")
        arrays_path = manifest_path.parent / arrays_name
        if not arrays_path.is_file():
            raise FileNotFoundError(f"manifest arrays do not exist: {arrays_path}")
    return CANONICAL_SCENES
