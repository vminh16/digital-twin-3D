from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bts_nvs.data.holdout import (
    build_pose_holdout,
    load_holdout_split,
    save_holdout_split,
)
from bts_nvs.data.manifest import (
    build_scene_manifest,
    load_scene_manifest,
    save_scene_manifest,
)


def prepare_scene_artifacts(
    scene_root: Path,
    artifact_dir: Path,
) -> tuple[Path, Path]:
    scene = Path(scene_root)
    artifact = Path(artifact_dir)
    manifest_path = artifact / "manifest.json"
    holdout_path = artifact / "holdout.json"

    if manifest_path.is_file():
        manifest = load_scene_manifest(manifest_path, scene)
    else:
        manifest = build_scene_manifest(scene)
        save_scene_manifest(manifest, artifact)
    if manifest.scene_id != scene.name:
        raise ValueError("manifest scene_id does not match scene directory")

    if holdout_path.is_file():
        load_holdout_split(holdout_path, manifest)
    else:
        save_holdout_split(build_pose_holdout(manifest), holdout_path)
    return manifest_path, holdout_path


def prepare_all_artifacts(
    scenes_root: Path,
    manifests_root: Path,
    *,
    expected_scenes: int | None = None,
    require_expected: bool = False,
) -> tuple[str, ...]:
    root = Path(scenes_root)
    scenes = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "train" / "images").is_dir()
        ),
        key=lambda path: path.name,
    )
    if expected_scenes is not None and expected_scenes <= 0:
        raise ValueError("expected_scenes must be positive")
    if require_expected and len(scenes) != expected_scenes:
        raise ValueError(f"expected {expected_scenes} scenes, found {len(scenes)}")

    for scene in scenes:
        prepare_scene_artifacts(scene, Path(manifests_root) / scene.name)
    return tuple(scene.name for scene in scenes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Phase 4 scene artifacts")
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--expected_scenes", type=int, default=13)
    parser.add_argument("--require_expected", action="store_true")
    args = parser.parse_args(argv)
    scenes = prepare_all_artifacts(
        args.scenes_root,
        args.manifests_root,
        expected_scenes=args.expected_scenes,
        require_expected=args.require_expected,
    )
    print(f"Prepared Phase 4 artifacts for {len(scenes)} scene(s): {', '.join(scenes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
