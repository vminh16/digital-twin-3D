from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bts_nvs.data.holdout import (
    build_research_holdout,
    load_holdout_split,
    save_holdout_split,
)
from bts_nvs.data.manifest import load_scene_manifest


RESEARCH_HOLDOUT_NAME = "holdout_research_v3.json"


def prepare_research_holdout(scene_root: Path, manifest_dir: Path) -> Path:
    scene = Path(scene_root)
    artifact = Path(manifest_dir)
    manifest = load_scene_manifest(artifact / "manifest.json", scene)
    if manifest.scene_id != scene.name:
        raise ValueError("manifest scene_id does not match scene directory")
    output = artifact / RESEARCH_HOLDOUT_NAME
    if output.is_file():
        load_holdout_split(output, manifest)
    else:
        save_holdout_split(build_research_holdout(manifest), output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare v3 targeted research holdout")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = prepare_research_holdout(args.scene_root, args.manifest_dir)
    print(f"Research holdout ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
