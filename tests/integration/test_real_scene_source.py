from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from bts_nvs.data.manifest import load_scene_source_data


@pytest.mark.real_data
def test_public_scene_source_smoke():
    try:
        installed_version = version("pycolmap")
    except PackageNotFoundError:
        pytest.skip("pycolmap is not installed")
    if installed_version != "4.1.0":
        pytest.skip(f"requires pycolmap==4.1.0, found {installed_version}")

    repo_root = Path(__file__).resolve().parents[2]
    scene_root = repo_root / "data" / "phase1" / "public_set" / "HCM0181"
    if not scene_root.is_dir():
        pytest.skip("local public dataset is not available")

    source = load_scene_source_data(scene_root)

    assert len(source.train_images) == 240
    assert len(source.test_poses) == 60
    assert len(source.test_distortions) == 60
    assert source.sparse_points
