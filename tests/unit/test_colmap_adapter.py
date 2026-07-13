import sys
from types import SimpleNamespace

import numpy as np
import pytest

import bts_nvs.data.colmap as colmap_module
from bts_nvs.data.colmap import read_colmap_model


class _Rigid:
    def matrix(self):
        return np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0]],
            dtype=np.float64,
        )


def _sparse_dir(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes(b"fixture")
    return sparse


def test_pycolmap_adapter_converts_objects_to_internal_records(tmp_path, monkeypatch):
    sparse = _sparse_dir(tmp_path)
    camera = SimpleNamespace(
        model=SimpleNamespace(name="SIMPLE_RADIAL"),
        params=np.array([926.0, 660.0, 494.5, -0.114]),
        width=1320,
        height=989,
    )
    image = SimpleNamespace(
        has_pose=True,
        name="image.JPG",
        camera_id=1,
        cam_from_world=lambda: _Rigid(),
    )
    point = SimpleNamespace(
        xyz=np.array([1.0, 2.0, 3.0]),
        color=np.array([10, 20, 30], dtype=np.uint8),
        error=0.25,
        track=SimpleNamespace(
            elements=[SimpleNamespace(image_id=7), SimpleNamespace(image_id=8)]
        ),
    )
    reconstruction = SimpleNamespace(
        cameras={1: camera},
        images={7: image},
        points3D={100: point},
    )
    fake_pycolmap = SimpleNamespace(Reconstruction=lambda path: reconstruction)
    monkeypatch.setattr(colmap_module, "version", lambda package: "4.1.0")
    monkeypatch.setitem(sys.modules, "pycolmap", fake_pycolmap)

    model = read_colmap_model(sparse)

    assert model.cameras[1].distortion.coefficients == (-0.114,)
    np.testing.assert_allclose(model.images[7].world_to_camera[:3, 3], [1, 2, 3])
    assert model.points3d[100].image_ids == (7, 8)
    assert model.points3d[100].rgb.dtype == np.uint8


def test_pycolmap_adapter_rejects_wrong_installed_version(tmp_path, monkeypatch):
    sparse = _sparse_dir(tmp_path)
    monkeypatch.setattr(colmap_module, "version", lambda package: "4.0.4")

    with pytest.raises(RuntimeError, match="4.1.0"):
        read_colmap_model(sparse)


def test_pycolmap_adapter_requires_all_binary_files(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="cameras.bin"):
        read_colmap_model(sparse)
