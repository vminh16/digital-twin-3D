import numpy as np

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.colmap import ColmapCameraRecord, ColmapImageRecord, ColmapPointRecord
from bts_nvs.data.diagnostics import SceneDiagnostics, build_scene_diagnostics
from bts_nvs.data.manifest import SceneSourceData


def test_scene_diagnostics_use_train_supported_manifest_points(monkeypatch, tmp_path):
    identity = np.eye(4)
    intrinsics = CameraIntrinsics(8, 6, 5.0, 5.0, 4.0, 3.0)
    distortion = CameraDistortion("PINHOLE", ())
    camera = ColmapCameraRecord(1, intrinsics, distortion)
    train_a = ColmapImageRecord(10, "a.png", 1, identity)
    train_b = ColmapImageRecord(11, "b.png", 1, identity)
    source = SceneSourceData(
        scene_id="scene",
        train_image_paths=("train/images/a.png", "train/images/b.png"),
        train_image_names=("a.png", "b.png"),
        train_images=(train_a, train_b),
        test_poses=(),
        test_distortions=(),
        sparse_points=(
            ColmapPointRecord(7, [1, 2, 3], [1, 2, 3], 0.25, (10, 11, 20)),
            ColmapPointRecord(9, [4, 5, 6], [4, 5, 6], 0.5, (10,)),
        ),
        cameras={1: camera},
    )
    monkeypatch.setattr(
        "bts_nvs.data.diagnostics.load_scene_source_data", lambda _: source
    )

    diagnostics = build_scene_diagnostics(tmp_path)

    assert isinstance(diagnostics, SceneDiagnostics)
    np.testing.assert_array_equal(diagnostics.point_ids, [7, 9])
    np.testing.assert_allclose(diagnostics.reprojection_errors, [0.25, 0.5])
    np.testing.assert_array_equal(diagnostics.track_lengths, [3, 1])
    np.testing.assert_array_equal(diagnostics.train_support_counts, [2, 1])
    assert diagnostics.train_observations_per_image == {"a.png": 2, "b.png": 1}
    assert diagnostics.point_ids.dtype == np.int64
    assert not diagnostics.point_ids.flags.writeable
