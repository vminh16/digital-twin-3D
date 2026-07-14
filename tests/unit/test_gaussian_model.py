import numpy as np
import pytest
import torch
from dataclasses import replace

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.manifest import SceneManifest
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.initialization import initialize_from_manifest
from bts_nvs.models.normalization import (
    normalize_c2w,
    normalize_points_torch,
    normalize_w2c,
    torch_invert_rigid_transform,
)


def test_gaussian_parameters_validation():
    # Correct shapes: N=5
    means = torch.randn(5, 3)
    scales = torch.randn(5, 3)
    quats = torch.randn(5, 4)
    opacities = torch.randn(5)
    sh0 = torch.randn(5, 1, 3)
    shN = torch.randn(5, 15, 3)

    # Should construct successfully
    gp = GaussianParameters(means, scales, quats, opacities, sh0, shN)
    assert gp.num_gaussians == 5

    # Should raise error on size mismatch
    with pytest.raises(ValueError, match="scales"):
        GaussianParameters(means, torch.randn(4, 3), quats, opacities, sh0, shN)
    with pytest.raises(ValueError, match="quats"):
        GaussianParameters(means, scales, torch.randn(5, 3), opacities, sh0, shN)
    with pytest.raises(ValueError, match="opacities"):
        GaussianParameters(means, scales, quats, torch.randn(5, 1), sh0, shN)
    with pytest.raises(ValueError, match="sh0"):
        GaussianParameters(means, scales, quats, opacities, torch.randn(5, 2, 3), shN)
    with pytest.raises(ValueError, match="shN"):
        GaussianParameters(means, scales, quats, opacities, sh0, torch.randn(5, 14, 3))
    with pytest.raises(ValueError, match="means"):
        GaussianParameters(torch.randn(5, 4), scales, quats, opacities, sh0, shN)
    with pytest.raises(ValueError, match="quaternion"):
        GaussianParameters(means, scales, torch.zeros(5, 4), opacities, sh0, shN)
    invalid_means = means.clone()
    invalid_means[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        GaussianParameters(invalid_means, scales, quats, opacities, sh0, shN)


def test_gaussian_parameters_getters_and_covariance():
    means = torch.zeros(3, 3)
    scales = torch.tensor([[0.0, 1.0, -1.0], [0.5, 0.5, 0.5], [-1.0, -1.0, -1.0]])
    # Quaternions: WXYZ. Let's make some non-unit ones to verify unit normalization.
    quats = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    opacities = torch.tensor([0.0, -10.0, 10.0])
    sh0 = torch.zeros(3, 1, 3)
    shN = torch.zeros(3, 15, 3)

    gp = GaussianParameters(means, scales, quats, opacities, sh0, shN)

    # 1. Verification of get_means
    torch.testing.assert_close(gp.get_means(), means)

    # 2. Verification of get_scales: should be exp(s)
    expected_scales = torch.exp(scales)
    torch.testing.assert_close(gp.get_scales(), expected_scales)

    # 3. Verification of get_quats: should be unit norm
    norm_quats = gp.get_quats()
    norms = torch.norm(norm_quats, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))

    # 4. Verification of get_opacities: should be sigmoid
    expected_opacities = torch.sigmoid(opacities)
    torch.testing.assert_close(gp.get_opacities(), expected_opacities)
    assert (gp.get_opacities() > 0.0).all()
    assert (gp.get_opacities() < 1.0).all()

    # 5. Verification of get_shs
    torch.testing.assert_close(gp.get_shs(), torch.cat((sh0, shN), dim=1))
    assert gp.sh0.is_leaf and gp.shN.is_leaf
    assert set(gp.state_dict()) == {
        "means",
        "scales",
        "quats",
        "opacities",
        "sh0",
        "shN",
    }
    optimizers = {
        "sh0": torch.optim.Adam([gp.sh0], lr=2.5e-3),
        "shN": torch.optim.Adam([gp.shN], lr=1.25e-4),
    }
    assert optimizers["sh0"].param_groups[0]["params"] == [gp.sh0]
    assert optimizers["shN"].param_groups[0]["params"] == [gp.shN]

    # 6. Verification of get_covariance: should be positive definite and symmetric
    cov = gp.get_covariance()
    assert cov.shape == (3, 3, 3)
    
    # Check symmetry
    torch.testing.assert_close(cov, cov.transpose(-1, -2), atol=1e-6, rtol=1e-6)
    
    # Check positive-definiteness via eigenvalues (must be strictly positive)
    for i in range(3):
        eigenvalues = torch.linalg.eigvalsh(cov[i])
        assert (eigenvalues > 0.0).all()


def test_rigid_transform_inversion():
    # Create random batch of translation vector and rotation angles
    # Roll, pitch, yaw
    T = torch.eye(4).repeat(5, 1, 1)
    for i in range(5):
        # random rotation matrix
        q = torch.randn(4)
        q = q / q.norm()
        qw, qx, qy, qz = q
        R = torch.tensor([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        t = torch.randn(3, 1)
        T[i, :3, :3] = R
        T[i, :3, 3:4] = t

    T_inv = torch_invert_rigid_transform(T)
    
    # Multiplying them should yield Identity
    identity = torch.eye(4).repeat(5, 1, 1)
    torch.testing.assert_close(T @ T_inv, identity, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(T_inv @ T, identity, atol=1e-5, rtol=1e-5)


def test_normalization_transforms():
    points = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=torch.float32)
    
    # Scale=2.0, center=[1.0, 2.0, 3.0]
    scale = 2.0
    center = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    transform = torch.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    
    # Expected points: s(X - center)
    # [0.0, 0.0, 0.0], [6.0, 6.0, 6.0]
    expected_points = scale * (points - center)
    norm_points = normalize_points_torch(points, transform)
    torch.testing.assert_close(norm_points, expected_points)
    
    # Verify C2W and W2C normalization
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([5.0, 6.0, 7.0])
    
    c2w_norm = normalize_c2w(c2w, transform)
    # C2W center should be normalized: 2 * (C - center)
    # 2 * ([5, 6, 7] - [1, 2, 3]) = [8, 8, 8]
    torch.testing.assert_close(c2w_norm[:3, 3], torch.tensor([8.0, 8.0, 8.0]))
    torch.testing.assert_close(c2w_norm[:3, :3], c2w[:3, :3]) # rotation unchanged
    
    w2c = torch_invert_rigid_transform(c2w)
    w2c_norm = normalize_w2c(w2c, transform)
    # Check that inverting w2c_norm yields c2w_norm
    torch.testing.assert_close(torch_invert_rigid_transform(w2c_norm), c2w_norm)


def _mock_manifest():
    identity = np.eye(4, dtype=np.float64)
    intrinsics = CameraIntrinsics(8, 6, 5.0, 5.0, 4.0, 3.0)
    distortion = CameraDistortion("SIMPLE_RADIAL", (0.5,))
    
    # 5 Points
    sparse_points = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [0.9, 1.9, 2.9],
        [1.5, 2.5, 3.5],
        [2.0, 3.0, 4.0]
    ], dtype=np.float64)
    sparse_colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255]
    ], dtype=np.uint8)
    
    # Normalization
    scale = 2.0
    center = np.array([1.0, 2.0, 3.0])
    normalization_transform = np.eye(4)
    normalization_transform[:3, :3] *= scale
    normalization_transform[:3, 3] = -scale * center
    inverse_normalization_transform = np.eye(4)
    inverse_normalization_transform[:3, :3] /= scale
    inverse_normalization_transform[:3, 3] = center
    
    manifest = SceneManifest(
        schema_version=1,
        scene_id="scene",
        train_image_paths=("train/images/a.png",),
        train_image_names=("a.png",),
        train_world_to_camera=identity[None],
        train_camera_to_world=identity[None],
        train_intrinsics=(intrinsics,),
        train_distortion=(distortion,),
        test_image_names=("target.JPG",),
        test_output_names=("target.png",),
        test_world_to_camera=identity[None],
        test_intrinsics=(intrinsics,),
        test_distortion=(distortion,),
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
        normalization_transform=normalization_transform,
        inverse_normalization_transform=inverse_normalization_transform,
    )
    return manifest


def test_initialization_from_manifest():
    manifest = _mock_manifest()
    
    gp = initialize_from_manifest(manifest)
    
    # 1. Size
    assert gp.num_gaussians == 5
    
    # 2. Normalization of means
    # First point [1, 2, 3] normalized should be [0, 0, 0] because center is [1, 2, 3]
    torch.testing.assert_close(gp.get_means()[0], torch.zeros(3))
    
    # 3. Scales log initialization check
    # KNN on CPU check
    scales = gp.get_scales()
    assert (scales > 0.0).all()
    # Check isotropic (all scale components of a Gaussian are identical)
    torch.testing.assert_close(scales[:, 0], scales[:, 1])
    torch.testing.assert_close(scales[:, 0], scales[:, 2])
    
    # 4. Quaternions: WXYZ identity [1, 0, 0, 0]
    quats = gp.get_quats()
    expected_quats = torch.zeros((5, 4))
    expected_quats[:, 0] = 1.0
    torch.testing.assert_close(quats, expected_quats)
    
    # 5. Opacities should evaluate to 0.1
    opacities = gp.get_opacities()
    torch.testing.assert_close(opacities, torch.full((5,), 0.1))
    
    # 6. Colors: DC term (degree 0) matches RGB colors
    # First point color is [255, 0, 0] (RGB: [1, 0, 0])
    # DC term evaluated should yield [1, 0, 0]
    shs = gp.get_shs()
    # gsplat spherical harmonics utility signature evaluation mock:
    # color = shs[:, 0, :] * 0.28209479177387814 + 0.5
    shs_eval = shs[:, 0, :] * 0.28209479177387814 + 0.5
    expected_rgb = torch.tensor(manifest.sparse_colors, dtype=torch.float32) / 255.0
    torch.testing.assert_close(shs_eval, expected_rgb)
    
    # 7. Check deterministic
    gp2 = initialize_from_manifest(manifest)
    torch.testing.assert_close(gp.means, gp2.means)
    torch.testing.assert_close(gp.scales, gp2.scales)
    torch.testing.assert_close(gp.quats, gp2.quats)
    torch.testing.assert_close(gp.opacities, gp2.opacities)
    torch.testing.assert_close(gp.sh0, gp2.sh0)
    torch.testing.assert_close(gp.shN, gp2.shN)


def test_initial_scale_uses_normalized_world_distances():
    manifest = _mock_manifest()
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    manifest = replace(
        manifest,
        sparse_points=points,
        sparse_colors=np.zeros((4, 3), dtype=np.uint8),
    )

    model = initialize_from_manifest(manifest)

    # The manifest normalization scale is 2, so the first point's three
    # unit-distance neighbors are all distance 2 in normalized world space.
    torch.testing.assert_close(model.get_scales()[0], torch.full((3,), 2.0))
    expected_second = (2.0 + 4.0 * np.sqrt(2.0)) / 3.0
    torch.testing.assert_close(
        model.get_scales()[1], torch.full((3,), expected_second)
    )


def test_initialization_does_not_mutate_manifest():
    manifest = _mock_manifest()
    
    points_before = manifest.sparse_points.copy()
    colors_before = manifest.sparse_colors.copy()
    transform_before = manifest.normalization_transform.copy()
    
    _ = initialize_from_manifest(manifest)
    
    np.testing.assert_array_equal(manifest.sparse_points, points_before)
    np.testing.assert_array_equal(manifest.sparse_colors, colors_before)
    np.testing.assert_array_equal(manifest.normalization_transform, transform_before)
