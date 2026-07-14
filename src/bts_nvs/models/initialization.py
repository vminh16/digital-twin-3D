import numpy as np
import torch
from scipy.spatial import KDTree

from bts_nvs.data.manifest import SceneManifest
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.normalization import normalize_points_torch


def initialize_from_manifest(
    manifest: SceneManifest,
    device: torch.device = torch.device("cpu"),
) -> GaussianParameters:
    """Initializes 3D Gaussian Splatting parameters from a SceneManifest.
    
    Args:
        manifest (SceneManifest): Preprocessed and validated scene manifest.
        device (torch.device): Target device for PyTorch tensors.
        
    Returns:
        GaussianParameters: Initialized parameters container on the target device.
    """
    xyz_raw = torch.from_numpy(manifest.sparse_points.copy())
    rgb_raw = torch.from_numpy(manifest.sparse_colors.copy())
    
    N = xyz_raw.shape[0]
    if N == 0:
        raise ValueError("Cannot initialize Gaussian parameters from an empty point cloud.")
        
    # 1. Normalize world points using the scene normalization transform
    transform = torch.from_numpy(manifest.normalization_transform.copy())
    means = normalize_points_torch(xyz_raw, transform)
    
    # 2. Compute isotropic scale using 3-NN average distance on CPU
    xyz_np = means.detach().cpu().numpy()
    if N >= 4:
        tree = KDTree(xyz_np)
        # Query k=4 neighbors because index 0 is the query point itself
        distances, _ = tree.query(xyz_np, k=4)
        # distances[:, 1:4] shape is (N, 3)
        d_i = np.clip(
            np.mean(distances[:, 1:4], axis=1), a_min=1e-7, a_max=None
        )
    else:
        # Fallback for tiny point clouds (e.g. unit tests or synthetic fixtures)
        d_i = np.ones((N,), dtype=np.float64) * 0.1
        
    scales_np = np.log(d_i[:, None] * np.ones((1, 3)))
    scales = torch.from_numpy(scales_np)
    
    # 3. Initialize rotations to identity quaternions [1.0, 0.0, 0.0, 0.0]
    quats = torch.zeros((N, 4), dtype=torch.float32)
    quats[:, 0] = 1.0
    
    # 4. Initialize opacities to 0.1 (logit ≈ -2.1972)
    opacities = torch.full((N,), -2.1972245773362196, dtype=torch.float32)
    
    # 5. Initialize Spherical Harmonics coefficients: DC term (degree 0) mapped from RGB [0,1]
    rgb = rgb_raw.float() / 255.0
    sh0 = ((rgb - 0.5) / 0.28209479177387814).unsqueeze(1)
    shN = torch.zeros((N, 15, 3), dtype=torch.float32)
    
    # 6. Transfer to target device and create the parameters container
    return GaussianParameters(
        means=means.to(device),
        scales=scales.to(device),
        quats=quats.to(device),
        opacities=opacities.to(device),
        sh0=sh0.to(device),
        shN=shN.to(device),
    )
