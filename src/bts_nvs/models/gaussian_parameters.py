import torch
import torch.nn as nn


class GaussianParameters(nn.Module):
    """Container for 3D Gaussian Representation parameters.
    
    Manages parameters optimized during training and exposes physical attributes.
    """
    def __init__(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor,
    ) -> None:
        """Initializes raw parameters as PyTorch nn.Parameters.
        
        Args:
            means (torch.Tensor): Initial coordinates of shape (N, 3).
            scales (torch.Tensor): Initial log-scales of shape (N, 3).
            quats (torch.Tensor): Initial quaternions (WXYZ) of shape (N, 4).
            opacities (torch.Tensor): Initial opacity logits of shape (N,).
            sh0 (torch.Tensor): Degree-zero SH coefficients of shape (N, 1, 3).
            shN (torch.Tensor): Higher-order SH coefficients of shape (N, 15, 3).
        """
        super().__init__()
        
        # Verify shapes
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(f"means must have shape (N, 3), got {means.shape}")
        N = means.shape[0]
        if scales.shape != (N, 3):
            raise ValueError(f"scales must have shape ({N}, 3), got {scales.shape}")
        if quats.shape != (N, 4):
            raise ValueError(f"quats must have shape ({N}, 4), got {quats.shape}")
        if opacities.shape != (N,):
            raise ValueError(f"opacities must have shape ({N},), got {opacities.shape}")
        if sh0.shape != (N, 1, 3):
            raise ValueError(f"sh0 must have shape ({N}, 1, 3), got {sh0.shape}")
        if shN.shape != (N, 15, 3):
            raise ValueError(f"shN must have shape ({N}, 15, 3), got {shN.shape}")

        tensors = {
            "means": means,
            "scales": scales,
            "quaternions": quats,
            "opacities": opacities,
            "sh0": sh0,
            "shN": shN,
        }
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
        if torch.any(torch.linalg.vector_norm(quats, dim=-1) < 1e-12):
            raise ValueError("quaternion norm is too small")
            
        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(scales.float())
        self.quats = nn.Parameter(quats.float())
        self.opacities = nn.Parameter(opacities.float())
        self.sh0 = nn.Parameter(sh0.float())
        self.shN = nn.Parameter(shN.float())

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def get_means(self) -> torch.Tensor:
        """Exposes raw coordinates (N, 3)."""
        return self.means

    def get_scales(self) -> torch.Tensor:
        """Computes positive scales exp(s) of shape (N, 3)."""
        return torch.exp(self.scales)

    def get_quats(self) -> torch.Tensor:
        """Computes normalized quaternions of shape (N, 4) with unit norm."""
        return self.quats / torch.norm(self.quats, dim=-1, keepdim=True).clamp_min(1e-12)

    def get_opacities(self) -> torch.Tensor:
        """Computes opacities sigmoid(o) of shape (N,) in (0, 1)."""
        return torch.sigmoid(self.opacities)

    def get_shs(self) -> torch.Tensor:
        """Exposes raw Spherical Harmonics coefficients of shape (N, 16, 3)."""
        return torch.cat((self.sh0, self.shN), dim=1)

    def get_covariance(self) -> torch.Tensor:
        """Computes the 3D covariance matrix Sigma = R S S^T R^T of shape (N, 3, 3)."""
        q = self.get_quats()
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Precompute quaternion products
        qw2 = qw * qw
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz
        
        qwx = qw * qx
        qwy = qw * qy
        qwz = qw * qz
        qxy = qx * qy
        qxz = qx * qz
        qyz = qy * qz
        
        # Compute components of rotation matrix R
        r00 = 1.0 - 2.0 * (qy2 + qz2)
        r01 = 2.0 * (qxy - qwz)
        r02 = 2.0 * (qxz + qwy)
        
        r10 = 2.0 * (qxy + qwz)
        r11 = 1.0 - 2.0 * (qx2 + qz2)
        r12 = 2.0 * (qyz - qwx)
        
        r20 = 2.0 * (qxz - qwy)
        r21 = 2.0 * (qyz + qwx)
        r22 = 1.0 - 2.0 * (qx2 + qy2)
        
        # Stack into rotation matrix R: (N, 3, 3)
        R = torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1)
        ], dim=-2)
        
        scales = self.get_scales()  # (N, 3)
        S = torch.diag_embed(scales)  # (N, 3, 3)
        
        R_S = R @ S
        covariance = R_S @ R_S.transpose(-1, -2)
        return covariance
