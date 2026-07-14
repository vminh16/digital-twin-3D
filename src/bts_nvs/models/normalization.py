import torch


def torch_invert_rigid_transform(transform: torch.Tensor) -> torch.Tensor:
    """Inverts a batch of 4x4 homogeneous rigid transformations.
    
    Args:
        transform (torch.Tensor): Tensor of shape (..., 4, 4).
        
    Returns:
        torch.Tensor: Inverted transformation of shape (..., 4, 4).
    """
    if transform.shape[-2:] != (4, 4):
        raise ValueError("Transform matrix must have shape (..., 4, 4)")
        
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3:4]
    
    rotation_inv = rotation.transpose(-1, -2)
    translation_inv = -rotation_inv @ translation
    
    inv = torch.eye(4, dtype=transform.dtype, device=transform.device)
    # Expand to batch shape
    inv = inv.expand_as(transform).clone()
    inv[..., :3, :3] = rotation_inv
    inv[..., :3, 3:4] = translation_inv
    return inv


def normalize_points_torch(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Normalizes raw 3D points using the scene normalization transform.
    
    Args:
        points (torch.Tensor): Raw 3D points of shape (..., 3).
        transform (torch.Tensor): Scene normalization transform matrix of shape (4, 4).
        
    Returns:
        torch.Tensor: Normalized 3D points of shape (..., 3).
    """
    if transform.shape != (4, 4):
        raise ValueError("Normalization transform must have shape (4, 4)")
        
    scale = transform[0, 0]
    translation = transform[:3, 3] # -scale * center
    
    return points * scale + translation


def normalize_c2w(poses: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Normalizes Camera-to-World (C2W) poses.
    
    Args:
        poses (torch.Tensor): C2W matrices of shape (..., 4, 4).
        transform (torch.Tensor): Scene normalization transform matrix of shape (4, 4).
        
    Returns:
        torch.Tensor: Normalized C2W matrices of shape (..., 4, 4).
    """
    if transform.shape != (4, 4):
        raise ValueError("Normalization transform must have shape (4, 4)")
        
    scale = transform[0, 0]
    translation = transform[:3, 3]
    
    normalized = poses.clone()
    normalized[..., :3, 3] = poses[..., :3, 3] * scale + translation
    return normalized


def normalize_w2c(poses: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Normalizes World-to-Camera (W2C) poses.
    
    Args:
        poses (torch.Tensor): W2C matrices of shape (..., 4, 4).
        transform (torch.Tensor): Scene normalization transform matrix of shape (4, 4).
        
    Returns:
        torch.Tensor: Normalized W2C matrices of shape (..., 4, 4).
    """
    c2w = torch_invert_rigid_transform(poses)
    c2w_norm = normalize_c2w(c2w, transform)
    return torch_invert_rigid_transform(c2w_norm)
