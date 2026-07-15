from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch


@dataclass(frozen=True)
class RenderResult:
    """Encapsulates the outputs returned by the Gaussian Splatting renderer."""

    rgb: torch.Tensor
    """Rendered RGB image tensor of shape (H, W, 3)."""

    alpha: torch.Tensor
    """Rendered accumulated alpha transparency mask of shape (H, W, 1)."""

    depth: Optional[torch.Tensor] = None
    """Optional rendered depth map tensor of shape (H, W, 1)."""

    info: Optional[Dict[str, Any]] = None
    """Optional dictionary containing intermediate tensors from gsplat for backward pass."""

    def __post_init__(self) -> None:
        # Validate shapes and dimensions
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError(f"RGB tensor must have shape (H, W, 3), got {self.rgb.shape}")
        if self.alpha.ndim != 3 or self.alpha.shape[-1] != 1:
            raise ValueError(f"Alpha tensor must have shape (H, W, 1), got {self.alpha.shape}")
        if self.rgb.shape[:2] != self.alpha.shape[:2]:
            raise ValueError(
                f"RGB and Alpha height/width dimensions must match: "
                f"RGB {self.rgb.shape[:2]} != Alpha {self.alpha.shape[:2]}"
            )
        if self.depth is not None:
            if self.depth.ndim != 3 or self.depth.shape[-1] != 1:
                raise ValueError(f"Depth tensor must have shape (H, W, 1), got {self.depth.shape}")
            if self.depth.shape[:2] != self.rgb.shape[:2]:
                raise ValueError(
                    f"Depth and RGB height/width dimensions must match: "
                    f"Depth {self.depth.shape[:2]} != RGB {self.rgb.shape[:2]}"
                )
