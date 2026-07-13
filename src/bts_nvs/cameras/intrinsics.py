from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise ValueError("height must be a positive integer")

        values = np.asarray([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("focal lengths must be positive")

        object.__setattr__(self, "fx", float(self.fx))
        object.__setattr__(self, "fy", float(self.fy))
        object.__setattr__(self, "cx", float(self.cx))
        object.__setattr__(self, "cy", float(self.cy))

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def resized(self, *, width: int, height: int) -> CameraIntrinsics:
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("height must be a positive integer")

        scale_x = width / self.width
        scale_y = height / self.height
        return CameraIntrinsics(
            width=width,
            height=height,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
        )
