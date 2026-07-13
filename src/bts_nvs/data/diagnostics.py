from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .manifest import load_scene_source_data


def _readonly(values: object, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SceneDiagnostics:
    point_ids: np.ndarray
    reprojection_errors: np.ndarray
    track_lengths: np.ndarray
    train_support_counts: np.ndarray
    train_observations_per_image: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_ids", _readonly(self.point_ids, np.int64))
        object.__setattr__(self, "reprojection_errors", _readonly(self.reprojection_errors, np.float64))
        object.__setattr__(self, "track_lengths", _readonly(self.track_lengths, np.int64))
        object.__setattr__(self, "train_support_counts", _readonly(self.train_support_counts, np.int64))
        object.__setattr__(
            self,
            "train_observations_per_image",
            MappingProxyType(dict(self.train_observations_per_image)),
        )


def build_scene_diagnostics(scene_root: Path) -> SceneDiagnostics:
    source = load_scene_source_data(scene_root)
    names_by_id = {image.image_id: image.name for image in source.train_images}
    observations = {name: 0 for name in source.train_image_names}
    support_counts: list[int] = []
    for point in source.sparse_points:
        support = 0
        for image_id in point.image_ids:
            name = names_by_id.get(image_id)
            if name is not None:
                support += 1
                observations[name] += 1
        support_counts.append(support)
    return SceneDiagnostics(
        point_ids=[point.point_id for point in source.sparse_points],
        reprojection_errors=[point.reprojection_error for point in source.sparse_points],
        track_lengths=[len(point.image_ids) for point in source.sparse_points],
        train_support_counts=support_counts,
        train_observations_per_image=observations,
    )

