from __future__ import annotations

import math
from pathlib import Path

from bts_nvs.experiments.provenance import load_json_artifact


def validate_observation_mapping_artifact(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"required initialization_diagnostics.json does not exist: {path}"
        )
    diagnostics = load_json_artifact(path)
    if diagnostics.get("mode") != "continuous-reprojection":
        raise ValueError("initialization diagnostics mode does not match candidate")
    for field in ("scale", "legacy_scale"):
        _validate_scale(diagnostics.get(field), field)
    for field in ("fit_observation_count", "color_comparison_point_count"):
        value = diagnostics.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"initialization diagnostics {field} must be a positive integer"
            )

    mapped_p95 = _finite_nonnegative(
        diagnostics.get("mapped_reprojection_p95_px"),
        "initialization mapped reprojection p95",
    )
    legacy_p95 = _finite_nonnegative(
        diagnostics.get("legacy_reprojection_p95_px"),
        "initialization legacy reprojection p95",
    )
    if mapped_p95 >= legacy_p95:
        raise ValueError(
            "initialization reprojection residual does not improve over legacy"
        )
    for suffix in ("mean", "median", "p90"):
        selected = _finite_nonnegative(
            diagnostics.get(f"selected_color_mae_{suffix}"),
            f"initialization selected color MAE {suffix}",
        )
        legacy = _finite_nonnegative(
            diagnostics.get(f"legacy_color_mae_{suffix}"),
            f"initialization legacy color MAE {suffix}",
        )
        if selected >= legacy:
            raise ValueError(
                f"initialization color MAE {suffix} does not improve over legacy"
            )


def _validate_scale(value: object, field: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(element, bool)
            or not isinstance(element, (int, float))
            or not math.isfinite(float(element))
            or float(element) <= 0.0
            for element in value
        )
    ):
        raise ValueError(
            f"initialization diagnostics {field} must be two positive values"
        )


def _finite_nonnegative(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field} must be finite and nonnegative")
    return float(value)
