from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from bts_nvs.rendering.spectral_math import spectral_entropy_and_condition


@torch.no_grad()
def build_spectral_diagnostics(trainer) -> dict[str, object]:
    scales = trainer.gaussians.get_scales().detach()
    opacity = trainer.gaussians.get_opacities().detach()
    entropy, condition = spectral_entropy_and_condition(scales)
    threshold = float(trainer.config["spectral_entropy_threshold"])
    low = entropy < threshold
    opacity_total = opacity.sum().clamp_min(1e-12)
    state = trainer.strategy_state
    return {
        "schema_version": 1,
        "candidate_id": trainer.config["candidate_id"],
        "entropy_threshold": threshold,
        "split_k": float(trainer.config["spectral_split_k"]),
        "split_k0": float(trainer.config["spectral_split_k0"]),
        "cap_max": int(trainer.config["spectral_cap_max"]),
        "entropy": _quantiles(entropy),
        "condition": _quantiles(condition),
        "low_entropy": {
            "count": int(low.sum()),
            "fraction": float(low.float().mean()),
            "opacity_mass": float(opacity[low].sum()),
            "opacity_fraction": float(opacity[low].sum() / opacity_total),
        },
        "density": {
            "adc_clone_count": int(state.get("adc_clone_count", 0)),
            "adc_split_count": int(state.get("adc_split_count", 0)),
            "spectral_split_count": int(state.get("spectral_split_count", 0)),
            "cap_hit_step": state.get("spectral_cap_hit_step"),
            "child_condition_violations": int(
                state.get("spectral_child_condition_violations", 0)
            ),
        },
    }


def validate_spectral_diagnostics(
    report: Mapping[str, object],
    *,
    candidate_id: str,
    threshold: float,
    cap_max: int,
) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("spectral diagnostics schema_version is invalid")
    if report.get("candidate_id") != candidate_id:
        raise ValueError("spectral diagnostics candidate_id does not match")
    if report.get("entropy_threshold") != threshold:
        raise ValueError("spectral diagnostics threshold does not match config")
    if report.get("cap_max") != cap_max:
        raise ValueError("spectral diagnostics cap does not match config")
    for section in ("entropy", "condition", "low_entropy", "density"):
        if not isinstance(report.get(section), Mapping):
            raise ValueError(f"spectral diagnostics {section} is invalid")
    density = report["density"]
    assert isinstance(density, Mapping)
    for field in (
        "adc_clone_count",
        "adc_split_count",
        "spectral_split_count",
        "child_condition_violations",
    ):
        value = density.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"spectral diagnostics {field} is invalid")
    if density["child_condition_violations"] != 0:
        raise ValueError("spectral split increased child condition number")


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {name: 0.0 for name in ("min", "p50", "p95", "p99", "max")}
    result = {
        "min": float(values.min()),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("spectral diagnostics contain non-finite values")
    return result
