from __future__ import annotations

import json

import pytest

from bts_nvs.training.backend_qualification import (
    compare_backend_profiles,
    load_backend_profile,
    write_backend_profile,
)


def _profile(
    backend: str,
    precision: str,
    *,
    cuda_ms: float,
    loss_scale: float = 1.0,
    count_delta: int = 0,
) -> dict:
    losses = [1.0 - index * 0.0005 for index in range(1000)]
    losses[-100:] = [value * loss_scale for value in losses[-100:]]
    counts = [1000] * 599 + [1100 + count_delta] * 401
    return {
        "schema_version": 1,
        "optimizer_backend": backend,
        "precision": precision,
        "steps": 1000,
        "device_name": "NVIDIA L4",
        "device_capability": [8, 9],
        "median_cuda_step_ms": cuda_ms,
        "peak_vram_bytes": 5_000_000_000,
        "sample_indices": [index % 7 for index in range(1000)],
        "losses": losses,
        "gaussian_counts": counts,
        "density_event_steps": [600, 700, 800, 900, 1000],
        "gradient_audits": [
            {
                "step": step,
                "finite": True,
                "strategy_gradient_unscaled": True,
                "loss_scale": 65536.0 if precision == "amp-fp16" else 1.0,
                "projected_grad_max": 0.001,
                "leaf_grad_max": {"means": 0.002},
                "parameter_dtypes": {"means": "torch.float32"},
                "render_dtype": "torch.float32",
                "loss_dtype": "torch.float32",
            }
            for step in (1, 499, 500, 501, 600, 1000)
        ],
    }


def test_backend_profile_round_trip_is_deterministic_and_standard_json(tmp_path):
    path = tmp_path / "profile.json"
    profile = _profile("adam", "fp32", cuda_ms=10.0)
    write_backend_profile(path, profile)
    first = path.read_bytes()
    write_backend_profile(path, profile)

    assert path.read_bytes() == first
    assert load_backend_profile(path) == profile
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_backend_comparator_selects_amp_only_for_incremental_speedup():
    report = compare_backend_profiles(
        _profile("adam", "fp32", cuda_ms=10.0),
        _profile("adam-fused", "fp32", cuda_ms=8.5),
        _profile("adam-fused", "amp-fp16", cuda_ms=7.5),
    )

    assert report["accepted"] is True
    assert report["selected_optimizer_backend"] == "adam-fused"
    assert report["selected_precision"] == "amp-fp16"
    assert report["fused_speedup_fraction"] == pytest.approx(0.15)
    assert report["amp_incremental_speedup_fraction"] == pytest.approx(1 - 7.5 / 8.5)


def test_backend_comparator_falls_back_in_priority_order():
    reference = _profile("adam", "fp32", cuda_ms=10.0)
    fused = _profile("adam-fused", "fp32", cuda_ms=9.5)
    amp = _profile("adam-fused", "amp-fp16", cuda_ms=9.2)
    report = compare_backend_profiles(reference, fused, amp)
    assert (report["selected_optimizer_backend"], report["selected_precision"]) == (
        "adam",
        "fp32",
    )

    fused["median_cuda_step_ms"] = 8.8
    report = compare_backend_profiles(reference, fused, amp)
    assert (report["selected_optimizer_backend"], report["selected_precision"]) == (
        "adam-fused",
        "fp32",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["sample_indices"].__setitem__(0, 99),
        lambda p: p["gradient_audits"][0].__setitem__("finite", False),
        lambda p: p["gradient_audits"][0].__setitem__(
            "strategy_gradient_unscaled", False
        ),
        lambda p: p.__setitem__("density_event_steps", [600, 700]),
        lambda p: p["gaussian_counts"].__setitem__(-1, 1200),
        lambda p: p["losses"].__setitem__(
            slice(-100, None), [value * 1.1 for value in p["losses"][-100:]]
        ),
    ],
)
def test_backend_comparator_rejects_semantic_drift(mutation):
    reference = _profile("adam", "fp32", cuda_ms=10.0)
    fused = _profile("adam-fused", "fp32", cuda_ms=8.0)
    mutation(fused)

    report = compare_backend_profiles(
        reference,
        fused,
        _profile("adam-fused", "amp-fp16", cuda_ms=7.0),
    )
    assert report["fused_correctness_passed"] is False


def test_backend_profile_rejects_non_finite_json(tmp_path):
    profile = _profile("adam", "fp32", cuda_ms=10.0)
    profile["losses"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        write_backend_profile(tmp_path / "profile.json", profile)
