from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
QUALIFICATION_STEPS = 1000
AUDIT_STEPS = (1, 499, 500, 501, 600, 1000)
DENSITY_EVENT_STEPS = (600, 700, 800, 900, 1000)
MAX_VRAM_BYTES = 20 * 1024**3


def _finite_number(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validate_profile(profile: dict) -> None:
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported backend profile schema")
    if profile.get("steps") != QUALIFICATION_STEPS:
        raise ValueError("backend profile must contain exactly 1000 steps")
    identity = (profile.get("optimizer_backend"), profile.get("precision"))
    if identity not in {
        ("adam", "fp32"),
        ("adam-fused", "fp32"),
        ("adam-fused", "amp-fp16"),
    }:
        raise ValueError("invalid backend profile identity")
    if profile.get("device_capability") != [8, 9]:
        raise ValueError("backend qualification requires NVIDIA L4 capability 8.9")
    if not isinstance(profile.get("device_name"), str) or "L4" not in profile["device_name"]:
        raise ValueError("backend qualification requires NVIDIA L4")
    _finite_number(profile.get("median_cuda_step_ms"), "median CUDA time", nonnegative=True)
    _finite_number(profile.get("peak_vram_bytes"), "peak VRAM", nonnegative=True)

    for key in ("sample_indices", "losses", "gaussian_counts"):
        values = profile.get(key)
        if not isinstance(values, list) or len(values) != QUALIFICATION_STEPS:
            raise ValueError(f"backend profile {key} must contain 1000 values")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in profile["sample_indices"]):
        raise ValueError("sample indices must be integers")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in profile["gaussian_counts"]):
        raise ValueError("Gaussian counts must be positive integers")
    for value in profile["losses"]:
        _finite_number(value, "loss")

    events = profile.get("density_event_steps")
    if not isinstance(events, list) or any(
        isinstance(step, bool) or not isinstance(step, int) for step in events
    ):
        raise ValueError("backend profile has invalid density event steps")
    audits = profile.get("gradient_audits")
    if not isinstance(audits, list) or [entry.get("step") for entry in audits] != list(AUDIT_STEPS):
        raise ValueError("backend profile has invalid gradient audit steps")
    for audit in audits:
        if not isinstance(audit.get("finite"), bool):
            raise ValueError("gradient finite flag must be boolean")
        if not isinstance(audit.get("strategy_gradient_unscaled"), bool):
            raise ValueError("strategy gradient flag must be boolean")
        _finite_number(audit.get("loss_scale"), "loss scale", nonnegative=True)
        _finite_number(audit.get("projected_grad_max"), "projected gradient", nonnegative=True)
        leaf = audit.get("leaf_grad_max")
        if not isinstance(leaf, dict) or not leaf:
            raise ValueError("leaf gradient audit must be non-empty")
        for value in leaf.values():
            _finite_number(value, "leaf gradient", nonnegative=True)
        dtypes = audit.get("parameter_dtypes")
        if not isinstance(dtypes, dict) or not dtypes:
            raise ValueError("parameter dtype audit must be non-empty")


def write_backend_profile(path: Path, profile: dict) -> None:
    _validate_profile(profile)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, output)


def load_backend_profile(path: Path) -> dict:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("backend profile must contain a JSON object")
    _validate_profile(profile)
    return profile


def _correctness(reference: dict, candidate: dict) -> bool:
    reference_final_count = reference["gaussian_counts"][-1]
    count_delta = abs(candidate["gaussian_counts"][-1] - reference_final_count)
    count_close = count_delta / reference_final_count <= 0.01
    reference_tail = float(np.mean(reference["losses"][-100:]))
    candidate_tail = float(np.mean(candidate["losses"][-100:]))
    tail_close = abs(candidate_tail - reference_tail) / max(abs(reference_tail), 1e-12) <= 0.02
    audits_pass = all(
        audit["finite"]
        and audit["strategy_gradient_unscaled"]
        and all(dtype == "torch.float32" for dtype in audit["parameter_dtypes"].values())
        for audit in candidate["gradient_audits"]
    )
    return bool(
        candidate["sample_indices"] == reference["sample_indices"]
        and candidate["density_event_steps"] == reference["density_event_steps"]
        and np.allclose(
            candidate["losses"][:599],
            reference["losses"][:599],
            rtol=2e-3,
            atol=1e-6,
        )
        and count_close
        and tail_close
        and audits_pass
        and candidate["peak_vram_bytes"] < MAX_VRAM_BYTES
    )


def compare_backend_profiles(reference: dict, fused: dict, amp: dict) -> dict:
    for profile in (reference, fused, amp):
        _validate_profile(profile)
    if (reference["optimizer_backend"], reference["precision"]) != ("adam", "fp32"):
        raise ValueError("first backend profile must be adam/fp32")
    if (fused["optimizer_backend"], fused["precision"]) != ("adam-fused", "fp32"):
        raise ValueError("second backend profile must be adam-fused/fp32")
    if (amp["optimizer_backend"], amp["precision"]) != ("adam-fused", "amp-fp16"):
        raise ValueError("third backend profile must be adam-fused/amp-fp16")

    reference_ok = all(
        audit["finite"] and audit["strategy_gradient_unscaled"]
        for audit in reference["gradient_audits"]
    ) and reference["peak_vram_bytes"] < MAX_VRAM_BYTES
    fused_ok = _correctness(reference, fused)
    amp_ok = _correctness(reference, amp)
    fused_speedup = 1.0 - fused["median_cuda_step_ms"] / reference["median_cuda_step_ms"]
    amp_incremental = 1.0 - amp["median_cuda_step_ms"] / fused["median_cuda_step_ms"]

    selected_backend = "adam"
    selected_precision = "fp32"
    if fused_ok and fused_speedup >= 0.10:
        selected_backend = "adam-fused"
        if amp_ok and amp_incremental >= 0.05:
            selected_precision = "amp-fp16"

    return {
        "accepted": bool(reference_ok),
        "amp_correctness_passed": amp_ok,
        "amp_incremental_speedup_fraction": amp_incremental,
        "fused_correctness_passed": fused_ok,
        "fused_speedup_fraction": fused_speedup,
        "selected_optimizer_backend": selected_backend,
        "selected_precision": selected_precision,
    }


def write_backend_comparison(path: Path, report: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
