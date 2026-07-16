from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np


PROFILE_SCHEMA_VERSION = 1
PROFILE_WARMUP_STEPS = 50
PROFILE_MEASURED_STEPS = 500


def _validate_profile(profile: dict) -> None:
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported input profile schema")
    if profile.get("warmup_steps") != PROFILE_WARMUP_STEPS:
        raise ValueError("input profile must use 50 warm-up steps")
    if profile.get("measured_steps") != PROFILE_MEASURED_STEPS:
        raise ValueError("input profile must use 500 measured steps")
    identity = profile.get("training_identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise ValueError("invalid training identity hash")
    if not isinstance(profile.get("cache_images"), bool):
        raise ValueError("profile cache_images must be boolean")
    for key in (
        "mean_wall_step_ms",
        "median_wall_step_ms",
        "median_cuda_step_ms",
        "cpu_preprocessing_fraction",
        "peak_vram_bytes",
    ):
        value = profile.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid profile value: {key}")
    for key in ("sample_indices", "losses", "gaussian_counts"):
        if (
            not isinstance(profile.get(key), list)
            or len(profile[key]) != PROFILE_MEASURED_STEPS
        ):
            raise ValueError(f"invalid profile trace: {key}")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for key in ("sample_indices", "gaussian_counts")
        for value in profile[key]
    ):
        raise ValueError("profile index/count traces must contain integers")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in profile["losses"]
    ):
        raise ValueError("profile loss trace must be finite")


def write_input_profile(path: Path, profile: dict) -> None:
    _validate_profile(profile)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, output)


def load_input_profile(path: Path) -> dict:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("input profile must contain a JSON object")
    _validate_profile(profile)
    return profile


def compare_input_profiles(uncached: dict, cached: dict) -> dict:
    _validate_profile(uncached)
    _validate_profile(cached)
    same_domain = (
        uncached["cache_images"] is False
        and cached["cache_images"] is True
        and uncached["training_identity_sha256"]
        == cached["training_identity_sha256"]
        and uncached["sample_indices"] == cached["sample_indices"]
        and uncached["gaussian_counts"] == cached["gaussian_counts"]
    )
    losses_equal = bool(
        len(uncached["losses"]) == len(cached["losses"])
        and np.allclose(
            uncached["losses"],
            cached["losses"],
            rtol=1e-4,
            atol=1e-6,
        )
    )
    trace_equal = bool(same_domain and losses_equal)
    uncached_wall = float(uncached["median_wall_step_ms"])
    cached_wall = float(cached["median_wall_step_ms"])
    speedup = 0.0 if uncached_wall == 0.0 else 1.0 - cached_wall / uncached_wall
    performance = bool(
        speedup >= 0.10 or cached["cpu_preprocessing_fraction"] < 0.10
    )
    return {
        "accepted": bool(trace_equal and performance),
        "performance_gate_passed": performance,
        "speedup_fraction": speedup,
        "trace_equal": trace_equal,
    }
