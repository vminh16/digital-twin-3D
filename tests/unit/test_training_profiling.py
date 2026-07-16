import json

import pytest

from bts_nvs.training.profiling import (
    compare_input_profiles,
    equivalence_steps_before_refinement,
    write_input_profile,
)
from bts_nvs.training.compare_input_profiles import main as compare_main


def _profile(*, wall_ms=10.0, preprocess_fraction=0.2, losses=(1.0, 0.9)):
    losses = tuple(losses)
    if len(losses) == 2:
        losses = losses * 250
    return {
        "cache_images": False,
        "equivalence_steps": 449,
        "schema_version": 2,
        "training_identity_sha256": "a" * 64,
        "warmup_steps": 50,
        "measured_steps": 500,
        "mean_wall_step_ms": wall_ms,
        "median_wall_step_ms": wall_ms,
        "median_cuda_step_ms": 8.0,
        "cpu_preprocessing_fraction": preprocess_fraction,
        "peak_vram_bytes": 1024,
        "sample_indices": [1, 0] * 250,
        "losses": list(losses),
        "gaussian_counts": [10, 10] * 250,
    }


def test_profile_json_is_deterministic_and_standard_compliant(tmp_path):
    path = tmp_path / "profile.json"
    write_input_profile(path, _profile())
    first = path.read_bytes()
    write_input_profile(path, _profile())

    assert path.read_bytes() == first
    assert json.loads(first)["measured_steps"] == 500
    assert b"NaN" not in first and b"Infinity" not in first


def test_equivalence_boundary_stops_before_first_refinement():
    assert equivalence_steps_before_refinement(500) == 449
    assert equivalence_steps_before_refinement(1000) == 500
    with pytest.raises(ValueError, match="before refinement"):
        equivalence_steps_before_refinement(51)


def test_comparator_accepts_speedup_and_equal_trace():
    cached = _profile(wall_ms=8.9)
    cached["cache_images"] = True
    report = compare_input_profiles(_profile(), cached)

    assert report["trace_equal"] is True
    assert report["performance_gate_passed"] is True
    assert report["accepted"] is True


def test_comparator_accepts_low_cached_preprocessing_fraction():
    cached = _profile(wall_ms=10.0, preprocess_fraction=0.09)
    cached["cache_images"] = True
    report = compare_input_profiles(_profile(), cached)
    assert report["accepted"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_indices", [0, 1] * 250),
        ("gaussian_counts", [10, 11] * 250),
    ],
)
def test_comparator_rejects_changed_optimization_domain(field, value):
    cached = _profile()
    cached["cache_images"] = True
    cached[field] = value

    report = compare_input_profiles(_profile(), cached)

    assert report["accepted"] is False
    assert report["trace_equal"] is False


def test_comparator_uses_locked_loss_tolerance():
    close = _profile(losses=(1.00005, 0.90005))
    far = _profile(losses=(1.01, 0.9))
    close["cache_images"] = True
    far["cache_images"] = True
    assert compare_input_profiles(_profile(), close)["trace_equal"]
    assert not compare_input_profiles(_profile(), far)["trace_equal"]


def test_comparator_rejects_different_training_identity():
    cached = _profile()
    cached["cache_images"] = True
    cached["training_identity_sha256"] = "b" * 64

    assert compare_input_profiles(_profile(), cached)["accepted"] is False


def test_comparator_ignores_only_post_refinement_topology_divergence():
    uncached = _profile()
    cached = _profile(wall_ms=8.0)
    cached["cache_images"] = True
    cached["gaussian_counts"][449:] = [214462] * 51
    uncached["gaussian_counts"][449:] = [214461] * 51
    cached["losses"][449:] = [0.2] * 51
    uncached["losses"][449:] = [0.21] * 51

    report = compare_input_profiles(uncached, cached)

    assert report["trace_equal"] is True
    assert report["accepted"] is True
    assert report["equivalence_steps"] == 449
    assert report["final_gaussian_count_delta"] == 1


def test_comparator_rejects_divergence_before_refinement():
    uncached = _profile()
    cached = _profile(wall_ms=8.0)
    cached["cache_images"] = True
    cached["gaussian_counts"][448] += 1

    assert compare_input_profiles(uncached, cached)["trace_equal"] is False


def test_compare_cli_writes_report_and_returns_gate_status(tmp_path):
    uncached = tmp_path / "uncached.json"
    cached = tmp_path / "cached.json"
    output = tmp_path / "comparison.json"
    write_input_profile(uncached, _profile())
    cached_profile = _profile(wall_ms=8.0)
    cached_profile["cache_images"] = True
    write_input_profile(cached, cached_profile)

    code = compare_main(
        [
            "--uncached",
            str(uncached),
            "--cached",
            str(cached),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert json.loads(output.read_text())["accepted"] is True
