from __future__ import annotations

import json

import pytest

from bts_nvs.experiments.provenance import (
    canonical_json_sha256,
    load_json_artifact,
    save_json_artifact,
)


def test_semantically_equal_records_have_the_same_hash() -> None:
    first = {"candidate_id": "B0-reference", "values": [1, 2]}
    second = {"values": [1, 2], "candidate_id": "B0-reference"}

    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_artifact_save_is_atomic_canonical_and_hash_checked(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    record = {"schema_version": 1, "candidate_id": "B0-reference"}

    digest = save_json_artifact(record, path)
    first = path.read_bytes()

    assert load_json_artifact(path, expected_sha256=digest) == record
    assert first.endswith(b"\n") and b"\r\n" not in first
    save_json_artifact(dict(reversed(tuple(record.items()))), path)
    assert path.read_bytes() == first

    with pytest.raises(ValueError, match="SHA-256"):
        load_json_artifact(path, expected_sha256="0" * 64)


@pytest.mark.parametrize("record", [[], "record", 1, None])
def test_provenance_requires_a_top_level_object(record) -> None:
    with pytest.raises(ValueError, match="object"):
        canonical_json_sha256(record)


@pytest.mark.parametrize(
    "expected_sha256",
    [True, "", "A" * 64, "a" * 63, "g" * 64],
)
def test_load_rejects_malformed_expected_hash(tmp_path, expected_sha256) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_json_artifact(path, expected_sha256=expected_sha256)


@pytest.mark.parametrize("text", ["[]", "null", "not-json"])
def test_load_rejects_invalid_or_non_object_json(tmp_path, text) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_json_artifact(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_save_preserves_existing_artifact(tmp_path, value) -> None:
    path = tmp_path / "artifact.json"
    original = {"status": "locked"}
    save_json_artifact(original, path)
    before = path.read_bytes()

    with pytest.raises(ValueError):
        save_json_artifact({"metric": value}, path)

    assert path.read_bytes() == before
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_save_and_load_reject_non_object_records(tmp_path) -> None:
    with pytest.raises(ValueError, match="object"):
        save_json_artifact([1, 2], tmp_path / "artifact.json")

    path = tmp_path / "artifact.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_json_artifact(path)
