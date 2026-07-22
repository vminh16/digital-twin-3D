from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _record(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("provenance record must be a JSON object")
    return value


def _canonical_bytes(record: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("provenance record must contain finite JSON values") from error


def canonical_json_sha256(record: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(_record(record))).hexdigest()


def save_json_artifact(record: Mapping[str, object], path: Path) -> str:
    validated = _record(record)
    digest = canonical_json_sha256(validated)
    try:
        text = (
            json.dumps(
                validated,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("provenance record must contain finite JSON values") from error

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_artifact(
    path: Path,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or _SHA256.fullmatch(expected_sha256) is None
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    try:
        parsed = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ValueError("artifact must contain a finite JSON object") from error
    record = dict(_record(parsed))
    if expected_sha256 is not None:
        actual = canonical_json_sha256(record)
        if not hmac.compare_digest(actual, expected_sha256):
            raise ValueError("artifact SHA-256 does not match expected SHA-256")
    return record
