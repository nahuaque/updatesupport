"""Shared serialization and provenance for the finance snapshot formats.

Workflow modules retain their own schema, evidence, and replay validation.
These helpers make no provider calls and do not interpret financial evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
from typing import Any

import updatesupport as us


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_digest(payload: Any) -> str:
    return sha256(canonical_json(payload).encode()).hexdigest()


def runtime_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("updatesupport", "updatesupport-finance", "numpy", "scipy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "uninstalled"
    # Version strings alone cannot distinguish unreleased editable checkouts.
    for package, root in (
        ("updatesupport", Path(us.__file__).parent),
        ("updatesupport-finance", Path(__file__).parent),
    ):
        digest = sha256()
        for source in sorted(root.rglob("*.py")):
            contents = source.read_bytes()
            digest.update(source.relative_to(root).as_posix().encode() + b"\0")
            digest.update(len(contents).to_bytes(8, "big"))
            digest.update(contents)
        result[f"{package}_source_sha256"] = digest.hexdigest()
    return result


def mapping_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    before_label: str = "before",
    after_label: str = "after",
) -> dict[str, Any]:
    return {
        key: {before_label: before.get(key), after_label: after.get(key)}
        for key in sorted(before.keys() | after.keys())
        if key not in before or key not in after or before[key] != after[key]
    }


def validate_capture_references(references: Any, *, message: str) -> None:
    if not isinstance(references, dict) or any(
        not isinstance(k, str)
        or not isinstance(v, str)
        or not re.fullmatch(r"[0-9a-f]{64}", v)
        for k, v in references.items()
    ):
        raise ValueError(message)


def encode_portable(value: Any) -> Any:
    """Preserve tuple-keyed Q matrices without converting states to strings."""
    if isinstance(value, Mapping):
        return {
            "mapping": [
                [encode_portable(k), encode_portable(v)] for k, v in value.items()
            ]
        }
    if isinstance(value, tuple):
        return {"tuple": [encode_portable(x) for x in value]}
    if isinstance(value, list):
        return [encode_portable(x) for x in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("portfolio snapshots require portable built-in Q values")


def decode_portable(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"mapping"}:
            pairs = [
                (decode_portable(k), decode_portable(v)) for k, v in value["mapping"]
            ]
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError("duplicate keys in portable Q mapping")
            return result
        if set(value) == {"tuple"}:
            return tuple(decode_portable(x) for x in value["tuple"])
        raise ValueError("invalid portable Q encoding")
    if isinstance(value, list):
        return [decode_portable(x) for x in value]
    return value
