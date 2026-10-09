"""Portable portfolio evidence, configuration, results, and offline replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import updatesupport as us
from updatesupport.exports import _json_ready

from ._snapshot_common import (
    canonical_json,
    decode_portable,
    encode_portable,
    mapping_changes,
    payload_digest,
    runtime_versions,
    validate_capture_references,
)
from .compilation import _restore_compiled_portfolio
from .headline import portfolio_headline_report


def _configuration(payload):
    config = dict(payload["configuration"])
    config["q"] = us.QSpec.from_value(decode_portable(config["q"])).to_preset()
    return config


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Canonical JSON bundle; replay performs no provider calls.

    Embedded facts are the caller's supplied evidence archive. Availability
    timestamps and capture hashes do not independently authenticate a vendor
    archive or prove that an upstream holdings page was complete.
    """

    payload_json: str

    def __post_init__(self):
        payload = json.loads(self.payload_json)
        required = {
            "schema",
            "portfolio",
            "configuration",
            "capture_references",
            "versions",
            "result",
            "result_sha256",
        }
        if set(payload) != required or payload["schema"] != 1:
            raise ValueError("invalid portfolio snapshot schema")
        validate_capture_references(
            payload["capture_references"],
            message="capture references must map names to SHA256 hex digests",
        )
        compiled = _restore_compiled_portfolio(payload["portfolio"])
        if canonical_json(compiled.as_dict()) != canonical_json(payload["portfolio"]):
            raise ValueError(
                "stored portfolio selection/coverage differs from recompilation"
            )
        config = _configuration(payload)
        # Validate the declarative claim even for an empty covered universe.
        from .headline import _claim, _validate_configuration

        _validate_configuration(config)
        if payload["result_sha256"] != payload_digest(payload["result"]):
            raise ValueError("stored result digest mismatch")
        result = payload["result"]
        if (
            result["headline"] != config["headline"]
            or result["scope"] != config["scope"]
            or canonical_json(result["coverage"])
            != canonical_json(compiled.coverage.as_dict())
        ):
            raise ValueError("stored result scope/evidence differs from inputs")
        if result["audit"] is not None:
            if canonical_json(result["audit"]["claim"]) != canonical_json(
                _json_ready(_claim(config).as_dict())
            ):
                raise ValueError("stored claim differs from replay configuration")
        object.__setattr__(self, "payload_json", canonical_json(payload))

    @classmethod
    def from_json(cls, text):
        return cls(text)

    def to_json(self, **kwargs):
        return json.dumps(self.as_dict(), **kwargs) if kwargs else self.payload_json

    def as_dict(self):
        return json.loads(self.payload_json)

    @property
    def fingerprint(self):
        return hashlib.sha256(self.payload_json.encode()).hexdigest()

    def replay(self):
        payload = self.as_dict()
        return portfolio_headline_report(
            _restore_compiled_portfolio(payload["portfolio"]), **_configuration(payload)
        )

    def runtime_differences(self):
        stored, current = self.as_dict()["versions"], runtime_versions()
        return mapping_changes(
            stored, current, before_label="stored", after_label="current"
        )


def capture_portfolio_snapshot(
    report, *, capture_references: Mapping[str, str] | None = None
):
    """Capture a headline report, its complete normalized evidence and Q spec."""
    config = dict(report.configuration)
    config["q"] = encode_portable(us.QSpec.from_value(config["q"]).as_dict())
    # The shared JSON exporter converts tuple state keys in core report objects.
    # Store the result as a structured artifact using the same export path.
    result = json.loads(report.to_json())
    payload = {
        "schema": 1,
        "portfolio": report.portfolio.as_dict(),
        "configuration": config,
        "capture_references": dict(capture_references or {}),
        "versions": runtime_versions(),
        "result": result,
        "result_sha256": payload_digest(result),
    }
    return PortfolioSnapshot(canonical_json(payload))


def compare_portfolio_snapshots(before: PortfolioSnapshot, after: PortfolioSnapshot):
    """Identify evidence/configuration/runtime/result changes without as-of inference."""
    left, right = before.as_dict(), after.as_dict()
    changes = {
        key: left[key] != right[key]
        for key in (
            "portfolio",
            "configuration",
            "capture_references",
            "versions",
            "result",
        )
    }
    return {
        "before": before.fingerprint,
        "after": after.fingerprint,
        "changed": any(changes.values()),
        "changes": changes,
    }
