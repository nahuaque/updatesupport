"""Portable portfolio evidence, configuration, results, and offline replay."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping

import updatesupport as us
from updatesupport.exports import _json_ready

from .compilation import (
    FundamentalObservation,
    PortfolioMetricPolicy,
    TaxonomyAssignment,
    compile_portfolio_evidence,
)
from .coverage import PortfolioPosition, PortfolioUniverse
from .evidence import DisclosureFact
from .headline import portfolio_headline_report
from .snapshots import _canonical, _versions


def _encode(value):
    """Preserve tuple-keyed Q matrices without converting states to strings."""
    if isinstance(value, Mapping):
        return {"mapping": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode(x) for x in value]}
    if isinstance(value, list):
        return [_encode(x) for x in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("portfolio snapshots require portable built-in Q values")


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"mapping"}:
            pairs = [(_decode(k), _decode(v)) for k, v in value["mapping"]]
            result = dict(pairs)
            if len(result) != len(pairs):
                raise ValueError("duplicate keys in portable Q mapping")
            return result
        if set(value) == {"tuple"}:
            return tuple(_decode(x) for x in value["tuple"])
        raise ValueError("invalid portable Q encoding")
    if isinstance(value, list):
        return [_decode(x) for x in value]
    return value


def _compile(payload):
    data = payload["portfolio"]
    raw_universe = dict(data["universe"])
    raw_universe["positions"] = [
        PortfolioPosition(**p) for p in raw_universe["positions"]
    ]
    return compile_portfolio_evidence(
        PortfolioUniverse(**raw_universe),
        observations=[
            FundamentalObservation(
                DisclosureFact(**x["fact"]),
                x["period_kind"],
                x["normalized_period_end"],
            )
            for x in data["observations"]
        ],
        taxonomy=[TaxonomyAssignment(**t) for t in data["taxonomy"]],
        policy=PortfolioMetricPolicy(**data["policy"]),
    )


def _configuration(payload):
    config = dict(payload["configuration"])
    config["q"] = us.QSpec.from_value(_decode(config["q"])).to_preset()
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
        if not isinstance(payload["capture_references"], dict) or any(
            not isinstance(k, str)
            or not isinstance(v, str)
            or not re.fullmatch(r"[0-9a-f]{64}", v)
            for k, v in payload["capture_references"].items()
        ):
            raise ValueError("capture references must map names to SHA256 hex digests")
        compiled = _compile(payload)
        if _canonical(compiled.as_dict()) != _canonical(payload["portfolio"]):
            raise ValueError(
                "stored portfolio selection/coverage differs from recompilation"
            )
        config = _configuration(payload)
        # Validate the declarative claim even for an empty covered universe.
        from .headline import _claim, _validate_configuration

        _validate_configuration(config)
        if (
            payload["result_sha256"]
            != hashlib.sha256(_canonical(payload["result"]).encode()).hexdigest()
        ):
            raise ValueError("stored result digest mismatch")
        result = payload["result"]
        if (
            result["headline"] != config["headline"]
            or result["scope"] != config["scope"]
            or _canonical(result["coverage"]) != _canonical(compiled.coverage.as_dict())
        ):
            raise ValueError("stored result scope/evidence differs from inputs")
        if result["audit"] is not None:
            if _canonical(result["audit"]["claim"]) != _canonical(
                _json_ready(_claim(config).as_dict())
            ):
                raise ValueError("stored claim differs from replay configuration")
        object.__setattr__(self, "payload_json", _canonical(payload))

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
        return portfolio_headline_report(_compile(payload), **_configuration(payload))

    def runtime_differences(self):
        stored, current = self.as_dict()["versions"], _versions()
        return {
            k: {"stored": stored.get(k), "current": current.get(k)}
            for k in set(stored) | set(current)
            if stored.get(k) != current.get(k)
        }


def capture_portfolio_snapshot(
    report, *, capture_references: Mapping[str, str] | None = None
):
    """Capture a headline report, its complete normalized evidence and Q spec."""
    config = dict(report.configuration)
    config["q"] = _encode(us.QSpec.from_value(config["q"]).as_dict())
    # The shared JSON exporter converts tuple state keys in core report objects.
    # Store the result as a structured artifact using the same export path.
    result = json.loads(report.to_json())
    payload = {
        "schema": 1,
        "portfolio": report.portfolio.as_dict(),
        "configuration": config,
        "capture_references": dict(capture_references or {}),
        "versions": _versions(),
        "result": result,
        "result_sha256": hashlib.sha256(_canonical(result).encode()).hexdigest(),
    }
    return PortfolioSnapshot(_canonical(payload))


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
