"""Versioned, data-only persistence for frozen reporting policies."""

from __future__ import annotations

from dataclasses import fields
import json
from math import isclose, isfinite
from typing import TYPE_CHECKING, Any, Mapping

from .claim import ClaimSpec
from .rollup import CategoricalRollupCandidate, CategoricalRollupDesign
from .spec import QSpec

if TYPE_CHECKING:
    from .policy import FrozenPublicReportPolicy

_FORMAT = "updatesupport.frozen_public_report_policy"
_VERSION = 1


def _plain(value: Any) -> Any:
    """Accept only values whose meaning survives ordinary JSON encoding."""

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError("portable policies require finite numbers")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("portable policy mappings require string keys")
        return {key: _plain(item) for key, item in value.items()}
    raise TypeError(
        f"portable policies do not support {type(value).__name__}; "
        "use JSON scalars, sequences, and string-keyed mappings"
    )


def _category(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_category(item) for item in value)
    if isinstance(value, Mapping):
        raise TypeError("policy categories must be JSON scalars or tuples")
    return _plain(value)


def _validate_claim(claim: ClaimSpec) -> None:
    if not isinstance(claim.target, str) or not claim.target:
        raise TypeError(
            "portable policies require column targets; RowMetric and "
            "ProcedureTarget callables cannot be restored from JSON"
        )
    for q in (claim.primary_q, *claim.q_presets):
        _plain(QSpec.from_value(q).as_dict())


def _validate_policy(policy: FrozenPublicReportPolicy) -> None:
    if not policy.recommended_public:
        raise ValueError("a portable policy must have a public representation")
    indices = [row.claim_index for row in policy.claims]
    if any(type(index) is not int or index <= 0 for index in indices):
        raise ValueError("policy claim indices must be positive integers")
    if len(set(indices)) != len(indices):
        raise ValueError("policy claim indices must be unique")
    if not isfinite(policy.coverage) or not 0 < policy.coverage <= 1:
        raise ValueError("policy calibration coverage must be in (0, 1]")
    for row in policy.claims:
        claim = row.claim
        _validate_claim(claim)
        if tuple(claim.public) != policy.recommended_public:
            raise ValueError("frozen claim public columns must match the policy")
        if claim.candidate_refinements or claim.must_include or claim.must_exclude:
            raise ValueError(
                "frozen claims must not contain refinement search requirements"
            )
        radius = row.calibrated_radius
        if not isfinite(radius) or not 0 <= radius <= 1:
            raise ValueError("calibrated TV radius must be finite and in [0, 1]")
        q = QSpec.from_value(claim.primary_q)
        if q.name != "tv_budget" or q.radius != radius:
            raise ValueError("frozen primary Q must match the calibrated TV radius")
        support = row.reference_support
        if not support or not isclose(
            sum(cell.mass for cell in support), 1.0, abs_tol=1e-9
        ):
            raise ValueError("reference support masses must sum to one")
        states = set()
        public_indices = [claim.hidden.index(column) for column in claim.public]
        for cell in support:
            if not isinstance(cell.state, tuple) or not isinstance(
                cell.public_value, tuple
            ):
                raise ValueError("reference states and public values must be tuples")
            _category(cell.state)
            _category(cell.public_value)
            if len(cell.state) != len(claim.hidden):
                raise ValueError("reference state dimensions must match hidden columns")
            if (
                tuple(cell.state[index] for index in public_indices)
                != cell.public_value
            ):
                raise ValueError("reference public values must match the hidden state")
            if cell.state in states:
                raise ValueError("reference support must not contain duplicate states")
            states.add(cell.state)
            if not isfinite(cell.mass) or cell.mass <= 0:
                raise ValueError("reference support masses must be finite and positive")
    if policy.rollup is not None:
        _validate_claim(policy.rollup.claim)
        categories = policy.rollup.categories
        if not isinstance(categories, tuple) or not categories:
            raise ValueError("rollup categories must be a nonempty tuple")
        for value in categories:
            _category(value)
        if len(set(categories)) != len(categories):
            raise ValueError("rollup categories must be unique")
        for candidate in (
            policy.rollup.selected,
            policy.rollup.base,
            *policy.rollup.best_by_group_count,
            *policy.rollup.frontier,
        ):
            if not isinstance(candidate.groups, tuple) or any(
                not isinstance(group, tuple) for group in candidate.groups
            ):
                raise ValueError("rollup groups must be tuples of categories")
            flattened = [value for group in candidate.groups for value in group]
            if (
                not candidate.groups
                or any(not group for group in candidate.groups)
                or len(flattened) != len(categories)
                or set(flattened) != set(categories)
            ):
                raise ValueError("rollup groups must partition the saved categories")


def policy_to_dict(policy: FrozenPublicReportPolicy) -> dict[str, Any]:
    _validate_policy(policy)
    return _plain(
        {
            "format": _FORMAT,
            "schema_version": _VERSION,
            **policy.as_dict(),
        }
    )


def _construct(cls: Any, payload: Mapping[str, Any], **overrides: Any) -> Any:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{cls.__name__} must be a mapping")
    values = {
        field.name: payload[field.name]
        for field in fields(cls)
        if field.name in payload
    }
    values.update(overrides)
    return cls(**values)


def _load_rollup(payload: Mapping[str, Any] | None) -> CategoricalRollupDesign | None:
    if payload is None:
        return None

    def candidate(row: Mapping[str, Any]) -> CategoricalRollupCandidate:
        return _construct(
            CategoricalRollupCandidate, row, groups=_category(row["groups"])
        )

    return _construct(
        CategoricalRollupDesign,
        payload,
        claim=ClaimSpec.from_dict(payload["claim"]),
        categories=_category(payload["categories"]),
        selected=candidate(payload["selected"]),
        base=candidate(payload["base"]),
        best_by_group_count=tuple(
            candidate(row) for row in payload["best_by_group_count"]
        ),
        frontier=tuple(candidate(row) for row in payload["frontier"]),
        limitations=tuple(payload["limitations"]),
    )


def policy_from_dict(payload: Mapping[str, Any]) -> FrozenPublicReportPolicy:
    from .policy import FrozenClaimPolicy, FrozenPublicReportPolicy, FrozenSupportCell

    if not isinstance(payload, Mapping) or payload.get("format") != _FORMAT:
        raise ValueError(
            "expected a versioned frozen policy from to_dict() or to_json()"
        )
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != _VERSION
    ):
        raise ValueError(
            f"unsupported frozen policy schema_version: {payload.get('schema_version')!r}"
        )
    _plain(payload)
    try:
        claims = tuple(
            _construct(
                FrozenClaimPolicy,
                row,
                claim=ClaimSpec.from_dict(row["claim"]),
                reference_support=tuple(
                    _construct(
                        FrozenSupportCell,
                        cell,
                        state=_category(cell["state"]),
                        public_value=_category(cell["public_value"]),
                    )
                    for cell in row["reference_support"]
                ),
            )
            for row in payload["claims"]
        )
        policy = _construct(
            FrozenPublicReportPolicy,
            payload,
            claims=claims,
            rollup=_load_rollup(payload["rollup"]),
        )
        restored = policy_to_dict(policy)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"invalid frozen policy contract: {exc}") from exc
    if restored["fingerprint"] != payload.get("fingerprint"):
        raise ValueError("frozen policy fingerprint mismatch")
    # Do not silently discard unknown fields, changed derived values, or omitted
    # settings that could acquire different defaults in a later package version.
    if json.dumps(restored, sort_keys=True) != json.dumps(
        _plain(payload), sort_keys=True
    ):
        raise ValueError(
            "frozen policy contains unknown, missing, or inconsistent fields"
        )
    return policy


def policy_from_json(text: str) -> FrozenPublicReportPolicy:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate policy JSON key: {key}")
            result[key] = value
        return result

    return policy_from_dict(json.loads(text, object_pairs_hook=unique_object))
