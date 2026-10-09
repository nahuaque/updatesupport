"""Offline, versioned disclosure run bundles and explicit run comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import updatesupport as us

from ._snapshot_common import canonical_json, mapping_changes, runtime_versions
from .disclosure import DisclosureAuditPack, disclosure_audit_pack
from .evidence import DisclosureFact, _timestamp, validate_disclosure_evidence
from .measurements import validate_normalization_records


@dataclass(frozen=True)
class DisclosureSnapshot:
    """Immutable JSON bundle containing inputs, evidence, results, and versions.

    Snapshots never select or reconstruct vendor filing versions. An adapter
    must supply the chosen evidence; availability and context checks then guard
    the bundle. Replay uses the current installed runtime, whose versions can
    be compared with ``runtime_differences()``.
    """

    payload_json: str

    def __post_init__(self) -> None:
        payload = json.loads(self.payload_json)
        if (
            not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
        ):
            raise ValueError("unsupported disclosure snapshot schema_version")
        required = {
            "as_of",
            "facts",
            "problem",
            "target",
            "tier",
            "claim",
            "assumptions",
            "versions",
            "result",
        }
        if required - payload.keys():
            raise ValueError("incomplete disclosure snapshot")
        payload["as_of"] = _timestamp(payload["as_of"])
        problem = us.NamedLinearFeasibilityProblem(**payload["problem"])
        facts = tuple(DisclosureFact(**row) for row in payload["facts"])
        context = payload.get("context", {})
        if not isinstance(context, dict):
            raise ValueError("snapshot context must be a mapping")
        validate_normalization_records(context.get("normalizations", ()), facts)
        if "scope" in context:
            from .briefs import DisclosureScope

            DisclosureScope(**context["scope"])
        diagnostics = validate_disclosure_evidence(
            facts, problem=problem, as_of=payload["as_of"]
        )
        if diagnostics:
            raise ValueError(
                "invalid snapshot evidence: "
                + "; ".join(row.code for row in diagnostics)
            )
        if payload["target"] not in {row.name for row in problem.targets}:
            raise ValueError("snapshot references an unknown target")
        if payload["tier"] not in {row.name for row in problem.scenarios}:
            raise ValueError("snapshot references an unknown tier")
        if payload["claim"] is not None:
            claim = us.NamedLinearClaim(**payload["claim"])
            if claim.target != payload["target"] or claim.scenario != payload["tier"]:
                raise ValueError("snapshot claim must match target and tier")
        result = payload["result"]
        if (
            result["triangulation_report"]["problem"] != payload["problem"]
            or result["target"] != payload["target"]
            or result["tier"] != payload["tier"]
            or result["evidence"] != payload["facts"]
            or result["assumptions"] != payload["assumptions"]
            or (
                None
                if result["claim_audit"] is None
                else result["claim_audit"]["claim"]
            )
            != payload["claim"]
        ):
            raise ValueError("stored result does not match snapshot inputs")
        object.__setattr__(self, "payload_json", canonical_json(payload))

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), allow_nan=False, **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> DisclosureSnapshot:
        return cls(payload)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the complete canonical bundle, including stored results."""
        return sha256(self.payload_json.encode()).hexdigest()

    def runtime_differences(self) -> dict[str, dict[str, str | None]]:
        return mapping_changes(self.as_dict()["versions"], runtime_versions())

    def replay(self) -> DisclosureAuditPack:
        payload = self.as_dict()
        problem = us.NamedLinearFeasibilityProblem(**payload["problem"])
        return disclosure_audit_pack(
            us.solve_named_linear_feasibility(problem),
            target=payload["target"],
            tier=payload["tier"],
            claim=payload["claim"],
            evidence=[DisclosureFact(**row) for row in payload["facts"]],
            assumptions=payload["assumptions"],
        )


def capture_disclosure_snapshot(
    problem: us.NamedLinearFeasibilityProblem,
    *,
    facts: Sequence[DisclosureFact],
    as_of: str,
    target: str,
    tier: str,
    claim: us.NamedLinearClaim | None = None,
    assumptions: Sequence[str] = (),
    context: Mapping[str, Any] | None = None,
) -> DisclosureSnapshot:
    """Validate, solve, and capture an offline audit in one portable bundle."""
    cutoff = _timestamp(as_of)
    diagnostics = validate_disclosure_evidence(facts, problem=problem, as_of=cutoff)
    if diagnostics:
        raise ValueError(
            "invalid snapshot evidence: " + "; ".join(row.code for row in diagnostics)
        )
    if claim is not None and (claim.target != target or claim.scenario != tier):
        raise ValueError("snapshot claim must match target and tier")
    pack = disclosure_audit_pack(
        us.solve_named_linear_feasibility(problem),
        target=target,
        tier=tier,
        claim=claim,
        evidence=facts,
        assumptions=assumptions,
    )
    return DisclosureSnapshot(
        canonical_json(
            {
                "schema_version": 1,
                "as_of": cutoff,
                "facts": [row.as_dict() for row in facts],
                "problem": problem.as_dict(),
                "target": target,
                "tier": tier,
                "claim": None if claim is None else claim.as_dict(),
                "assumptions": list(assumptions),
                "versions": runtime_versions(),
                "result": pack.as_dict(),
                **({"context": dict(context)} if context is not None else {}),
            }
        )
    )


def compare_disclosure_snapshots(
    before: DisclosureSnapshot,
    after: DisclosureSnapshot,
) -> dict[str, Any]:
    """Separate evidence, model, assumptions, runtime, and result changes.

    Differences are descriptive, not causal attribution. Stable fact IDs match
    records; a revision with a new ID appears as a removal and an addition.
    Optional saved scopes guard period/cohort comparisons; missing scopes remain
    unassessed. This does not assign causes to observed differences.
    """
    left, right = before.as_dict(), after.as_dict()
    from .briefs import compare_disclosure_scopes

    comparability = compare_disclosure_scopes(
        left.get("context", {}).get("scope"), right.get("context", {}).get("scope")
    )
    ltarget = next(t for t in left["problem"]["targets"] if t["name"] == left["target"])
    rtarget = next(
        t for t in right["problem"]["targets"] if t["name"] == right["target"]
    )
    if any(ltarget[k] != rtarget[k] for k in ("expression", "unit", "scale")):
        comparability = {
            "status": "not_comparable",
            "comparable": False,
            "reasons": [*comparability["reasons"], "different target definitions"],
            "causal": False,
        }

    def named(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
        return {row[key]: row for row in rows}

    def verdict(payload: Mapping[str, Any]) -> str | None:
        audit = payload["result"]["claim_audit"]
        return None if audit is None else audit["verdict"]

    return {
        "before_fingerprint": before.fingerprint,
        "after_fingerprint": after.fingerprint,
        "comparability": comparability,
        "context_changes": mapping_changes(
            left.get("context", {}), right.get("context", {})
        ),
        "as_of": {"before": left["as_of"], "after": right["as_of"]},
        "fact_changes": mapping_changes(
            named(left["facts"], "fact_id"), named(right["facts"], "fact_id")
        ),
        "constraint_changes": mapping_changes(
            named(left["problem"]["constraints"], "name"),
            named(right["problem"]["constraints"], "name"),
        ),
        "model_changes": mapping_changes(
            {k: v for k, v in left["problem"].items() if k != "constraints"},
            {k: v for k, v in right["problem"].items() if k != "constraints"},
        ),
        "assumptions": {"before": left["assumptions"], "after": right["assumptions"]},
        "claim": {"before": left["claim"], "after": right["claim"]},
        "selection": {
            "before": [left["target"], left["tier"]],
            "after": [right["target"], right["tier"]],
        },
        "runtime_changes": mapping_changes(left["versions"], right["versions"]),
        "interval": {
            "before": left["result"]["interval"],
            "after": right["result"]["interval"],
        },
        "verdict": {"before": verdict(left), "after": verdict(right)},
    }
