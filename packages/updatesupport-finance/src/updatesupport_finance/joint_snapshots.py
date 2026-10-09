"""Offline joint workflow replay and fixed schema/measurement contracts."""

from dataclasses import dataclass
import hashlib
import json

import updatesupport as us

from .joint_portfolio import JointPortfolioEvidence, joint_portfolio_report
from .compilation import _restore_compiled_portfolio
from ._snapshot_common import (
    canonical_json,
    decode_portable,
    encode_portable,
    mapping_changes,
    payload_digest,
    runtime_versions,
    validate_capture_references,
)


def _evidence(payload):
    result = JointPortfolioEvidence(
        {n: _restore_compiled_portfolio(p) for n, p in payload["metrics"].items()}
    )
    if canonical_json(result.as_dict()) != canonical_json(payload):
        raise ValueError("stored joint evidence differs from recompilation")
    return result


def _config(payload):
    config = decode_portable(payload)
    config["conditional_refinements"] = config.pop("refinements")
    return config


@dataclass(frozen=True)
class JointPortfolioSnapshot:
    """Evidence, descriptors, domains, Q, schemas, results, and runtime versions.

    Hashes detect accidental corruption; they do not authenticate a provider
    archive or establish holdings completeness. Replay makes no provider calls.
    """

    payload_json: str

    def __post_init__(self):
        p = json.loads(self.payload_json)
        if (
            set(p)
            != {
                "format",
                "schema",
                "evidence",
                "configuration",
                "result",
                "result_sha256",
                "capture_references",
                "versions",
            }
            or p["schema"] != 1
            or p["format"] != "updatesupport_finance.joint_portfolio"
        ):
            raise ValueError("invalid joint snapshot schema")
        validate_capture_references(
            p["capture_references"], message="capture references must be SHA256 digests"
        )
        _evidence(p["evidence"])
        if payload_digest(p["result"]) != p["result_sha256"]:
            raise ValueError("stored joint result digest mismatch")
        # Recompute the complete finite result to validate configuration and
        # evidence/result agreement. Comparison across different runtimes is
        # also available explicitly through replay_matches().
        replay = joint_portfolio_report(
            _evidence(p["evidence"]), **_config(p["configuration"])
        )
        if canonical_json(
            json.loads(replay.to_json())["configuration"]
        ) != canonical_json(p["result"]["configuration"]) or canonical_json(
            replay.as_dict()["scopes"]
        ) != canonical_json(p["result"]["scopes"]):
            raise ValueError(
                "stored joint result scope/configuration differs from inputs"
            )
        object.__setattr__(self, "payload_json", canonical_json(p))

    @classmethod
    def from_json(cls, text):
        return cls(text)

    def as_dict(self):
        return json.loads(self.payload_json)

    def to_json(self, **kwargs):
        return (
            json.dumps(self.as_dict(), allow_nan=False, **kwargs)
            if kwargs
            else self.payload_json
        )

    @property
    def fingerprint(self):
        return hashlib.sha256(self.payload_json.encode()).hexdigest()

    def replay(self):
        p = self.as_dict()
        return joint_portfolio_report(
            _evidence(p["evidence"]), **_config(p["configuration"])
        )

    def replay_matches(self):
        return canonical_json(json.loads(self.replay().to_json())) == canonical_json(
            self.as_dict()["result"]
        )

    def runtime_differences(self):
        stored, current = self.as_dict()["versions"], runtime_versions()
        return mapping_changes(
            stored, current, before_label="stored", after_label="current"
        )


def capture_joint_portfolio_snapshot(report, *, capture_references=None):
    result = json.loads(report.to_json())
    return JointPortfolioSnapshot(
        canonical_json(
            {
                "format": "updatesupport_finance.joint_portfolio",
                "schema": 1,
                "evidence": report.evidence.as_dict(),
                "configuration": encode_portable(dict(report.configuration)),
                "result": result,
                "result_sha256": payload_digest(result),
                "capture_references": dict(capture_references or {}),
                "versions": runtime_versions(),
            }
        )
    )


def compare_joint_portfolio_snapshots(left, right):
    a, b = left.as_dict(), right.as_dict()
    return {
        f"{key}_changed": canonical_json(a[key]) != canonical_json(b[key])
        for key in ("evidence", "configuration", "result", "versions")
    }


@dataclass(frozen=True)
class FrozenJointPortfolioContract:
    """Fixed report/Q/measurement policies with fresh per-batch coverage.

    No historical radius or historical taxonomy is inferred. New share classes
    retain issuer facts and separate holdings values before support aggregation.
    """

    snapshot: JointPortfolioSnapshot
    candidate_index: int
    contracts: dict

    def __post_init__(self):
        expected = self._contracts(self.snapshot.replay(), self.candidate_index)
        if set(expected) != set(self.contracts) or any(
            canonical_json(expected[n].as_dict())
            != canonical_json(self.contracts[n].as_dict())
            for n in expected
        ):
            raise ValueError("frozen contracts differ from the reference report")

    @staticmethod
    def _contracts(report, index):
        candidate = report.candidates[index]
        config = report.configuration
        hidden = (*config["hidden"], *(p["name"] for p in config["refinements"]))
        result = {}
        for n in report.evidence.metrics:
            qs = config["q_presets"][n]
            claim = us.claim(
                n,
                target=n,
                public=candidate["public_columns"],
                hidden=hidden,
                weight="weight",
                min_cell_weight=0,
                q=qs[0],
                q_presets=qs,
            )
            result[n] = us.FrozenReportContract.freeze(claim, report.rows)
        return result

    @classmethod
    def freeze(cls, report, *, candidate_index=None):
        if candidate_index is None:
            if report.selected is None:
                raise ValueError(
                    "choose a candidate explicitly for a raw or unmet report"
                )
            candidate_index = report.candidates.index(report.selected)
        if not isinstance(candidate_index, int) or not 0 <= candidate_index < len(
            report.candidates
        ):
            raise ValueError("invalid report candidate index")
        return cls(
            capture_joint_portfolio_snapshot(report),
            candidate_index,
            cls._contracts(report, candidate_index),
        )

    def audit(self, evidence: JointPortfolioEvidence | None):
        def inconclusive(reason, **details):
            return {"status": "inconclusive", "reason": reason, **details}

        if evidence is None:
            return inconclusive("no_snapshot")
        reference = self.snapshot.replay()
        config = reference.configuration
        if (
            set(evidence.metrics) != set(reference.evidence.metrics)
            or evidence.universe.currency != reference.evidence.universe.currency
            or evidence.universe.eligible_types
            != reference.evidence.universe.eligible_types
            or any(
                evidence.metrics[n].policy.as_dict()
                != reference.evidence.metrics[n].policy.as_dict()
                for n in evidence.metrics
            )
        ):
            return inconclusive("measurement_or_scope_definition_changed")

        def eligible(book):
            return [
                p
                for p in book.universe.positions
                if p.value and p.security_type in book.universe.eligible_types
            ]

        known_issuers = {
            p.issuer_id for p in eligible(reference.evidence) if p.issuer_id
        }
        old_unresolved = {
            p.position_id for p in eligible(reference.evidence) if not p.issuer_id
        }
        current_positions = eligible(evidence)
        previous = {p.position_id: p for p in reference.evidence.universe.positions}
        changed_mapping = [
            p.position_id
            for p in evidence.universe.positions
            if p.value
            and p.position_id in previous
            and previous[p.position_id].issuer_id is not None
            and p.issuer_id != previous[p.position_id].issuer_id
        ]
        if changed_mapping:
            return inconclusive("issuer_mapping_changed", position_ids=changed_mapping)
        changed_eligibility = [
            p.position_id
            for p in evidence.universe.positions
            if p.value
            and p.position_id in previous
            and p.security_type != previous[p.position_id].security_type
        ]
        if changed_eligibility:
            return inconclusive(
                "security_eligibility_changed", position_ids=changed_eligibility
            )
        unresolved = [
            p.position_id
            for p in current_positions
            if not p.issuer_id and p.position_id not in old_unresolved
        ]
        if unresolved:
            return inconclusive("unresolved_securities", position_ids=unresolved)
        new_issuers = sorted(
            {
                p.issuer_id
                for p in current_positions
                if p.issuer_id and p.issuer_id not in known_issuers
            }
        )
        if new_issuers:
            return inconclusive("new_issuers", issuer_ids=new_issuers)
        rows = evidence.rows
        try:
            for p in config["refinements"]:
                rows = us.ConditionalRefinement.from_dict(p).transform(rows)
            checks = {n: c.audit(rows) for n, c in self.contracts.items()}
        except KeyError:
            return inconclusive("descriptor_schema_changed")
        if any(c["status"] == "inconclusive" for c in checks.values()):
            return inconclusive("unsupported_support", support_checks=checks)
        selected = reference.candidates[self.candidate_index]
        metrics, criteria = {}, []
        for n, c in self.contracts.items():
            f = us.public_representation_frontier(
                rows,
                base_public=c.claim.public,
                hidden=c.claim.hidden,
                target=n,
                weight="weight",
                min_cell_weight=0,
                q_presets=[
                    us.QSpec.from_value(q).to_preset() for q in config["q_presets"][n]
                ],
                candidate_refinements=(),
            ).candidates[0]
            common = (
                min(s.lower for s in f.scenarios),
                max(s.upper for s in f.scenarios),
            )
            interval = (
                common
                if config["scope"] == "common"
                else evidence.lift_interval(n, common)
            )
            passed = []
            if n in config["ambiguity_limits"]:
                passed.append(
                    interval is not None
                    and interval[1] - interval[0]
                    <= config["ambiguity_limits"][n] + 1e-10
                )
            if n in config["decisions"]:
                rule = us.DecisionRule.from_value(config["decisions"][n])
                passed.append(
                    interval is not None
                    and rule.evaluate(interval[0])
                    == rule.evaluate(interval[1])
                    == rule.pass_label
                )
            criteria.extend(passed)
            metrics[n] = {
                "common_interval": common,
                "eligible_interval": evidence.lift_interval(n, common),
                "contract_met": all(passed) if passed else None,
                "scope": evidence.metric_scope(n),
            }
        return {
            "status": "pass"
            if criteria and all(criteria)
            else "review"
            if criteria
            else "evaluated_without_criteria",
            "reason": "frozen_contract_evaluated",
            "evaluation_scope": config["scope"],
            "ambiguity_limits": config["ambiguity_limits"],
            "decisions": config["decisions"],
            "public_columns": selected["public_columns"],
            "support_checks": checks,
            "metrics": metrics,
        }

    def as_dict(self):
        return {
            "format": "updatesupport_finance.frozen_joint_contract",
            "schema": 1,
            "snapshot": self.snapshot.as_dict(),
            "candidate_index": self.candidate_index,
            "contracts": {n: c.as_dict() for n, c in self.contracts.items()},
        }

    def to_json(self, **kwargs):
        return json.dumps(self.as_dict(), allow_nan=False, **kwargs)

    @classmethod
    def from_json(cls, text):
        p = json.loads(text)
        if (
            set(p) != {"format", "schema", "snapshot", "candidate_index", "contracts"}
            or p["format"] != "updatesupport_finance.frozen_joint_contract"
            or p["schema"] != 1
        ):
            raise ValueError("invalid frozen joint contract schema")
        return cls(
            JointPortfolioSnapshot(canonical_json(p["snapshot"])),
            p["candidate_index"],
            {
                n: us.FrozenReportContract.from_dict(c)
                for n, c in p["contracts"].items()
            },
        )
