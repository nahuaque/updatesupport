"""Frozen out-of-sample validation for calibrated public-report designs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any, Hashable, Mapping, Sequence

if TYPE_CHECKING:
    from .calibrated_design import CalibratedPublicReportDesign

from .artifacts import ReportArtifactMixin
from .breaking import MinimumClaimBreakingWitnessReport
from .claim import ClaimAudit, ClaimSpec
from .data import (
    GroupedProblem,
    _hashable_category,
    _iter_records,
    _record_value,
)
from .rollup import CategoricalRollupDesign


@dataclass(frozen=True)
class FrozenSupportCell:
    """One retained design-period hidden cell in a frozen claim policy."""

    state: tuple[Hashable, ...]
    public_value: tuple[Hashable, ...]
    mass: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "public_value": self.public_value,
            "mass": self.mass,
        }


@dataclass(frozen=True)
class FrozenClaimPolicy:
    """One selected and historically calibrated claim contract."""

    claim_index: int
    claim: ClaimSpec
    calibrated_radius: float
    reference_support: tuple[FrozenSupportCell, ...]

    @property
    def estimate_name(self) -> str:
        return self.claim.estimate_name

    @property
    def selected_public(self) -> tuple[str, ...]:
        return tuple(self.claim.public)

    @property
    def reference_public_cells(self) -> tuple[tuple[Hashable, ...], ...]:
        return tuple(
            sorted(
                {cell.public_value for cell in self.reference_support},
                key=str,
            )
        )

    @property
    def reference_hidden_cells(self) -> tuple[tuple[Hashable, ...], ...]:
        return tuple(cell.state for cell in self.reference_support)

    @property
    def reference_public_law(self) -> dict[tuple[Hashable, ...], float]:
        result: dict[tuple[Hashable, ...], float] = defaultdict(float)
        for cell in self.reference_support:
            result[cell.public_value] += cell.mass
        return dict(result)

    @property
    def reference_cell_weights(self) -> dict[tuple[Hashable, ...], float]:
        return {cell.state: cell.mass for cell in self.reference_support}

    @property
    def reference_public_map(
        self,
    ) -> dict[tuple[Hashable, ...], tuple[Hashable, ...]]:
        return {cell.state: cell.public_value for cell in self.reference_support}

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_index": self.claim_index,
            "estimate_name": self.estimate_name,
            "selected_public": self.selected_public,
            "calibrated_radius": self.calibrated_radius,
            "reference_public_cell_count": len(self.reference_public_cells),
            "reference_hidden_cell_count": len(self.reference_hidden_cells),
            "claim": self.claim.as_dict(),
            "reference_support": [cell.as_dict() for cell in self.reference_support],
        }


@dataclass(frozen=True)
class FrozenSupportDrift:
    """Observed composition movement relative to frozen design-period support."""

    calibrated_radius: float
    actual_tv_radius: float | None
    support_compatible: bool
    within_calibrated_radius: bool | None
    reference_public_cell_count: int
    current_public_cell_count: int
    reference_hidden_cell_count: int
    current_hidden_cell_count: int
    new_public_cells: tuple[tuple[Hashable, ...], ...] = ()
    missing_reference_public_cells: tuple[tuple[Hashable, ...], ...] = ()
    new_hidden_cells: tuple[tuple[Hashable, ...], ...] = ()
    missing_reference_hidden_cells: tuple[tuple[Hashable, ...], ...] = ()
    reason: str = ""

    @property
    def status(self) -> str:
        if not self.support_compatible:
            return "unsupported_support"
        if self.within_calibrated_radius is False:
            return "radius_breach"
        if self.missing_reference_hidden_cells:
            return "support_contraction"
        return "within_radius"

    @property
    def public_cell_change(self) -> int:
        return self.current_public_cell_count - self.reference_public_cell_count

    @property
    def hidden_cell_change(self) -> int:
        return self.current_hidden_cell_count - self.reference_hidden_cell_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "calibrated_radius": self.calibrated_radius,
            "actual_tv_radius": self.actual_tv_radius,
            "support_compatible": self.support_compatible,
            "within_calibrated_radius": self.within_calibrated_radius,
            "reference_public_cell_count": self.reference_public_cell_count,
            "current_public_cell_count": self.current_public_cell_count,
            "public_cell_change": self.public_cell_change,
            "reference_hidden_cell_count": self.reference_hidden_cell_count,
            "current_hidden_cell_count": self.current_hidden_cell_count,
            "hidden_cell_change": self.hidden_cell_change,
            "new_public_cells": self.new_public_cells,
            "missing_reference_public_cells": self.missing_reference_public_cells,
            "new_hidden_cells": self.new_hidden_cells,
            "missing_reference_hidden_cells": self.missing_reference_hidden_cells,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FrozenPolicyClaimResult:
    """One frozen claim evaluated on one future batch."""

    policy_claim: FrozenClaimPolicy
    drift: FrozenSupportDrift
    audit: ClaimAudit | None = None
    breaking_witness: MinimumClaimBreakingWitnessReport | None = None

    @property
    def claim_index(self) -> int:
        return self.policy_claim.claim_index

    @property
    def estimate_name(self) -> str:
        return self.policy_claim.estimate_name

    @property
    def status(self) -> str:
        if not self.drift.support_compatible or self.audit is None:
            return "inconclusive"
        if self.audit.inconclusive:
            return "inconclusive"
        if self.audit.failed or self.drift.within_calibrated_radius is False:
            return "review"
        return "pass"

    @property
    def certified(self) -> bool:
        return self.status == "pass"

    @property
    def breaking_tv_distance(self) -> float | None:
        if self.breaking_witness is None:
            return None
        return self.breaking_witness.witness_tv_distance

    @property
    def breaking_radius_multiple(self) -> float | None:
        distance = self.breaking_tv_distance
        radius = self.policy_claim.calibrated_radius
        if distance is None or radius <= 0.0:
            return None
        return distance / radius

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_index": self.claim_index,
            "estimate_name": self.estimate_name,
            "status": self.status,
            "certified": self.certified,
            "selected_public": self.policy_claim.selected_public,
            "calibrated_radius": self.policy_claim.calibrated_radius,
            "actual_tv_radius": self.drift.actual_tv_radius,
            "within_calibrated_radius": self.drift.within_calibrated_radius,
            "support_status": self.drift.status,
            "claim_audit_status": None if self.audit is None else self.audit.status,
            "observed_value": None if self.audit is None else self.audit.observed_value,
            "lower": None if self.audit is None else self.audit.interval.lower,
            "upper": None if self.audit is None else self.audit.interval.upper,
            "ambiguity": None if self.audit is None else self.audit.ambiguity,
            "decision_invariant": (
                None
                if self.audit is None or self.audit.decision is None
                else self.audit.decision.invariant
            ),
            "breaking_tv_distance": self.breaking_tv_distance,
            "breaking_radius_multiple": self.breaking_radius_multiple,
            "drift": self.drift.as_dict(),
            "audit": None if self.audit is None else self.audit.as_dict(),
            "breaking_witness": (
                None
                if self.breaking_witness is None
                else self.breaking_witness.as_dict()
            ),
        }

    def table_row(self) -> dict[str, Any]:
        payload = self.as_dict()
        return {
            key: value
            for key, value in payload.items()
            if key not in {"drift", "audit", "breaking_witness"}
        }


@dataclass(frozen=True)
class FrozenPolicyAudit(ReportArtifactMixin):
    """One future batch evaluated under a frozen public-report policy."""

    policy_name: str
    policy_fingerprint: str
    period: Hashable | None
    row_count: int
    claim_results: tuple[FrozenPolicyClaimResult, ...]
    unseen_rollup_categories: tuple[Hashable, ...] = ()
    transform_error: str | None = None
    title: str = "Frozen Public-Report Policy Audit"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim_results:
            raise ValueError("claim_results cannot be empty")
        object.__setattr__(self, "claim_results", tuple(self.claim_results))
        object.__setattr__(
            self,
            "unseen_rollup_categories",
            tuple(self.unseen_rollup_categories),
        )
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def status(self) -> str:
        statuses = {row.status for row in self.claim_results}
        if self.transform_error is not None or "inconclusive" in statuses:
            return "inconclusive"
        if "review" in statuses:
            return "review"
        return "pass"

    @property
    def claim_count(self) -> int:
        return len(self.claim_results)

    @property
    def pass_count(self) -> int:
        return sum(row.status == "pass" for row in self.claim_results)

    @property
    def review_count(self) -> int:
        return sum(row.status == "review" for row in self.claim_results)

    @property
    def inconclusive_count(self) -> int:
        return sum(row.status == "inconclusive" for row in self.claim_results)

    @property
    def support_compatible(self) -> bool:
        return self.transform_error is None and all(
            row.drift.support_compatible for row in self.claim_results
        )

    @property
    def radius_breach_count(self) -> int:
        return sum(
            row.drift.within_calibrated_radius is False for row in self.claim_results
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "policy_name": self.policy_name,
            "policy_fingerprint": self.policy_fingerprint,
            "period": self.period,
            "status": self.status,
            "row_count": self.row_count,
            "claim_count": self.claim_count,
            "pass_count": self.pass_count,
            "review_count": self.review_count,
            "inconclusive_count": self.inconclusive_count,
            "support_compatible": self.support_compatible,
            "radius_breach_count": self.radius_breach_count,
            "unseen_rollup_categories": self.unseen_rollup_categories,
            "transform_error": self.transform_error,
            "claim_results": [row.as_dict() for row in self.claim_results],
            "limitations": self.limitations,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        tables = {
            "summary": (
                {
                    key: value
                    for key, value in self.as_dict().items()
                    if key not in {"claim_results", "limitations"}
                },
            ),
            "claim_outcomes": tuple(row.table_row() for row in self.claim_results),
            "support_drift": tuple(
                {
                    "claim_index": row.claim_index,
                    "estimate_name": row.estimate_name,
                    **row.drift.as_dict(),
                }
                for row in self.claim_results
            ),
            "breaking_witnesses": tuple(
                {
                    "claim_index": row.claim_index,
                    "estimate_name": row.estimate_name,
                    **_without_nested_witness_rows(row.breaking_witness.as_dict()),
                }
                for row in self.claim_results
                if row.breaking_witness is not None
            ),
            "breaking_transfers": tuple(
                {
                    "claim_index": row.claim_index,
                    "estimate_name": row.estimate_name,
                    **transfer.as_dict(),
                }
                for row in self.claim_results
                if row.breaking_witness is not None
                for transfer in row.breaking_witness.transfers
            ),
            "limitations": tuple(
                {"limitation": limitation} for limitation in self.limitations
            ),
        }
        return tables

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Verdict",
            "",
            f"- Policy: {self.policy_name}",
            f"- Policy fingerprint: `{self.policy_fingerprint}`",
            f"- Period: {_format_period(self.period)}",
            f"- Status: **{self.status.upper()}**",
            f"- Rows: {self.row_count}",
            f"- Claims passed: {self.pass_count}/{self.claim_count}",
            f"- Claims requiring review: {self.review_count}",
            f"- Inconclusive claims: {self.inconclusive_count}",
            f"- Support compatible: {_yes_no(self.support_compatible)}",
            f"- Calibrated-radius breaches: {self.radius_breach_count}",
            "",
            "## Claim Outcomes",
            "",
            "| claim | calibrated TV | observed TV | ambiguity | audit | support | policy | nearest break | headroom |",
            "|:---|---:|---:|---:|:---:|:---:|:---:|---:|---:|",
        ]
        for row in self.claim_results:
            lines.append(
                "| "
                f"{row.estimate_name} | "
                f"{row.policy_claim.calibrated_radius:.4f} | "
                f"{_format_float(row.drift.actual_tv_radius)} | "
                f"{_format_float(None if row.audit is None else row.audit.ambiguity)} | "
                f"{'n/a' if row.audit is None else row.audit.status} | "
                f"{row.drift.status} | "
                f"{row.status} | "
                f"{_format_float(row.breaking_tv_distance)} | "
                f"{_format_multiple(row.breaking_radius_multiple)} |"
            )
        if self.transform_error is not None:
            lines.extend(
                [
                    "",
                    "## Unsupported Rollup Input",
                    "",
                    self.transform_error,
                ]
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "The policy reuses the selected public schema, rollup mapping, "
                "claim thresholds, and historical TV radii without recalibration "
                "or representation search. Observed TV compares this batch with "
                "the frozen design-period composition after restoring the "
                "design-period public law.",
                "",
                "`REVIEW` means a claim failed or the observed compatible shift "
                "exceeded its calibrated radius. `INCONCLUSIVE` means the frozen "
                "support contract could not evaluate the batch or an underlying "
                "claim audit was itself inconclusive.",
                "",
                "## Limitations",
                "",
            ]
        )
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines)


@dataclass(frozen=True)
class FrozenPolicyBacktest(ReportArtifactMixin):
    """Ordered holdout-period evaluation of one frozen policy."""

    policy_name: str
    policy_fingerprint: str
    period_column: str
    period_order: tuple[Hashable, ...]
    audits: tuple[FrozenPolicyAudit, ...]
    title: str = "Frozen Public-Report Policy Backtest"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.audits:
            raise ValueError("audits cannot be empty")
        object.__setattr__(self, "period_order", tuple(self.period_order))
        object.__setattr__(self, "audits", tuple(self.audits))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def period_count(self) -> int:
        return len(self.audits)

    @property
    def pass_count(self) -> int:
        return sum(row.status == "pass" for row in self.audits)

    @property
    def review_count(self) -> int:
        return sum(row.status == "review" for row in self.audits)

    @property
    def inconclusive_count(self) -> int:
        return sum(row.status == "inconclusive" for row in self.audits)

    @property
    def pass_rate(self) -> float:
        return self.pass_count / self.period_count

    @property
    def support_compatibility_rate(self) -> float:
        return sum(row.support_compatible for row in self.audits) / self.period_count

    @property
    def claim_evaluation_count(self) -> int:
        return sum(row.claim_count for row in self.audits)

    @property
    def claim_pass_rate(self) -> float:
        if self.claim_evaluation_count == 0:
            return 0.0
        return sum(row.pass_count for row in self.audits) / self.claim_evaluation_count

    @property
    def radius_breach_count(self) -> int:
        return sum(row.radius_breach_count for row in self.audits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "policy_name": self.policy_name,
            "policy_fingerprint": self.policy_fingerprint,
            "period_column": self.period_column,
            "period_order": self.period_order,
            "period_count": self.period_count,
            "pass_count": self.pass_count,
            "review_count": self.review_count,
            "inconclusive_count": self.inconclusive_count,
            "pass_rate": self.pass_rate,
            "support_compatibility_rate": self.support_compatibility_rate,
            "claim_evaluation_count": self.claim_evaluation_count,
            "claim_pass_rate": self.claim_pass_rate,
            "radius_breach_count": self.radius_breach_count,
            "audits": [row.as_dict() for row in self.audits],
            "limitations": self.limitations,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "summary": (
                {
                    key: value
                    for key, value in self.as_dict().items()
                    if key not in {"audits", "limitations"}
                },
            ),
            "periods": tuple(
                {
                    "period": audit.period,
                    "status": audit.status,
                    "row_count": audit.row_count,
                    "claim_count": audit.claim_count,
                    "pass_count": audit.pass_count,
                    "review_count": audit.review_count,
                    "inconclusive_count": audit.inconclusive_count,
                    "support_compatible": audit.support_compatible,
                    "radius_breach_count": audit.radius_breach_count,
                    "unseen_rollup_categories": audit.unseen_rollup_categories,
                }
                for audit in self.audits
            ),
            "claim_period_outcomes": tuple(
                {
                    "period": audit.period,
                    **result.table_row(),
                }
                for audit in self.audits
                for result in audit.claim_results
            ),
            "support_drift": tuple(
                {
                    "period": audit.period,
                    "claim_index": result.claim_index,
                    "estimate_name": result.estimate_name,
                    **result.drift.as_dict(),
                }
                for audit in self.audits
                for result in audit.claim_results
            ),
            "limitations": tuple(
                {"limitation": limitation} for limitation in self.limitations
            ),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Summary",
            "",
            f"- Policy: {self.policy_name}",
            f"- Policy fingerprint: `{self.policy_fingerprint}`",
            f"- Holdout periods: {self.period_count}",
            f"- Period pass rate: {self.pass_rate:.1%}",
            f"- Claim pass rate: {self.claim_pass_rate:.1%}",
            f"- Support compatibility rate: {self.support_compatibility_rate:.1%}",
            f"- Calibrated-radius breaches: {self.radius_breach_count}",
            "",
            "## Period Outcomes",
            "",
            "| period | rows | status | claims passed | review | inconclusive | support compatible | radius breaches |",
            "|:---|---:|:---:|---:|---:|---:|:---:|---:|",
        ]
        for audit in self.audits:
            lines.append(
                "| "
                f"{audit.period} | "
                f"{audit.row_count} | "
                f"{audit.status} | "
                f"{audit.pass_count}/{audit.claim_count} | "
                f"{audit.review_count} | "
                f"{audit.inconclusive_count} | "
                f"{_yes_no(audit.support_compatible)} | "
                f"{audit.radius_breach_count} |"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "Every holdout period is evaluated against the same frozen "
                "design-period policy. No radius, rollup, schema, or threshold is "
                "re-estimated from holdout outcomes.",
                "",
                "## Limitations",
                "",
            ]
        )
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines)


@dataclass(frozen=True)
class FrozenPublicReportPolicy(ReportArtifactMixin):
    """Immutable reporting contract derived from a calibrated design."""

    name: str
    source_design_status: str
    source_period_column: str
    coverage: float
    historical_row_count: int
    design_row_count: int
    recommended_public: tuple[str, ...]
    claims: tuple[FrozenClaimPolicy, ...]
    rollup: CategoricalRollupDesign | None = None
    title: str = "Frozen Public-Report Policy"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claims:
            raise ValueError("claims cannot be empty")
        object.__setattr__(
            self,
            "recommended_public",
            tuple(self.recommended_public),
        )
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @classmethod
    def from_design(
        cls,
        design: CalibratedPublicReportDesign,
        *,
        require_certified: bool = True,
        title: str = "Frozen Public-Report Policy",
    ) -> FrozenPublicReportPolicy:
        """Freeze a calibrated design without refitting any component."""

        from .calibrated_design import CalibratedPublicReportDesign

        if not isinstance(design, CalibratedPublicReportDesign):
            raise TypeError("design must be a CalibratedPublicReportDesign")
        if require_certified and not design.all_claims_certified:
            raise ValueError(
                "cannot freeze a design that does not certify every claim; "
                "pass require_certified=False to retain a best-effort policy"
            )
        if design.recommended_public is None:
            raise ValueError("design has no recommended public representation")

        claims = tuple(_freeze_claim_result(result) for result in design.claim_results)
        limitations = (
            "The policy freezes a design-period contract; it does not update its "
            "rollup, public schema, target definition, thresholds, or TV radii.",
            "Observed TV isolates within-public recomposition by restoring the "
            "design-period public law. It does not measure changes in public "
            "bucket shares.",
            "New retained hidden cells, new public cells, missing positive-mass "
            "reference public fibers, or unseen rollup categories invalidate "
            "the calibrated support comparison and produce an inconclusive result.",
            "Target values are recompiled from each evaluated batch. The policy "
            "audits current reporting stability but does not attribute target or "
            "model drift to hidden composition.",
            "A calibrated-radius breach is a review trigger, not a probability "
            "statement or a statistical test.",
        )
        return cls(
            name=design.name,
            source_design_status=design.status,
            source_period_column=design.period_column,
            coverage=design.coverage,
            historical_row_count=design.historical_row_count,
            design_row_count=design.current_row_count,
            recommended_public=tuple(design.recommended_public),
            claims=claims,
            rollup=design.rollup,
            title=title,
            limitations=limitations,
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "source_design_status": self.source_design_status,
            "source_period_column": self.source_period_column,
            "coverage": self.coverage,
            "recommended_public": self.recommended_public,
            "claims": [claim.as_dict() for claim in self.claims],
            "rollup": None if self.rollup is None else self.rollup.as_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return sha256(encoded).hexdigest()[:16]

    def audit(
        self,
        data: Any,
        *,
        period: Hashable | None = None,
        include_breaking_witness: bool = True,
        threshold_margin: float = 1e-8,
        title: str = "Frozen Public-Report Policy Audit",
    ) -> FrozenPolicyAudit:
        """Evaluate one future batch without refitting the frozen policy."""

        records = tuple(_iter_records(data))
        if not records:
            raise ValueError("data must contain at least one row")

        unseen_categories = _unseen_rollup_categories(records, self.rollup)
        if unseen_categories:
            error = (
                f"Unseen categories in frozen rollup column "
                f"`{self.rollup.column}`: "
                + ", ".join(str(value) for value in unseen_categories)
            )
            claim_results = tuple(
                FrozenPolicyClaimResult(
                    policy_claim=claim,
                    drift=_unsupported_drift(
                        claim,
                        reason=error,
                    ),
                )
                for claim in self.claims
            )
            return FrozenPolicyAudit(
                policy_name=self.name,
                policy_fingerprint=self.fingerprint,
                period=period,
                row_count=len(records),
                claim_results=claim_results,
                unseen_rollup_categories=unseen_categories,
                transform_error=error,
                title=title,
                limitations=self.limitations,
            )

        transformed = records if self.rollup is None else self.rollup.transform(records)
        claim_results: list[FrozenPolicyClaimResult] = []
        for policy_claim in self.claims:
            audit = policy_claim.claim.audit(transformed)
            drift = _support_drift(
                policy_claim,
                audit.primary.grouped,
            )
            breaking_witness = None
            if (
                include_breaking_witness
                and drift.support_compatible
                and audit.claim.decision is not None
            ):
                breaking_witness = audit.breaking_witness(
                    threshold_margin=threshold_margin
                )
            claim_results.append(
                FrozenPolicyClaimResult(
                    policy_claim=policy_claim,
                    drift=drift,
                    audit=audit,
                    breaking_witness=breaking_witness,
                )
            )
        return FrozenPolicyAudit(
            policy_name=self.name,
            policy_fingerprint=self.fingerprint,
            period=period,
            row_count=len(records),
            claim_results=tuple(claim_results),
            title=title,
            limitations=self.limitations,
        )

    def backtest(
        self,
        data: Any,
        *,
        period: str,
        period_order: Sequence[Hashable] | None = None,
        include_breaking_witness: bool = False,
        threshold_margin: float = 1e-8,
        title: str = "Frozen Public-Report Policy Backtest",
    ) -> FrozenPolicyBacktest:
        """Apply the unchanged policy to ordered holdout periods."""

        if not isinstance(period, str) or not period:
            raise ValueError("period must be a non-empty column name")
        if any(
            period in policy_claim.claim.hidden or period in policy_claim.claim.public
            for policy_claim in self.claims
        ):
            raise ValueError(
                "period must not be part of the frozen hidden/public representation"
            )
        records = tuple(_iter_records(data))
        if not records:
            raise ValueError("data must contain at least one row")

        rows_by_period: dict[Hashable, list[Mapping[str, Any]]] = defaultdict(list)
        observed_order: list[Hashable] = []
        for row_number, row in enumerate(records, start=1):
            period_value = _hashable_category(
                _record_value(row, period, row_number=row_number)
            )
            if period_value not in rows_by_period:
                observed_order.append(period_value)
            rows_by_period[period_value].append(row)
        ordered_periods = _resolve_period_order(observed_order, period_order)
        audits = tuple(
            self.audit(
                rows_by_period[period_value],
                period=period_value,
                include_breaking_witness=include_breaking_witness,
                threshold_margin=threshold_margin,
                title=f"{self.name} Policy Audit: {period_value}",
            )
            for period_value in ordered_periods
        )
        return FrozenPolicyBacktest(
            policy_name=self.name,
            policy_fingerprint=self.fingerprint,
            period_column=period,
            period_order=ordered_periods,
            audits=audits,
            title=title,
            limitations=(
                *self.limitations,
                "Holdout periods are evaluated independently against the frozen "
                "design-period baseline, not against a rolling or recalibrated "
                "reference period.",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "name": self.name,
            "fingerprint": self.fingerprint,
            "source_design_status": self.source_design_status,
            "source_period_column": self.source_period_column,
            "coverage": self.coverage,
            "historical_row_count": self.historical_row_count,
            "design_row_count": self.design_row_count,
            "recommended_public": self.recommended_public,
            "claim_count": len(self.claims),
            "rollup_applied": (
                self.rollup is not None and self.rollup.uses_rollup_column
            ),
            "rollup": None if self.rollup is None else self.rollup.as_dict(),
            "claims": [claim.as_dict() for claim in self.claims],
            "limitations": self.limitations,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        tables = {
            "summary": (
                {
                    key: value
                    for key, value in self.as_dict().items()
                    if key not in {"rollup", "claims", "limitations"}
                },
            ),
            "claims": tuple(
                {
                    key: value
                    for key, value in claim.as_dict().items()
                    if key not in {"claim", "reference_support"}
                }
                for claim in self.claims
            ),
            "reference_support": tuple(
                {
                    "claim_index": claim.claim_index,
                    "estimate_name": claim.estimate_name,
                    **cell.as_dict(),
                }
                for claim in self.claims
                for cell in claim.reference_support
            ),
            "limitations": tuple(
                {"limitation": limitation} for limitation in self.limitations
            ),
        }
        if self.rollup is not None:
            tables["rollup_mapping"] = tuple(
                {
                    "category": category,
                    "group": group,
                }
                for category, group in self.rollup.selected_mapping.items()
            )
        return tables

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Contract",
            "",
            f"- Policy: {self.name}",
            f"- Fingerprint: `{self.fingerprint}`",
            f"- Source design status: `{self.source_design_status}`",
            f"- Claims: {len(self.claims)}",
            f"- Historical calibration coverage: {self.coverage:.1%}",
            f"- Public representation: `{' + '.join(self.recommended_public)}`",
            "- Rollup mapping frozen: "
            f"{_yes_no(self.rollup is not None and self.rollup.uses_rollup_column)}",
            "",
            "## Frozen Claims",
            "",
            "| claim | public representation | calibrated TV | public cells | hidden cells |",
            "|:---|:---|---:|---:|---:|",
        ]
        for claim in self.claims:
            lines.append(
                "| "
                f"{claim.estimate_name} | "
                f"{' + '.join(claim.selected_public)} | "
                f"{claim.calibrated_radius:.4f} | "
                f"{len(claim.reference_public_cells)} | "
                f"{len(claim.reference_hidden_cells)} |"
            )
        lines.extend(
            [
                "",
                "## Use",
                "",
                "Apply `policy.audit(rows)` to one future batch or "
                "`policy.backtest(rows, period=...)` to ordered holdout periods. "
                "Neither operation changes this contract.",
                "",
                "## Limitations",
                "",
            ]
        )
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines)


def _freeze_claim_result(result: Any) -> FrozenClaimPolicy:
    grouped = result.audit.primary.grouped
    claim = replace(
        result.audit.claim,
        candidate_refinements=(),
        must_include=(),
        must_exclude=(),
        max_added_columns=None,
    )
    support = tuple(
        FrozenSupportCell(
            state=_as_cell(state),
            public_value=_as_cell(grouped.problem.public_map[state]),
            mass=float(grouped.cell_weights[state]),
        )
        for state in sorted(grouped.problem.states, key=str)
    )
    return FrozenClaimPolicy(
        claim_index=result.claim_index,
        claim=claim,
        calibrated_radius=float(result.calibrated_radius),
        reference_support=support,
    )


def _support_drift(
    policy_claim: FrozenClaimPolicy,
    current: GroupedProblem,
) -> FrozenSupportDrift:
    reference_states = set(policy_claim.reference_hidden_cells)
    current_states = set(_as_cell(state) for state in current.problem.states)
    reference_public = set(policy_claim.reference_public_cells)
    current_public = set(
        _as_cell(public_value) for public_value in current.problem.public_values
    )
    new_public = tuple(sorted(current_public - reference_public, key=str))
    missing_public = tuple(sorted(reference_public - current_public, key=str))
    new_hidden = tuple(sorted(current_states - reference_states, key=str))
    missing_hidden = tuple(sorted(reference_states - current_states, key=str))

    actual_tv = None
    reference_public_law = policy_claim.reference_public_law
    if not missing_public:
        restandardized: dict[tuple[Hashable, ...], float] = defaultdict(float)
        current_public_law = {
            _as_cell(key): float(value) for key, value in current.public_law.items()
        }
        for raw_state, mass in current.cell_weights.items():
            state = _as_cell(raw_state)
            public_value = _as_cell(current.problem.public_map[raw_state])
            reference_mass = reference_public_law.get(public_value, 0.0)
            if reference_mass <= current.problem.tol:
                continue
            current_mass = current_public_law[public_value]
            restandardized[state] += reference_mass * float(mass) / current_mass
        state_union = reference_states | set(restandardized)
        reference_weights = policy_claim.reference_cell_weights
        actual_tv = 0.5 * sum(
            abs(reference_weights.get(state, 0.0) - restandardized.get(state, 0.0))
            for state in state_union
        )

    support_compatible = not (new_public or missing_public or new_hidden)
    within_radius = (
        None
        if not support_compatible or actual_tv is None
        else actual_tv <= policy_claim.calibrated_radius + current.problem.tol
    )
    if new_public or new_hidden:
        reason = "the batch contains retained support absent from the frozen design"
    elif missing_public:
        reason = (
            "the batch is missing one or more positive-mass design-period public fibers"
        )
    elif within_radius is False:
        reason = "the compatible recomposition exceeds the calibrated TV radius"
    elif missing_hidden:
        reason = (
            "the batch contracts hidden support but remains evaluable under the "
            "frozen policy"
        )
    else:
        reason = "the batch remains within the frozen support and TV contract"
    return FrozenSupportDrift(
        calibrated_radius=policy_claim.calibrated_radius,
        actual_tv_radius=actual_tv,
        support_compatible=support_compatible,
        within_calibrated_radius=within_radius,
        reference_public_cell_count=len(reference_public),
        current_public_cell_count=len(current_public),
        reference_hidden_cell_count=len(reference_states),
        current_hidden_cell_count=len(current_states),
        new_public_cells=new_public,
        missing_reference_public_cells=missing_public,
        new_hidden_cells=new_hidden,
        missing_reference_hidden_cells=missing_hidden,
        reason=reason,
    )


def _unsupported_drift(
    policy_claim: FrozenClaimPolicy,
    *,
    reason: str,
) -> FrozenSupportDrift:
    return FrozenSupportDrift(
        calibrated_radius=policy_claim.calibrated_radius,
        actual_tv_radius=None,
        support_compatible=False,
        within_calibrated_radius=None,
        reference_public_cell_count=len(policy_claim.reference_public_cells),
        current_public_cell_count=0,
        reference_hidden_cell_count=len(policy_claim.reference_hidden_cells),
        current_hidden_cell_count=0,
        reason=reason,
    )


def _unseen_rollup_categories(
    records: Sequence[Mapping[str, Any]],
    rollup: CategoricalRollupDesign | None,
) -> tuple[Hashable, ...]:
    if rollup is None or not rollup.uses_rollup_column:
        return ()
    known = set(rollup.selected_mapping)
    unseen = {
        _hashable_category(_record_value(row, rollup.column, row_number=row_number))
        for row_number, row in enumerate(records, start=1)
    } - known
    return tuple(sorted(unseen, key=str))


def _resolve_period_order(
    observed_order: Sequence[Hashable],
    requested: Sequence[Hashable] | None,
) -> tuple[Hashable, ...]:
    observed = tuple(observed_order)
    if requested is None:
        try:
            return tuple(sorted(observed))
        except TypeError:
            return tuple(sorted(observed, key=str))
    normalized = tuple(_hashable_category(value) for value in requested)
    if len(set(normalized)) != len(normalized):
        raise ValueError("period_order must not contain duplicates")
    if set(normalized) != set(observed):
        raise ValueError("period_order must contain every observed period exactly once")
    return normalized


def _without_nested_witness_rows(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"cells", "transfers", "limitations"}
    }


def _as_cell(value: Hashable) -> tuple[Hashable, ...]:
    return value if isinstance(value, tuple) else (value,)


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_multiple(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def _format_period(value: Hashable | None) -> str:
    return "not supplied" if value is None else str(value)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
