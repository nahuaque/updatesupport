"""Historically calibrated public-report design orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Hashable, Mapping, Sequence

from .artifacts import ReportArtifactMixin
from .breaking import MinimumClaimBreakingWitnessReport
from .calibration import HistoricalTVCalibrationReport, calibrate_tv_radius
from .claim import ClaimAudit, ClaimSpec, PublicReportDesign
from .data import _iter_records
from .portfolio import (
    ClaimPortfolio,
    SharedRepresentationDesign,
    design_shared_representation,
)
from .presets import q_saturated
from .rollup import CategoricalRollupDesign, design_categorical_rollup


@dataclass(frozen=True)
class CalibratedClaimDesignResult:
    """Selected-report outcome for one historically calibrated claim."""

    claim_index: int
    calibration: HistoricalTVCalibrationReport
    audit: ClaimAudit
    breaking_witness: MinimumClaimBreakingWitnessReport | None = None

    @property
    def estimate_name(self) -> str:
        return self.audit.claim.estimate_name

    @property
    def calibrated_radius(self) -> float:
        return self.calibration.calibrated_radius

    @property
    def selected_public(self) -> tuple[str, ...]:
        return tuple(self.audit.claim.public)

    @property
    def certified(self) -> bool:
        return self.audit.passed

    @property
    def breaking_tv_distance(self) -> float | None:
        if self.breaking_witness is None:
            return None
        return self.breaking_witness.witness_tv_distance

    @property
    def breaking_radius_multiple(self) -> float | None:
        distance = self.breaking_tv_distance
        if distance is None or self.calibrated_radius <= 0.0:
            return None
        return distance / self.calibrated_radius

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_index": self.claim_index,
            "estimate_name": self.estimate_name,
            "selected_public": self.selected_public,
            "calibrated_radius": self.calibrated_radius,
            "rolling_shift_coverage": self.calibration.rolling_shift_coverage,
            "rolling_target_coverage": self.calibration.rolling_target_coverage,
            "claim_status": self.audit.status,
            "certified": self.certified,
            "observed_value": self.audit.observed_value,
            "lower": self.audit.interval.lower,
            "upper": self.audit.interval.upper,
            "ambiguity": self.audit.ambiguity,
            "breaking_witness_status": None
            if self.breaking_witness is None
            else self.breaking_witness.status,
            "breaking_tv_distance": self.breaking_tv_distance,
            "breaking_radius_multiple": self.breaking_radius_multiple,
            "calibration": self.calibration.as_dict(),
            "audit": self.audit.as_dict(),
            "breaking_witness": None
            if self.breaking_witness is None
            else self.breaking_witness.as_dict(),
        }

    def table_row(self) -> dict[str, Any]:
        """Return a compact row for governance and review tables."""

        payload = self.as_dict()
        return {
            key: value
            for key, value in payload.items()
            if key not in {"calibration", "audit", "breaking_witness"}
        }


@dataclass(frozen=True)
class CalibratedPublicReportDesign(ReportArtifactMixin):
    """Calibrated stress, rollup, schema, and breaking-witness design report."""

    name: str
    period_column: str
    coverage: float
    historical_row_count: int
    current_row_count: int
    candidate_refinements: tuple[str, ...]
    claim_results: tuple[CalibratedClaimDesignResult, ...]
    public_design: PublicReportDesign | None = None
    shared_design: SharedRepresentationDesign | None = None
    rollup: CategoricalRollupDesign | None = None
    rollup_anchor_claim_index: int | None = None
    title: str = "Calibrated Public-Report Design"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim_results:
            raise ValueError("claim_results cannot be empty")
        if (self.public_design is None) == (self.shared_design is None):
            raise ValueError(
                "exactly one of public_design or shared_design must be supplied"
            )
        object.__setattr__(
            self, "candidate_refinements", tuple(self.candidate_refinements)
        )
        object.__setattr__(self, "claim_results", tuple(self.claim_results))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def design_kind(self) -> str:
        return "shared" if self.shared_design is not None else "single_claim"

    @property
    def claim_count(self) -> int:
        return len(self.claim_results)

    @property
    def certified_claim_count(self) -> int:
        return sum(row.certified for row in self.claim_results)

    @property
    def all_claims_certified(self) -> bool:
        return self.certified_claim_count == self.claim_count

    @property
    def status(self) -> str:
        if self.all_claims_certified:
            return "calibrated_design_found"
        if all(row.audit.inconclusive for row in self.claim_results):
            return "inconclusive"
        return "best_effort"

    @property
    def recommended_public(self) -> tuple[str, ...] | None:
        if self.shared_design is not None:
            return self.shared_design.recommended_public
        if self.public_design is None:
            return None
        return self.public_design.recommended_public

    @property
    def rollup_applied(self) -> bool:
        return self.rollup is not None and self.rollup.uses_rollup_column

    @property
    def calibrated_radii(self) -> tuple[float, ...]:
        return tuple(row.calibrated_radius for row in self.claim_results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "name": self.name,
            "status": self.status,
            "design_kind": self.design_kind,
            "period_column": self.period_column,
            "coverage": self.coverage,
            "historical_row_count": self.historical_row_count,
            "current_row_count": self.current_row_count,
            "claim_count": self.claim_count,
            "certified_claim_count": self.certified_claim_count,
            "all_claims_certified": self.all_claims_certified,
            "candidate_refinements": self.candidate_refinements,
            "recommended_public": self.recommended_public,
            "calibrated_radii": self.calibrated_radii,
            "rollup_applied": self.rollup_applied,
            "rollup_anchor_claim_index": self.rollup_anchor_claim_index,
            "rollup": None if self.rollup is None else self.rollup.as_dict(),
            "public_design": None
            if self.public_design is None
            else self.public_design.as_dict(),
            "shared_design": None
            if self.shared_design is None
            else self.shared_design.as_dict(),
            "claim_results": [row.as_dict() for row in self.claim_results],
            "limitations": self.limitations,
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Return flattened calibration, design, audit, and witness tables."""

        tables: dict[str, tuple[dict[str, Any], ...]] = {
            "summary": (
                {
                    "title": self.title,
                    "name": self.name,
                    "status": self.status,
                    "design_kind": self.design_kind,
                    "period_column": self.period_column,
                    "coverage": self.coverage,
                    "historical_row_count": self.historical_row_count,
                    "current_row_count": self.current_row_count,
                    "claim_count": self.claim_count,
                    "certified_claim_count": self.certified_claim_count,
                    "all_claims_certified": self.all_claims_certified,
                    "candidate_refinements": self.candidate_refinements,
                    "recommended_public": self.recommended_public,
                    "calibrated_radii": self.calibrated_radii,
                    "rollup_applied": self.rollup_applied,
                    "rollup_anchor_claim_index": self.rollup_anchor_claim_index,
                },
            ),
            "claim_outcomes": tuple(row.table_row() for row in self.claim_results),
            "calibration_transitions": tuple(
                {
                    "claim_index": result.claim_index,
                    "estimate_name": result.estimate_name,
                    **transition.as_dict(),
                }
                for result in self.claim_results
                for transition in result.calibration.transitions
            ),
            "calibration_backtests": tuple(
                {
                    "claim_index": result.claim_index,
                    "estimate_name": result.estimate_name,
                    **backtest.as_dict(),
                }
                for result in self.claim_results
                for backtest in result.calibration.backtests
            ),
            "breaking_witnesses": tuple(
                {
                    "claim_index": result.claim_index,
                    "estimate_name": result.estimate_name,
                    **_without_nested_witness_rows(result.breaking_witness.as_dict()),
                }
                for result in self.claim_results
                if result.breaking_witness is not None
            ),
            "breaking_transfers": tuple(
                {
                    "claim_index": result.claim_index,
                    "estimate_name": result.estimate_name,
                    **transfer.as_dict(),
                }
                for result in self.claim_results
                if result.breaking_witness is not None
                for transfer in result.breaking_witness.transfers
            ),
            "limitations": tuple(
                {"limitation": limitation} for limitation in self.limitations
            ),
        }
        if self.rollup is not None:
            tables.update(_prefixed_tables("rollup", self.rollup.to_tables()))
        if self.shared_design is not None:
            tables.update(
                _prefixed_tables("shared_design", self.shared_design.to_tables())
            )
        if self.public_design is not None:
            tables.update(
                _prefixed_tables("public_design", self.public_design.to_tables())
            )
        return tables

    def to_markdown(self) -> str:
        """Render the end-to-end calibrated design for an analyst review."""

        lines = [
            f"# {self.title}",
            "",
            "## Recommendation",
            "",
            f"- Design: {self.name}",
            f"- Status: **{self.status.upper()}**",
            f"- Claims certified: {self.certified_claim_count}/{self.claim_count}",
            f"- Historical calibration coverage target: {self.coverage:.1%}",
            f"- Recommended public representation: `{_columns(self.recommended_public)}`",
        ]
        if self.rollup is not None:
            lines.extend(
                [
                    f"- Categorical rollup: `{self.rollup.column}` -> "
                    f"`{self.rollup.output_column}`",
                    f"- Selected rollup groups: {self.rollup.selected.group_count}",
                    f"- Rollup applied to design: {_yes_no(self.rollup_applied)}",
                ]
            )

        lines.extend(
            [
                "",
                "## Claim Outcomes",
                "",
                "| claim | calibrated TV | selected ambiguity | status | nearest break | radius multiple |",
                "|:---|---:|---:|:---:|---:|---:|",
            ]
        )
        for row in self.claim_results:
            lines.append(
                "| "
                f"{row.estimate_name} | "
                f"{row.calibrated_radius:.4f} | "
                f"{row.audit.ambiguity:.4f} | "
                f"{row.audit.status} | "
                f"{_format_float(row.breaking_tv_distance)} | "
                f"{_format_multiple(row.breaking_radius_multiple)} |"
            )

        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "The stress radius is estimated from historical consecutive-period "
                "hidden recompositions after holding the earlier public law fixed. "
                "The public-representation search then uses that calibrated TV "
                "radius instead of an arbitrary stress budget.",
                "",
            ]
        )
        if self.shared_design is not None:
            lines.append(
                "One shared public schema is evaluated against every claim under "
                "its own calibrated target audit. The selected schema is therefore "
                "a common reporting contract, not a collection of incompatible "
                "claim-specific recommendations."
            )
        else:
            lines.append(
                "The selected schema is the smallest evaluated public report that "
                "supports the calibrated single-claim audit, when such a schema was "
                "found."
            )
        if self.rollup is not None:
            lines.extend(
                [
                    "",
                    "The categorical rollup is learned exactly under saturated Q "
                    "for the declared anchor claim, then the resulting mapping is "
                    "validated under calibrated TV stress for every claim. The "
                    "anchor rollup cannot certify the portfolio by itself.",
                ]
            )
        if any(row.breaking_witness is not None for row in self.claim_results):
            lines.extend(
                [
                    "",
                    "For threshold claims, `nearest break` is the direct minimum-TV "
                    "fixed-public recomposition that reaches the failing side of the "
                    "decision. `radius multiple` compares that distance with the "
                    "historically calibrated stress radius. It is descriptive "
                    "separation, not a failure probability.",
                ]
            )

        lines.extend(["", "## Assumptions And Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in self.limitations)
        return "\n".join(lines)


def design_calibrated_public_report(
    historical_data: Any,
    current_data: Any,
    claim_or_portfolio: ClaimSpec | ClaimPortfolio | Mapping[str, Any],
    *,
    period: str,
    period_order: Sequence[Hashable] | None = None,
    coverage: float = 0.90,
    min_train_transitions: int = 3,
    calibration_backend: str = "cvxpy",
    calibration_solver: str | None = None,
    calibration_solver_options: Mapping[str, Any] | None = None,
    candidate_refinements: Sequence[str] | None = None,
    max_added_columns: int | None = None,
    bucket_budget: int | None = None,
    max_evaluations: int | None = 4096,
    rollup_column: str | None = None,
    rollup_claim_index: int = 0,
    rollup_max_groups: int | None = None,
    rollup_max_categories: int = 9,
    rollup_output_column: str | None = None,
    threshold_margin: float = 1e-8,
    tolerance: float = 1e-9,
    title: str = "Calibrated Public-Report Design",
) -> CalibratedPublicReportDesign:
    """Compose historical TV calibration with public-report design intelligence."""

    source_name, source_claims, portfolio_candidates = _coerce_subject(
        claim_or_portfolio
    )
    history = tuple(_iter_records(historical_data))
    current = tuple(_iter_records(current_data))
    if not history:
        raise ValueError("historical_data must contain at least one row")
    if not current:
        raise ValueError("current_data must contain at least one row")
    if rollup_column is not None and (
        rollup_claim_index < 0 or rollup_claim_index >= len(source_claims)
    ):
        raise ValueError("rollup_claim_index is outside the claim collection")

    resolved_candidates = _resolve_candidates(
        source_claims,
        portfolio_candidates=portfolio_candidates,
        explicit=candidate_refinements,
    )
    working_claims = source_claims
    rollup = None
    if rollup_column is not None:
        anchor = replace(
            working_claims[rollup_claim_index],
            q=q_saturated(),
            q_presets=(q_saturated(),),
        )
        rollup = design_categorical_rollup(
            (*history, *current),
            anchor,
            column=rollup_column,
            max_groups=rollup_max_groups,
            max_categories=rollup_max_categories,
            bucket_budget=bucket_budget,
            output_column=rollup_output_column,
            tolerance=tolerance,
            title=f"{source_name} Categorical Rollup",
        )
        if rollup.uses_rollup_column:
            transformed = rollup.transform((*history, *current))
            history = transformed[: len(history)]
            current = transformed[len(history) :]
            working_claims = tuple(
                _claim_with_rollup(claim, rollup) for claim in working_claims
            )
            resolved_candidates = _rewrite_candidates(
                resolved_candidates,
                old=rollup.column,
                new=rollup.output_column,
            )
        elif rollup.column not in resolved_candidates:
            resolved_candidates = (*resolved_candidates, rollup.column)

    working_claims = tuple(
        replace(claim, candidate_refinements=resolved_candidates)
        for claim in working_claims
    )
    calibrations = tuple(
        calibrate_tv_radius(
            history,
            claim,
            period=period,
            period_order=period_order,
            coverage=coverage,
            min_train_transitions=min_train_transitions,
            backend=calibration_backend,
            solver=calibration_solver,
            solver_options=calibration_solver_options,
            tolerance=tolerance,
            title=f"{claim.estimate_name} Historical TV Calibration",
        )
        for claim in working_claims
    )
    calibrated_claims = tuple(
        replace(report.calibrated_claim, q=None, q_presets=(report.q,))
        for report in calibrations
    )

    public_design: PublicReportDesign | None = None
    shared_design: SharedRepresentationDesign | None = None
    if len(calibrated_claims) == 1:
        single_overrides = {}
        if max_added_columns is not None:
            single_overrides["max_added_columns"] = max_added_columns
        if bucket_budget is not None:
            single_overrides["bucket_budget"] = bucket_budget
        if max_evaluations is not None:
            single_overrides["max_evaluations"] = max_evaluations
        public_design = calibrated_claims[0].design(current, **single_overrides)
        selected_public = public_design.recommended_public or tuple(
            calibrated_claims[0].public
        )
        selected_claim = _selected_claim(public_design.audit.claim, selected_public)
        selected_audits = (selected_claim.audit(current),)
    else:
        calibrated_portfolio = ClaimPortfolio(
            claims=calibrated_claims,
            name=source_name,
            description=(
                claim_or_portfolio.description
                if isinstance(claim_or_portfolio, ClaimPortfolio)
                else None
            ),
            candidate_refinements=resolved_candidates,
        )
        shared_design = design_shared_representation(
            current,
            calibrated_portfolio,
            candidate_refinements=resolved_candidates,
            max_added_columns=max_added_columns,
            bucket_budget=bucket_budget,
            max_evaluations=max_evaluations,
            tolerance=tolerance,
            title=f"{source_name} Calibrated Shared Representation",
        )
        selected_audits = tuple(
            _selected_claim(claim, shared_design.recommended_public).audit(current)
            for claim in calibrated_claims
        )

    claim_results = tuple(
        CalibratedClaimDesignResult(
            claim_index=index,
            calibration=calibration,
            audit=audit,
            breaking_witness=(
                None
                if audit.claim.decision is None
                else audit.breaking_witness(threshold_margin=threshold_margin)
            ),
        )
        for index, (calibration, audit) in enumerate(
            zip(calibrations, selected_audits, strict=True),
            start=1,
        )
    )
    limitations = (
        "The calibrated radius is an empirical historical quantile, not a "
        "guarantee against future regime changes.",
        "Calibration, rollup selection, and representation search are conditional "
        "on the retained hidden columns and observed support.",
        "Each claim's TV radius is calibrated at its declared base public "
        "representation and held fixed across candidate refinements as a common "
        "stress severity; it is not re-estimated for each candidate schema and may "
        "therefore be conservative after refinement.",
        "When requested, one anchor claim chooses the global categorical rollup "
        "under saturated Q; every claim is subsequently revalidated under its "
        "calibrated TV stress test.",
        "The current and historical rows are both used to fit an optional rollup "
        "mapping and establish its category vocabulary, so the rollup step is "
        "design analysis rather than an out-of-sample validation exercise.",
        "Minimum breaking witnesses preserve the selected public law but do not "
        "impose the calibrated TV radius; comparing their TV distance with that "
        "radius is a sensitivity diagnostic, not a probability statement.",
        "Hidden means retained but not publicly reported, not statistically "
        "unobserved.",
    )
    return CalibratedPublicReportDesign(
        name=source_name,
        period_column=period,
        coverage=float(coverage),
        historical_row_count=len(history),
        current_row_count=len(current),
        candidate_refinements=resolved_candidates,
        claim_results=claim_results,
        public_design=public_design,
        shared_design=shared_design,
        rollup=rollup,
        rollup_anchor_claim_index=(rollup_claim_index + 1 if rollup else None),
        title=title,
        limitations=limitations,
    )


def _coerce_subject(
    value: ClaimSpec | ClaimPortfolio | Mapping[str, Any],
) -> tuple[str, tuple[ClaimSpec, ...], tuple[str, ...] | None]:
    if isinstance(value, ClaimPortfolio):
        candidates = (
            None
            if value.candidate_refinements is None
            else tuple(value.candidate_refinements)
        )
        return value.name, tuple(value.claims), candidates
    if isinstance(value, ClaimSpec):
        return value.estimate_name, (value,), tuple(value.candidate_refinements)
    if isinstance(value, Mapping):
        if "claims" in value:
            return _coerce_subject(ClaimPortfolio.from_dict(value))
        return _coerce_subject(ClaimSpec.from_dict(value))
    raise TypeError(
        "claim_or_portfolio must be a ClaimSpec, ClaimPortfolio, or mapping"
    )


def _resolve_candidates(
    claims: Sequence[ClaimSpec],
    *,
    portfolio_candidates: Sequence[str] | None,
    explicit: Sequence[str] | None,
) -> tuple[str, ...]:
    values = (
        explicit
        if explicit is not None
        else portfolio_candidates
        if portfolio_candidates is not None
        else tuple(column for claim in claims for column in claim.candidate_refinements)
    )
    return _unique_strings(values, name="candidate_refinements")


def _claim_with_rollup(
    claim: ClaimSpec,
    rollup: CategoricalRollupDesign,
) -> ClaimSpec:
    output = rollup.output_column
    hidden = tuple(claim.hidden)
    if rollup.column not in hidden:
        raise ValueError(
            f"rollup column {rollup.column!r} is missing from claim "
            f"{claim.estimate_name!r}"
        )
    hidden_sets = tuple(
        _append_unique(tuple(hidden_set), output)
        for hidden_set in (claim.hidden_sets or (hidden,))
    )
    return replace(
        claim,
        hidden=_append_unique(hidden, output),
        hidden_sets=hidden_sets,
        candidate_refinements=_rewrite_candidates(
            claim.candidate_refinements,
            old=rollup.column,
            new=output,
        ),
        must_include=_replace_name(claim.must_include, rollup.column, output),
        must_exclude=_replace_name(claim.must_exclude, rollup.column, output),
    )


def _selected_claim(claim: ClaimSpec, public: Sequence[str]) -> ClaimSpec:
    selected_public = tuple(public)
    remaining_candidates = tuple(
        column
        for column in claim.candidate_refinements
        if column not in selected_public
    )
    return replace(
        claim,
        public=selected_public,
        candidate_refinements=remaining_candidates,
        must_include=(),
    )


def _rewrite_candidates(
    values: Sequence[str],
    *,
    old: str,
    new: str,
) -> tuple[str, ...]:
    rewritten = _replace_name(values, old, new)
    return _append_unique(rewritten, new)


def _replace_name(values: Sequence[str], old: str, new: str) -> tuple[str, ...]:
    return _unique_strings(
        (new if value == old else value for value in values),
        name="columns",
    )


def _append_unique(values: Sequence[str], value: str) -> tuple[str, ...]:
    return tuple(values) if value in values else (*values, value)


def _unique_strings(values: Sequence[str] | Any, *, name: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must contain non-empty strings")
        if value not in result:
            result.append(value)
    return tuple(result)


def _prefixed_tables(
    prefix: str,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    return {
        f"{prefix}_{name}": tuple(dict(row) for row in rows)
        for name, rows in tables.items()
    }


def _without_nested_witness_rows(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"cells", "transfers", "limitations"}
    }


def _columns(columns: Sequence[str] | None) -> str:
    return "none found" if columns is None else " + ".join(columns)


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_multiple(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
