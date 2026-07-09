"""Conformal-prediction reporting stability helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .artifacts import ReportArtifactMixin
from .claim import ClaimAudit, ClaimSpec, PublicReportDesign, claim


@dataclass(frozen=True)
class ConformalTargetSpec:
    """One conformal-derived aggregate target to audit."""

    column: str
    label: str
    role: str
    target_description: str
    ambiguity_limit: float | None = None

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("column must be non-empty")
        if not self.label:
            raise ValueError("label must be non-empty")
        if not self.role:
            raise ValueError("role must be non-empty")
        if not self.target_description:
            raise ValueError("target_description must be non-empty")
        if self.ambiguity_limit is not None and self.ambiguity_limit < 0:
            raise ValueError("ambiguity_limit must be non-negative")

    @classmethod
    def from_value(
        cls,
        value: str | Mapping[str, Any] | "ConformalTargetSpec",
        *,
        ambiguity_limit: float | None = None,
    ) -> "ConformalTargetSpec":
        """Build a conformal target spec from a string, mapping, or spec."""

        if isinstance(value, ConformalTargetSpec):
            if ambiguity_limit is None:
                return value
            return cls(
                column=value.column,
                label=value.label,
                role=value.role,
                target_description=value.target_description,
                ambiguity_limit=ambiguity_limit,
            )
        if isinstance(value, str):
            return cls(
                column=value,
                label=value,
                role="custom",
                target_description=f"aggregate value of `{value}`",
                ambiguity_limit=ambiguity_limit,
            )
        if isinstance(value, Mapping):
            payload = dict(value)
            if ambiguity_limit is not None and "ambiguity_limit" not in payload:
                payload["ambiguity_limit"] = ambiguity_limit
            return cls(**payload)
        raise TypeError("conformal targets must be strings, mappings, or specs")

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "label": self.label,
            "role": self.role,
            "target_description": self.target_description,
            "ambiguity_limit": self.ambiguity_limit,
        }


@dataclass(frozen=True)
class ConformalTargetAudit:
    """Audit output for one conformal-derived target column."""

    spec: ConformalTargetSpec
    claim: ClaimSpec
    design: PublicReportDesign

    @property
    def audit(self) -> ClaimAudit:
        return self.design.audit

    @property
    def status(self) -> str:
        return self.audit.status

    @property
    def design_status(self) -> str:
        return self.design.status

    @property
    def observed_value(self) -> float:
        return self.audit.observed_value

    @property
    def lower(self) -> float:
        return self.audit.interval.lower

    @property
    def upper(self) -> float:
        return self.audit.interval.upper

    @property
    def ambiguity(self) -> float:
        return self.audit.ambiguity

    @property
    def public_adequate(self) -> bool:
        return self.audit.primary.public_adequate

    @property
    def recommended_public(self) -> tuple[str, ...] | None:
        return self.design.recommended_public

    @property
    def recommended_label(self) -> str | None:
        return self.design.recommended_label

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.spec.as_dict(),
            "status": self.status,
            "design_status": self.design_status,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "public_adequate": self.public_adequate,
            "recommended_public": self.recommended_public,
            "recommended_label": self.recommended_label,
            "claim": self.claim.as_dict(),
            "design": self.design.as_dict(),
        }


@dataclass(frozen=True)
class ConformalReportingStabilityReport(ReportArtifactMixin):
    """Multi-target stability report for conformal-prediction outputs."""

    target_audits: tuple[ConformalTargetAudit, ...]
    public: tuple[str, ...]
    hidden: tuple[str, ...]
    weight: str | None = None
    source: str = "conformal"
    source_rows: int | None = None
    metadata: Mapping[str, Any] | None = None
    title: str = "Conformal Reporting Stability"

    @property
    def target_count(self) -> int:
        return len(self.target_audits)

    @property
    def pass_count(self) -> int:
        return sum(row.status == "pass" for row in self.target_audits)

    @property
    def fail_count(self) -> int:
        return sum(row.status == "fail" for row in self.target_audits)

    @property
    def inconclusive_count(self) -> int:
        return sum(row.status == "inconclusive" for row in self.target_audits)

    @property
    def needs_refinement_count(self) -> int:
        return sum(
            row.design_status in {"repair_available", "representation_available"}
            for row in self.target_audits
        )

    @property
    def status(self) -> str:
        if not self.target_audits:
            return "empty"
        if self.fail_count == 0 and self.inconclusive_count == 0:
            return "pass"
        if self.needs_refinement_count:
            return "needs_refinement"
        if self.fail_count:
            return "fail"
        return "inconclusive"

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "status": self.status,
            "source": self.source,
            "source_rows": self.source_rows,
            "public": self.public,
            "hidden": self.hidden,
            "weight": self.weight,
            "target_count": self.target_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "inconclusive_count": self.inconclusive_count,
            "needs_refinement_count": self.needs_refinement_count,
            "metadata": dict(self.metadata or {}),
            "targets": [row.as_dict() for row in self.target_audits],
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "summary": (
                {
                    "title": self.title,
                    "status": self.status,
                    "source": self.source,
                    "source_rows": self.source_rows,
                    "public": self.public,
                    "hidden": self.hidden,
                    "weight": self.weight,
                    "target_count": self.target_count,
                    "pass_count": self.pass_count,
                    "fail_count": self.fail_count,
                    "inconclusive_count": self.inconclusive_count,
                    "needs_refinement_count": self.needs_refinement_count,
                },
            ),
            "targets": tuple(_target_row(row) for row in self.target_audits),
            "refinement_recommendations": tuple(
                {
                    "target": row.spec.column,
                    "target_label": row.spec.label,
                    **recommendation.as_dict(),
                }
                for row in self.target_audits
                for recommendation in row.audit.recommend_refinements()
            ),
            "limitations": tuple(
                {
                    "target": row.spec.column,
                    "target_label": row.spec.label,
                    "limitation": limitation,
                }
                for row in self.target_audits
                for limitation in row.audit.limitations
            ),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Summary",
            "",
            f"- Source: `{self.source}`",
            f"- Source rows: {_format_optional_int(self.source_rows)}",
            f"- Public columns: `{_column_label(self.public)}`",
            f"- Hidden columns: `{_column_label(self.hidden)}`",
            f"- Targets audited: {self.target_count}",
            (
                f"- Target outcomes: {self.pass_count} pass, {self.fail_count} fail, "
                f"{self.inconclusive_count} inconclusive"
            ),
            f"- Overall status: `{self.status}`",
            "",
            "## Interpretation",
            "",
            "Conformal prediction quantifies row-level model uncertainty through "
            "intervals or prediction sets. This report audits aggregate "
            "conformal-derived quantities under hidden subgroup recomposition: "
            "the public distribution is held fixed while retained, not-publicly-"
            "reported cells inside each public bucket may shift according to the "
            "declared Q stress test.",
            "",
            "These intervals are hidden-composition stability intervals, not "
            "conformal prediction intervals and not statistical confidence "
            "intervals.",
            "",
            "## Target Audits",
            "",
        ]
        if not self.target_audits:
            lines.append("No conformal target columns were available to audit.")
            return "\n".join(lines)

        lines.extend(
            [
                "| target | role | status | observed | lower | upper | ambiguity | public adequate | recommended public |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | --- |",
            ]
        )
        for row in self.target_audits:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_escape_table(row.spec.label)}`",
                        _escape_table(row.spec.role),
                        row.status,
                        f"{row.observed_value:.4f}",
                        f"{row.lower:.4f}",
                        f"{row.upper:.4f}",
                        f"{row.ambiguity:.4f}",
                        "yes" if row.public_adequate else "no",
                        "`" + _escape_table(row.recommended_label or "") + "`",
                    ]
                )
                + " |"
            )

        recommendations = [
            (row, recommendation)
            for row in self.target_audits
            for recommendation in row.audit.recommend_refinements(top=3)
        ]
        if recommendations:
            lines.extend(["", "## Top Refinement Signals", ""])
            lines.extend(
                [
                    "| target | refinement | after ambiguity | reduction | signal |",
                    "| --- | --- | ---: | ---: | --- |",
                ]
            )
            for row, recommendation in recommendations:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{_escape_table(row.spec.label)}`",
                            f"`{_escape_table(recommendation.label)}`",
                            f"{recommendation.after_ambiguity:.4f}",
                            _format_optional_float(recommendation.reduction),
                            _escape_table(recommendation.reason),
                        ]
                    )
                    + " |"
                )

        return "\n".join(lines)


def conformal_reporting_stability(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    weight: str | None = None,
    targets: Sequence[str | Mapping[str, Any] | ConformalTargetSpec] | None = None,
    candidate_refinements: Sequence[str] = (),
    ambiguity_limit: float | None = None,
    ambiguity_limits: Mapping[str, float] | None = None,
    q: Any | None = None,
    q_presets: Sequence[Any] = ("saturated",),
    min_cell_weight: float = 1.0,
    top: int = 10,
    include_attribution: bool = False,
    title: str = "Conformal Reporting Stability",
    **claim_kwargs: Any,
) -> ConformalReportingStabilityReport:
    """Audit useful conformal-derived aggregate targets in one report.

    ``data`` may be a :class:`ConformalAdapterResult` or ordinary row-like
    data. When adapter metadata is available, target columns are discovered
    automatically. Otherwise pass ``targets`` explicitly.
    """

    rows, source, source_rows, metadata = _extract_conformal_rows(data)
    target_specs = _resolve_target_specs(
        data,
        targets=targets,
        ambiguity_limit=ambiguity_limit,
        ambiguity_limits=ambiguity_limits,
    )

    target_audits: list[ConformalTargetAudit] = []
    for spec in target_specs:
        spec_claim = claim(
            spec.label,
            public=public,
            hidden=hidden,
            target=spec.column,
            weight=weight,
            q=q,
            q_presets=q_presets,
            candidate_refinements=candidate_refinements,
            ambiguity_limit=spec.ambiguity_limit,
            min_cell_weight=min_cell_weight,
            top=top,
            target_description=spec.target_description,
            observed_label=f"Observed {spec.label}",
            title=f"{spec.label} Claim Audit",
            **claim_kwargs,
        )
        design = spec_claim.design(
            rows,
            top=top,
            include_attribution=include_attribution,
            title=f"{spec.label} Public Report Design",
        )
        target_audits.append(
            ConformalTargetAudit(
                spec=spec,
                claim=spec_claim,
                design=design,
            )
        )

    return ConformalReportingStabilityReport(
        target_audits=tuple(target_audits),
        public=tuple(public),
        hidden=tuple(hidden),
        weight=weight,
        source=source,
        source_rows=source_rows,
        metadata=metadata,
        title=title,
    )


def _extract_conformal_rows(
    data: Any,
) -> tuple[Any, str, int | None, Mapping[str, Any]]:
    rows = getattr(data, "rows", data)
    source = str(getattr(data, "source", "conformal"))
    source_rows = getattr(data, "source_rows", None)
    metadata = getattr(data, "metadata", None)
    return rows, source, source_rows, dict(metadata or {})


def _resolve_target_specs(
    data: Any,
    *,
    targets: Sequence[str | Mapping[str, Any] | ConformalTargetSpec] | None,
    ambiguity_limit: float | None,
    ambiguity_limits: Mapping[str, float] | None,
) -> tuple[ConformalTargetSpec, ...]:
    limits = dict(ambiguity_limits or {})
    if targets is not None:
        return tuple(
            ConformalTargetSpec.from_value(
                target,
                ambiguity_limit=limits.get(
                    target.column
                    if isinstance(target, ConformalTargetSpec)
                    else target
                    if isinstance(target, str)
                    else str(target.get("column", "")),
                    ambiguity_limit,
                ),
            )
            for target in targets
        )
    discovered = _discover_conformal_targets(data)
    return tuple(
        ConformalTargetSpec(
            column=spec.column,
            label=spec.label,
            role=spec.role,
            target_description=spec.target_description,
            ambiguity_limit=limits.get(spec.column, ambiguity_limit),
        )
        for spec in discovered
    )


def _discover_conformal_targets(data: Any) -> tuple[ConformalTargetSpec, ...]:
    metadata = getattr(data, "metadata", {}) or {}
    kind = metadata.get("kind")
    specs: list[ConformalTargetSpec] = []
    if kind == "regression_interval":
        _append_if_present(
            specs,
            data,
            "prediction_column",
            label="mean prediction",
            role="prediction",
            description="aggregate point prediction",
        )
        _append_if_present(
            specs,
            data,
            "lower_column",
            label="mean lower conformal bound",
            role="lower_bound",
            description="aggregate lower conformal bound",
        )
        _append_if_present(
            specs,
            data,
            "upper_column",
            label="mean upper conformal bound",
            role="upper_bound",
            description="aggregate upper conformal bound",
        )
        _append_if_present(
            specs,
            data,
            "interval_width_column",
            label="mean interval width",
            role="interval_width",
            description="aggregate conformal interval width",
        )
        _append_if_present(
            specs,
            data,
            "covered_column",
            label="coverage rate",
            role="coverage",
            description="empirical conformal coverage rate",
        )
        _append_if_present(
            specs,
            data,
            "miscovered_column",
            label="miscoverage rate",
            role="miscoverage",
            description="empirical conformal miscoverage rate",
        )
        _append_if_present(
            specs,
            data,
            "crosses_threshold_column",
            label="threshold-crossing interval rate",
            role="threshold_crossing",
            description="share of intervals crossing the supplied threshold",
        )
    elif kind == "classification_prediction_set":
        _append_if_present(
            specs,
            data,
            "prediction_set_size_column",
            label="mean prediction-set size",
            role="prediction_set_size",
            description="aggregate conformal prediction-set size",
        )
        _append_if_present(
            specs,
            data,
            "ambiguous_set_column",
            label="ambiguous prediction-set rate",
            role="ambiguous_prediction_set",
            description="share of rows with multi-label conformal prediction sets",
        )
        _append_if_present(
            specs,
            data,
            "covered_column",
            label="prediction-set coverage rate",
            role="coverage",
            description="empirical prediction-set coverage rate",
        )
        _append_if_present(
            specs,
            data,
            "miscovered_column",
            label="prediction-set miscoverage rate",
            role="miscoverage",
            description="empirical prediction-set miscoverage rate",
        )
        contains_positive = metadata.get("contains_positive_label_column")
        if contains_positive is not None:
            specs.append(
                ConformalTargetSpec(
                    column=str(contains_positive),
                    label="positive-label containment rate",
                    role="positive_label_containment",
                    target_description=(
                        "share of prediction sets containing the supplied "
                        "positive label"
                    ),
                )
            )
    return tuple(specs)


def _append_if_present(
    specs: list[ConformalTargetSpec],
    data: Any,
    attr: str,
    *,
    label: str,
    role: str,
    description: str,
) -> None:
    column = getattr(data, attr, None)
    if column is not None:
        specs.append(
            ConformalTargetSpec(
                column=str(column),
                label=label,
                role=role,
                target_description=description,
            )
        )


def _target_row(row: ConformalTargetAudit) -> dict[str, Any]:
    return {
        "target": row.spec.column,
        "label": row.spec.label,
        "role": row.spec.role,
        "status": row.status,
        "design_status": row.design_status,
        "observed_value": row.observed_value,
        "lower": row.lower,
        "upper": row.upper,
        "ambiguity": row.ambiguity,
        "ambiguity_limit": row.spec.ambiguity_limit,
        "public_adequate": row.public_adequate,
        "recommended_public": row.recommended_public,
        "recommended_label": row.recommended_label,
    }


def _column_label(columns: Sequence[str]) -> str:
    return " + ".join(str(column) for column in columns)


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_optional_int(value: int | None) -> str:
    return "unknown" if value is None else str(value)
