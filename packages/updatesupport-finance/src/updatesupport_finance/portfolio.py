"""Portfolio compilers and report profiles for financial model-risk review."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import updatesupport as us

from .presets import finance_sensitivity_grid


DEFAULT_MODEL_RISK_LIMITATIONS = (
    "The report audits representation stability for a supplied portfolio risk "
    "metric; it does not validate PD, LGD, EAD, calibration, discrimination, "
    "backtesting, overrides, or governance controls.",
    "The hidden-composition interval is a sensitivity or partial-identification "
    "result under the selected Q stress test, not a sampling confidence interval.",
    "Statistical or model-estimation uncertainty is included only when supplied "
    "explicitly through report metadata or hidden-cell standard errors.",
    "Conclusions depend on the retained hidden state space, exposure weights, "
    "minimum-cell filtering, and the selected admissible-shift preset.",
    "Refinement recommendations identify reporting variables that reduce "
    "hidden-composition ambiguity; they are not automatically policy, pricing, "
    "underwriting, or causal adjustment recommendations.",
    "CVXPY dual diagnostics, when present, are local solver diagnostics rather "
    "than global guarantees for every possible constraint relaxation.",
)


@dataclass(frozen=True)
class ModelRiskMetadata:
    """Model-review metadata for a financial representation-stability report."""

    model_id: str | None = None
    portfolio_name: str | None = None
    as_of_date: str | None = None
    intended_use: str | None = None
    owner: str | None = None
    reviewer: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "model_id": self.model_id,
            "portfolio_name": self.portfolio_name,
            "as_of_date": self.as_of_date,
            "intended_use": self.intended_use,
            "owner": self.owner,
            "reviewer": self.reviewer,
        }


@dataclass(frozen=True)
class ReviewThresholds:
    """Simple model-risk review thresholds for report triage."""

    ambiguity_limit: float | None = None
    public_adequacy_required: bool = False

    def __post_init__(self) -> None:
        if self.ambiguity_limit is not None and self.ambiguity_limit < 0:
            raise ValueError("ambiguity_limit must be non-negative")

    def evaluate(
        self, core_report: us.PublicDescentReport
    ) -> tuple[str, tuple[str, ...]]:
        reasons = []
        if (
            self.ambiguity_limit is not None
            and core_report.interval.diameter > self.ambiguity_limit
        ):
            reasons.append(
                "transport ambiguity "
                f"{core_report.interval.diameter:.4f} exceeds limit "
                f"{self.ambiguity_limit:.4f}"
            )
        if self.public_adequacy_required and not core_report.public_adequate:
            reasons.append("public adequacy is required but was not satisfied")
        status = "attention required" if reasons else "pass"
        return status, tuple(reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ambiguity_limit": self.ambiguity_limit,
            "public_adequacy_required": self.public_adequacy_required,
        }


@dataclass(frozen=True)
class ModelRiskReport:
    """Finance-specific wrapper around a core public-descent report."""

    core: us.PublicDescentReport
    metadata: ModelRiskMetadata = ModelRiskMetadata()
    thresholds: ReviewThresholds = ReviewThresholds()
    statistical_uncertainty: us.StatisticalUncertainty | None = None
    composition_uncertainty: us.HiddenCompositionUncertaintyReport | None = None
    reviewer_notes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = DEFAULT_MODEL_RISK_LIMITATIONS

    def __post_init__(self) -> None:
        object.__setattr__(self, "reviewer_notes", tuple(self.reviewer_notes))
        object.__setattr__(self, "limitations", tuple(self.limitations))

    @property
    def review_status(self) -> str:
        status, _reasons = self.thresholds.evaluate(self.core)
        return status

    @property
    def review_reasons(self) -> tuple[str, ...]:
        _status, reasons = self.thresholds.evaluate(self.core)
        return reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.core.title,
            "review_status": self.review_status,
            "review_reasons": self.review_reasons,
            "metadata": self.metadata.as_dict(),
            "thresholds": self.thresholds.as_dict(),
            "reported_portfolio_risk_estimate": self.core.observed_value,
            "hidden_composition_lower": self.core.interval.lower,
            "hidden_composition_upper": self.core.interval.upper,
            "hidden_composition_ambiguity": self.core.interval.diameter,
            "concentration_stress": _concentration_stress_summary(self.core),
            "statistical_uncertainty": None
            if self.statistical_uncertainty is None
            else self.statistical_uncertainty.as_dict(),
            "composition_uncertainty": None
            if self.composition_uncertainty is None
            else self.composition_uncertainty.as_dict(),
            "estimator_uncertainty": None
            if self.core.estimator_uncertainty is None
            else self.core.estimator_uncertainty.as_dict(),
            "top_refinements": [row.as_dict() for row in self.core.refinements[:5]],
            "top_dual_diagnostics": [
                row.as_dict()
                for row in self.core.interval.dual_summary(top=8, min_magnitude=1e-8)
            ],
            "data_diagnostics": [row.as_dict() for row in self.core.diagnostics],
            "limitations": self.limitations,
            "reviewer_notes": self.reviewer_notes,
            "core": self.core.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return us.report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        tables = {
            "finance_model_risk": (
                {
                    "title": self.core.title,
                    "review_status": self.review_status,
                    "model_id": self.metadata.model_id,
                    "portfolio_name": self.metadata.portfolio_name,
                    "as_of_date": self.metadata.as_of_date,
                    "intended_use": self.metadata.intended_use,
                    "owner": self.metadata.owner,
                    "reviewer": self.metadata.reviewer,
                    "reported_portfolio_risk_estimate": self.core.observed_value,
                    "hidden_composition_lower": self.core.interval.lower,
                    "hidden_composition_upper": self.core.interval.upper,
                    "hidden_composition_ambiguity": self.core.interval.diameter,
                    "public_adequate": self.core.public_adequate,
                    "q_name": self.core.grouped.q_name,
                    "q_description": self.core.grouped.q_description,
                },
            )
        }
        tables["finance_review_reasons"] = tuple(
            {"reason": reason} for reason in self.review_reasons
        )
        tables["finance_concentration_stress"] = (
            _concentration_stress_summary(self.core),
        )
        if self.statistical_uncertainty is not None:
            tables["finance_statistical_uncertainty"] = (
                self.statistical_uncertainty.as_dict(),
            )
        if self.composition_uncertainty is not None:
            tables.update(
                {
                    f"finance_model_assisted_{name}": rows
                    for name, rows in self.composition_uncertainty.to_tables().items()
                }
            )
        if self.core.estimator_uncertainty is not None:
            tables["finance_estimator_uncertainty"] = (
                self.core.estimator_uncertainty.as_dict(),
            )
        tables["finance_refinement_recommendations"] = tuple(
            row.as_dict() for row in self.core.refinements
        )
        duals = self.core.interval.dual_summary(top=8, min_magnitude=1e-8)
        tables["finance_dual_diagnostics"] = tuple(row.as_dict() for row in duals)
        tables["finance_data_diagnostics"] = tuple(
            row.as_dict() for row in self.core.diagnostics
        )
        tables["finance_reviewer_notes"] = tuple(
            {"note": note} for note in self.reviewer_notes
        )
        tables["finance_limitations"] = tuple(
            {"limitation": limitation} for limitation in self.limitations
        )
        tables.update(
            {f"core_{name}": rows for name, rows in self.core.to_tables().items()}
        )
        return tables

    def to_dataframes(self) -> dict[str, Any]:
        return us.tables_to_dataframes(self.to_tables())

    def to_markdown(self) -> str:
        lines = [f"# {self.core.title}", ""]
        lines.extend(self._metadata_markdown())
        lines.extend(self._threshold_markdown())
        lines.extend(self._finance_interpretation_markdown())
        lines.extend(self._reported_estimate_markdown())
        lines.extend(self._uncertainty_markdown())
        lines.extend(self._model_assisted_uncertainty_markdown())
        lines.extend(self._hidden_ambiguity_markdown())
        lines.extend(self._concentration_stress_markdown())
        lines.extend(self._refinement_markdown())
        lines.extend(self._dual_diagnostics_markdown())
        lines.extend(self._data_diagnostics_markdown())
        lines.extend(self._limitations_and_notes_markdown())
        core_markdown = self.core.to_markdown().splitlines()
        if core_markdown and core_markdown[0].startswith("# "):
            core_markdown = core_markdown[2:]
        lines.extend(["", "## Core Update-Support Audit", ""])
        lines.extend(core_markdown)
        return "\n".join(lines)

    def _metadata_markdown(self) -> list[str]:
        rows = [
            ("Model ID", self.metadata.model_id),
            ("Portfolio", self.metadata.portfolio_name),
            ("As-of date", self.metadata.as_of_date),
            ("Intended use", self.metadata.intended_use),
            ("Owner", self.metadata.owner),
            ("Reviewer", self.metadata.reviewer),
        ]
        present = [(label, value) for label, value in rows if value]
        if not present:
            return []
        lines = ["## Model-Risk Context", ""]
        lines.extend(f"- {label}: {value}" for label, value in present)
        lines.append("")
        return lines

    def _threshold_markdown(self) -> list[str]:
        lines = [
            "## Review Status",
            "",
            f"- Status: {self.review_status}",
        ]
        if self.thresholds.ambiguity_limit is not None:
            lines.append(f"- Ambiguity limit: {self.thresholds.ambiguity_limit:.4f}")
        lines.append(
            "- Public adequacy required: "
            f"{'yes' if self.thresholds.public_adequacy_required else 'no'}"
        )
        if self.review_reasons:
            lines.append("- Reasons:")
            lines.extend(f"  - {reason}" for reason in self.review_reasons)
        lines.append("")
        return lines

    def _finance_interpretation_markdown(self) -> list[str]:
        core = self.core
        return [
            "## Financial Model-Risk Interpretation",
            "",
            "This report asks whether the reported public portfolio segmentation "
            "still supports the portfolio risk estimate when hidden composition "
            "inside those public segments is allowed to shift under the selected "
            "Q stress test.",
            "",
            f"- Reported portfolio risk estimate: {core.observed_value:.4f}",
            f"- Hidden-composition ambiguity: {core.interval.diameter:.4f}",
            "- Observed-law partial-ID interval: "
            f"[{core.interval.lower:.4f}, {core.interval.upper:.4f}]",
            f"- Public segmentation adequate: {'yes' if core.public_adequate else 'no'}",
            "",
            "A failing or attention-required status is not a statistical "
            "confidence result. It means the current public risk buckets may "
            "leave material hidden-composition ambiguity for this metric and "
            "stress-test choice.",
        ]

    def _reported_estimate_markdown(self) -> list[str]:
        core = self.core
        grouped = core.grouped
        return [
            "",
            "## Reported Portfolio Risk Estimate",
            "",
            f"- {core.observed_label}: {core.observed_value:.4f}",
            f"- Metric: aggregate {core.target_description}.",
            f"- Public segmentation: `{_column_label(grouped.public_columns)}`.",
            f"- Hidden state definition: `{_column_label(grouped.hidden_columns)}`.",
            f"- Retained hidden cells: {len(grouped.problem.states)}.",
            f"- Public cells: {len(grouped.problem.public_values)}.",
            f"- Exposure/row weight total retained by compiler: {grouped.total_weight:.4f}.",
        ]

    def _uncertainty_markdown(self) -> list[str]:
        lines = ["", "## Supplied Statistical / Model Uncertainty", ""]
        if self.statistical_uncertainty is None:
            lines.append(
                "- No external statistical or model uncertainty summary was supplied."
            )
        else:
            row = self.statistical_uncertainty
            lines.append(f"- Label: {row.label}.")
            if row.estimate is not None:
                lines.append(f"- External estimate: {row.estimate:.4f}.")
            if row.standard_error is not None:
                lines.append(f"- Standard error: {row.standard_error:.4f}.")
            if row.lower is not None and row.upper is not None:
                lines.append(
                    f"- External interval: [{row.lower:.4f}, {row.upper:.4f}]."
                )
            if row.confidence_level is not None:
                lines.append(f"- Confidence level: {row.confidence_level:.3f}.")
            if row.method is not None:
                lines.append(f"- Method: {row.method}.")

        uncertainty = self.core.estimator_uncertainty
        if uncertainty is None:
            lines.append(
                "- No hidden-cell metric standard errors were supplied, so no "
                "estimator-uncertainty-aware hidden ambiguity adjustment is shown."
            )
        else:
            lines.extend(
                [
                    "- Hidden-cell metric standard errors were supplied.",
                    "- Estimator-uncertainty-aware conservative interval: "
                    f"[{uncertainty.conservative_lower:.4f}, "
                    f"{uncertainty.conservative_upper:.4f}].",
                    "- Estimator-uncertainty-aware conservative ambiguity: "
                    f"{uncertainty.conservative_diameter:.4f}.",
                ]
            )
            if uncertainty.confidence_core is not None:
                core = uncertainty.confidence_core
                if core.nonempty:
                    lines.append(
                        f"- SOCP confidence core: [{core.lower:.4f}, {core.upper:.4f}]."
                    )
                else:
                    lines.append(
                        f"- SOCP confidence core: empty by {core.empty_gap:.4f}."
                    )
        return lines

    def _model_assisted_uncertainty_markdown(self) -> list[str]:
        lines = ["", "## Model-Assisted Portfolio Uncertainty", ""]
        report = self.composition_uncertainty
        if report is None:
            lines.append(
                "- No posterior/bootstrap hidden-composition uncertainty report "
                "was supplied or requested."
            )
            lines.append(
                "- The fixed-public-law ambiguity section below remains a "
                "worst-case admissible-shift result, not a sampling uncertainty "
                "summary."
            )
            return lines

        ambiguity = report.ambiguity_summary
        observed = report.observed_summary
        lines.extend(
            [
                f"- Method: `{report.joint_model.method}`.",
                f"- Draws: {report.successful_draws}/{report.draw_count} successful.",
                f"- Hidden cells in fitted joint model: {report.joint_model.cell_count}.",
                "- Public law preserved in draws: "
                f"{'yes' if report.preserve_public_law else 'no'}.",
                "- Posterior/bootstrap observed estimate: "
                f"mean={_format_optional_float(observed.mean)}, "
                f"{100.0 * report.confidence_level:g}% interval="
                f"[{_format_optional_float(observed.lower)}, "
                f"{_format_optional_float(observed.upper)}].",
                "- Posterior/bootstrap ambiguity: "
                f"mean={_format_optional_float(ambiguity.mean)}, "
                f"{100.0 * report.confidence_level:g}% interval="
                f"[{_format_optional_float(ambiguity.lower)}, "
                f"{_format_optional_float(ambiguity.upper)}].",
                "- Public adequacy rate across draws: "
                f"{_format_optional_rate(report.public_adequate_rate)}.",
            ]
        )
        if report.ambiguity_limit is not None:
            lines.append(
                "- Ambiguity-limit failure rate across draws: "
                f"{_format_optional_rate(report.failure_rate)}."
            )
        lines.extend(
            [
                "- Separation: this section resamples hidden composition from a "
                "fitted joint model. Supplied statistical/model uncertainty and "
                "hidden-cell metric standard errors are reported separately in "
                "the section above; fixed-public-law hidden-composition ambiguity "
                "is reported separately below.",
            ]
        )
        return lines

    def _hidden_ambiguity_markdown(self) -> list[str]:
        core = self.core
        return [
            "",
            "## Hidden-Composition Ambiguity",
            "",
            "- Fixed-public-law interval: "
            f"[{core.interval.lower:.4f}, {core.interval.upper:.4f}].",
            f"- Hidden-composition ambiguity: {core.interval.diameter:.4f}.",
            f"- Public segmentation adequate: {'yes' if core.public_adequate else 'no'}.",
            "- Observed estimate position: "
            f"{'inside' if core.interval_contains_observed else 'outside'} the "
            "fixed-public-law interval.",
            f"- Q preset: `{core.grouped.q_name}`.",
            f"- Q interpretation: {core.grouped.q_description}.",
        ]

    def _concentration_stress_markdown(self) -> list[str]:
        summary = _concentration_stress_summary(self.core)
        lines = ["", "## Concentration-Stress Ambiguity", ""]
        lines.append(f"- Stress type: {summary['stress_type']}.")
        lines.append(f"- Q preset: `{summary['q_name']}`.")
        lines.append(f"- Interval width under this stress: {summary['ambiguity']:.4f}.")
        if summary["moment_count"] is not None:
            lines.append(f"- Balance/concentration moments: {summary['moment_count']}.")
        if summary["moment_names"]:
            lines.append(
                "- Moment examples: "
                + ", ".join(f"`{name}`" for name in summary["moment_names"][:5])
                + "."
            )
        lines.append(
            "- Interpretation: this is the same hidden-composition interval "
            "reported above, labeled in portfolio-concentration terms when the "
            "selected Q preset constrains factor or concentration moments."
        )
        return lines

    def _refinement_markdown(self) -> list[str]:
        lines = ["", "## Public Refinement Recommendations", ""]
        if not self.core.refinements:
            lines.append(
                "- No one-column refinement recommendations were computed or no "
                "candidate refinement reduced ambiguity."
            )
            return lines
        for row in self.core.refinements[:5]:
            lines.append(
                f"- Add `{row.column}`: ambiguity {row.before_ambiguity:.4f} -> "
                f"{row.after_ambiguity:.4f}; reduction {row.reduction:.4f} "
                f"({row.reduction_percent:.1f}%), public_cells={row.public_cells}."
            )
        return lines

    def _dual_diagnostics_markdown(self) -> list[str]:
        lines = ["", "## Dual Diagnostics", ""]
        duals = self.core.interval.dual_summary(top=8, min_magnitude=1e-8)
        if not duals:
            lines.append(
                "- No CVXPY dual diagnostics are available for this report. "
                "This is expected for non-CVXPY or mixed-integer solver paths."
            )
            return lines
        lines.append(
            "- Largest local solver multipliers; use as constraint-sensitivity "
            "diagnostics, not as standalone validation evidence."
        )
        for row in duals:
            lines.append(
                f"- {row.kind}: {row.name}, solve={row.solve}, "
                f"magnitude={row.magnitude:.4g}."
            )
        return lines

    def _data_diagnostics_markdown(self) -> list[str]:
        lines = ["", "## Data Diagnostics", ""]
        if not self.core.diagnostics:
            lines.append("- No pre-solve data diagnostics were raised.")
            return lines
        for row in self.core.diagnostics:
            columns = f" columns={', '.join(row.columns)}" if row.columns else ""
            count = "" if row.count is None else f" count={row.count}"
            lines.append(
                f"- {row.severity}: `{row.code}`{count}{columns} - {row.message}"
            )
        return lines

    def _limitations_and_notes_markdown(self) -> list[str]:
        lines = ["", "## Limitations / Reviewer Notes", ""]
        if self.limitations:
            lines.append("Limitations:")
            lines.extend(f"- {limitation}" for limitation in self.limitations)
        if self.reviewer_notes:
            lines.extend(["", "Reviewer notes:"])
            lines.extend(f"- {note}" for note in self.reviewer_notes)
        else:
            lines.extend(["", "Reviewer notes:", "- None supplied."])
        return lines


@dataclass(frozen=True)
class FinanceStabilityCertificate:
    """Finance-specific wrapper around a core representation certificate."""

    core: us.RepresentationStabilityCertificate
    metadata: ModelRiskMetadata = ModelRiskMetadata()
    q_profile: str = "credit_expected_loss"
    title: str = "Financial Model-Risk Segmentation Stability Certificate"

    @property
    def status(self) -> str:
        return self.core.status

    @property
    def passed(self) -> bool:
        return self.core.passed

    @property
    def failed(self) -> bool:
        return self.core.failed

    @property
    def inconclusive(self) -> bool:
        return self.core.inconclusive

    @property
    def certified_candidate(self):
        return self.core.certified_candidate

    @property
    def selected_candidate(self):
        return self.core.selected_candidate

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "status": self.status,
            "passed": self.passed,
            "failed": self.failed,
            "inconclusive": self.inconclusive,
            "q_profile": self.q_profile,
            "metadata": self.metadata.as_dict(),
            "core": self.core.as_dict(),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- Certification status: **{self.status.upper()}**",
            f"- Finance Q profile: `{self.q_profile}`",
        ]
        selected = self.core.selected_candidate
        if selected is not None:
            lines.extend(
                [
                    "- Selected public segmentation: "
                    f"`{_column_label(selected.public_columns)}`",
                    f"- Worst ambiguity: {selected.max_ambiguity:.4f}",
                ]
            )
        else:
            lines.append("- Selected public segmentation: none")
        lines.extend([""])
        lines.extend(self._metadata_markdown())
        lines.extend(self._finance_interpretation_markdown())
        core_markdown = self.core.to_markdown().splitlines()
        if core_markdown and core_markdown[0].startswith("# "):
            core_markdown = core_markdown[2:]
        lines.extend(["", "## Core Certificate Evidence", ""])
        lines.extend(core_markdown)
        return "\n".join(lines)

    def _metadata_markdown(self) -> list[str]:
        rows = [
            ("Model ID", self.metadata.model_id),
            ("Portfolio", self.metadata.portfolio_name),
            ("As-of date", self.metadata.as_of_date),
            ("Intended use", self.metadata.intended_use),
            ("Owner", self.metadata.owner),
            ("Reviewer", self.metadata.reviewer),
        ]
        present = [(label, value) for label, value in rows if value]
        if not present:
            return []
        lines = ["## Model-Risk Context", ""]
        lines.extend(f"- {label}: {value}" for label, value in present)
        lines.append("")
        return lines

    def _finance_interpretation_markdown(self) -> list[str]:
        return [
            "## Financial Certification Interpretation",
            "",
            "This certificate asks whether a reported public portfolio "
            "segmentation can support the selected risk metric under the "
            "declared hidden-composition and portfolio-concentration stress "
            "grid.",
            "",
            "A pass is a representation-stability statement. It is not a "
            "statistical confidence interval, model calibration result, "
            "backtest, or governance approval.",
        ]

    def to_json(self, **kwargs: Any) -> str:
        return us.report_to_json(self, **kwargs)

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        tables = {
            "finance_certificate": (
                {
                    "title": self.title,
                    "status": self.status,
                    "q_profile": self.q_profile,
                    "model_id": self.metadata.model_id,
                    "portfolio_name": self.metadata.portfolio_name,
                    "as_of_date": self.metadata.as_of_date,
                    "intended_use": self.metadata.intended_use,
                    "owner": self.metadata.owner,
                    "reviewer": self.metadata.reviewer,
                },
            )
        }
        tables.update(
            {f"core_{name}": rows for name, rows in self.core.to_tables().items()}
        )
        return tables

    def to_dataframes(self) -> dict[str, Any]:
        return us.tables_to_dataframes(self.to_tables())


def from_portfolio(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    metric: str | us.RowMetric,
    metric_standard_error: str | us.RowMetric | None = None,
    target_standard_error: str | us.RowMetric | None = None,
    exposure: str | None = None,
    weight: str | None = None,
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    q_radius: float | None = None,
    target_confidence_multiplier: float = 1.96,
) -> us.GroupedProblem:
    """Compile a financial portfolio into an update-support grouped problem."""

    if exposure is not None and weight is not None and exposure != weight:
        raise ValueError("use either exposure or weight, not both")
    resolved_standard_error = _resolve_metric_standard_error(
        metric_standard_error,
        target_standard_error,
    )
    return us.from_dataframe(
        data,
        public=public,
        hidden=hidden,
        target=metric,
        target_standard_error=resolved_standard_error,
        weight=weight if weight is not None else exposure,
        min_cell_weight=min_cell_weight,
        q=q,
        q_radius=q_radius,
        target_confidence_multiplier=target_confidence_multiplier,
    )


def model_risk_report(
    data: Any | us.GroupedProblem,
    *,
    source_data: Any | None = None,
    public: Sequence[str] | None = None,
    hidden: Sequence[str] | None = None,
    metric: str | us.RowMetric | None = None,
    metric_standard_error: str | us.RowMetric | None = None,
    target_standard_error: str | us.RowMetric | None = None,
    exposure: str | None = None,
    weight: str | None = None,
    candidate_refinements: Sequence[str] | None = None,
    top: int = 10,
    min_cell_weight: float = 1.0,
    title: str = "Financial Model-Risk Representation Stability Report",
    observed_label: str = "Reported portfolio risk estimate",
    q: Any | None = None,
    q_radius: float | None = None,
    target_confidence_multiplier: float = 1.96,
    metadata: ModelRiskMetadata | None = None,
    thresholds: ReviewThresholds | None = None,
    model_id: str | None = None,
    portfolio_name: str | None = None,
    as_of_date: str | None = None,
    intended_use: str | None = None,
    owner: str | None = None,
    reviewer: str | None = None,
    ambiguity_limit: float | None = None,
    public_adequacy_required: bool = False,
    statistical_estimate: float | None = None,
    statistical_standard_error: float | None = None,
    statistical_interval: Sequence[float] | None = None,
    statistical_confidence_level: float | None = None,
    statistical_method: str | None = None,
    statistical_label: str = "Supplied statistical/model uncertainty",
    composition_uncertainty: us.HiddenCompositionUncertaintyReport | None = None,
    composition_uncertainty_draws: int | None = None,
    composition_uncertainty_method: str = "bayesian_bootstrap",
    composition_uncertainty_seed: int | None = None,
    composition_uncertainty_confidence_level: float = 0.9,
    composition_uncertainty_preserve_public_law: bool = True,
    composition_uncertainty_effective_sample_size: float | None = None,
    composition_uncertainty_smoothing: float = 1e-9,
    composition_uncertainty_title: str = "Model-Assisted Portfolio Uncertainty Report",
    reviewer_notes: Sequence[str] = (),
    limitations: Sequence[str] | None = None,
) -> ModelRiskReport:
    """Build a model-risk report for a financial portfolio segmentation."""

    if exposure is not None and weight is not None and exposure != weight:
        raise ValueError("use either exposure or weight, not both")
    effective_weight = weight if weight is not None else exposure
    resolved_standard_error = _resolve_metric_standard_error(
        metric_standard_error,
        target_standard_error,
    )
    if (
        composition_uncertainty is not None
        and composition_uncertainty_draws is not None
    ):
        raise ValueError(
            "pass either composition_uncertainty or composition_uncertainty_draws, "
            "not both"
        )
    if isinstance(data, us.GroupedProblem):
        target_description = "portfolio risk metric"
        if resolved_standard_error is not None:
            raise ValueError(
                "metric_standard_error can only be supplied when data contains "
                "raw portfolio rows; precompiled GroupedProblem inputs must "
                "already include target standard errors"
            )
        if composition_uncertainty_draws is not None:
            raise ValueError(
                "composition_uncertainty_draws requires raw portfolio rows; pass a "
                "precomputed composition_uncertainty for GroupedProblem inputs"
            )
    else:
        if metric is None:
            raise TypeError(
                "model_risk_report() missing required keyword argument: 'metric'"
            )
        target_description = (
            metric.description
            if isinstance(metric, us.RowMetric) and metric.description
            else "portfolio risk metric"
        )

    resolved_composition_uncertainty = composition_uncertainty
    if composition_uncertainty_draws is not None:
        if public is None:
            raise TypeError("public is required for composition_uncertainty_draws")
        if hidden is None:
            raise TypeError("hidden is required for composition_uncertainty_draws")
        if metric is None:
            raise TypeError("metric is required for composition_uncertainty_draws")
        resolved_composition_uncertainty = model_assisted_portfolio_uncertainty(
            data,
            public=public,
            hidden=hidden,
            metric=metric,
            exposure=exposure,
            weight=weight,
            draws=composition_uncertainty_draws,
            seed=composition_uncertainty_seed,
            method=composition_uncertainty_method,
            min_cell_weight=min_cell_weight,
            q="saturated" if q is None else q,
            ambiguity_limit=ambiguity_limit,
            confidence_level=composition_uncertainty_confidence_level,
            preserve_public_law=composition_uncertainty_preserve_public_law,
            effective_sample_size=composition_uncertainty_effective_sample_size,
            smoothing=composition_uncertainty_smoothing,
            title=composition_uncertainty_title,
        )

    core_report = us.public_descent_report(
        data,
        source_data=source_data,
        public=public,
        hidden=hidden,
        target=metric,
        target_standard_error=resolved_standard_error,
        weight=effective_weight,
        candidate_refinements=candidate_refinements,
        top=top,
        min_cell_weight=min_cell_weight,
        title=title,
        target_description=target_description,
        observed_label=observed_label,
        row_count_label="Portfolio rows",
        q=q,
        q_radius=q_radius,
        target_confidence_multiplier=target_confidence_multiplier,
    )
    report_metadata = metadata or ModelRiskMetadata(
        model_id=model_id,
        portfolio_name=portfolio_name,
        as_of_date=as_of_date,
        intended_use=intended_use,
        owner=owner,
        reviewer=reviewer,
    )
    report_thresholds = thresholds or ReviewThresholds(
        ambiguity_limit=ambiguity_limit,
        public_adequacy_required=public_adequacy_required,
    )
    return ModelRiskReport(
        core=core_report,
        metadata=report_metadata,
        thresholds=report_thresholds,
        statistical_uncertainty=_statistical_uncertainty(
            estimate=statistical_estimate,
            standard_error=statistical_standard_error,
            interval=statistical_interval,
            confidence_level=statistical_confidence_level,
            method=statistical_method,
            label=statistical_label,
        ),
        composition_uncertainty=resolved_composition_uncertainty,
        reviewer_notes=tuple(reviewer_notes),
        limitations=(
            DEFAULT_MODEL_RISK_LIMITATIONS
            if limitations is None
            else tuple(limitations)
        ),
    )


def model_assisted_portfolio_uncertainty(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    metric: str | us.RowMetric,
    exposure: str | None = None,
    weight: str | None = None,
    draws: int = 500,
    seed: int | None = None,
    method: str = "bayesian_bootstrap",
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    ambiguity_limit: float | None = None,
    confidence_level: float = 0.9,
    preserve_public_law: bool = True,
    effective_sample_size: float | None = None,
    smoothing: float = 1e-9,
    title: str = "Model-Assisted Portfolio Uncertainty Report",
) -> us.HiddenCompositionUncertaintyReport:
    """Bootstrap/posterior uncertainty over portfolio hidden composition.

    This is a finance-facing wrapper around
    :func:`updatesupport.hidden_composition_uncertainty`. It preserves finance
    vocabulary such as ``metric`` and ``exposure`` while returning the core
    model-assisted report object.
    """

    if exposure is not None and weight is not None and exposure != weight:
        raise ValueError("use either exposure or weight, not both")
    return us.hidden_composition_uncertainty(
        data,
        public=public,
        hidden=hidden,
        target=metric,
        weight=weight if weight is not None else exposure,
        draws=draws,
        seed=seed,
        method=method,
        min_cell_weight=min_cell_weight,
        q=q,
        ambiguity_limit=ambiguity_limit,
        confidence_level=confidence_level,
        preserve_public_law=preserve_public_law,
        effective_sample_size=effective_sample_size,
        smoothing=smoothing,
        title=title,
    )


def certify_portfolio_segmentation(
    data: Any,
    *,
    public: Sequence[str] | None = None,
    base_public: Sequence[str] | None = None,
    hidden: Sequence[str],
    metric: str | us.RowMetric,
    exposure: str | None = None,
    weight: str | None = None,
    candidate_refinements: Sequence[str] | None = None,
    q_profile: str = "credit_expected_loss",
    q_presets: Sequence[Any] | None = None,
    ambiguity_limit: float,
    bucket_budget: int | None = None,
    min_cell_weight: float = 1.0,
    min_cell_weights: Sequence[float] | None = None,
    hidden_sets: Sequence[Sequence[str]] | None = None,
    search: str = "exhaustive",
    exact_required: bool = True,
    max_added_columns: int | None = None,
    max_evaluations: int | None = None,
    beam_width: int = 12,
    factors: str | Sequence[str] | dict[str, str] | None = None,
    portfolio_mix_radius: float | None = 0.35,
    tv_radius: float | None = 0.10,
    factor_radius: float | None = 0.20,
    region: str | None = "region",
    regional_radius: float | None = 0.10,
    include_tv: bool = True,
    include_regional: bool = True,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: dict[str, Any] | None = None,
    title: str = "Financial Model-Risk Segmentation Stability Certificate",
    core_title: str = "Representation Stability Certificate",
    frontier_title: str = "Finance Portfolio Segmentation Frontier Evidence",
    metadata: ModelRiskMetadata | None = None,
    model_id: str | None = None,
    portfolio_name: str | None = None,
    as_of_date: str | None = None,
    intended_use: str | None = None,
    owner: str | None = None,
    reviewer: str | None = None,
    **frontier_kwargs: Any,
) -> FinanceStabilityCertificate:
    """Certify a public portfolio segmentation against a finance stress grid."""

    if exposure is not None and weight is not None and exposure != weight:
        raise ValueError("use either exposure or weight, not both")
    selected_public = _resolve_public(public=public, base_public=base_public)
    if q_presets is None:
        q_presets = finance_sensitivity_grid(
            data,
            hidden=hidden,
            exposure=exposure,
            weight=weight,
            profile=q_profile,
            portfolio_mix_radius=portfolio_mix_radius,
            tv_radius=tv_radius,
            factors=factors,
            factor_radius=factor_radius,
            region=region,
            regional_radius=regional_radius,
            include_tv=include_tv,
            include_regional=include_regional,
            backend=backend,
            solver=solver,
            solver_options=solver_options,
        )
    core = us.certify_public_representation(
        data,
        base_public=selected_public,
        hidden=hidden,
        target=metric,
        weight=weight if weight is not None else exposure,
        candidate_refinements=candidate_refinements,
        q_presets=q_presets,
        ambiguity_limit=ambiguity_limit,
        bucket_budget=bucket_budget,
        min_cell_weight=min_cell_weight,
        min_cell_weights=min_cell_weights,
        hidden_sets=hidden_sets,
        search=search,
        exact_required=exact_required,
        max_added_columns=max_added_columns,
        max_evaluations=max_evaluations,
        beam_width=beam_width,
        title=core_title,
        frontier_title=frontier_title,
        **frontier_kwargs,
    )
    certificate_metadata = metadata or ModelRiskMetadata(
        model_id=model_id,
        portfolio_name=portfolio_name,
        as_of_date=as_of_date,
        intended_use=intended_use,
        owner=owner,
        reviewer=reviewer,
    )
    return FinanceStabilityCertificate(
        core=core,
        metadata=certificate_metadata,
        q_profile=q_profile,
        title=title,
    )


def _concentration_stress_summary(
    core: us.PublicDescentReport,
) -> dict[str, Any]:
    q = core.grouped.q
    moment_names: tuple[str, ...] = ()
    if q is not None and getattr(q, "name", None) == "covariate_balance":
        cost = getattr(q, "cost", None)
        if isinstance(cost, Mapping):
            moment_names = tuple(str(name) for name in cost)
        has_factor = any(name.startswith("factor:") for name in moment_names)
        has_region = any(name.startswith("region:") for name in moment_names)
        if has_factor and has_region:
            stress_type = "factor and regional concentration stress"
        elif has_factor:
            stress_type = "factor-exposure concentration stress"
        elif has_region:
            stress_type = "regional concentration stress"
        else:
            stress_type = "covariate-balance concentration stress"
        moment_count = len(moment_names)
    else:
        stress_type = "not a concentration-balance preset"
        moment_count = None
    return {
        "stress_type": stress_type,
        "q_name": core.grouped.q_name,
        "ambiguity": core.interval.diameter,
        "moment_count": moment_count,
        "moment_names": moment_names,
    }


def _resolve_metric_standard_error(
    metric_standard_error: str | us.RowMetric | None,
    target_standard_error: str | us.RowMetric | None,
) -> str | us.RowMetric | None:
    if (
        metric_standard_error is not None
        and target_standard_error is not None
        and metric_standard_error != target_standard_error
    ):
        raise TypeError(
            "use either 'metric_standard_error' or 'target_standard_error', not both"
        )
    return (
        metric_standard_error
        if metric_standard_error is not None
        else target_standard_error
    )


def _statistical_uncertainty(
    *,
    estimate: float | None,
    standard_error: float | None,
    interval: Sequence[float] | None,
    confidence_level: float | None,
    method: str | None,
    label: str,
) -> us.StatisticalUncertainty | None:
    lower = None
    upper = None
    if interval is not None:
        values = tuple(float(value) for value in interval)
        if len(values) != 2:
            raise ValueError("statistical_interval must contain exactly two values")
        lower, upper = values
        if lower > upper:
            raise ValueError("statistical_interval lower bound exceeds upper bound")
    if confidence_level is not None:
        confidence_level = float(confidence_level)
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("statistical_confidence_level must be between 0 and 1")
    if (
        estimate is None
        and standard_error is None
        and lower is None
        and upper is None
        and confidence_level is None
        and method is None
    ):
        return None
    return us.StatisticalUncertainty(
        estimate=None if estimate is None else float(estimate),
        standard_error=None if standard_error is None else float(standard_error),
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method=method,
        label=label,
    )


def _format_optional_float(value: float | None) -> str:
    return "not available" if value is None else f"{value:.4f}"


def _format_optional_rate(value: float | None) -> str:
    return "not evaluated" if value is None else f"{100.0 * value:.1f}%"


def _resolve_public(
    *,
    public: Sequence[str] | None,
    base_public: Sequence[str] | None,
) -> Sequence[str]:
    if (
        public is not None
        and base_public is not None
        and tuple(public) != tuple(base_public)
    ):
        raise TypeError("use either 'public' or 'base_public', not both")
    selected = public if public is not None else base_public
    if selected is None:
        raise TypeError(
            "certify_portfolio_segmentation() missing required keyword argument: "
            "'public'"
        )
    return selected


def _column_label(columns: Sequence[str]) -> str:
    return " x ".join(columns) if columns else "(none)"
