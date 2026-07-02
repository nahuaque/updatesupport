"""Portfolio compilers and report profiles for financial model-risk review."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import updatesupport as us


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


@dataclass(frozen=True)
class ModelRiskReport:
    """Finance-specific wrapper around a core public-descent report."""

    core: us.PublicDescentReport
    metadata: ModelRiskMetadata = ModelRiskMetadata()
    thresholds: ReviewThresholds = ReviewThresholds()

    @property
    def review_status(self) -> str:
        status, _reasons = self.thresholds.evaluate(self.core)
        return status

    @property
    def review_reasons(self) -> tuple[str, ...]:
        _status, reasons = self.thresholds.evaluate(self.core)
        return reasons

    def to_markdown(self) -> str:
        lines = [f"# {self.core.title}", ""]
        lines.extend(self._metadata_markdown())
        lines.extend(self._threshold_markdown())
        lines.extend(self._finance_interpretation_markdown())
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


def from_portfolio(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    metric: str | us.RowMetric,
    exposure: str | None = None,
    weight: str | None = None,
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    q_radius: float | None = None,
) -> us.GroupedProblem:
    """Compile a financial portfolio into an update-support grouped problem."""

    if exposure is not None and weight is not None and exposure != weight:
        raise ValueError("use either exposure or weight, not both")
    return us.from_dataframe(
        data,
        public=public,
        hidden=hidden,
        target=metric,
        weight=weight if weight is not None else exposure,
        min_cell_weight=min_cell_weight,
        q=q,
        q_radius=q_radius,
    )


def model_risk_report(
    data: Any | us.GroupedProblem,
    *,
    source_data: Any | None = None,
    public: Sequence[str] | None = None,
    hidden: Sequence[str] | None = None,
    metric: str | us.RowMetric | None = None,
    exposure: str | None = None,
    weight: str | None = None,
    candidate_refinements: Sequence[str] | None = None,
    top: int = 10,
    min_cell_weight: float = 1.0,
    title: str = "Financial Model-Risk Representation Stability Report",
    observed_label: str = "Reported portfolio risk estimate",
    q: Any | None = None,
    q_radius: float | None = None,
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
) -> ModelRiskReport:
    """Build a model-risk report for a financial portfolio segmentation."""

    if exposure is not None and weight is not None and exposure != weight:
        raise ValueError("use either exposure or weight, not both")
    effective_weight = weight if weight is not None else exposure
    if isinstance(data, us.GroupedProblem):
        target_description = "portfolio risk metric"
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

    core_report = us.public_descent_report(
        data,
        source_data=source_data,
        public=public,
        hidden=hidden,
        target=metric,
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
    )
