"""Portfolio compilers and report profiles for financial model-risk review."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import updatesupport as us

from .presets import finance_sensitivity_grid


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
