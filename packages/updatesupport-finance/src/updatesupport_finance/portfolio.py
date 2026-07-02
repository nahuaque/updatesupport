"""Portfolio compilers and report profiles for financial model-risk review."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import updatesupport as us


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
) -> us.PublicDescentReport:
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

    return us.public_descent_report(
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
