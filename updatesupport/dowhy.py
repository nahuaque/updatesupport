"""DoWhy integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .artifacts import ReportArtifactMixin
from .data import GroupedProblem
from .report import PublicDescentReport, audit_effects

DEFAULT_REFUTATION_TYPE = "UpdateSupport representation stability"


@dataclass(frozen=True)
class DoWhyRepresentationAudit(ReportArtifactMixin):
    """Update-support audit packaged for a DoWhy causal workflow."""

    report: PublicDescentReport
    estimate: Any | None = None
    refutation_type: str = DEFAULT_REFUTATION_TYPE

    @property
    def estimated_effect(self) -> float:
        """Scalar effect estimate used as the DoWhy refutation baseline."""

        return _estimated_effect_value(
            self.estimate, fallback=self.report.observed_value
        )

    @property
    def new_effect(self) -> tuple[float, float]:
        """Update-support partial-ID interval reported as the refuted effect."""

        return (self.report.interval.lower, self.report.interval.upper)

    @property
    def ambiguity(self) -> float:
        """Width of the update-support partial-ID interval."""

        return self.report.interval.diameter

    def to_refutation(self, *, refutation_class: type[Any] | None = None) -> Any:
        """Return a DoWhy ``CausalRefutation`` carrying this audit's interval."""

        return dowhy_refutation_from_report(
            self.report,
            estimate=self.estimate,
            refutation_type=self.refutation_type,
            refutation_class=refutation_class,
        )

    def to_markdown(self) -> str:
        """Render the underlying public-descent report."""

        return self.report.to_markdown()

    def as_dict(self) -> dict[str, Any]:
        return {
            "refutation_type": self.refutation_type,
            "estimated_effect": self.estimated_effect,
            "new_effect": self.new_effect,
            "ambiguity": self.ambiguity,
            "report": self.report.as_dict(),
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        from .exports import report_tables

        tables = report_tables(self.report)
        tables["dowhy_refutation"] = (
            {
                "refutation_type": self.refutation_type,
                "estimated_effect": self.estimated_effect,
                "new_effect": self.new_effect,
                "ambiguity": self.ambiguity,
            },
        )
        return tables


def audit_dowhy_effects(
    data: Any | GroupedProblem,
    *,
    estimate: Any | None = None,
    refutation_type: str = DEFAULT_REFUTATION_TYPE,
    source_data: Any | None = None,
    public: Sequence[str] | None = None,
    hidden: Sequence[str] | None = None,
    effect: str | None = None,
    weight: str | None = None,
    public_columns: Sequence[str] | None = None,
    hidden_columns: Sequence[str] | None = None,
    effect_column: str | None = None,
    weight_column: str | None = None,
    candidate_refinements: Sequence[str] | None = None,
    candidate_columns: Sequence[str] | None = None,
    top: int = 10,
    min_cell_weight: float = 1.0,
    title: str = "DoWhy Effect Representation Stability Audit",
    effect_description: str = "estimated causal effect",
    observed_label: str = "Observed effect estimate",
    row_count: int | None = None,
    row_count_label: str = "Rows",
    q: Any | None = None,
    q_radius: float | None = None,
) -> DoWhyRepresentationAudit:
    """Audit a DoWhy-compatible effect target for reporting stability.

    DoWhy identifies, estimates, and refutes causal effects. This helper assumes
    that workflow has already produced a row-level, subgroup-level, or
    hidden-cell-level effect target, then runs the update-support representation
    audit on that supplied target.
    """

    report = audit_effects(
        data,
        source_data=source_data,
        public=public,
        hidden=hidden,
        effect=effect,
        weight=weight,
        public_columns=public_columns,
        hidden_columns=hidden_columns,
        effect_column=effect_column,
        weight_column=weight_column,
        candidate_refinements=candidate_refinements,
        candidate_columns=candidate_columns,
        top=top,
        min_cell_weight=min_cell_weight,
        title=title,
        effect_description=effect_description,
        observed_label=observed_label,
        row_count=row_count,
        row_count_label=row_count_label,
        q=q,
        q_radius=q_radius,
    )
    return DoWhyRepresentationAudit(
        report=report,
        estimate=estimate,
        refutation_type=refutation_type,
    )


def dowhy_refutation_from_report(
    report: PublicDescentReport,
    *,
    estimate: Any | None = None,
    estimated_effect: float | None = None,
    refutation_type: str = DEFAULT_REFUTATION_TYPE,
    refutation_class: type[Any] | None = None,
) -> Any:
    """Convert a public-descent report into a DoWhy ``CausalRefutation``.

    ``new_effect`` is the update-support partial-ID interval, not a point
    estimate from a second causal estimator. Extra update-support metadata is
    attached to the returned object for downstream inspection.
    """

    baseline = (
        float(estimated_effect)
        if estimated_effect is not None
        else _estimated_effect_value(estimate, fallback=report.observed_value)
    )
    interval = (report.interval.lower, report.interval.upper)

    if refutation_class is None:
        try:
            from dowhy.causal_refuter import CausalRefutation
        except ImportError as exc:
            raise ImportError(
                "DoWhy is required to create a CausalRefutation. "
                "Install it with: uv sync --extra dowhy"
            ) from exc
        refutation_class = CausalRefutation

    refutation = refutation_class(
        estimated_effect=baseline,
        new_effect=interval,
        refutation_type=refutation_type,
    )
    _attach_metadata(
        refutation,
        updatesupport_report=report,
        updatesupport_interval=interval,
        updatesupport_ambiguity=report.interval.diameter,
        updatesupport_public_adequate=report.public_adequate,
    )
    return refutation


def _estimated_effect_value(estimate: Any | None, *, fallback: float) -> float:
    if estimate is None:
        return float(fallback)

    direct = _try_float(estimate)
    if direct is not None:
        return direct

    for attribute in ("value", "estimate", "estimated_effect"):
        if hasattr(estimate, attribute):
            value = _try_float(getattr(estimate, attribute))
            if value is not None:
                return value

    raise ValueError(
        "estimate must be numeric or expose a numeric 'value', 'estimate', "
        "or 'estimated_effect' attribute"
    )


def _try_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except (TypeError, ValueError):
            return None
    return None


def _attach_metadata(target: Any, **metadata: Any) -> None:
    for name, value in metadata.items():
        try:
            setattr(target, name, value)
        except AttributeError:
            continue
