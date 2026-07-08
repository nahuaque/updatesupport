"""Model-assisted joint public/hidden distribution utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt
from typing import Any, Hashable, Mapping, Sequence

from .artifacts import ReportArtifactMixin
from .data import TabularTarget, from_dataframe
from .report import public_descent_report


_MODEL_WEIGHT_COLUMN = "__updatesupport_joint_weight__"
_MODEL_TARGET_COLUMN = "__updatesupport_joint_target__"


@dataclass(frozen=True)
class JointCell:
    """One retained hidden cell in a fitted nonparametric joint distribution."""

    hidden_cell: tuple[Hashable, ...]
    public_value: tuple[Hashable, ...]
    probability: float
    total_weight: float
    target_value: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_cell": self.hidden_cell,
            "public_value": self.public_value,
            "probability": self.probability,
            "total_weight": self.total_weight,
            "target_value": self.target_value,
        }


@dataclass(frozen=True)
class JointDistributionDraw:
    """One model-assisted draw of hidden-cell masses."""

    draw_index: int
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    cells: tuple[JointCell, ...]
    probabilities: tuple[float, ...]
    total_weight: float
    weight_column: str = _MODEL_WEIGHT_COLUMN
    target_column: str = _MODEL_TARGET_COLUMN

    def records(self) -> tuple[dict[str, Any], ...]:
        """Return weighted hidden-cell records consumable by report helpers."""

        records = []
        for cell, probability in zip(self.cells, self.probabilities, strict=True):
            row = {
                column: value
                for column, value in zip(
                    self.hidden_columns,
                    cell.hidden_cell,
                    strict=True,
                )
            }
            row[self.weight_column] = probability * self.total_weight
            row[self.target_column] = cell.target_value
            records.append(row)
        return tuple(records)

    def as_dict(self) -> dict[str, Any]:
        return {
            "draw_index": self.draw_index,
            "public_columns": self.public_columns,
            "hidden_columns": self.hidden_columns,
            "probabilities": self.probabilities,
            "total_weight": self.total_weight,
            "weight_column": self.weight_column,
            "target_column": self.target_column,
        }


@dataclass(frozen=True)
class NonparametricJointDistribution:
    """Fitted empirical public/hidden cell law with bootstrap draw support."""

    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    target_name: str
    cells: tuple[JointCell, ...]
    total_weight: float
    rows_seen: int
    method: str = "bayesian_bootstrap"
    effective_sample_size: float | None = None
    smoothing: float = 1e-9

    def __post_init__(self) -> None:
        method = _normalize_joint_method(self.method)
        object.__setattr__(self, "method", method)
        if not self.cells:
            raise ValueError("cells must contain at least one joint cell")
        if self.total_weight <= 0:
            raise ValueError("total_weight must be positive")
        if self.rows_seen <= 0:
            raise ValueError("rows_seen must be positive")
        if self.effective_sample_size is not None and self.effective_sample_size <= 0:
            raise ValueError("effective_sample_size must be positive")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be positive")

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def draw(
        self,
        *,
        draw_index: int = 1,
        seed: int | None = None,
        weight_column: str = _MODEL_WEIGHT_COLUMN,
        target_column: str = _MODEL_TARGET_COLUMN,
    ) -> JointDistributionDraw:
        """Draw one full-joint weighted cell composition."""

        rng = random.Random(seed)  # nosec B311
        return self._draw_with_rng(
            rng,
            draw_index=draw_index,
            weight_column=weight_column,
            target_column=target_column,
        )

    def iter_draws(
        self,
        count: int,
        *,
        seed: int | None = None,
        weight_column: str = _MODEL_WEIGHT_COLUMN,
        target_column: str = _MODEL_TARGET_COLUMN,
    ) -> tuple[JointDistributionDraw, ...]:
        """Return ``count`` independent full-joint model-assisted draws."""

        if count < 0:
            raise ValueError("count must be non-negative")
        rng = random.Random(seed)  # nosec B311
        return tuple(
            self._draw_with_rng(
                rng,
                draw_index=index,
                weight_column=weight_column,
                target_column=target_column,
            )
            for index in range(1, count + 1)
        )

    def hidden_composition_draw(
        self,
        *,
        draw_index: int = 1,
        seed: int | None = None,
        weight_column: str = _MODEL_WEIGHT_COLUMN,
        target_column: str = _MODEL_TARGET_COLUMN,
    ) -> JointDistributionDraw:
        """Draw hidden-cell masses while preserving the fitted public law."""

        rng = random.Random(seed)  # nosec B311
        return self._hidden_composition_draw_with_rng(
            rng,
            draw_index=draw_index,
            weight_column=weight_column,
            target_column=target_column,
        )

    def iter_hidden_composition_draws(
        self,
        count: int,
        *,
        seed: int | None = None,
        weight_column: str = _MODEL_WEIGHT_COLUMN,
        target_column: str = _MODEL_TARGET_COLUMN,
    ) -> tuple[JointDistributionDraw, ...]:
        """Return hidden-composition draws with public masses held fixed."""

        if count < 0:
            raise ValueError("count must be non-negative")
        rng = random.Random(seed)  # nosec B311
        return tuple(
            self._hidden_composition_draw_with_rng(
                rng,
                draw_index=index,
                weight_column=weight_column,
                target_column=target_column,
            )
            for index in range(1, count + 1)
        )

    def draw_records(
        self,
        *,
        seed: int | None = None,
        weight_column: str = _MODEL_WEIGHT_COLUMN,
        target_column: str = _MODEL_TARGET_COLUMN,
    ) -> tuple[dict[str, Any], ...]:
        """Return one draw as weighted cell records."""

        return self.draw(
            seed=seed,
            weight_column=weight_column,
            target_column=target_column,
        ).records()

    def as_dict(self) -> dict[str, Any]:
        return {
            "public_columns": self.public_columns,
            "hidden_columns": self.hidden_columns,
            "target_name": self.target_name,
            "cell_count": self.cell_count,
            "total_weight": self.total_weight,
            "rows_seen": self.rows_seen,
            "method": self.method,
            "effective_sample_size": self.effective_sample_size,
            "smoothing": self.smoothing,
            "public_law": self.public_law,
            "cells": [cell.as_dict() for cell in self.cells],
        }

    def _draw_with_rng(
        self,
        rng: random.Random,
        *,
        draw_index: int,
        weight_column: str,
        target_column: str,
    ) -> JointDistributionDraw:
        if self.method == "empirical":
            probabilities = tuple(cell.probability for cell in self.cells)
        elif self.method == "bootstrap":
            probabilities = self._bootstrap_probabilities(rng)
        else:
            probabilities = self._bayesian_bootstrap_probabilities(rng)
        return JointDistributionDraw(
            draw_index=draw_index,
            public_columns=self.public_columns,
            hidden_columns=self.hidden_columns,
            cells=self.cells,
            probabilities=probabilities,
            total_weight=self.total_weight,
            weight_column=weight_column,
            target_column=target_column,
        )

    @property
    def public_law(self) -> dict[tuple[Hashable, ...], float]:
        law: dict[tuple[Hashable, ...], float] = {}
        for cell in self.cells:
            law[cell.public_value] = law.get(cell.public_value, 0.0) + cell.probability
        return law

    def _bayesian_bootstrap_probabilities(
        self,
        rng: random.Random,
    ) -> tuple[float, ...]:
        effective_n = (
            float(self.rows_seen)
            if self.effective_sample_size is None
            else float(self.effective_sample_size)
        )
        gammas = [
            rng.gammavariate(
                max(cell.probability * effective_n, self.smoothing),
                1.0,
            )
            for cell in self.cells
        ]
        total = sum(gammas)
        if total <= 0:
            return tuple(cell.probability for cell in self.cells)
        return tuple(value / total for value in gammas)

    def _bootstrap_probabilities(self, rng: random.Random) -> tuple[float, ...]:
        effective_n = (
            float(self.rows_seen)
            if self.effective_sample_size is None
            else float(self.effective_sample_size)
        )
        draw_count = max(1, int(round(effective_n)))
        weights = [cell.probability for cell in self.cells]
        sampled = rng.choices(
            range(len(self.cells)),
            weights=weights,
            k=draw_count,
        )
        counts = [0] * len(self.cells)
        for index in sampled:
            counts[index] += 1
        return tuple(count / draw_count for count in counts)

    def _hidden_composition_draw_with_rng(
        self,
        rng: random.Random,
        *,
        draw_index: int,
        weight_column: str,
        target_column: str,
    ) -> JointDistributionDraw:
        if self.method == "empirical":
            probabilities = tuple(cell.probability for cell in self.cells)
        else:
            probabilities = self._hidden_composition_probabilities(rng)
        return JointDistributionDraw(
            draw_index=draw_index,
            public_columns=self.public_columns,
            hidden_columns=self.hidden_columns,
            cells=self.cells,
            probabilities=probabilities,
            total_weight=self.total_weight,
            weight_column=weight_column,
            target_column=target_column,
        )

    def _hidden_composition_probabilities(
        self,
        rng: random.Random,
    ) -> tuple[float, ...]:
        probabilities = [0.0] * len(self.cells)
        for public_mass, indices in self._public_fiber_indices():
            conditional = [
                self.cells[index].probability / public_mass for index in indices
            ]
            if self.method == "bootstrap":
                fiber_probabilities = self._bootstrap_conditional_probabilities(
                    rng,
                    conditional,
                    public_mass=public_mass,
                )
            else:
                fiber_probabilities = self._bayesian_conditional_probabilities(
                    rng,
                    conditional,
                    public_mass=public_mass,
                )
            for index, conditional_probability in zip(
                indices,
                fiber_probabilities,
                strict=True,
            ):
                probabilities[index] = public_mass * conditional_probability
        return tuple(probabilities)

    def _public_fiber_indices(self) -> tuple[tuple[float, tuple[int, ...]], ...]:
        grouped: dict[tuple[Hashable, ...], list[int]] = {}
        for index, cell in enumerate(self.cells):
            grouped.setdefault(cell.public_value, []).append(index)
        fibers = []
        for indices in grouped.values():
            public_mass = sum(self.cells[index].probability for index in indices)
            if public_mass > 0:
                fibers.append((public_mass, tuple(indices)))
        return tuple(fibers)

    def _bayesian_conditional_probabilities(
        self,
        rng: random.Random,
        conditional: Sequence[float],
        *,
        public_mass: float,
    ) -> tuple[float, ...]:
        effective_n = self._effective_sample_size()
        gammas = [
            rng.gammavariate(
                max(probability * public_mass * effective_n, self.smoothing),
                1.0,
            )
            for probability in conditional
        ]
        total = sum(gammas)
        if total <= 0:
            return tuple(conditional)
        return tuple(value / total for value in gammas)

    def _bootstrap_conditional_probabilities(
        self,
        rng: random.Random,
        conditional: Sequence[float],
        *,
        public_mass: float,
    ) -> tuple[float, ...]:
        draw_count = max(1, int(round(public_mass * self._effective_sample_size())))
        sampled = rng.choices(
            range(len(conditional)),
            weights=conditional,
            k=draw_count,
        )
        counts = [0] * len(conditional)
        for index in sampled:
            counts[index] += 1
        return tuple(count / draw_count for count in counts)

    def _effective_sample_size(self) -> float:
        return (
            float(self.rows_seen)
            if self.effective_sample_size is None
            else float(self.effective_sample_size)
        )


@dataclass(frozen=True)
class UncertaintyMetricSummary:
    """Posterior/bootstrap summary for one scalar output."""

    metric: str
    count: int
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    lower: float | None
    median: float | None
    upper: float | None
    maximum: float | None
    confidence_level: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "count": self.count,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "lower": self.lower,
            "median": self.median,
            "upper": self.upper,
            "maximum": self.maximum,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class HiddenCompositionUncertaintyRow:
    """One posterior/bootstrap draw evaluated by a public-descent audit."""

    draw_index: int
    observed_value: float | None
    lower: float | None
    upper: float | None
    ambiguity: float | None
    public_adequate: bool | None
    status: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "draw_index": self.draw_index,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "public_adequate": self.public_adequate,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class HiddenCompositionUncertaintyReport(ReportArtifactMixin):
    """Posterior/bootstrap uncertainty over hidden-composition audits."""

    joint_model: NonparametricJointDistribution
    rows: tuple[HiddenCompositionUncertaintyRow, ...]
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    target_name: str
    q_name: str
    q_description: str
    ambiguity_limit: float | None = None
    confidence_level: float = 0.9
    seed: int | None = None
    preserve_public_law: bool = True
    title: str = "Hidden-Composition Uncertainty Report"

    @property
    def draw_count(self) -> int:
        return len(self.rows)

    @property
    def successful_draws(self) -> int:
        return sum(row.error is None for row in self.rows)

    @property
    def error_count(self) -> int:
        return sum(row.error is not None for row in self.rows)

    @property
    def failed_draws(self) -> int:
        return sum(row.status == "fail" for row in self.rows)

    @property
    def failure_rate(self) -> float | None:
        if self.ambiguity_limit is None:
            return None
        if self.successful_draws == 0:
            return None
        return self.failed_draws / self.successful_draws

    @property
    def public_adequate_rate(self) -> float | None:
        evaluated = [row for row in self.rows if row.public_adequate is not None]
        if not evaluated:
            return None
        return sum(bool(row.public_adequate) for row in evaluated) / len(evaluated)

    @property
    def observed_summary(self) -> UncertaintyMetricSummary:
        return _metric_summary(
            "observed_value",
            (row.observed_value for row in self.rows),
            confidence_level=self.confidence_level,
        )

    @property
    def lower_summary(self) -> UncertaintyMetricSummary:
        return _metric_summary(
            "lower",
            (row.lower for row in self.rows),
            confidence_level=self.confidence_level,
        )

    @property
    def upper_summary(self) -> UncertaintyMetricSummary:
        return _metric_summary(
            "upper",
            (row.upper for row in self.rows),
            confidence_level=self.confidence_level,
        )

    @property
    def ambiguity_summary(self) -> UncertaintyMetricSummary:
        return _metric_summary(
            "ambiguity",
            (row.ambiguity for row in self.rows),
            confidence_level=self.confidence_level,
        )

    @property
    def metric_summaries(self) -> tuple[UncertaintyMetricSummary, ...]:
        return (
            self.observed_summary,
            self.lower_summary,
            self.upper_summary,
            self.ambiguity_summary,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "draw_count": self.draw_count,
            "successful_draws": self.successful_draws,
            "error_count": self.error_count,
            "failed_draws": self.failed_draws,
            "failure_rate": self.failure_rate,
            "public_adequate_rate": self.public_adequate_rate,
            "public_columns": self.public_columns,
            "hidden_columns": self.hidden_columns,
            "target_name": self.target_name,
            "q_name": self.q_name,
            "q_description": self.q_description,
            "ambiguity_limit": self.ambiguity_limit,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
            "preserve_public_law": self.preserve_public_law,
            "joint_model": self.joint_model.as_dict(),
            "metric_summaries": [row.as_dict() for row in self.metric_summaries],
            "rows": [row.as_dict() for row in self.rows],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- Joint model method: {self.joint_model.method}",
            f"- Joint cells: {self.joint_model.cell_count}",
            f"- Draws: {self.successful_draws}/{self.draw_count} successful",
            f"- Draw errors: {self.error_count}",
            f"- Q preset: {self.q_name}",
            f"- Confidence level: {100.0 * self.confidence_level:g}%",
            f"- Public law preserved: {'yes' if self.preserve_public_law else 'no'}",
            "- Claim failure rate: "
            f"{_format_optional_rate(self.failure_rate, missing='not evaluated')}",
            "- Public adequacy rate: "
            f"{_format_optional_rate(self.public_adequate_rate, missing='not evaluated')}",
        ]
        if self.ambiguity_limit is not None:
            lines.append(f"- Ambiguity limit: {self.ambiguity_limit:.4f}")
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "This report samples hidden-cell masses from a fitted "
                "nonparametric joint distribution and reruns the public-descent "
                "audit on each sampled composition. By default it preserves the "
                "observed public law and resamples hidden composition within "
                "public fibers. It is "
                "model-assisted, not a distribution-free robustness guarantee.",
                "",
                "## Metric Summaries",
                "",
                "| metric | mean | sd | lower | median | upper | min | max |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in self.metric_summaries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.metric,
                        _format_optional_float(row.mean),
                        _format_optional_float(row.standard_deviation),
                        _format_optional_float(row.lower),
                        _format_optional_float(row.median),
                        _format_optional_float(row.upper),
                        _format_optional_float(row.minimum),
                        _format_optional_float(row.maximum),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## Draws",
                "",
                "| draw | status | observed | lower | upper | ambiguity | adequate |",
                "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in self.rows[:50]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.draw_index),
                        row.status,
                        _format_optional_float(row.observed_value),
                        _format_optional_float(row.lower),
                        _format_optional_float(row.upper),
                        _format_optional_float(row.ambiguity),
                        ""
                        if row.public_adequate is None
                        else ("yes" if row.public_adequate else "no"),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)


def fit_joint_distribution(
    data: Any,
    *,
    public: Sequence[str],
    hidden: Sequence[str],
    target: TabularTarget,
    weight: str | None = None,
    method: str = "bayesian_bootstrap",
    min_cell_weight: float = 1.0,
    effective_sample_size: float | None = None,
    smoothing: float = 1e-9,
) -> NonparametricJointDistribution:
    """Fit a nonparametric joint law over retained public/hidden cells."""

    grouped = from_dataframe(
        data,
        public=public,
        hidden=hidden,
        target=target,
        weight=weight,
        min_cell_weight=min_cell_weight,
        q="observed",
    )
    cells = tuple(
        JointCell(
            hidden_cell=state,
            public_value=grouped.problem.public_map[state],
            probability=float(grouped.cell_weights[state]),
            total_weight=float(grouped.cell_weights[state] * grouped.total_weight),
            target_value=float(grouped.problem.estimand_map[state]),
        )
        for state in grouped.problem.states
    )
    rows_seen = (
        1 if grouped.diagnostics is None else max(1, int(grouped.diagnostics.rows_seen))
    )
    return NonparametricJointDistribution(
        public_columns=tuple(public),
        hidden_columns=tuple(hidden),
        target_name=_target_label(target),
        cells=cells,
        total_weight=float(grouped.total_weight),
        rows_seen=rows_seen,
        method=method,
        effective_sample_size=effective_sample_size,
        smoothing=smoothing,
    )


def hidden_composition_uncertainty(
    data: Any | NonparametricJointDistribution | None = None,
    *,
    public: Sequence[str] | None = None,
    hidden: Sequence[str] | None = None,
    target: TabularTarget | None = None,
    weight: str | None = None,
    joint_model: NonparametricJointDistribution | None = None,
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
    title: str = "Hidden-Composition Uncertainty Report",
) -> HiddenCompositionUncertaintyReport:
    """Summarize posterior/bootstrap uncertainty over hidden composition."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if ambiguity_limit is not None and ambiguity_limit < 0:
        raise ValueError("ambiguity_limit must be non-negative")

    model = joint_model
    if isinstance(data, NonparametricJointDistribution):
        if model is not None and model is not data:
            raise ValueError(
                "pass either data as a joint model or joint_model, not both"
            )
        model = data
    if model is None:
        if data is None:
            raise TypeError("data is required when joint_model is not supplied")
        if public is None:
            raise TypeError("public is required when fitting a joint model")
        if hidden is None:
            raise TypeError("hidden is required when fitting a joint model")
        if target is None:
            raise TypeError("target is required when fitting a joint model")
        model = fit_joint_distribution(
            data,
            public=public,
            hidden=hidden,
            target=target,
            weight=weight,
            method=method,
            min_cell_weight=min_cell_weight,
            effective_sample_size=effective_sample_size,
            smoothing=smoothing,
        )

    public_tuple = tuple(public) if public is not None else model.public_columns
    hidden_tuple = tuple(hidden) if hidden is not None else model.hidden_columns
    target_label = _target_label(target) if target is not None else model.target_name
    rows = []
    q_name = str(q)
    q_description = str(q)
    draw_iterator = (
        model.iter_hidden_composition_draws(draws, seed=seed)
        if preserve_public_law
        else model.iter_draws(draws, seed=seed)
    )
    for draw in draw_iterator:
        try:
            report = public_descent_report(
                draw.records(),
                public=public_tuple,
                hidden=hidden_tuple,
                target=draw.target_column,
                weight=draw.weight_column,
                min_cell_weight=0.0,
                q=q,
                top=0,
                title=f"{title} Draw {draw.draw_index}",
            )
        except Exception as exc:  # pragma: no cover - depends on caller data shape
            rows.append(
                HiddenCompositionUncertaintyRow(
                    draw_index=draw.draw_index,
                    observed_value=None,
                    lower=None,
                    upper=None,
                    ambiguity=None,
                    public_adequate=None,
                    status="error",
                    error=str(exc),
                )
            )
            continue
        q_name = report.grouped.q_name
        q_description = report.grouped.q_description
        status = "inconclusive"
        if ambiguity_limit is not None:
            status = "pass" if report.interval.diameter <= ambiguity_limit else "fail"
        rows.append(
            HiddenCompositionUncertaintyRow(
                draw_index=draw.draw_index,
                observed_value=report.observed_value,
                lower=report.interval.lower,
                upper=report.interval.upper,
                ambiguity=report.interval.diameter,
                public_adequate=report.public_adequate,
                status=status,
            )
        )

    return HiddenCompositionUncertaintyReport(
        joint_model=model,
        rows=tuple(rows),
        public_columns=public_tuple,
        hidden_columns=hidden_tuple,
        target_name=target_label,
        q_name=q_name,
        q_description=q_description,
        ambiguity_limit=ambiguity_limit,
        confidence_level=confidence_level,
        seed=seed,
        preserve_public_law=preserve_public_law,
        title=title,
    )


def _normalize_joint_method(method: str) -> str:
    key = method.strip().lower().replace("-", "_")
    aliases = {
        "bayesian": "bayesian_bootstrap",
        "bayesian_bootstrap": "bayesian_bootstrap",
        "dirichlet": "bayesian_bootstrap",
        "posterior": "bayesian_bootstrap",
        "posterior_bootstrap": "bayesian_bootstrap",
        "bootstrap": "bootstrap",
        "multinomial": "bootstrap",
        "multinomial_bootstrap": "bootstrap",
        "nonparametric_bootstrap": "bootstrap",
        "empirical": "empirical",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            "method must be 'bayesian_bootstrap', 'bootstrap', or 'empirical'"
        ) from exc


def _metric_summary(
    metric: str,
    values: Sequence[float | None] | Any,
    *,
    confidence_level: float,
) -> UncertaintyMetricSummary:
    finite = sorted(float(value) for value in values if value is not None)
    if not finite:
        return UncertaintyMetricSummary(
            metric=metric,
            count=0,
            mean=None,
            standard_deviation=None,
            minimum=None,
            lower=None,
            median=None,
            upper=None,
            maximum=None,
            confidence_level=confidence_level,
        )
    mean = sum(finite) / len(finite)
    variance = (
        0.0
        if len(finite) == 1
        else sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    )
    alpha = (1.0 - confidence_level) / 2.0
    return UncertaintyMetricSummary(
        metric=metric,
        count=len(finite),
        mean=mean,
        standard_deviation=sqrt(variance),
        minimum=finite[0],
        lower=_quantile(finite, alpha),
        median=_quantile(finite, 0.5),
        upper=_quantile(finite, 1.0 - alpha),
        maximum=finite[-1],
        confidence_level=confidence_level,
    )


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values cannot be empty")
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]
    position = probability * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _format_optional_rate(value: float | None, *, missing: str = "") -> str:
    return missing if value is None else f"{100.0 * value:.1f}%"


def _target_label(target: Any) -> str:
    if isinstance(target, str):
        return target
    return str(getattr(target, "name", type(target).__name__))


def joint_draw_records(
    draw: JointDistributionDraw | Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return records from a draw or pass through an existing record sequence."""

    if isinstance(draw, JointDistributionDraw):
        return draw.records()
    return tuple(draw)


__all__ = [
    "HiddenCompositionUncertaintyReport",
    "HiddenCompositionUncertaintyRow",
    "JointCell",
    "JointDistributionDraw",
    "NonparametricJointDistribution",
    "UncertaintyMetricSummary",
    "fit_joint_distribution",
    "hidden_composition_uncertainty",
    "joint_draw_records",
]
