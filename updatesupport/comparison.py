"""Robust comparison and ranking reports under hidden recomposition."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Any, Hashable, Mapping, Sequence

from .artifacts import ReportArtifactMixin
from .claim import DecisionResult, threshold_decision
from .data import _iter_records, _record_value, _row_weight
from .metrics import RowMetric, evaluate_target, target_name
from .report import public_descent_report

ComparisonTarget = str | RowMetric

_VALUE_COLUMN = "__updatesupport_value__"
_MARGIN_COLUMN = "__updatesupport_margin__"
_WEIGHT_COLUMN = "__updatesupport_weight__"


@dataclass(frozen=True)
class ComparisonItemResult:
    """One alternative's hidden-composition interval."""

    item: Hashable
    rank: int
    observed_value: float
    lower: float
    upper: float
    ambiguity: float
    public_adequate: bool
    q_name: str
    q_description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "rank": self.rank,
            "observed_value": self.observed_value,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "public_adequate": self.public_adequate,
            "q_name": self.q_name,
            "q_description": self.q_description,
        }


@dataclass(frozen=True)
class PairwiseComparisonResult:
    """One pairwise margin audit in observed-rank order."""

    preferred_item: Hashable
    compared_item: Hashable
    preferred_rank: int
    compared_rank: int
    observed_margin: float
    lower: float
    upper: float
    ambiguity: float
    margin_threshold: float
    decision: DecisionResult
    q_name: str
    q_description: str

    @property
    def robust_order(self) -> bool:
        return (
            self.decision.invariant
            and self.decision.certified_decision == self.decision.rule.pass_label
        )

    @property
    def threshold_crossed(self) -> bool:
        return self.decision.threshold_crossed

    def as_dict(self) -> dict[str, Any]:
        return {
            "preferred_item": self.preferred_item,
            "compared_item": self.compared_item,
            "preferred_rank": self.preferred_rank,
            "compared_rank": self.compared_rank,
            "observed_margin": self.observed_margin,
            "lower": self.lower,
            "upper": self.upper,
            "ambiguity": self.ambiguity,
            "margin_threshold": self.margin_threshold,
            "robust_order": self.robust_order,
            "threshold_crossed": self.threshold_crossed,
            "q_name": self.q_name,
            "q_description": self.q_description,
            "decision_result": self.decision.as_dict(),
        }


@dataclass(frozen=True)
class RobustComparisonReport(ReportArtifactMixin):
    """Robust comparison/ranking report for several alternatives."""

    title: str
    item_column: str
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    target: str
    target_description: str
    observed_label: str
    higher_is_better: bool
    margin_threshold: float
    q_name: str
    q_description: str
    observed_order: tuple[Hashable, ...]
    observed_winner: Hashable
    certified_winner: Hashable | None
    status: str
    item_results: tuple[ComparisonItemResult, ...]
    pairwise_results: tuple[PairwiseComparisonResult, ...]

    @property
    def winner_stable(self) -> bool:
        return self.certified_winner == self.observed_winner

    @property
    def full_ranking_stable(self) -> bool:
        return bool(self.pairwise_results) and all(
            row.robust_order for row in self.pairwise_results
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "item_column": self.item_column,
            "public_columns": self.public_columns,
            "hidden_columns": self.hidden_columns,
            "target": self.target,
            "target_description": self.target_description,
            "observed_label": self.observed_label,
            "higher_is_better": self.higher_is_better,
            "margin_threshold": self.margin_threshold,
            "q_name": self.q_name,
            "q_description": self.q_description,
            "observed_order": self.observed_order,
            "observed_winner": self.observed_winner,
            "certified_winner": self.certified_winner,
            "winner_stable": self.winner_stable,
            "full_ranking_stable": self.full_ranking_stable,
            "status": self.status,
            "items": [row.as_dict() for row in self.item_results],
            "pairwise_margins": [row.as_dict() for row in self.pairwise_results],
        }

    def to_tables(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Return named tables for structured export."""

        return {
            "summary": (
                {
                    "title": self.title,
                    "item_column": self.item_column,
                    "target": self.target,
                    "target_description": self.target_description,
                    "observed_label": self.observed_label,
                    "higher_is_better": self.higher_is_better,
                    "margin_threshold": self.margin_threshold,
                    "q_name": self.q_name,
                    "q_description": self.q_description,
                    "observed_winner": self.observed_winner,
                    "certified_winner": self.certified_winner,
                    "winner_stable": self.winner_stable,
                    "full_ranking_stable": self.full_ranking_stable,
                    "status": self.status,
                    "public_columns": self.public_columns,
                    "hidden_columns": self.hidden_columns,
                    "observed_order": self.observed_order,
                },
            ),
            "items": tuple(row.as_dict() for row in self.item_results),
            "pairwise_margins": tuple(row.as_dict() for row in self.pairwise_results),
        }

    def to_markdown(self, *, max_pairwise_rows: int = 20) -> str:
        """Render an analyst-facing Markdown interpretation."""

        direction = "higher is better" if self.higher_is_better else "lower is better"
        lines = [
            f"# {self.title}",
            "",
            "## Summary",
            "",
            f"- Objective: {direction}",
            f"- Observed winner: `{self.observed_winner}`",
            "- Certified winner: "
            + (
                f"`{self.certified_winner}`"
                if self.certified_winner is not None
                else "none"
            ),
            f"- Full ranking stable: {_yes_no(self.full_ranking_stable)}",
            f"- Q: `{self.q_name}`",
            f"- Status: `{self.status}`",
            "",
            "## Interpretation",
            "",
            _interpretation(self),
            "",
            "A pairwise row is certified when the lower endpoint of the "
            "preferred-minus-compared margin remains above the margin threshold "
            "under the same admissible hidden-composition shift. This is not the "
            "same thing as comparing independent item intervals.",
            "",
            "## Item Intervals",
            "",
            "| rank | item | observed | lower | upper | ambiguity |",
            "|---:|:---|---:|---:|---:|---:|",
        ]
        for row in self.item_results:
            lines.append(
                "| "
                f"{row.rank} | "
                f"{row.item} | "
                f"{_format_float(row.observed_value)} | "
                f"{_format_float(row.lower)} | "
                f"{_format_float(row.upper)} | "
                f"{_format_float(row.ambiguity)} |"
            )

        lines.extend(
            [
                "",
                "## Pairwise Margins",
                "",
                "| preferred | compared | observed margin | lower | upper | certified |",
                "|:---|:---|---:|---:|---:|:---:|",
            ]
        )
        displayed = self.pairwise_results[:max_pairwise_rows]
        for row in displayed:
            lines.append(
                "| "
                f"{row.preferred_item} | "
                f"{row.compared_item} | "
                f"{_format_float(row.observed_margin)} | "
                f"{_format_float(row.lower)} | "
                f"{_format_float(row.upper)} | "
                f"{_yes_no(row.robust_order)} |"
            )
        hidden_count = len(self.pairwise_results) - len(displayed)
        if hidden_count > 0:
            lines.append(f"| ... | ... | ... | ... | ... | {hidden_count} more rows |")

        lines.extend(
            [
                "",
                "## Assumptions And Limitations",
                "",
                "- Alternatives must share the same retained hidden-cell support and "
                "cell weights in this first API slice.",
                "- `hidden` columns are observed by the analyst but omitted from the "
                "coarse public comparison being stress-tested.",
                "- The result is relative to the chosen refinement and Q family; it "
                "does not certify robustness to every possible composition shift.",
            ]
        )
        return "\n".join(lines)


def robust_comparison_report(
    data: Any,
    *,
    item: str,
    public: Sequence[str],
    hidden: Sequence[str],
    target: ComparisonTarget,
    weight: str | None = None,
    items: Sequence[Hashable] | None = None,
    q: Any = "saturated",
    q_radius: float | None = None,
    higher_is_better: bool = True,
    margin_threshold: float = 0.0,
    min_cell_weight: float = 1.0,
    weight_tolerance: float = 1e-9,
    top: int = 5,
    title: str = "Robust Comparison Report",
    target_description: str = "target value",
    observed_label: str = "Observed value",
) -> RobustComparisonReport:
    """Audit whether a comparison or ranking survives hidden recomposition.

    Input data should be long-form: one row per alternative and hidden cell, or
    multiple rows that aggregate to that shape. Pairwise ranking certificates use
    margin targets, not independent interval overlap.
    """

    if len(public) == 0:
        raise ValueError("public must contain at least one column")
    if len(hidden) == 0:
        raise ValueError("hidden must contain at least one column")
    missing_public = [column for column in public if column not in hidden]
    if missing_public:
        raise ValueError(
            f"public columns must also be hidden columns: {missing_public!r}"
        )
    if item in hidden:
        raise ValueError("item column should not be included in hidden columns")
    if min_cell_weight < 0:
        raise ValueError("min_cell_weight must be non-negative")
    if weight_tolerance < 0:
        raise ValueError("weight_tolerance must be non-negative")
    if not isfinite(float(margin_threshold)):
        raise ValueError("margin_threshold must be finite")

    compiled = _compile_long_form_comparison(
        data,
        item=item,
        public=tuple(public),
        hidden=tuple(hidden),
        target=target,
        weight=weight,
        items=items,
        weight_tolerance=float(weight_tolerance),
    )
    if len(compiled.items) < 2:
        raise ValueError("robust comparison requires at least two alternatives")

    item_results_by_item: dict[Hashable, ComparisonItemResult] = {}
    q_name = "unknown"
    q_description = ""
    for item_value in compiled.items:
        rows = _item_rows(compiled, item_value)
        report = public_descent_report(
            rows,
            public=public,
            hidden=hidden,
            target=_VALUE_COLUMN,
            weight=_WEIGHT_COLUMN,
            q=q,
            q_radius=q_radius,
            min_cell_weight=min_cell_weight,
            top=top,
            title=f"{title}: {item_value}",
            target_description=target_description,
            observed_label=observed_label,
        )
        q_name = report.grouped.q_name
        q_description = report.grouped.q_description
        item_results_by_item[item_value] = ComparisonItemResult(
            item=item_value,
            rank=0,
            observed_value=report.observed_value,
            lower=report.interval.lower,
            upper=report.interval.upper,
            ambiguity=report.interval.diameter,
            public_adequate=report.public_adequate,
            q_name=report.grouped.q_name,
            q_description=report.grouped.q_description,
        )

    observed_order = tuple(
        sorted(
            compiled.items,
            key=lambda item_value: item_results_by_item[item_value].observed_value,
            reverse=higher_is_better,
        )
    )
    ranked_items = tuple(
        _replace_item_rank(item_results_by_item[item_value], rank)
        for rank, item_value in enumerate(observed_order, start=1)
    )

    rank_by_item = {row.item: row.rank for row in ranked_items}
    pairwise: list[PairwiseComparisonResult] = []
    certified_label = "order_certified"
    uncertified_label = "order_not_certified"
    decision = threshold_decision(
        ">=",
        float(margin_threshold),
        label=f"pairwise margin >= {float(margin_threshold):g}",
        pass_label=certified_label,
        fail_label=uncertified_label,
    )
    for preferred_index, preferred_item in enumerate(observed_order):
        for compared_item in observed_order[preferred_index + 1 :]:
            rows = _pairwise_rows(
                compiled,
                preferred_item,
                compared_item,
                higher_is_better=higher_is_better,
            )
            report = public_descent_report(
                rows,
                public=public,
                hidden=hidden,
                target=_MARGIN_COLUMN,
                weight=_WEIGHT_COLUMN,
                q=q,
                q_radius=q_radius,
                min_cell_weight=min_cell_weight,
                top=top,
                title=f"{title}: {preferred_item} vs {compared_item}",
                target_description="pairwise comparison margin",
                observed_label="Observed pairwise margin",
            )
            pair_decision = decision.interval_result(
                observed_value=report.observed_value,
                lower=report.interval.lower,
                upper=report.interval.upper,
            )
            pairwise.append(
                PairwiseComparisonResult(
                    preferred_item=preferred_item,
                    compared_item=compared_item,
                    preferred_rank=rank_by_item[preferred_item],
                    compared_rank=rank_by_item[compared_item],
                    observed_margin=report.observed_value,
                    lower=report.interval.lower,
                    upper=report.interval.upper,
                    ambiguity=report.interval.diameter,
                    margin_threshold=float(margin_threshold),
                    decision=pair_decision,
                    q_name=report.grouped.q_name,
                    q_description=report.grouped.q_description,
                )
            )

    observed_winner = observed_order[0]
    winner_pairwise = [row for row in pairwise if row.preferred_item == observed_winner]
    winner_stable = all(row.robust_order for row in winner_pairwise)
    full_ranking_stable = all(row.robust_order for row in pairwise)
    if full_ranking_stable:
        status = "full_ranking_stable"
    elif winner_stable:
        status = "winner_stable"
    else:
        status = "ambiguous_winner"

    return RobustComparisonReport(
        title=title,
        item_column=item,
        public_columns=tuple(public),
        hidden_columns=tuple(hidden),
        target=target_name(target),
        target_description=target_description,
        observed_label=observed_label,
        higher_is_better=bool(higher_is_better),
        margin_threshold=float(margin_threshold),
        q_name=q_name,
        q_description=q_description,
        observed_order=observed_order,
        observed_winner=observed_winner,
        certified_winner=observed_winner if winner_stable else None,
        status=status,
        item_results=ranked_items,
        pairwise_results=tuple(pairwise),
    )


def robust_ranking_report(*args: Any, **kwargs: Any) -> RobustComparisonReport:
    """Alias for :func:`robust_comparison_report`."""

    return robust_comparison_report(*args, **kwargs)


@dataclass(frozen=True)
class _CompiledComparison:
    public: tuple[str, ...]
    hidden: tuple[str, ...]
    items: tuple[Hashable, ...]
    hidden_keys: tuple[tuple[Hashable, ...], ...]
    public_by_hidden: Mapping[tuple[Hashable, ...], tuple[Hashable, ...]]
    value_by_cell_item: Mapping[tuple[tuple[Hashable, ...], Hashable], float]
    weight_by_hidden: Mapping[tuple[Hashable, ...], float]


def _compile_long_form_comparison(
    data: Any,
    *,
    item: str,
    public: tuple[str, ...],
    hidden: tuple[str, ...],
    target: ComparisonTarget,
    weight: str | None,
    items: Sequence[Hashable] | None,
    weight_tolerance: float,
) -> _CompiledComparison:
    requested_items = None if items is None else tuple(items)
    requested_item_set = None if requested_items is None else set(requested_items)
    seen_items: list[Hashable] = []
    seen_item_set: set[Hashable] = set()
    public_by_hidden: dict[tuple[Hashable, ...], tuple[Hashable, ...]] = {}
    hidden_order: list[tuple[Hashable, ...]] = []
    hidden_seen: set[tuple[Hashable, ...]] = set()
    weight_sum: dict[tuple[tuple[Hashable, ...], Hashable], float] = defaultdict(float)
    target_sum: dict[tuple[tuple[Hashable, ...], Hashable], float] = defaultdict(float)

    for row_number, row in enumerate(_iter_records(data), start=1):
        item_value = _record_value(row, item, row_number=row_number)
        if requested_item_set is not None and item_value not in requested_item_set:
            continue
        if item_value not in seen_item_set:
            seen_item_set.add(item_value)
            seen_items.append(item_value)
        hidden_key = tuple(
            _record_value(row, column, row_number=row_number) for column in hidden
        )
        public_key = tuple(
            _record_value(row, column, row_number=row_number) for column in public
        )
        existing_public = public_by_hidden.get(hidden_key)
        if existing_public is not None and existing_public != public_key:
            raise ValueError(
                f"hidden cell {hidden_key!r} maps to multiple public cells"
            )
        public_by_hidden[hidden_key] = public_key
        if hidden_key not in hidden_seen:
            hidden_seen.add(hidden_key)
            hidden_order.append(hidden_key)
        row_weight = _row_weight(row, weight, row_number=row_number)
        row_target = evaluate_target(
            row,
            target,
            get_value=lambda record, column: _record_value(
                record,
                column,
                row_number=row_number,
            ),
        )
        key = (hidden_key, item_value)
        weight_sum[key] += row_weight
        target_sum[key] += row_weight * row_target

    selected_items = (
        requested_items if requested_items is not None else tuple(seen_items)
    )
    missing_items = [
        item_value for item_value in selected_items if item_value not in seen_item_set
    ]
    if missing_items:
        raise ValueError(f"requested comparison items not found: {missing_items!r}")

    missing_cells: list[tuple[tuple[Hashable, ...], Hashable]] = []
    weight_by_hidden: dict[tuple[Hashable, ...], float] = {}
    value_by_cell_item: dict[tuple[tuple[Hashable, ...], Hashable], float] = {}
    for hidden_key in hidden_order:
        cell_weights: list[float] = []
        for item_value in selected_items:
            key = (hidden_key, item_value)
            cell_weight = weight_sum.get(key, 0.0)
            if cell_weight <= 0:
                missing_cells.append(key)
                continue
            cell_weights.append(cell_weight)
            value_by_cell_item[key] = target_sum[key] / cell_weight
        if len(cell_weights) != len(selected_items):
            continue
        min_weight = min(cell_weights)
        max_weight = max(cell_weights)
        tolerance = weight_tolerance * max(1.0, abs(max_weight))
        if max_weight - min_weight > tolerance:
            raise ValueError(
                "comparison alternatives must share the same hidden-cell weights; "
                f"hidden cell {hidden_key!r} has weights {cell_weights!r}"
            )
        weight_by_hidden[hidden_key] = sum(cell_weights) / len(cell_weights)

    if missing_cells:
        preview = ", ".join(
            f"{hidden_key!r}/{item_value!r}"
            for hidden_key, item_value in missing_cells[:5]
        )
        suffix = (
            "" if len(missing_cells) <= 5 else f", and {len(missing_cells) - 5} more"
        )
        raise ValueError(
            "comparison alternatives must share the same hidden-cell support; "
            f"missing cells: {preview}{suffix}"
        )

    return _CompiledComparison(
        public=public,
        hidden=hidden,
        items=tuple(selected_items),
        hidden_keys=tuple(hidden_order),
        public_by_hidden=public_by_hidden,
        value_by_cell_item=value_by_cell_item,
        weight_by_hidden=weight_by_hidden,
    )


def _item_rows(
    compiled: _CompiledComparison,
    item_value: Hashable,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for hidden_key in compiled.hidden_keys:
        row = _hidden_row(compiled, hidden_key)
        row[_VALUE_COLUMN] = compiled.value_by_cell_item[(hidden_key, item_value)]
        row[_WEIGHT_COLUMN] = compiled.weight_by_hidden[hidden_key]
        rows.append(row)
    return tuple(rows)


def _pairwise_rows(
    compiled: _CompiledComparison,
    preferred_item: Hashable,
    compared_item: Hashable,
    *,
    higher_is_better: bool,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for hidden_key in compiled.hidden_keys:
        preferred_value = compiled.value_by_cell_item[(hidden_key, preferred_item)]
        compared_value = compiled.value_by_cell_item[(hidden_key, compared_item)]
        margin = (
            preferred_value - compared_value
            if higher_is_better
            else compared_value - preferred_value
        )
        row = _hidden_row(compiled, hidden_key)
        row[_MARGIN_COLUMN] = margin
        row[_WEIGHT_COLUMN] = compiled.weight_by_hidden[hidden_key]
        rows.append(row)
    return tuple(rows)


def _hidden_row(
    compiled: _CompiledComparison,
    hidden_key: tuple[Hashable, ...],
) -> dict[str, Any]:
    return {column: hidden_key[index] for index, column in enumerate(compiled.hidden)}


def _replace_item_rank(
    row: ComparisonItemResult,
    rank: int,
) -> ComparisonItemResult:
    return ComparisonItemResult(
        item=row.item,
        rank=rank,
        observed_value=row.observed_value,
        lower=row.lower,
        upper=row.upper,
        ambiguity=row.ambiguity,
        public_adequate=row.public_adequate,
        q_name=row.q_name,
        q_description=row.q_description,
    )


def _interpretation(report: RobustComparisonReport) -> str:
    if report.full_ranking_stable:
        return (
            "Every observed pairwise ordering remains certified under the chosen "
            "hidden-composition stress test. The observed winner and full ranking "
            "are stable relative to this refinement and Q family."
        )
    if report.winner_stable:
        return (
            "The observed winner beats every lower-ranked alternative under the "
            "chosen stress test, but at least one lower part of the ranking is not "
            "fully certified."
        )
    challengers = [
        row.compared_item
        for row in report.pairwise_results
        if row.preferred_item == report.observed_winner and not row.robust_order
    ]
    challenger_text = ", ".join(str(item) for item in challengers) or "another item"
    return (
        f"The observed winner is not certified. At least {challenger_text} can "
        "become competitive under an admissible recomposition of the hidden cells."
    )


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(float(value))
    if magnitude != 0 and (magnitude < 1e-4 or magnitude >= 1e6):
        return f"{float(value):.4e}"
    return f"{float(value):.4f}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
