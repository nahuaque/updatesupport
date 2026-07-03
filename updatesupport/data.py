"""Tabular data compilers for finite update-support problems."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Hashable, Iterable, Mapping, Sequence

from .metrics import RowMetric, evaluate_target, target_description, target_name
from .problem import FiniteProblem
from .presets import QPreset, resolve_q_environment
from .targets import (
    LinearTarget,
    ProcedureTarget,
    ProcedureTargetContext,
    RatioTarget,
    UnsupportedTargetError,
    raise_if_unsupported_target,
)

TabularTarget = str | RowMetric | ProcedureTarget


@dataclass(frozen=True)
class DataDiagnostic:
    """One data preflight diagnostic emitted before solving."""

    code: str
    severity: str
    message: str
    count: int | None = None
    columns: tuple[str, ...] = ()
    public_value: tuple[Hashable, ...] | None = None
    hidden_cell: tuple[Hashable, ...] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "count": self.count,
            "columns": self.columns,
            "public_value": self.public_value,
            "hidden_cell": self.hidden_cell,
            "details": self.details,
        }


@dataclass(frozen=True)
class DataDiagnostics:
    """Pre-solve diagnostics for a compiled tabular problem."""

    rows_seen: int
    total_weight: float
    retained_weight: float
    dropped_weight: float
    hidden_cells: int
    retained_hidden_cells: int
    dropped_hidden_cells: int
    public_cells: int
    singleton_public_fibers: int
    constant_public_fibers: int
    diagnostics: tuple[DataDiagnostic, ...] = ()

    @property
    def dropped_weight_share(self) -> float:
        if self.total_weight <= 0:
            return 0.0
        return self.dropped_weight / self.total_weight

    @property
    def warning_count(self) -> int:
        return sum(1 for row in self.diagnostics if row.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for row in self.diagnostics if row.severity == "info")

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_seen": self.rows_seen,
            "total_weight": self.total_weight,
            "retained_weight": self.retained_weight,
            "dropped_weight": self.dropped_weight,
            "dropped_weight_share": self.dropped_weight_share,
            "hidden_cells": self.hidden_cells,
            "retained_hidden_cells": self.retained_hidden_cells,
            "dropped_hidden_cells": self.dropped_hidden_cells,
            "public_cells": self.public_cells,
            "singleton_public_fibers": self.singleton_public_fibers,
            "constant_public_fibers": self.constant_public_fibers,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "diagnostics": [row.as_dict() for row in self.diagnostics],
        }


@dataclass(frozen=True)
class GroupedProblem:
    """A finite problem compiled from grouped tabular observations."""

    problem: FiniteProblem
    public_law: dict[tuple[Hashable, ...], float]
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    target_column: str | RowMetric
    target_functional: LinearTarget | RatioTarget
    total_weight: float
    cell_weights: dict[tuple[Hashable, ...], float]
    q: QPreset | None = None
    q_name: str = "saturated"
    q_description: str = (
        "arbitrary reweighting among retained hidden cells inside each observed "
        "public cell"
    )
    diagnostics: DataDiagnostics | None = None
    target_procedure: ProcedureTarget | None = None
    target_procedure_context: ProcedureTargetContext | None = None


def from_dataframe(
    data: Any,
    *,
    public: Sequence[str] | None = None,
    hidden: Sequence[str] | None = None,
    target: TabularTarget | None = None,
    weight: str | None = None,
    public_columns: Sequence[str] | None = None,
    hidden_columns: Sequence[str] | None = None,
    target_column: TabularTarget | None = None,
    weight_column: str | None = None,
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    q_radius: float | None = None,
) -> GroupedProblem:
    """Compile tabular observations into a finite update-support problem.

    ``data`` may be a pandas-like dataframe with ``to_dict("records")`` or an
    iterable of row mappings. Each hidden cell receives the weighted empirical
    mean of ``target`` as its estimand value. If ``target`` is a
    :class:`ProcedureTarget`, the procedure is compiled once for the selected
    public representation and must return a column name or ``RowMetric``. The
    returned problem uses an environment built from the selected ``q`` preset.
    The default is
    ``q="saturated"``, which fixes the observed public law and allows arbitrary
    reweighting inside retained public fibers.

    The ``*_columns`` keyword names are accepted as compatibility aliases for
    ``public``, ``hidden``, ``target``, and ``weight``.
    """

    public = _resolve_sequence_arg(
        public, public_columns, primary_name="public", alias_name="public_columns"
    )
    hidden = _resolve_sequence_arg(
        hidden, hidden_columns, primary_name="hidden", alias_name="hidden_columns"
    )
    target = _resolve_scalar_arg(
        target, target_column, primary_name="target", alias_name="target_column"
    )
    weight = _resolve_scalar_arg(
        weight, weight_column, primary_name="weight", alias_name="weight_column"
    )
    if public is None:
        raise TypeError("from_dataframe() missing required keyword argument: 'public'")
    if hidden is None:
        raise TypeError("from_dataframe() missing required keyword argument: 'hidden'")
    if target is None:
        raise TypeError("from_dataframe() missing required keyword argument: 'target'")
    if min_cell_weight < 0:
        raise ValueError("min_cell_weight must be non-negative")

    public_columns_tuple = tuple(public)
    hidden_columns_tuple = tuple(hidden)
    if not public_columns_tuple:
        raise ValueError("public must contain at least one column")
    if not hidden_columns_tuple:
        raise ValueError("hidden must contain at least one column")

    missing_public = [
        column for column in public_columns_tuple if column not in hidden_columns_tuple
    ]
    if missing_public:
        raise ValueError(
            f"public columns must also be hidden columns: {missing_public!r}"
        )

    target_procedure = target if isinstance(target, ProcedureTarget) else None
    target_procedure_context = None
    if target_procedure is not None:
        data = _repeatable_data_for_procedure(data)
        target_procedure_context = ProcedureTargetContext(
            data=data,
            public=public_columns_tuple,
            hidden=hidden_columns_tuple,
            weight=weight,
            min_cell_weight=float(min_cell_weight),
            q=q,
            q_radius=q_radius,
        )
        target = _compile_procedure_target(
            target_procedure,
            target_procedure_context,
        )

    cell_weight: dict[tuple[Hashable, ...], float] = defaultdict(float)
    cell_target_sum: dict[tuple[Hashable, ...], float] = defaultdict(float)
    public_by_cell: dict[tuple[Hashable, ...], tuple[Hashable, ...]] = {}
    missing_category_counts: dict[str, int] = defaultdict(int)
    zero_weight_rows = 0
    observed_weight = 0.0

    rows_seen = 0
    for row_number, row in enumerate(_iter_records(data), start=1):
        rows_seen += 1
        values_by_column: dict[str, Hashable] = {}
        for column in hidden_columns_tuple:
            raw_value = _record_value(row, column, row_number=row_number)
            if _is_missing(raw_value):
                missing_category_counts[column] += 1
            values_by_column[column] = _hashable_category(raw_value)
        hidden_key = tuple(values_by_column[column] for column in hidden_columns_tuple)
        public_key = tuple(values_by_column[column] for column in public_columns_tuple)
        row_weight = _row_weight(row, weight, row_number=row_number)
        if row_weight == 0:
            zero_weight_rows += 1
        observed_weight += row_weight
        target_value = _target_value(row, target, row_number=row_number)
        cell_weight[hidden_key] += row_weight
        cell_target_sum[hidden_key] += row_weight * target_value
        public_by_cell[hidden_key] = public_key

    if rows_seen == 0:
        raise ValueError("data must contain at least one row")

    kept_cells = [
        cell
        for cell, mass in cell_weight.items()
        if mass >= min_cell_weight and mass > 0
    ]
    if not kept_cells:
        raise ValueError("no hidden cells remain after applying min_cell_weight")

    total_weight = sum(cell_weight[cell] for cell in kept_cells)
    states = tuple(sorted(kept_cells, key=str))
    public_map = {cell: public_by_cell[cell] for cell in states}
    estimand = {cell: cell_target_sum[cell] / cell_weight[cell] for cell in states}
    target_functional = LinearTarget(
        estimand,
        name=target_name(target),
        description=target_description(target),
        source="from_dataframe",
    )
    normalized_cell_weight = {cell: cell_weight[cell] / total_weight for cell in states}

    public_law: dict[tuple[Hashable, ...], float] = defaultdict(float)
    for cell, mass in normalized_cell_weight.items():
        public_law[public_map[cell]] += mass

    diagnostics = _data_diagnostics(
        rows_seen=rows_seen,
        observed_weight=observed_weight,
        retained_weight=total_weight,
        cell_weight=cell_weight,
        public_by_cell=public_by_cell,
        retained_cells=states,
        estimand=estimand,
        missing_category_counts=missing_category_counts,
        zero_weight_rows=zero_weight_rows,
        min_cell_weight=min_cell_weight,
    )

    q_environment = resolve_q_environment(
        q,
        public_law=dict(public_law),
        public_map=public_map,
        cell_weights=normalized_cell_weight,
        q_radius=q_radius,
    )
    problem = FiniteProblem(
        states=states,
        public=public_map,
        estimand=target_functional,
        environments=q_environment.environment,
    )
    return GroupedProblem(
        problem=problem,
        public_law=dict(public_law),
        public_columns=public_columns_tuple,
        hidden_columns=hidden_columns_tuple,
        target_column=target,
        target_functional=target_functional,
        target_procedure=target_procedure,
        target_procedure_context=target_procedure_context,
        total_weight=total_weight,
        cell_weights=normalized_cell_weight,
        q=q_environment.preset,
        q_name=q_environment.name,
        q_description=q_environment.description,
        diagnostics=diagnostics,
    )


def _data_diagnostics(
    *,
    rows_seen: int,
    observed_weight: float,
    retained_weight: float,
    cell_weight: Mapping[tuple[Hashable, ...], float],
    public_by_cell: Mapping[tuple[Hashable, ...], tuple[Hashable, ...]],
    retained_cells: Sequence[tuple[Hashable, ...]],
    estimand: Mapping[tuple[Hashable, ...], float],
    missing_category_counts: Mapping[str, int],
    zero_weight_rows: int,
    min_cell_weight: float,
) -> DataDiagnostics:
    retained_set = set(retained_cells)
    dropped_cells = tuple(cell for cell in cell_weight if cell not in retained_set)
    dropped_weight = sum(cell_weight[cell] for cell in dropped_cells)

    diagnostics: list[DataDiagnostic] = []
    if zero_weight_rows:
        diagnostics.append(
            DataDiagnostic(
                code="zero_weight_rows",
                severity="warning",
                message=(
                    "Rows with zero weight were read but do not affect retained "
                    "hidden-cell masses or target values."
                ),
                count=zero_weight_rows,
            )
        )
    missing_columns = tuple(
        column for column, count in missing_category_counts.items() if count
    )
    if missing_columns:
        diagnostics.append(
            DataDiagnostic(
                code="missing_category_values",
                severity="warning",
                message="Missing category values were encoded as 'NA'.",
                count=sum(
                    missing_category_counts[column] for column in missing_columns
                ),
                columns=missing_columns,
                details={
                    column: missing_category_counts[column]
                    for column in missing_columns
                },
            )
        )
    positive_dropped = tuple(cell for cell in dropped_cells if cell_weight[cell] > 0)
    if positive_dropped:
        diagnostics.append(
            DataDiagnostic(
                code="min_cell_weight_dropped_cells",
                severity="warning",
                message=(
                    "Hidden cells were dropped before solving because their weight "
                    "was below min_cell_weight."
                ),
                count=len(positive_dropped),
                details={
                    "min_cell_weight": min_cell_weight,
                    "dropped_weight": dropped_weight,
                    "dropped_weight_share": (
                        0.0
                        if observed_weight <= 0
                        else dropped_weight / observed_weight
                    ),
                },
            )
        )

    retained_by_public: dict[tuple[Hashable, ...], list[tuple[Hashable, ...]]] = (
        defaultdict(list)
    )
    for cell in retained_cells:
        retained_by_public[public_by_cell[cell]].append(cell)

    singleton_public_values = tuple(
        public_value
        for public_value, cells in retained_by_public.items()
        if len(cells) == 1
    )
    if singleton_public_values:
        diagnostics.append(
            DataDiagnostic(
                code="singleton_public_fibers",
                severity="info",
                message=(
                    "Some retained public cells contain only one retained hidden "
                    "cell; those cells cannot contribute within-fiber ambiguity."
                ),
                count=len(singleton_public_values),
                details={"public_values": singleton_public_values[:10]},
            )
        )

    constant_public_values = []
    for public_value, cells in retained_by_public.items():
        if len(cells) <= 1:
            continue
        values = [estimand[cell] for cell in cells]
        if max(values) - min(values) <= 1e-12:
            constant_public_values.append(public_value)
    if constant_public_values:
        diagnostics.append(
            DataDiagnostic(
                code="constant_target_public_fibers",
                severity="info",
                message=(
                    "Some retained public cells have constant hidden-cell target "
                    "values; those cells cannot contribute target ambiguity under "
                    "within-public-cell reweighting."
                ),
                count=len(constant_public_values),
                details={"public_values": tuple(constant_public_values[:10])},
            )
        )

    return DataDiagnostics(
        rows_seen=rows_seen,
        total_weight=observed_weight,
        retained_weight=retained_weight,
        dropped_weight=dropped_weight,
        hidden_cells=len(cell_weight),
        retained_hidden_cells=len(retained_cells),
        dropped_hidden_cells=len(dropped_cells),
        public_cells=len(retained_by_public),
        singleton_public_fibers=len(singleton_public_values),
        constant_public_fibers=len(constant_public_values),
        diagnostics=tuple(diagnostics),
    )


def _iter_records(data: Any) -> Iterable[Mapping[str, Any]]:
    if hasattr(data, "to_dict"):
        try:
            records = data.to_dict("records")
        except TypeError:
            records = data.to_dict(orient="records")
        yield from records
        return
    yield from data


def _repeatable_data_for_procedure(data: Any) -> Any:
    if hasattr(data, "to_dict"):
        return data
    if isinstance(data, Sequence) and not isinstance(data, str | bytes):
        return data
    return tuple(data)


def _compile_procedure_target(
    procedure: ProcedureTarget,
    context: ProcedureTargetContext,
) -> str | RowMetric:
    compiled = procedure.compile(context)
    if isinstance(compiled, ProcedureTarget):
        raise UnsupportedTargetError(
            "ProcedureTarget compilers must return a column name string or "
            "RowMetric, not another ProcedureTarget."
        )
    raise_if_unsupported_target(
        compiled,
        context=f"ProcedureTarget({procedure.name!r}) compiler",
    )
    if isinstance(compiled, RowMetric):
        return compiled
    if isinstance(compiled, str):
        return compiled
    raise TypeError(
        "ProcedureTarget compilers must return a column name string or RowMetric"
    )


def _resolve_sequence_arg(
    primary: Sequence[str] | None,
    alias: Sequence[str] | None,
    *,
    primary_name: str,
    alias_name: str,
) -> Sequence[str] | None:
    if primary is not None and alias is not None and tuple(primary) != tuple(alias):
        raise TypeError(f"use either {primary_name!r} or {alias_name!r}, not both")
    return primary if primary is not None else alias


def _resolve_scalar_arg(
    primary: Any | None,
    alias: Any | None,
    *,
    primary_name: str,
    alias_name: str,
) -> Any | None:
    if primary is not None and alias is not None and primary != alias:
        raise TypeError(f"use either {primary_name!r} or {alias_name!r}, not both")
    return primary if primary is not None else alias


def _record_value(row: Mapping[str, Any], column: str, *, row_number: int) -> Any:
    try:
        return row[column]
    except KeyError as exc:
        raise ValueError(f"row {row_number} is missing column {column!r}") from exc


def _row_weight(
    row: Mapping[str, Any], weight_column: str | None, *, row_number: int
) -> float:
    if weight_column is None:
        return 1.0
    value = float(_record_value(row, weight_column, row_number=row_number))
    if not isfinite(value):
        raise ValueError(f"row {row_number} has non-finite weight")
    if value < 0:
        raise ValueError("row weights must be non-negative")
    return value


def _target_value(
    row: Mapping[str, Any],
    target_column: str | RowMetric,
    *,
    row_number: int,
) -> float:
    try:
        return evaluate_target(
            row,
            target_column,
            get_value=lambda record, column: _record_value(
                record,
                column,
                row_number=row_number,
            ),
        )
    except ValueError as exc:
        raise ValueError(f"row {row_number} has invalid target") from exc


def _hashable_category(value: Any) -> Hashable:
    if _is_missing(value):
        return "NA"
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    text = str(value)
    if text in {"nan", "None", "<NA>", "NaT"}:
        return True
    try:
        unequal_to_self = value != value
    except TypeError:
        return False
    return bool(unequal_to_self) if isinstance(unequal_to_self, bool) else False
