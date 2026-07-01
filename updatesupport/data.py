"""Tabular data compilers for finite update-support problems."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Any, Hashable, Iterable, Mapping, Sequence

from .problem import FiniteProblem
from .presets import QPreset, resolve_q_environment


@dataclass(frozen=True)
class GroupedProblem:
    """A finite problem compiled from grouped tabular observations."""

    problem: FiniteProblem
    public_law: dict[tuple[Hashable, ...], float]
    public_columns: tuple[str, ...]
    hidden_columns: tuple[str, ...]
    target_column: str
    total_weight: float
    cell_weights: dict[tuple[Hashable, ...], float]
    q: QPreset | None = None
    q_name: str = "saturated"
    q_description: str = (
        "arbitrary reweighting among retained hidden cells inside each observed "
        "public cell"
    )


def from_dataframe(
    data: Any,
    *,
    public: Sequence[str] | None = None,
    hidden: Sequence[str] | None = None,
    target: str | None = None,
    weight: str | None = None,
    public_columns: Sequence[str] | None = None,
    hidden_columns: Sequence[str] | None = None,
    target_column: str | None = None,
    weight_column: str | None = None,
    min_cell_weight: float = 1.0,
    q: Any = "saturated",
    q_radius: float | None = None,
) -> GroupedProblem:
    """Compile tabular observations into a finite update-support problem.

    ``data`` may be a pandas-like dataframe with ``to_dict("records")`` or an
    iterable of row mappings. Each hidden cell receives the weighted empirical
    mean of ``target`` as its estimand value. The returned problem uses a
    environment built from the selected ``q`` preset. The default is
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
        column
        for column in public_columns_tuple
        if column not in hidden_columns_tuple
    ]
    if missing_public:
        raise ValueError(
            f"public columns must also be hidden columns: {missing_public!r}"
        )

    cell_weight: dict[tuple[Hashable, ...], float] = defaultdict(float)
    cell_target_sum: dict[tuple[Hashable, ...], float] = defaultdict(float)
    public_by_cell: dict[tuple[Hashable, ...], tuple[Hashable, ...]] = {}

    rows_seen = 0
    for row_number, row in enumerate(_iter_records(data), start=1):
        rows_seen += 1
        hidden_key = tuple(
            _hashable_category(
                _record_value(row, column, row_number=row_number)
            )
            for column in hidden_columns_tuple
        )
        public_key = tuple(
            _hashable_category(
                _record_value(row, column, row_number=row_number)
            )
            for column in public_columns_tuple
        )
        row_weight = _row_weight(row, weight, row_number=row_number)
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
    estimand = {
        cell: cell_target_sum[cell] / cell_weight[cell]
        for cell in states
    }
    normalized_cell_weight = {
        cell: cell_weight[cell] / total_weight
        for cell in states
    }

    public_law: dict[tuple[Hashable, ...], float] = defaultdict(float)
    for cell, mass in normalized_cell_weight.items():
        public_law[public_map[cell]] += mass

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
        estimand=estimand,
        environments=q_environment.environment,
    )
    return GroupedProblem(
        problem=problem,
        public_law=dict(public_law),
        public_columns=public_columns_tuple,
        hidden_columns=hidden_columns_tuple,
        target_column=target,
        total_weight=total_weight,
        cell_weights=normalized_cell_weight,
        q=q_environment.preset,
        q_name=q_environment.name,
        q_description=q_environment.description,
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
    primary: str | None,
    alias: str | None,
    *,
    primary_name: str,
    alias_name: str,
) -> str | None:
    if primary is not None and alias is not None and primary != alias:
        raise TypeError(f"use either {primary_name!r} or {alias_name!r}, not both")
    return primary if primary is not None else alias


def _record_value(
    row: Mapping[str, Any], column: str, *, row_number: int
) -> Any:
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


def _target_value(row: Mapping[str, Any], target_column: str, *, row_number: int) -> float:
    value = float(_record_value(row, target_column, row_number=row_number))
    if not isfinite(value):
        raise ValueError(f"row {row_number} has non-finite target")
    return value


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
