"""Finance-oriented Q preset aliases."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Hashable, Mapping, Sequence

import updatesupport as us

MomentMap = dict[str, dict[tuple[Hashable, ...], float]]


def q_portfolio_mix_shift(radius: float = 0.5) -> us.QPreset:
    """Constrain each hidden portfolio-cell mass around its observed mass."""

    return us.q_bounded_shift(radius)


def q_exposure_weighted_tv(
    radius: float,
    *,
    backend: str = "cvxpy",
) -> us.QPreset:
    """Constrain total hidden portfolio mass shift under the observed weights."""

    return us.q_tv_budget(radius, backend=backend)


def portfolio_factor_moments(
    data: Any,
    *,
    hidden: Sequence[str],
    factors: str | Sequence[str] | Mapping[str, str],
    exposure: str | None = None,
    weight: str | None = None,
    prefix: str | None = "factor",
) -> MomentMap:
    """Compute exposure-weighted factor moments by hidden portfolio cell."""

    hidden_columns = _hidden_columns(hidden)
    factor_items = _factor_items(factors, prefix=prefix)
    weight_column = _resolve_weight_column(exposure=exposure, weight=weight)
    cell_weight: dict[tuple[Hashable, ...], float] = defaultdict(float)
    sums: dict[str, dict[tuple[Hashable, ...], float]] = {
        name: defaultdict(float) for name, _column in factor_items
    }

    rows_seen = 0
    for row_number, row in enumerate(_iter_records(data), start=1):
        rows_seen += 1
        hidden_key = _hidden_key(row, hidden_columns, row_number=row_number)
        row_weight = _row_weight(row, weight_column, row_number=row_number)
        if row_weight == 0.0:
            continue
        cell_weight[hidden_key] += row_weight
        for name, column in factor_items:
            sums[name][hidden_key] += row_weight * _number(
                row,
                column,
                row_number=row_number,
            )

    _require_rows(rows_seen)
    return {
        name: {
            cell: weighted_sum[cell] / cell_weight[cell]
            for cell in cell_weight
            if cell_weight[cell] > 0.0
        }
        for name, weighted_sum in sums.items()
    }


def portfolio_concentration_moments(
    data: Any,
    *,
    hidden: Sequence[str],
    category: str,
    exposure: str | None = None,
    weight: str | None = None,
    categories: Sequence[Hashable] | None = None,
    prefix: str | None = None,
) -> MomentMap:
    """Compute exposure-weighted categorical concentration moments."""

    hidden_columns = _hidden_columns(hidden)
    if not category:
        raise ValueError("category column must be non-empty")
    weight_column = _resolve_weight_column(exposure=exposure, weight=weight)
    moment_prefix = category if prefix is None else prefix
    requested_categories = None
    if categories is not None:
        requested_categories = tuple(_hashable_category(value) for value in categories)
        if not requested_categories:
            raise ValueError("categories must contain at least one value")

    cell_weight: dict[tuple[Hashable, ...], float] = defaultdict(float)
    category_weight: dict[Hashable, dict[tuple[Hashable, ...], float]] = defaultdict(
        lambda: defaultdict(float)
    )
    observed_categories: set[Hashable] = set()

    rows_seen = 0
    for row_number, row in enumerate(_iter_records(data), start=1):
        rows_seen += 1
        hidden_key = _hidden_key(row, hidden_columns, row_number=row_number)
        category_value = _hashable_category(
            _record_value(row, category, row_number=row_number)
        )
        row_weight = _row_weight(row, weight_column, row_number=row_number)
        if row_weight == 0.0:
            continue
        observed_categories.add(category_value)
        cell_weight[hidden_key] += row_weight
        category_weight[category_value][hidden_key] += row_weight

    _require_rows(rows_seen)
    category_values = (
        requested_categories
        if requested_categories is not None
        else tuple(sorted(observed_categories, key=str))
    )
    return {
        f"{moment_prefix}:{category_value}": {
            cell: category_weight[category_value][cell] / cell_weight[cell]
            for cell in cell_weight
            if cell_weight[cell] > 0.0
        }
        for category_value in category_values
    }


def q_factor_exposure_shift(
    radius: float,
    data: Any,
    *,
    hidden: Sequence[str],
    factors: str | Sequence[str] | Mapping[str, str],
    exposure: str | None = None,
    weight: str | None = None,
    baseline: Mapping[Hashable, float] | Sequence[float] | None = None,
    scale: Mapping[Hashable, float] | Sequence[float] | float | None = None,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> us.QPreset:
    """Constrain standardized portfolio factor-exposure drift."""

    moments = portfolio_factor_moments(
        data,
        hidden=hidden,
        factors=factors,
        exposure=exposure,
        weight=weight,
    )
    return us.q_covariate_balance(
        radius,
        moments,
        baseline=baseline,
        scale=scale,
        backend=backend,
        solver=solver,
        solver_options=solver_options,
    )


def q_regional_concentration_shift(
    radius: float,
    data: Any,
    *,
    hidden: Sequence[str],
    region: str = "region",
    exposure: str | None = None,
    weight: str | None = None,
    regions: Sequence[Hashable] | None = None,
    baseline: Mapping[Hashable, float] | Sequence[float] | None = None,
    scale: Mapping[Hashable, float] | Sequence[float] | float | None = None,
    backend: str = "cvxpy",
    solver: str | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> us.QPreset:
    """Constrain standardized regional concentration drift."""

    moments = portfolio_concentration_moments(
        data,
        hidden=hidden,
        category=region,
        exposure=exposure,
        weight=weight,
        categories=regions,
        prefix="region",
    )
    return us.q_covariate_balance(
        radius,
        moments,
        baseline=baseline,
        scale=scale,
        backend=backend,
        solver=solver,
        solver_options=solver_options,
    )


def _hidden_columns(hidden: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(hidden)
    if not columns:
        raise ValueError("hidden must contain at least one column")
    return columns


def _factor_items(
    factors: str | Sequence[str] | Mapping[str, str],
    *,
    prefix: str | None,
) -> tuple[tuple[str, str], ...]:
    if isinstance(factors, str):
        raw_items = ((factors, factors),)
    elif isinstance(factors, Mapping):
        raw_items = tuple((str(name), str(column)) for name, column in factors.items())
    else:
        raw_items = tuple((str(column), str(column)) for column in factors)
    if not raw_items:
        raise ValueError("factors must contain at least one column")
    return tuple(
        (str(name) if prefix is None else f"{prefix}:{name}", column)
        for name, column in raw_items
    )


def _resolve_weight_column(
    *,
    exposure: str | None,
    weight: str | None,
) -> str | None:
    if exposure is not None and weight is not None and exposure != weight:
        raise ValueError("use either exposure or weight, not both")
    return weight if weight is not None else exposure


def _iter_records(data: Any):
    if hasattr(data, "to_dict"):
        try:
            yield from data.to_dict("records")
        except TypeError:
            yield from data.to_dict(orient="records")
        return
    yield from data


def _hidden_key(
    row: Mapping[str, Any],
    hidden: Sequence[str],
    *,
    row_number: int,
) -> tuple[Hashable, ...]:
    return tuple(
        _hashable_category(_record_value(row, column, row_number=row_number))
        for column in hidden
    )


def _record_value(row: Mapping[str, Any], column: str, *, row_number: int) -> Any:
    try:
        return row[column]
    except KeyError as exc:
        raise ValueError(f"row {row_number} is missing column {column!r}") from exc


def _row_weight(
    row: Mapping[str, Any],
    weight_column: str | None,
    *,
    row_number: int,
) -> float:
    if weight_column is None:
        return 1.0
    value = _number(row, weight_column, row_number=row_number)
    if value < 0.0:
        raise ValueError("row weights must be non-negative")
    return value


def _number(row: Mapping[str, Any], column: str, *, row_number: int) -> float:
    value = float(_record_value(row, column, row_number=row_number))
    if not isfinite(value):
        raise ValueError(f"row {row_number} has non-finite value in {column!r}")
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


def _require_rows(rows_seen: int) -> None:
    if rows_seen == 0:
        raise ValueError("data must contain at least one row")
