"""Estimator-output adapters for causal reporting audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_EFFECT_COLUMN = "tau_hat"


@dataclass(frozen=True)
class EstimatorAdapterResult:
    """Rows with an attached effect column plus lightweight adapter metadata."""

    rows: tuple[dict[str, Any], ...]
    effect_column: str
    source: str
    effect_kind: str
    source_rows: int
    estimator_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def audit_effects(self, **kwargs: Any):
        """Run :func:`updatesupport.audit_effects` on the adapted rows."""

        from .report import audit_effects

        return audit_effects(
            self.rows,
            effect=self.effect_column,
            row_count=self.source_rows,
            **kwargs,
        )

    def causal_reporting_stability(self, **kwargs: Any):
        """Run :func:`updatesupport.causal_reporting_stability` on the rows."""

        from .report import causal_reporting_stability

        return causal_reporting_stability(
            self.rows,
            effect=self.effect_column,
            **kwargs,
        )


def adapt_dataframe_effects(
    data: Any,
    *,
    effect: str | None = None,
    effect_values: Iterable[Any] | None = None,
    effect_column: str = DEFAULT_EFFECT_COLUMN,
    source: str = "dataframe",
) -> EstimatorAdapterResult:
    """Adapt generic dataframe or row outputs into update-support effect rows.

    Use ``effect`` when the effect is already a column in ``data``. Use
    ``effect_values`` when an estimator returned a separate vector.
    """

    records = _records_from_data(data)
    if effect_values is None:
        source_column = effect or effect_column
        rows = _copy_existing_effect_column(
            records,
            source_column=source_column,
            effect_column=effect_column,
        )
        metadata = {"source_column": source_column}
    else:
        rows = _attach_effect_values(
            records,
            effect_values=effect_values,
            effect_column=effect_column,
        )
        metadata = {"source_column": None}

    return EstimatorAdapterResult(
        rows=tuple(rows),
        effect_column=effect_column,
        source=source,
        effect_kind="row-level effect",
        source_rows=len(rows),
        metadata=metadata,
    )


def adapt_econml_effects(
    estimator: Any,
    data: Any,
    X: Any,
    *,
    effect_column: str = DEFAULT_EFFECT_COLUMN,
    effect_kwargs: Mapping[str, Any] | None = None,
    source: str = "econml",
) -> EstimatorAdapterResult:
    """Attach ``estimator.effect(X)`` output to rows for an EconML workflow."""

    effect_values = estimator.effect(X, **dict(effect_kwargs or {}))
    rows = _attach_effect_values(
        _records_from_data(data),
        effect_values=effect_values,
        effect_column=effect_column,
    )
    return EstimatorAdapterResult(
        rows=tuple(rows),
        effect_column=effect_column,
        source=source,
        effect_kind="conditional treatment effect",
        source_rows=len(rows),
        estimator_name=_estimator_name(estimator),
        metadata={
            "effect_method": "effect",
            "effect_kwargs": dict(effect_kwargs or {}),
        },
    )


def adapt_dowhy_effects(
    estimate: Any,
    data: Any,
    *,
    effect_values: Iterable[Any] | None = None,
    effect_column: str = DEFAULT_EFFECT_COLUMN,
    allow_scalar: bool = True,
    source: str = "dowhy",
) -> EstimatorAdapterResult:
    """Adapt DoWhy estimates or externally computed DoWhy effect values.

    DoWhy commonly returns a scalar average effect. If ``effect_values`` is not
    supplied and ``allow_scalar`` is true, that scalar is repeated on every row.
    For heterogeneous reporting audits, pass row-level or subgroup-level
    ``effect_values`` instead.
    """

    records = _records_from_data(data)
    if effect_values is None:
        if not allow_scalar:
            raise ValueError(
                "DoWhy adapter needs effect_values when allow_scalar is false"
            )
        scalar = _scalar_estimate_value(estimate, name="DoWhy estimate")
        effect_values = [scalar] * len(records)
        effect_kind = "scalar causal estimate"
        metadata = {"estimated_effect": scalar}
    else:
        effect_kind = "row-level causal effect"
        metadata = {"estimated_effect": _optional_scalar_estimate(estimate)}

    rows = _attach_effect_values(
        records,
        effect_values=effect_values,
        effect_column=effect_column,
    )
    return EstimatorAdapterResult(
        rows=tuple(rows),
        effect_column=effect_column,
        source=source,
        effect_kind=effect_kind,
        source_rows=len(rows),
        estimator_name=_estimator_name(estimate),
        metadata=metadata,
    )


def adapt_doubleml_effects(
    model: Any,
    data: Any,
    *,
    effect_values: Iterable[Any] | None = None,
    effect_column: str = DEFAULT_EFFECT_COLUMN,
    coef_index: int = 0,
    allow_scalar: bool = True,
    source: str = "doubleml",
) -> EstimatorAdapterResult:
    """Adapt DoubleML model output or externally computed effect values.

    DoubleML's common estimators expose scalar coefficients. If no
    ``effect_values`` are supplied, this adapter repeats ``model.coef`` on every
    row. Pass explicit row-level or group-level effect values when available.
    """

    records = _records_from_data(data)
    if effect_values is None:
        if not allow_scalar:
            raise ValueError(
                "DoubleML adapter needs effect_values when allow_scalar is false"
            )
        scalar = _indexed_scalar_estimate_value(
            model,
            name="DoubleML model",
            index=coef_index,
        )
        effect_values = [scalar] * len(records)
        effect_kind = "scalar causal estimate"
        metadata = {"coef": scalar, "coef_index": coef_index}
    else:
        effect_kind = "row-level causal effect"
        metadata = {
            "coef": _optional_indexed_scalar(model, index=coef_index),
            "coef_index": coef_index,
        }

    rows = _attach_effect_values(
        records,
        effect_values=effect_values,
        effect_column=effect_column,
    )
    return EstimatorAdapterResult(
        rows=tuple(rows),
        effect_column=effect_column,
        source=source,
        effect_kind=effect_kind,
        source_rows=len(rows),
        estimator_name=_estimator_name(model),
        metadata=metadata,
    )


def _records_from_data(data: Any) -> list[dict[str, Any]]:
    if hasattr(data, "to_dict"):
        try:
            records = data.to_dict("records")
        except TypeError:
            records = data.to_dict(orient="records")
        return [dict(row) for row in records]

    if isinstance(data, Mapping):
        raise TypeError("data must be a table or iterable of row mappings")

    return [dict(row) for row in data]


def _copy_existing_effect_column(
    records: Sequence[Mapping[str, Any]],
    *,
    source_column: str,
    effect_column: str,
) -> list[dict[str, Any]]:
    rows = []
    for row in records:
        if source_column not in row:
            raise ValueError(f"missing effect column: {source_column!r}")
        output = dict(row)
        output[effect_column] = _as_float(row[source_column], name=source_column)
        rows.append(output)
    return rows


def _attach_effect_values(
    records: Sequence[Mapping[str, Any]],
    *,
    effect_values: Iterable[Any],
    effect_column: str,
) -> list[dict[str, Any]]:
    values = _flat_values(effect_values)
    if len(values) != len(records):
        raise ValueError(
            "effect_values must contain one value per row "
            f"({len(values)} values for {len(records)} rows)"
        )
    rows = []
    for row, effect in zip(records, values, strict=True):
        output = dict(row)
        output[effect_column] = _as_float(effect, name=effect_column)
        rows.append(output)
    return rows


def _flat_values(values: Iterable[Any]) -> list[Any]:
    if isinstance(values, str):
        raise TypeError("effect_values must be numeric, not a string")
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()
    flattened = []
    for value in list(values):
        if isinstance(value, str):
            flattened.append(value)
        elif _is_singleton_sequence(value):
            flattened.append(value[0])
        else:
            flattened.append(value)
    return flattened


def _is_singleton_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, Mapping)):
        return False
    if not isinstance(value, Sequence):
        return False
    return len(value) == 1


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _scalar_estimate_value(estimate: Any, *, name: str) -> float:
    direct = _try_float(estimate)
    if direct is not None:
        return direct

    for attribute in ("value", "estimate", "estimated_effect", "coef", "coef_"):
        if hasattr(estimate, attribute):
            value = _try_float(getattr(estimate, attribute))
            if value is not None:
                return value
            indexed = _try_indexed_float(getattr(estimate, attribute), 0)
            if indexed is not None:
                return indexed

    raise ValueError(f"{name} must expose a numeric scalar effect")


def _indexed_scalar_estimate_value(estimate: Any, *, name: str, index: int) -> float:
    for attribute in ("coef", "coef_", "effect", "effects"):
        if hasattr(estimate, attribute):
            value = getattr(estimate, attribute)
            direct = _try_float(value)
            if direct is not None and index == 0:
                return direct
            indexed = _try_indexed_float(value, index)
            if indexed is not None:
                return indexed
    return _scalar_estimate_value(estimate, name=name)


def _optional_scalar_estimate(estimate: Any) -> float | None:
    try:
        return _scalar_estimate_value(estimate, name="estimate")
    except ValueError:
        return None


def _optional_indexed_scalar(estimate: Any, *, index: int) -> float | None:
    try:
        return _indexed_scalar_estimate_value(estimate, name="estimate", index=index)
    except ValueError:
        return None


def _try_float(value: Any) -> float | None:
    if isinstance(value, Real):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _try_indexed_float(value: Any, index: int) -> float | None:
    try:
        candidate = value[index]
    except (TypeError, KeyError, IndexError):
        return None
    return _try_float(candidate)


def _estimator_name(estimator: Any) -> str:
    return estimator.__class__.__name__
