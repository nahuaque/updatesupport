"""Estimator-output adapters for reporting audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_EFFECT_COLUMN = "tau_hat"
DEFAULT_PREDICTION_COLUMN = "y_pred"
DEFAULT_LOWER_COLUMN = "y_lower"
DEFAULT_UPPER_COLUMN = "y_upper"
DEFAULT_INTERVAL_WIDTH_COLUMN = "interval_width"
DEFAULT_COVERED_COLUMN = "covered"
DEFAULT_MISCOVERED_COLUMN = "miscovered"
DEFAULT_CROSSES_THRESHOLD_COLUMN = "crosses_threshold"
DEFAULT_PREDICTION_SET_SIZE_COLUMN = "prediction_set_size"
DEFAULT_AMBIGUOUS_SET_COLUMN = "ambiguous_set"


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


@dataclass(frozen=True)
class ConformalAdapterResult:
    """Rows with attached conformal prediction targets and metadata."""

    rows: tuple[dict[str, Any], ...]
    source: str
    source_rows: int
    prediction_column: str | None = None
    lower_column: str | None = None
    upper_column: str | None = None
    interval_width_column: str | None = None
    covered_column: str | None = None
    miscovered_column: str | None = None
    crosses_threshold_column: str | None = None
    prediction_set_size_column: str | None = None
    ambiguous_set_column: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def claim(self, estimate_name: str, *, target: str, **kwargs: Any):
        """Build a claim over one adapted conformal target column."""

        from .claim import claim

        return claim(estimate_name, target=target, **kwargs)

    def design(self, claim_or_spec: Any, **kwargs: Any):
        """Run ``claim.design(...)`` against the adapted rows."""

        if not hasattr(claim_or_spec, "design"):
            from .claim import ClaimSpec

            claim_or_spec = ClaimSpec.from_dict(claim_or_spec)
        return claim_or_spec.design(self.rows, **kwargs)

    def audit(self, claim_or_spec: Any, **kwargs: Any):
        """Run ``claim.audit(...)`` against the adapted rows."""

        if not hasattr(claim_or_spec, "audit"):
            from .claim import ClaimSpec

            claim_or_spec = ClaimSpec.from_dict(claim_or_spec)
        return claim_or_spec.audit(self.rows, **kwargs)

    def reporting_stability(self, **kwargs: Any):
        """Audit useful conformal-derived targets from this adapter result."""

        from .conformal import conformal_reporting_stability

        return conformal_reporting_stability(self, **kwargs)


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


def adapt_conformal_regression(
    data: Any,
    *,
    prediction: str | Iterable[Any] | None = None,
    lower: str | Iterable[Any] | None = None,
    upper: str | Iterable[Any] | None = None,
    interval: Any | None = None,
    interval_index: int = 0,
    y_true: str | Iterable[Any] | None = None,
    threshold: float | None = None,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    lower_column: str = DEFAULT_LOWER_COLUMN,
    upper_column: str = DEFAULT_UPPER_COLUMN,
    interval_width_column: str = DEFAULT_INTERVAL_WIDTH_COLUMN,
    covered_column: str = DEFAULT_COVERED_COLUMN,
    miscovered_column: str = DEFAULT_MISCOVERED_COLUMN,
    crosses_threshold_column: str = DEFAULT_CROSSES_THRESHOLD_COLUMN,
    source: str = "conformal_regression",
) -> ConformalAdapterResult:
    """Attach conformal regression interval targets to tabular rows.

    Inputs may be existing column names or array-like values. ``interval`` is a
    convenience for conformal libraries that return one interval array; it may
    have shape ``(n, 2)`` or MAPIE-style ``(n, 2, n_levels)``.
    """

    records = _records_from_data(data)
    if interval is not None:
        if lower is not None or upper is not None:
            raise ValueError("use either interval or lower/upper, not both")
        lower_values, upper_values = _interval_bounds(
            interval,
            interval_index=interval_index,
        )
    else:
        if lower is None or upper is None:
            raise ValueError("lower and upper are required when interval is omitted")
        lower_values = _column_or_values(records, lower, name="lower")
        upper_values = _column_or_values(records, upper, name="upper")

    prediction_values = (
        None
        if prediction is None
        else _column_or_values(records, prediction, name="prediction")
    )
    y_true_values = (
        None
        if y_true is None
        else _column_or_values(
            records,
            y_true,
            name="y_true",
        )
    )
    threshold_value = None if threshold is None else float(threshold)

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        lo = _as_float(lower_values[index], name="lower")
        hi = _as_float(upper_values[index], name="upper")
        if lo > hi:
            raise ValueError("lower interval endpoint cannot exceed upper endpoint")
        output = dict(row)
        output[lower_column] = lo
        output[upper_column] = hi
        output[interval_width_column] = hi - lo
        if prediction_values is not None:
            output[prediction_column] = _as_float(
                prediction_values[index],
                name="prediction",
            )
        if y_true_values is not None:
            observed = _as_float(y_true_values[index], name="y_true")
            covered = lo <= observed <= hi
            output[covered_column] = covered
            output[miscovered_column] = not covered
        if threshold_value is not None:
            output[crosses_threshold_column] = lo <= threshold_value <= hi
        rows.append(output)

    return ConformalAdapterResult(
        rows=tuple(rows),
        source=source,
        source_rows=len(rows),
        prediction_column=prediction_column if prediction is not None else None,
        lower_column=lower_column,
        upper_column=upper_column,
        interval_width_column=interval_width_column,
        covered_column=covered_column if y_true is not None else None,
        miscovered_column=miscovered_column if y_true is not None else None,
        crosses_threshold_column=(
            crosses_threshold_column if threshold is not None else None
        ),
        metadata={
            "kind": "regression_interval",
            "threshold": threshold_value,
            "interval_index": interval_index if interval is not None else None,
        },
    )


def adapt_conformal_classification(
    data: Any,
    *,
    prediction_sets: str | Iterable[Any],
    classes: Sequence[Any] | None = None,
    set_index: int = 0,
    prediction: str | Iterable[Any] | None = None,
    y_true: str | Iterable[Any] | None = None,
    positive_label: Any | None = None,
    prediction_column: str = DEFAULT_PREDICTION_COLUMN,
    prediction_set_size_column: str = DEFAULT_PREDICTION_SET_SIZE_COLUMN,
    covered_column: str = DEFAULT_COVERED_COLUMN,
    miscovered_column: str = DEFAULT_MISCOVERED_COLUMN,
    ambiguous_set_column: str = DEFAULT_AMBIGUOUS_SET_COLUMN,
    contains_positive_label_column: str = "contains_positive_label",
    source: str = "conformal_classification",
) -> ConformalAdapterResult:
    """Attach conformal classification prediction-set targets to rows."""

    records = _records_from_data(data)
    set_values = _prediction_set_values(
        _column_or_values(
            records,
            prediction_sets,
            name="prediction_sets",
            flatten=False,
        ),
        classes=classes,
        set_index=set_index,
    )
    prediction_values = (
        None
        if prediction is None
        else _column_or_values(records, prediction, name="prediction")
    )
    y_true_values = (
        None
        if y_true is None
        else _column_or_values(
            records,
            y_true,
            name="y_true",
        )
    )

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        prediction_set = set_values[index]
        output = dict(row)
        if prediction_values is not None:
            output[prediction_column] = prediction_values[index]
        output["prediction_set"] = tuple(prediction_set)
        output[prediction_set_size_column] = len(prediction_set)
        output[ambiguous_set_column] = len(prediction_set) > 1
        if y_true_values is not None:
            covered = y_true_values[index] in prediction_set
            output[covered_column] = covered
            output[miscovered_column] = not covered
        if positive_label is not None:
            output[contains_positive_label_column] = positive_label in prediction_set
        rows.append(output)

    return ConformalAdapterResult(
        rows=tuple(rows),
        source=source,
        source_rows=len(rows),
        prediction_column=prediction_column if prediction is not None else None,
        covered_column=covered_column if y_true is not None else None,
        miscovered_column=miscovered_column if y_true is not None else None,
        prediction_set_size_column=prediction_set_size_column,
        ambiguous_set_column=ambiguous_set_column,
        metadata={
            "kind": "classification_prediction_set",
            "classes": None if classes is None else tuple(classes),
            "set_index": set_index,
            "positive_label": positive_label,
            "contains_positive_label_column": contains_positive_label_column
            if positive_label is not None
            else None,
        },
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


def _column_or_values(
    records: Sequence[Mapping[str, Any]],
    source: str | Iterable[Any],
    *,
    name: str,
    flatten: bool = True,
) -> list[Any]:
    if isinstance(source, str):
        if not all(source in row for row in records):
            raise ValueError(f"missing {name} column: {source!r}")
        values = [row[source] for row in records]
    else:
        values = _flat_values(source) if flatten else _list_values(source)
    if len(values) != len(records):
        raise ValueError(
            f"{name} must contain one value per row "
            f"({len(values)} values for {len(records)} rows)"
        )
    return values


def _interval_bounds(
    interval: Any,
    *,
    interval_index: int,
) -> tuple[list[Any], list[Any]]:
    rows = _list_values(interval)
    lower: list[Any] = []
    upper: list[Any] = []
    for row in rows:
        values = _list_values(row)
        if len(values) != 2:
            raise ValueError("interval rows must contain lower and upper endpoints")
        lower.append(_indexed_interval_endpoint(values[0], interval_index))
        upper.append(_indexed_interval_endpoint(values[1], interval_index))
    return lower, upper


def _indexed_interval_endpoint(value: Any, interval_index: int) -> Any:
    values = _list_values(value)
    if len(values) == 1:
        return values[0]
    try:
        return values[interval_index]
    except IndexError as exc:
        raise ValueError("interval_index is out of bounds") from exc


def _prediction_set_values(
    values: Sequence[Any],
    *,
    classes: Sequence[Any] | None,
    set_index: int,
) -> list[tuple[Any, ...]]:
    if classes is None:
        return [_literal_prediction_set(value) for value in values]
    class_values = tuple(classes)
    if not class_values:
        raise ValueError("classes cannot be empty")
    return [
        tuple(
            label
            for label, included in zip(
                class_values,
                _class_mask(value, class_count=len(class_values), set_index=set_index),
                strict=True,
            )
            if included
        )
        for value in values
    ]


def _literal_prediction_set(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(key for key, included in value.items() if bool(included))
    return tuple(_list_values(value))


def _class_mask(value: Any, *, class_count: int, set_index: int) -> tuple[bool, ...]:
    values = _list_values(value)
    if len(values) != class_count:
        raise ValueError("prediction set mask length must match classes")
    mask = []
    for item in values:
        item_values = _list_values(item)
        if len(item_values) == 1:
            mask.append(bool(item_values[0]))
        else:
            try:
                mask.append(bool(item_values[set_index]))
            except IndexError as exc:
                raise ValueError("set_index is out of bounds") from exc
    return tuple(mask)


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


def _list_values(values: Any) -> list[Any]:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except TypeError:
        return [values]


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
