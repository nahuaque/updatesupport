# Conformal And MAPIE Integration

Conformal prediction and `updatesupport` answer different questions:

- conformal tools such as MAPIE quantify row-level prediction uncertainty,
- `updatesupport` audits whether an aggregate uncertainty, coverage, risk, or
  policy claim is stable under hidden subgroup recomposition.

The handoff is:

```text
model predictions -> conformal intervals or sets -> audit-ready row columns
    -> updatesupport claim design
```

`updatesupport` does not depend on MAPIE in core. Use MAPIE, another conformal
library, or your own conformal procedure upstream, then pass its arrays or
columns to the generic conformal adapters.

## Regression Intervals

For regression, conformal methods usually return a point prediction and a lower
and upper prediction interval:

```python
adapted = us.adapt_conformal_regression(
    df_audit,
    prediction=y_pred,
    lower=y_lower,
    upper=y_upper,
    y_true=y_observed,
    threshold=50000.0,
)
```

The adapter returns a `ConformalAdapterResult` whose rows include:

- `y_pred`,
- `y_lower`,
- `y_upper`,
- `interval_width`,
- `covered` and `miscovered`, when `y_true` is supplied,
- `crosses_threshold`, when `threshold` is supplied.

If a conformal library returns one interval tensor, pass it as `interval=...`.
The adapter accepts common shapes such as `(n, 2)` and MAPIE-style
`(n, 2, n_levels)`:

```python
adapted = us.adapt_conformal_regression(
    df_audit,
    prediction=y_pred,
    interval=y_intervals,
    interval_index=0,
    y_true=y_observed,
)
```

Then design the public report:

```python
claim = us.claim(
    "Intervals rarely cross the automation threshold",
    public=["region", "segment", "channel"],
    hidden=[
        "region",
        "segment",
        "channel",
        "cohort",
        "source_quality",
        "rep_team",
    ],
    target="crosses_threshold",
    weight="account_weight",
    candidate_refinements=["cohort", "source_quality", "rep_team"],
    decision=us.threshold_decision("<=", 0.20),
    ambiguity_limit=0.03,
)

design = adapted.design(claim)
print(design.to_markdown())
```

This separates the conformal prediction interval from the hidden-composition
stability interval. A model can have useful conformal intervals while the
reported automation claim is still unstable under hidden mix shift.

## Classification Prediction Sets

For classification, conformal methods often return prediction sets:

```python
adapted = us.adapt_conformal_classification(
    df_audit,
    prediction=y_pred,
    prediction_sets=prediction_sets,
    y_true=y_observed,
    positive_label="approve",
)
```

The adapter adds:

- `prediction_set`,
- `prediction_set_size`,
- `ambiguous_set`,
- `covered` and `miscovered`, when `y_true` is supplied,
- `contains_positive_label`, when `positive_label` is supplied.

If prediction sets are encoded as class-membership masks, pass the class order:

```python
adapted = us.adapt_conformal_classification(
    df_audit,
    classes=["approve", "review", "reject"],
    prediction_sets=prediction_set_masks,
    y_true=y_observed,
    positive_label="approve",
)
```

A natural policy claim is:

```python
claim = us.claim(
    "Prediction sets stay small enough for automation",
    public=["product", "region"],
    hidden=["product", "region", "language", "difficulty", "source_system"],
    target="prediction_set_size",
    weight="case_weight",
    candidate_refinements=["language", "difficulty", "source_system"],
    decision=us.threshold_decision("<=", 2.0),
)

design = adapted.design(claim)
```

## Useful Targets

Good conformal targets for `updatesupport` are aggregate quantities that a model
reviewer or operator would report:

- mean prediction,
- mean lower or upper conformal bound,
- mean interval width,
- interval-width tail rate,
- coverage or miscoverage rate,
- prediction-set size,
- ambiguous-set rate,
- threshold-crossing rate,
- abstention or manual-review rate,
- risk-control failure indicator.

Each target answers a different review question. For example, coverage asks
whether a nominal uncertainty claim is stable; threshold crossing asks whether a
decision policy remains stable; prediction-set size asks whether automation
burden remains stable.

## MAPIE Usage Pattern

The MAPIE-specific code should stay upstream:

```python
# Example shape only; exact MAPIE class names vary by MAPIE version.
mapie.fit(X_train, y_train)
y_pred, y_intervals = mapie.predict_interval(X_audit)

adapted = us.adapt_conformal_regression(
    df_audit,
    prediction=y_pred,
    interval=y_intervals,
    y_true=y_audit,
)
```

`updatesupport` only needs the resulting arrays. That avoids coupling core to
MAPIE's API surface while preserving the useful product combination:

> conformal prediction tells you how uncertain the model is; `updatesupport`
> tells you whether the uncertainty report is stable across the population you
> care about.
