# Historical TV-Radius Calibration

`calibrate_tv_radius(...)` estimates a total-variation stress radius from
observed consecutive-period hidden-composition changes and evaluates it with
rolling one-step backtests. It addresses the practical question:

> How large should the TV budget be if it is meant to cover a declared share of
> the recompositions seen in this historical series?

The result is an empirical calibration, not a universal or forward-looking
guarantee. It makes the source of the radius inspectable and backtestable.

## Basic Workflow

Install the CVXPY extra used by `q_tv_budget(...)`:

```bash
pip install "updatesupport[cvxpy]"
```

Define the reporting claim once, then calibrate it on period-labelled history:

```python
import updatesupport as us

claim = us.claim(
    "MQL-to-SQL conversion remains above the review floor",
    public=["reported_segment", "region"],
    hidden=[
        "reported_segment",
        "region",
        "lead_source",
        "industry",
        "rep_ramp_band",
    ],
    target="sql_conversion_rate",
    weight="mql_count",
    ambiguity_limit=0.02,
    decision=us.threshold_decision(">=", 0.18),
)

calibration = claim.calibrate_tv(
    historical_rows,
    period="quarter",
    coverage=0.90,
    min_train_transitions=4,
)

print(calibration.to_markdown())
print(calibration.calibrated_radius)
```

The returned report exposes the calibrated preset as `calibration.q`. It can
also apply the calibrated claim directly:

```python
current_audit = calibration.audit(current_period_rows)
current_design = calibration.design(current_period_rows)
```

The functional spelling is equivalent:

```python
calibration = us.calibrate_tv_radius(
    historical_rows,
    claim,
    period="quarter",
    coverage=0.90,
    min_train_transitions=4,
)
```

Use `period_order=[...]` when period labels do not have the desired natural
sort order.

## What Is Calibrated

Let `p_t(o)` be the public law in reference period `t`, and let
`c_t(d | o)` be the hidden-cell composition inside public cell `o`. The
reference hidden law is:

```text
q_t(d) = p_t(o) c_t(d | o)
```

The next period is restandardized to the reference public law:

```text
q_{t+1|t}(d) = p_t(o) c_{t+1}(d | o)
```

The historical transition radius is then:

```text
r_t = 0.5 * ||q_{t+1|t} - q_t||_1
```

This is the same total-variation geometry used by `q_tv_budget(r_t)`. Changes
in public bucket shares are removed before the distance is calculated, so the
radius measures within-public-cell recomposition rather than general
population drift.

The final radius is the higher empirical quantile of eligible transition
distances. For example, `coverage=0.90` selects a radius at least as large as
the empirical 90th percentile under the conservative `higher` quantile rule.

## Rolling Backtests

For transition `t -> t+1`, the rolling backtest calibrates its radius using
eligible transitions ending no later than `t`. The transition being evaluated
is never included in its own training set.

Each row separates:

- **Shift coverage**: whether the realized restandardized TV distance is no
  larger than the historically calibrated radius.
- **Target coverage**: whether the recomposed target value falls inside the TV
  audit interval built from reference-period target values.
- **Decision preservation**: when the claim has a decision rule, whether the
  realized recomposed value implies the same decision as the reference value.
- **Predicted ambiguity and decision invariance**: whether the calibrated audit
  itself meets the claim's ambiguity limit or certifies its decision.

Target backtesting deliberately holds reference-period hidden-cell target
values fixed. That isolates composition sensitivity. Using evaluation-period
target values would mix target or model drift into the TV calibration.

## Support Drift

An ordinary TV ball in `updatesupport` reweights the retained reference support;
it does not invent new hidden cells. A transition is therefore excluded from
radius calibration when:

- the evaluation period introduces a positive-mass hidden cell absent from the
  reference retained support, or
- a positive-mass reference public cell disappears, leaving no evaluation
  composition to restandardize.

These transitions remain visible in the report as `unsupported_support`. They
are not silently counted as ordinary radius misses. A high unsupported rate is
evidence that support expansion needs its own stress model or that the retained
cell definition is too brittle for historical calibration.

## Structured Exports

The report supports the standard artifact API:

```python
calibration.to_json()
calibration.to_tables()
calibration.to_dataframes()
```

The named tables are `summary`, `claim`, `transitions`, `backtests`, and
`limitations`.

## Interpretation Boundary

Historical coverage describes this sequence of retained compositions under
the selected period definition, filtering rule, and quantile level. It does
not guarantee coverage after a regime change, validate the target estimator,
or cover variables absent from the retained refinement.

