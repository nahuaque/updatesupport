# Using `updatesupport` With Causal Inference Libraries

`updatesupport` is not a causal inference library. It does not identify causal
effects, remove confounding, estimate propensity scores, or validate a causal
graph.

It sits after those steps as a representation-stability audit for reported
rates, effects, and subgroup summaries. This makes `updatesupport`
complementary to libraries such as [DoWhy], [EconML], [CausalML], [DoubleML],
and uplift-modeling packages.

## Where It Fits

A typical causal workflow is:

1. State the causal question.
2. Encode assumptions, often with a graph.
3. Identify the estimand.
4. Estimate the effect.
5. Run causal refuters, sensitivity checks, and inference.
6. Report aggregate or subgroup results.

`updatesupport` belongs at step 6. The causal library handles identification
and estimation; `updatesupport` handles the reporting representation.

The input to `updatesupport` can be an observed outcome rate, an estimated CATE,
an uplift score, a subgroup treatment effect, or any other linear target value
attached to hidden cells.

## Pattern: Audit Estimated Effects

Suppose a causal estimator has produced one estimated treatment effect per row:

```python
df["tau_hat"] = estimated_treatment_effects
df["tau_se"] = estimated_treatment_effect_standard_errors  # optional
```

Then `updatesupport` can audit the reporting categories with the full causal
reporting-stability suite:

```python
import updatesupport as us

suite = us.causal_reporting_stability(
    df,
    public=["age_band", "sex"],
    hidden=[
        "age_band",
        "sex",
        "education_band",
        "income_band",
        "region",
        "prior_usage_band",
    ],
    effect="tau_hat",
    effect_standard_error="tau_se",
    weight="sample_weight",
    candidate_refinements=[
        "education_band",
        "income_band",
        "region",
        "prior_usage_band",
    ],
    q=us.q_bounded_shift(0.5),
    min_cell_weight=25,
    sensitivity_min_cell_weights=[10, 25, 50],
    sensitivity_q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
    statistical_estimate=ate_hat,
    statistical_interval=(ci_low, ci_high),
    statistical_method="causal estimator bootstrap",
)

print(suite.to_markdown())
```

The Markdown suite separates:

- the causal estimate supplied by the causal library
- statistical uncertainty from the causal/statistical workflow
- hidden-composition ambiguity from the update-support stress test
- estimator-uncertainty-aware hidden ambiguity when effect standard errors are
  supplied
- public refinement recommendations for improving the reporting representation

## Estimator Adapters

The adapter helpers standardize common estimator outputs into row records with a
single effect column. They do not fit causal models or change the identification
argument; they only perform the handoff into `updatesupport`.

For generic dataframe or model-output tables:

```python
adapted = us.adapt_dataframe_effects(
    df,
    effect="estimated_effect",
    effect_column="tau_hat",
)

suite = adapted.causal_reporting_stability(
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "region", "income_band"],
    candidate_refinements=["region", "income_band"],
    weight="sample_weight",
    q=us.q_bounded_shift(0.5),
)
```

For [EconML]:

```python
est.fit(Y, T, X=X, W=W)
adapted = us.adapt_econml_effects(est, df, X)

report = adapted.audit_effects(
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "region", "income_band"],
    candidate_refinements=["region", "income_band"],
    weight="sample_weight",
)
```

For [DoWhy]:

```python
estimate = model.estimate_effect(...)

# If you have heterogeneous row-level effects, pass them explicitly.
adapted = us.adapt_dowhy_effects(
    estimate,
    df,
    effect_values=row_level_effects,
)
```

If `effect_values` is omitted, the [DoWhy] adapter repeats the scalar estimate on
each row. That is useful for documenting an average-effect handoff, but it will
not expose hidden CATE heterogeneity.

For [DoubleML]:

```python
dml.fit()

# Common estimators here expose scalar coefficients.
adapted = us.adapt_doubleml_effects(dml, df)

# Prefer explicit row-level or subgroup-level effects when your workflow has them.
adapted = us.adapt_doubleml_effects(
    dml,
    df,
    effect_values=row_or_group_effects,
)
```

Scalar [DoubleML] coefficients are repeated on every row by default. As with
[DoWhy], pass heterogeneous effect values when the reporting question is about
hidden treatment-effect variation.

## Folktables ACS Causal-Effect Example

The repository includes an executable ACS example that makes the handoff
concrete:

```bash
uv run --extra examples --extra causal python examples/folktables_acs_causal.py \
  --task income \
  --states CA \
  --year 2018 \
  --sample 50000
```

The example uses:

- treatment: BA or graduate degree versus less than BA
- outcome: the selected Folktables ACS task label
- public reporting categories: age band and sex
- hidden reporting refinements: occupation, weekly-hours band, race, marital
  status, class of worker, and relationship status when available

The built-in first stage fits an [EconML] `CausalForestDML` estimator and
computes one row-level effect target:

```python
df["__tau_hat__"] = estimator.effect(X)
```

That makes the integration surface visible, but it still does not claim that
education is causally identified in ACS.

In a real workflow, use the causal estimator that matches the identification
strategy:

```python
# Example shape, independent of the specific causal library.
df["tau_hat"] = causal_estimator.effect(X)
df["tau_se"] = causal_estimator.effect_standard_error(X)  # if available

report = us.audit_effects(
    df,
    public=["AGE_BAND", "SEX"],
    hidden=[
        "AGE_BAND",
        "SEX",
        "OCC_MAJOR",
        "WKHP_BAND",
        "RAC1P",
        "MAR",
        "COW",
        "RELP",
    ],
    effect="tau_hat",
    effect_standard_error="tau_se",
    weight="sample_weight",
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P", "MAR"],
    q=us.q_bounded_shift(0.5),
    min_cell_weight=25,
    title="Causal Effect Representation Stability Audit",
)
```

The important separation is:

> The causal library estimates the effect; `updatesupport` audits whether the
> public categories used to report that effect are stable to hidden-composition
> changes.

## With [DoWhy]

Use [DoWhy] for causal modeling, identification, estimation, and refutation.
Then use `updatesupport` to audit the reporting layer.

Sketch:

```python
from dowhy import CausalModel
import updatesupport as us

model = CausalModel(
    data=df,
    treatment="T",
    outcome="Y",
    graph=graph,
)

identified = model.identify_effect()
estimate = model.estimate_effect(
    identified,
    method_name="backdoor.propensity_score_matching",
)
```

If the estimator only returns a single ATE, `updatesupport` has little to audit
unless you also estimate effects by hidden cells, target units, or subgroups.
Once you have a row-level, cell-level, or subgroup-level target, feed that target
into the [DoWhy] adapter:

```python
df["tau_hat"] = subgroup_or_unit_level_effect

audit = us.audit_dowhy_effects(
    df,
    estimate=estimate,
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "education_band", "region", "income_band"],
    effect="tau_hat",
    weight="sample_weight",
    candidate_refinements=["education_band", "region", "income_band"],
)

print(audit.to_markdown())
```

If [DoWhy] is installed, the audit can also be converted into a [DoWhy]
`CausalRefutation`:

```python
representation_refutation = audit.to_refutation()
print(representation_refutation)
```

Install the optional [DoWhy] dependency with:

```bash
# with pip
pip install "updatesupport[dowhy]"

# with uv
uv add "updatesupport[dowhy]"
```

This is a [DoWhy]-compatible refutation object, not a registered
`model.refute_estimate(..., method_name=...)` plugin. The `new_effect` field is
the update-support partial-ID interval, and the object also carries
`updatesupport_report`, `updatesupport_interval`, `updatesupport_ambiguity`, and
`updatesupport_public_adequate` attributes for downstream inspection.

This refutation does not refute the causal graph itself. It stress-tests the
adequacy of the public representation used to report the estimate.

## With [EconML]

[EconML] is a natural fit because many estimators expose conditional treatment
effect predictions.

Sketch:

```python
from econml.dml import CausalForestDML
import updatesupport as us

est = CausalForestDML(...)
est.fit(Y, T, X=X, W=W)

adapted = us.adapt_econml_effects(est, df, X)

report = adapted.audit_effects(
    public=["age_band", "sex"],
    hidden=[
        "age_band",
        "sex",
        "income_band",
        "region",
        "prior_usage_band",
    ],
    weight="sample_weight",
    candidate_refinements=["income_band", "region", "prior_usage_band"],
)
```

Interpretation:

> The causal forest estimates heterogeneous treatment effects. `updatesupport`
> asks whether the coarse public reporting groups almost determine the aggregate
> reported effect, or whether hidden CATE heterogeneity inside those groups can
> move the answer.

## With [CausalML] or Uplift Models

For uplift or treatment-response models, use the predicted uplift or estimated
treatment effect as the target:

```python
df["tau_hat"] = uplift_model.predict(X)

report = us.audit_effects(
    df,
    public=["segment", "channel"],
    hidden=["segment", "channel", "region", "tenure_band", "spend_band"],
    effect="tau_hat",
    weight="sample_weight",
    candidate_refinements=["region", "tenure_band", "spend_band"],
)
```

This is useful for dashboards that report uplift by a small set of business
segments:

> If we only report uplift by segment and channel, are hidden region, tenure, or
> spend differences doing important work?

## Covariate-Balance Stress Tests

For causal model review, the most natural stress set is often a balance
tolerance rather than a direct mass-movement budget. Use
`q_covariate_balance(...)` when the question is:

> If the public report preserves the observed public buckets, but hidden
> covariate balance can drift within an L2 tolerance, how much can the reported
> effect move?

The preset constrains

```text
|| standardized_hidden_moment_shift ||_2 <= epsilon
```

Build hidden-cell moment summaries on the same hidden state space used by the
audit:

```python
hidden_cols = [
    "age_band",
    "sex",
    "education_band",
    "region",
    "prior_usage_band",
]
balance_cols = ["prior_usage_z", "baseline_risk_z", "propensity_logit_z"]

moment_frame = df.groupby(hidden_cols, observed=True)[balance_cols].mean()
moments = {
    column: {
        tuple(hidden_state): float(value)
        for hidden_state, value in moment_frame[column].items()
    }
    for column in balance_cols
}

suite = us.causal_reporting_stability(
    df,
    public=["age_band", "sex"],
    hidden=hidden_cols,
    effect="tau_hat",
    effect_standard_error="tau_se",
    weight="sample_weight",
    q=us.q_covariate_balance(0.25, moments),
    statistical_estimate=ate_hat,
    statistical_interval=(ci_low, ci_high),
    statistical_method="estimator bootstrap",
)
```

By default, the baseline is the observed hidden-distribution moment vector and
the scale is the observed weighted standard deviation of each moment across
hidden cells. Pass `baseline=...` or `scale=...` when a model-review standard
defines a different balance target or tolerance scale.

This pairs well with [EconML], [DoWhy], [DoubleML], and uplift-model outputs
because the causal estimator supplies `tau_hat`, while `updatesupport` reports
how much the aggregate effect could move under hidden balance drift. Keep this
separate from statistical uncertainty: the balance stress interval is a
hidden-composition sensitivity result, not a confidence interval.

## Sensitivity Grid

The suite runs this internally when `sensitivity_*` arguments are supplied. You
can also use a standalone sensitivity report to see whether conclusions depend
on `Q`, sparse-cell filtering, or hidden-column choices:

```python
sensitivity = us.sensitivity_report(
    df,
    public=["age_band", "sex"],
    hidden=[
        "age_band",
        "sex",
        "education_band",
        "income_band",
        "region",
        "prior_usage_band",
    ],
    target="tau_hat",
    weight="sample_weight",
    min_cell_weights=[1, 10, 25, 50],
    hidden_sets=[
        ["age_band", "sex", "education_band"],
        ["age_band", "sex", "education_band", "region"],
        ["age_band", "sex", "education_band", "region", "income_band"],
    ],
    q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
)

print(sensitivity.to_markdown())
```

The Markdown output summarizes successful and failed scenarios, reports the
ambiguity range across the grid, identifies the highest-ambiguity scenario, and
states whether public adequacy is stable or mixed across scenarios.

To ask which hidden variables consistently improve the reporting representation
across the grid, use `recommend_refinements_sensitivity(...)` with the same
public columns, hidden columns, candidate refinements, Q presets, and
`min_cell_weight` thresholds.

## Interpretation Boundary

The audit does not validate a causal effect, turn the transport interval into a
confidence interval, or solve confounding, overlap, selection, or model
misspecification. Refinement candidates are reporting refinements unless the
causal identification argument also supports using them as adjustment variables.

[CausalML]: https://causalml.readthedocs.io/en/latest/
[DoWhy]: https://www.pywhy.org/dowhy/
[DoubleML]: https://docs.doubleml.org/
[EconML]: https://www.pywhy.org/EconML/
