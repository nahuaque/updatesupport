# Using `updatesupport` With Causal Inference Libraries

`updatesupport` is not a causal inference library. It does not identify causal
effects, remove confounding, estimate propensity scores, or validate a causal
graph.

It sits after those steps as a representation-stability audit:

> Given the effect or rate we are willing to report, are the public categories
> used in the report stable enough, or could hidden composition inside those
> categories move the aggregate?

This makes `updatesupport` complementary to libraries such as DoWhy, EconML,
CausalML, DoubleML, and uplift-modeling packages.

## Where It Fits

A typical causal workflow is:

1. State the causal question.
2. Encode assumptions, often with a graph.
3. Identify the estimand.
4. Estimate the effect.
5. Run causal refuters, sensitivity checks, and inference.
6. Report aggregate or subgroup results.

`updatesupport` belongs at step 6. It audits whether the representation used for
the reported aggregate is adequate.

In plain terms:

```text
DoWhy / EconML / CausalML:
    causal assumptions + effect estimation + causal validation

updatesupport:
    public-category adequacy + hidden-composition transport ambiguity
```

The input to `updatesupport` can be an observed outcome rate, an estimated CATE,
an uplift score, a subgroup treatment effect, or any other linear target value
attached to hidden cells.

## Pattern: Audit Estimated Effects

Suppose a causal estimator has produced one estimated treatment effect per row:

```python
df["tau_hat"] = estimated_treatment_effects
```

Then `updatesupport` can audit the reporting categories:

```python
import updatesupport as us

report = us.public_descent_report(
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
    candidate_refinements=[
        "education_band",
        "income_band",
        "region",
        "prior_usage_band",
    ],
    q=us.q_bounded_shift(0.5),
    min_cell_weight=25,
    title="Treatment Effect Representation Adequacy Report",
)

print(report.to_markdown())
```

The report asks:

> If we only report the treatment effect by age band and sex, how much could the
> aggregate move if education, income, region, or prior usage changed inside
> those public cells?

## Folktables ACS Causal-Effect Example

The repository includes an executable ACS example that makes the handoff
concrete:

```bash
uv run --extra examples python examples/folktables_acs_causal.py \
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

The built-in first stage estimates one `__tau_hat__` value per hidden stratum
using a transparent treated-minus-control difference in weighted outcome means.
That is intentionally simple. It is there to make the integration surface
visible, not to claim that education is causally identified in ACS.

In a real workflow, replace the first stage with a causal library:

```python
# Example shape, independent of the specific causal library.
df["tau_hat"] = causal_estimator.effect(X)

report = us.public_descent_report(
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
    target="tau_hat",
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

## With DoWhy

Use DoWhy for causal modeling, identification, estimation, and refutation. Then
use `updatesupport` to audit the reporting layer.

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
into `updatesupport`:

```python
df["tau_hat"] = subgroup_or_unit_level_effect

report = us.public_descent_report(
    df,
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "education_band", "region", "income_band"],
    target="tau_hat",
    weight="sample_weight",
    candidate_refinements=["education_band", "region", "income_band"],
)
```

A future integration could be a DoWhy refuter named something like:

```python
model.refute_estimate(
    identified,
    estimate,
    method_name="updatesupport_representation_refuter",
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "education_band", "region", "income_band"],
)
```

That refuter would not refute the causal graph itself. It would refute, or
stress-test, the adequacy of the public representation used to report the
estimate.

## With EconML

EconML is a natural fit because many estimators expose conditional treatment
effect predictions.

Sketch:

```python
from econml.dml import CausalForestDML
import updatesupport as us

est = CausalForestDML(...)
est.fit(Y, T, X=X, W=W)

df["tau_hat"] = est.effect(X)

report = us.public_descent_report(
    df,
    public=["age_band", "sex"],
    hidden=[
        "age_band",
        "sex",
        "income_band",
        "region",
        "prior_usage_band",
    ],
    target="tau_hat",
    weight="sample_weight",
    candidate_refinements=["income_band", "region", "prior_usage_band"],
)
```

Interpretation:

> The causal forest estimates heterogeneous treatment effects. `updatesupport`
> asks whether the coarse public reporting groups almost determine the aggregate
> reported effect, or whether hidden CATE heterogeneity inside those groups can
> move the answer.

## With CausalML Or Uplift Models

For uplift or treatment-response models, use the predicted uplift or estimated
treatment effect as the target:

```python
df["tau_hat"] = uplift_model.predict(X)

report = us.public_descent_report(
    df,
    public=["segment", "channel"],
    hidden=["segment", "channel", "region", "tenure_band", "spend_band"],
    target="tau_hat",
    weight="sample_weight",
    candidate_refinements=["region", "tenure_band", "spend_band"],
)
```

This is useful for dashboards that report uplift by a small set of business
segments:

> If we only report uplift by segment and channel, are hidden region, tenure, or
> spend differences doing important work?

## Sensitivity Grid

Use a sensitivity report to see whether conclusions depend on `Q`, sparse-cell
filtering, or hidden-column choices:

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

## What Not To Claim

Do not say that `updatesupport` proves a causal effect is valid.

Do not interpret the transport interval as a confidence interval.

Do not use small ambiguity as evidence that confounding, overlap, selection, or
model misspecification are solved.

The right claim is narrower:

> Conditional on the target values supplied to `updatesupport`, this audit
> quantifies how much the reported aggregate can move under hidden-composition
> changes that preserve the public distribution.

## Good README Sentence

For causal workflows:

> Use DoWhy, EconML, CausalML, or DoubleML to estimate causal effects; then use
> `updatesupport` to audit whether the public categories used to report those
> effects are stable to hidden composition changes.
