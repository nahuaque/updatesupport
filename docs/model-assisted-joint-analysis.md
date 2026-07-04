# Model-Assisted Joint Analysis

Model-assisted joint analysis fits a nonparametric joint distribution over
retained public and hidden cells, then uses posterior/bootstrap draws to rerun
the audit. It complements adversarial Q-based ambiguity by summarizing outcomes
under sampled cell laws.

## Fit A Joint Distribution

```python
import updatesupport as us

joint = us.fit_joint_distribution(
    rows_or_frame,
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="income_over_threshold",
    weight="sample_weight",
    method="bayesian_bootstrap",
)
```

The fitted model stores retained hidden cells, their empirical joint
probabilities, and their hidden-cell target values. The default
`method="bayesian_bootstrap"` draws new cell masses from a Dirichlet distribution
centered on the empirical cell probabilities. Use `method="empirical"` when you
want deterministic draws equal to the fitted empirical joint law.
Use `method="bootstrap"` for an ordinary multinomial nonparametric bootstrap
over retained hidden cells.

You can inspect or reuse draws directly:

```python
draw = joint.draw(seed=123)
records = draw.records()
```

The records are weighted hidden-cell rows with generated target and weight
columns, so they can be fed back into the usual report helpers.

For update-support audits, the most direct draw is usually a
hidden-composition draw:

```python
draw = joint.hidden_composition_draw(seed=123)
```

That draw preserves the fitted public law and resamples hidden-cell shares
inside each public fiber. Plain `joint.draw(...)` samples the full joint law,
so public bucket masses can move too.

## Posterior / Bootstrap Uncertainty Report

Use `hidden_composition_uncertainty(...)` when you want a standalone uncertainty
report over hidden composition:

```python
uncertainty = us.hidden_composition_uncertainty(
    rows_or_frame,
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="income_over_threshold",
    weight="sample_weight",
    method="bayesian_bootstrap",
    draws=500,
    seed=123,
    q=us.q_tv_budget(0.10),
    ambiguity_limit=0.015,
    confidence_level=0.90,
)

print(uncertainty.to_markdown())
```

The report summarizes posterior/bootstrap uncertainty over:

- the observed aggregate value under sampled hidden-cell masses,
- lower and upper hidden-composition interval endpoints,
- ambiguity width,
- public adequacy rate,
- claim failure rate against `ambiguity_limit`.

By default, `hidden_composition_uncertainty(...)` uses
`preserve_public_law=True`. This keeps the public bucket distribution fixed and
resamples hidden composition within each public fiber. Set
`preserve_public_law=False` when you intentionally want full joint
public/hidden composition uncertainty.

It also emits structured tables:

```python
uncertainty.to_tables()["metric_summaries"]
uncertainty.to_tables()["draws"]
uncertainty.to_tables()["joint_cells"]
```

## Audit A Claim With Joint Draws

```python
claim = us.claim(
    estimate_name="Income-threshold target rate",
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="income_over_threshold",
    weight="sample_weight",
    q_presets=[us.q_tv_budget(0.10), us.q_chi_square_budget(0.25)],
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    ambiguity_limit=0.015,
)

verdict = us.audit_claim(
    rows_or_frame,
    claim,
    joint_model=joint,
    joint_draws=500,
    joint_seed=123,
)

print(verdict.to_markdown())
```

The claim report adds a **Model-Assisted Joint Analysis** section with:

- number of successful draws,
- failure rate against the claim's `ambiguity_limit`,
- public adequacy rate,
- ambiguity range and mean ambiguity across draws,
- per-draw status rows.

If `joint_model` is omitted but `joint_draws` is positive, `audit_claim(...)`
fits the joint model from the same data using the claim's public, hidden,
target, weight, and `min_cell_weight` settings.

Claim-level model-assisted analysis uses hidden-composition draws with the
public law preserved, matching the main reporting-stability question. Use the
standalone uncertainty report with `preserve_public_law=False` for monitoring
questions where future public bucket mix is also allowed to vary.

## Interpretation

Read the outputs as three distinct layers:

- **Observed-support ambiguity:** adversarial Q-based interval on the retained
  observed support.
- **Model-assisted hidden-composition draws:** plausible within-public-fiber
  compositions according to the fitted nonparametric joint model.
- **Statistical uncertainty:** external standard errors or intervals supplied
  by an upstream estimator.

The model-assisted layer introduces assumptions through the fitted joint cell
law and the chosen effective sample size. It is useful for plausibility,
future-composition stress tests, sparse-cell sensitivity, and monitoring, but it
should not be described as a distribution-free robustness guarantee.
