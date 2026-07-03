# Model-Assisted Joint Analysis

`updatesupport` can fit a nonparametric joint distribution over retained public
and hidden cells, then use draws from that fitted joint law to rerun reporting
stability checks.

This is a model-assisted layer. It does not replace adversarial Q-based
ambiguity. It answers a different question:

> Across plausible public/hidden compositions generated from the fitted joint
> cell law, how often would this public representation still support the claim?

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

You can inspect or reuse draws directly:

```python
draw = joint.draw(seed=123)
records = draw.records()
```

The records are weighted hidden-cell rows with generated target and weight
columns, so they can be fed back into the usual report helpers.

## Verify A Claim With Joint Draws

```python
claim = us.ReportingClaim(
    estimate_name="Income-threshold target rate",
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="income_over_threshold",
    weight="sample_weight",
    q_presets=[us.q_tv_budget(0.10), us.q_chi_square_budget(0.25)],
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    ambiguity_limit=0.015,
)

verdict = us.verify_claim(
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

If `joint_model` is omitted but `joint_draws` is positive, `verify_claim(...)`
fits the joint model from the same data using the claim's public, hidden,
target, weight, and `min_cell_weight` settings.

## Interpretation

Read the outputs as three distinct layers:

- **Observed-support ambiguity:** adversarial Q-based interval on the retained
  observed support.
- **Model-assisted joint draws:** plausible compositions according to the
  fitted nonparametric joint model.
- **Statistical uncertainty:** external standard errors or intervals supplied
  by an upstream estimator.

The model-assisted layer introduces assumptions through the fitted joint cell
law and the chosen effective sample size. It is useful for plausibility,
future-composition stress tests, sparse-cell sensitivity, and monitoring, but it
should not be described as a distribution-free robustness guarantee.
