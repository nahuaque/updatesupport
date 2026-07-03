# Reporting Claims

`ReportingClaim` is the highest-level review artifact in `updatesupport`.
Instead of choosing individual diagnostics up front, declare the aggregate claim
you want to defend and call `verify_claim(...)`.

```python
import updatesupport as us

claim = us.ReportingClaim(
    estimate_name="Income-threshold target rate",
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="income_over_threshold",
    weight="sample_weight",
    q_presets=[us.q_tv_budget(0.10), us.q_chi_square_budget(0.25)],
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    ambiguity_limit=0.015,
    decision=us.threshold_decision(
        ">=",
        0.12,
        label="income-threshold rate clears reporting floor",
    ),
    bucket_budget=40,
    statistical_interval=(0.119, 0.128),
)

verdict = us.verify_claim(rows_or_frame, claim)
print(verdict.to_markdown())
```

The verifier produces one report that separates:

- the reported causal or statistical estimate,
- statistical uncertainty, if supplied,
- hidden-composition ambiguity,
- decision invariance, if a threshold decision rule is supplied,
- a pass/fail/inconclusive verdict,
- a counterexample witness when the public representation is unstable,
- a repair representation when candidate refinements can stabilize the claim,
- refinement recommendations and limitations.

## Verdict Semantics

The current public representation **passes** when its primary
hidden-composition ambiguity is no larger than `ambiguity_limit`, any supplied
decision rule is invariant over the hidden-composition interval, and any
requested representation certificate also passes.

It **fails** when the current representation exceeds the ambiguity limit or
when the hidden-composition interval crosses the supplied decision threshold.
If a repair representation is found, the report still marks the original claim
as failed and shows the repair separately. This distinction is intentional: the
original public representation did not support the claim, but the report tells
you how to make a defensible version.

It is **inconclusive** when neither an `ambiguity_limit` nor a decision rule is
supplied, or when the repair search is heuristic and `exact_required=True`.

## Decision Invariance

Use `decision=...` when the practical claim is a conclusion or action rather
than a raw interval width:

```python
claim = us.ReportingClaim(
    estimate_name="Expected-loss estimate",
    public=["product", "region", "score_band"],
    hidden=["product", "region", "score_band", "channel", "vintage"],
    target="expected_loss",
    weight="ead",
    q_presets=[us.q_tv_budget(0.10)],
    decision=us.threshold_decision(
        "<=",
        0.025,
        label="expected loss within model-review tolerance",
    ),
    candidate_refinements=["channel", "vintage"],
)
```

For `pass_if <= threshold`, the decision is certified when the upper endpoint
of the hidden-composition interval is still at or below the threshold. It is
certified failed when the lower endpoint is already above the threshold. It is
not invariant when the interval crosses the threshold.

The verifier reports:

- the observed decision,
- lower- and upper-endpoint decisions,
- whether the decision is invariant,
- a decision-specific repair representation, if candidate refinements can make
  the observed decision invariant across all evaluated stress-test scenarios.

This is intentionally different from `ambiguity_limit`. A wide interval can
still support a decision if it stays on one side of the threshold, and a narrow
interval can fail decision invariance if it straddles the threshold.

## Exact Minimum Repairs

By default, claim verification uses exhaustive certificate search because it is
portable. When SCIP is installed and the Q presets are compatible convex
presets, request the exact-minimum repair path:

```python
claim = us.ReportingClaim(
    estimate_name="Expected-loss estimate",
    public=["product", "region"],
    hidden=["product", "region", "score_band", "ltv_band", "channel"],
    target="expected_loss",
    weight="ead",
    q_presets=[us.q_tv_budget(0.10), us.q_kl_budget(0.05)],
    candidate_refinements=["score_band", "ltv_band", "channel"],
    ambiguity_limit=0.005,
    bucket_budget=40,
    search="mip_minimum",
)
```

This routes the certificate through the SCIP master plus support-function oracle
path and records the exact-minimum guarantee in the embedded certificate trace.

## Structured Output

Claim verification reports support:

```python
verdict.as_dict()
verdict.to_json()
verdict.to_tables()
verdict.to_dataframes()
```

The tables include a top-level `summary`, the serialized `claim`, decision
`reasons`, `limitations`, prefixed primary public-descent evidence, and prefixed
certificate or witness evidence when those components are present.

## Model-Assisted Joint Draws

For plausibility analysis, fit a nonparametric joint distribution and pass it to
the verifier:

```python
joint = us.fit_joint_distribution(
    rows_or_frame,
    public=claim.public,
    hidden=claim.hidden,
    target=claim.target,
    weight=claim.weight,
)

verdict = us.verify_claim(
    rows_or_frame,
    claim,
    joint_model=joint,
    joint_draws=500,
    joint_seed=123,
)
```

The report adds a model-assisted section summarizing ambiguity ranges, public
adequacy rates, and claim failure rates across fitted-joint draws. This is a
model-assisted plausibility check, not a replacement for the adversarial Q
interval. See [Model-assisted joint analysis](model-assisted-joint-analysis.md).
