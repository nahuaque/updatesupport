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
- a pass/fail/inconclusive verdict,
- a counterexample witness when the public representation is unstable,
- a repair representation when candidate refinements can stabilize the claim,
- refinement recommendations and limitations.

## Verdict Semantics

The current public representation **passes** when its primary
hidden-composition ambiguity is no larger than `ambiguity_limit` and any
requested representation certificate also passes.

It **fails** when the current representation exceeds the ambiguity limit. If a
repair representation is found, the report still marks the original claim as
failed and shows the repair separately. This distinction is intentional: the
original public representation did not support the claim, but the report tells
you how to make a defensible version.

It is **inconclusive** when no `ambiguity_limit` is supplied or when the repair
search is heuristic and `exact_required=True`.

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
