# Representation Stability Certificates

`certify_public_representation(...)` turns public-representation frontier search
into a review-ready decision artifact for model reviews, dashboard releases,
causal reports, monitoring controls, or governance packets.

## Basic Use

```python
import updatesupport as us

certificate = us.certify_public_representation(
    rows_or_frame,
    base_public=["product", "region"],
    hidden=[
        "product",
        "region",
        "score_band",
        "ltv_band",
        "vintage",
        "channel",
    ],
    target="expected_loss",
    candidate_refinements=["score_band", "ltv_band", "vintage", "channel"],
    q_presets=["saturated", us.q_bounded_shift(0.5)],
    ambiguity_limit=0.0025,
    bucket_budget=80,
    search="exhaustive",
)

print(certificate.to_markdown())
```

For L2-budget stress tests, certificates can optionally use the experimental
residopt screening backend during frontier evaluation:

```python
certificate = us.certify_public_representation(
    rows_or_frame,
    base_public=["product", "region"],
    hidden=["product", "region", "score_band", "ltv_band", "vintage"],
    target="expected_loss",
    candidate_refinements=["score_band", "ltv_band", "vintage"],
    q_presets=[us.q_l2_budget(0.05)],
    ambiguity_limit=0.0025,
    screening_backend="residopt",
)
```

When screening is enabled, the certificate reports which frontier endpoints
were certified by conservative bounds, which endpoints required exact fallback,
and how many exact solves were avoided.

The returned `RepresentationStabilityCertificate` includes:

- `status`: `pass`, `fail`, or `inconclusive`;
- `certified_candidate`: the selected representation when the certificate
  passes;
- `selected_candidate`: the stable evaluated candidate, including provisional
  candidates from heuristic searches;
- `frontier`: the full underlying `PublicRepresentationFrontier`;
- `reasons` and `limitations`;
- `to_markdown()`, `to_json()`, `to_tables()`, and `to_dataframes()`.

## Status Meanings

`pass` means an evaluated representation satisfied:

- the supplied `ambiguity_limit`;
- the supplied `bucket_budget`, if any;
- the exact-search requirement, if `exact_required=True`.

`fail` means no evaluated representation satisfied the ambiguity limit and
bucket budget.

`inconclusive` means the search was heuristic while `exact_required=True`.
The run may still have found a stable evaluated representation, but the
certificate does not claim that unevaluated candidates were ruled out.

Set `exact_required=False` when you intentionally want a certificate over only
the evaluated candidates:

```python
certificate = us.certify_public_representation(
    rows_or_frame,
    base_public=["segment"],
    hidden=["segment", "region", "channel"],
    target="outcome_rate",
    candidate_refinements=["region", "channel"],
    ambiguity_limit=0.01,
    search="beam",
    exact_required=False,
)
```

## Relationship To Frontier Search

`public_representation_frontier(...)` is exploratory. It shows tradeoffs among
candidate public representations.

`certify_public_representation(...)` is decisional. It runs the same frontier
machinery, then selects the smallest candidate that meets the certificate
requirements and records the evidence.

The certificate does not hide the frontier: structured exports include prefixed
frontier tables so downstream review systems can inspect candidates, scenarios,
screened refinements, and search trace metadata.

## Interpretation Rules

A certificate is a representation-stability statement, not a statistical
confidence statement. It is conditional on:

- the retained support;
- the hidden columns and hidden-set scenarios;
- minimum-cell filtering;
- the compiled target;
- the declared `Q` stress-test grid;
- the searched candidate refinements and search mode.

It does not cover unseen hidden cells, future support drift, model-estimation
uncertainty, or survey-design uncertainty unless those are represented in the
supplied target or stress grid.
