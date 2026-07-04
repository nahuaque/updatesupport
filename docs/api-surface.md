# API Surface

`updatesupport` is organized around claim audits. The main user path is small:

```python
claim = us.claim(
    "reported estimate is stable enough to use",
    public=[...],
    hidden=[...],
    target="metric",
    candidate_refinements=[...],
    ambiguity_limit=0.01,
)

audit = claim.audit(rows_or_frame)
```

## Core API

The public user-facing surface is:

- `us.claim(...)`: build a `ClaimSpec`.
- `ClaimSpec.audit(...)`: run the audit.
- `us.audit_claim(...)`: functional equivalent of `ClaimSpec.audit(...)`.
- `ClaimAudit`: the report object returned by an audit.
- `ClaimAudit.recommend_refinements(...)`: claim-centered refinement ranking.
- `us.claim_tree(...)`: organize related `ClaimSpec`s into a nested claim tree.
- `us.audit_claim_tree(...)`: audit a nested claim tree in one call.
- `ClaimTreeAudit`: the report object for hierarchical claim reviews.
- `us.threshold_decision(...)`: add a decision-invariance rule.
- `us.from_dataframe(...)`: compile rows when you need to inspect the finite
  problem before auditing.

The claim audit composes the lower-level machinery: primary interval evidence,
counterexample witnesses, representation certificates, decision-invariant
repairs, nested claim reports, model-assisted joint draws, structured exports,
and limitations.

The package `__all__` is curated around this surface. Some backend classes and
diagnostic dataclasses are importable as direct attributes, but they are not
part of the star-import surface.

## Advanced Evidence Tools

Use these directly only when you intentionally want a lower-level artifact:

- `public_descent_report(...)`: primary hidden-composition interval evidence.
- `sensitivity_report(...)`: grid over Q presets, hidden sets, or sparsity
  thresholds.
- `recommend_refinements(...)`: one-column ambiguity-reduction screening.
- `recommend_refinement_interactions(...)`: small interaction-aware refinement
  search.
- `attribute_refinement_ambiguity(...)`: Shapley-style attribution of joint
  ambiguity reduction across candidate refinements.
- `recommend_refinements_sensitivity(...)`: refinement ranking aggregated over
  a sensitivity grid.
- `public_representation_frontier(...)`: public-bucket design frontier.
- `certify_public_representation(...)`: standalone representation certificate.
- `breakdown_point(...)`: stress radius where a claim or decision stops passing.
- `robust_comparison_report(...)`: robust pairwise/ranking comparison evidence.

These are implementation depth behind the claim workflow. They remain useful for
method development, diagnostics, and specialized notebooks, but they should not
be the first thing a new analyst has to learn.
