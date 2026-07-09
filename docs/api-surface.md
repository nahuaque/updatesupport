# API Surface

`updatesupport` is organized around claim-first public report design. The main
user path is small:

```python
claim = us.claim(
    "reported estimate is stable enough to use",
    public=[...],
    hidden=[...],
    target="metric",
    candidate_refinements=[...],
    ambiguity_limit=0.01,
)

design = claim.design(rows_or_frame)
```

## Core API

The public user-facing surface is:

- `us.claim(...)`: build a `ClaimSpec`.
- `ClaimSpec.design(...)`: audit the claim and design a defensible public
  representation.
- `us.design_public_report(...)`: functional equivalent of
  `ClaimSpec.design(...)`.
- `PublicReportDesign`: the report object returned by public-report design.
- `ClaimSpec.audit(...)`: run the audit.
- `us.audit_claim(...)`: functional equivalent of `ClaimSpec.audit(...)`.
- `ClaimAudit`: the report object returned by an audit.
- `ClaimSpec.calibrate_tv(...)`: calibrate a TV stress radius from historical
  period transitions and run rolling one-step backtests.
- `us.calibrate_tv_radius(...)`: functional equivalent of
  `ClaimSpec.calibrate_tv(...)`.
- `HistoricalTVCalibrationReport`: the calibration, rolling coverage evidence,
  calibrated Q preset, and current-period audit/design handoff.
- `ClaimAudit.recommend_refinements(...)`: claim-centered refinement ranking.
- `ClaimAudit.repair_plan(...)`: cost-aware action list for stabilizing a
  claim.
- `us.plan_claim_repair(...)`: functional helper for scripts; the method form
  `ClaimAudit.repair_plan(...)` is the preferred spelling once an audit exists.
- `ClaimRepairPlan`: the structured repair-plan report object.
- `us.claim_tree(...)`: organize related `ClaimSpec`s into a nested claim tree.
- `us.audit_claim_tree(...)`: audit a nested claim tree in one call.
- `ClaimTreeAudit`: the report object for hierarchical claim reviews.
- `us.threshold_decision(...)`: add a decision-invariance rule.
- `us.from_dataframe(...)`: compile rows when you need to inspect the finite
  problem before auditing.

Public-report design composes the lower-level machinery: claim audit evidence,
counterexample witnesses, representation certificates, frontier search,
decision-invariant repairs, repair plans, optional refinement attribution,
nested claim reports, model-assisted joint draws, structured exports, and
limitations.

The package `__all__` is intentionally narrower than the set of direct
attributes on `updatesupport`. It is the recommended star-import surface:
claim-first workflow, common report functions, Q presets, structured exports,
integration adapters, specs, and extension hooks. Diagnostic dataclasses,
backend reports, residopt internals, support-function internals, and named
linear feasibility objects remain importable directly or from their owning
modules, but they are not advertised through `from updatesupport import *`.

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
- `calibrate_tv_radius(...)`: historical TV-radius calibration and rolling
  one-step validation.
- `robust_comparison_report(...)`: robust pairwise/ranking comparison evidence.

These are implementation depth behind the claim workflow. They remain useful for
method development, diagnostics, and specialized notebooks, but they should not
be the first thing a new analyst has to learn.
