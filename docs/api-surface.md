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
- `ClaimSpec.design_categorical_rollup(...)`: find an exact global grouping of
  one retained categorical column under saturated Q.
- `us.design_categorical_rollup(...)`: functional equivalent of the claim
  method.
- `CategoricalRollupDesign`: selected category mapping, group-count tradeoffs,
  Pareto frontier, structured exports, and transformed-data audit handoff.
- `us.claim_portfolio(...)`: declare claims that must share one public schema.
- `ClaimPortfolio.design(...)`: run exact shared representation search.
- `us.design_shared_representation(...)`: functional equivalent of the
  portfolio method.
- `SharedRepresentationDesign`: selected common schema, per-claim outcomes,
  shared frontier, best-effort diagnostics, and full claim-audit handoff.
- `ClaimSpec.design_calibrated(...)`: calibrate historical TV stress and design
  the current public representation for one claim.
- `ClaimPortfolio.design_calibrated(...)`: apply the same workflow to one shared
  public representation across several claims.
- `us.design_calibrated_public_report(...)`: functional equivalent of the
  calibrated claim and portfolio methods.
- `CalibratedPublicReportDesign`: calibration backtests, optional categorical
  rollup, selected schema, current audits, nearest breaking witnesses, and
  structured exports.
- `CalibratedPublicReportDesign.freeze(...)`: preserve the selected schema,
  rollup mapping, calibrated claim presets, and design-period support as an
  out-of-sample reporting policy.
- `FrozenPublicReportPolicy.audit(...)`: evaluate one future batch without
  recalibration or representation search.
- `FrozenPublicReportPolicy.backtest(...)`: apply the same frozen contract to
  ordered holdout periods.
- `FrozenPublicReportPolicy.save(...)` and `.load(...)`: write and restore a
  versioned, fingerprint-validated portable policy. `to_dict`/`from_dict` and
  `to_json`/`from_json` support application-managed storage.
- `FrozenPolicyAudit` and `FrozenPolicyBacktest`: operational pass, review, or
  inconclusive verdicts with support-drift and calibrated-radius diagnostics.
- `ClaimSpec.breaking_witness(...)`: find the closest fixed-public hidden-cell
  recomposition that fails the claim's threshold decision.
- `ClaimAudit.breaking_witness(...)`: reuse an audit's compiled problem for the
  same inverse solve.
- `us.minimum_claim_breaking_witness(...)`: functional equivalent of the claim
  method.
- `MinimumClaimBreakingWitnessReport`: minimum distance, decision-flipping cell
  law, within-fiber transfer ledger, solver certificate, and exports.
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

`ClaimSpec.max_dropped_weight_share` optionally requires sufficient retained
input weight. `ClaimAudit.coverage` exposes that evidence and the requirement
result; coverage failures produce an inconclusive verdict and prevent certifying
refinement recommendations. The limit also applies to shared representation
and rollup candidates and persists in frozen policies.

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
- `minimum_claim_breaking_witness(...)`: direct inverse solve for the closest
  threshold-flipping composition in TV, L2, or Mahalanobis geometry.
- `calibrate_tv_radius(...)`: historical TV-radius calibration and rolling
  one-step validation.
- `design_categorical_rollup(...)`: exact restricted partition design for one
  categorical refinement under saturated Q.
- `design_shared_representation(...)`: exact common-schema search across
  several claims with claim-specific targets and stress scenarios.
- `design_calibrated_public_report(...)`: compose historical TV calibration,
  optional one-column rollup, single/shared schema search, and direct threshold
  breaking witnesses.
- `robust_comparison_report(...)`: robust pairwise/ranking comparison evidence.

These are implementation depth behind the claim workflow. They remain useful for
method development, diagnostics, and specialized notebooks, but they should not
be the first thing a new analyst has to learn.

## Finance API

Import financial workflows from `updatesupport_finance`. These APIs are available
in `updatesupport-finance>=0.1.6` with `updatesupport>=0.1.7`. They use the core
solver and accept reviewed evidence and policies without requiring a provider.

| Layer | Main API | Responsibility |
| --- | --- | --- |
| Source evidence | `DisclosureFact`, `disclosure_fact_constraint`, `link_disclosure_evidence` | Preserve source context, lineage and reported precision. |
| Measurement contracts | `MeasurementDefinition`, `DisclosureRequirement`, `normalize_disclosure_fact`, `compile_disclosure_evidence` | Normalize numerical units and quarantine incomplete or incompatible inputs. |
| Accounting models | `allocation_table`, `percentage_constraints`, `CashMeasure`, `time_window_allocation`, `CustomerAdvanceBridge` | Compile declared relationships, scopes and movement mappings. |
| Conditional analysis | `ConstraintPolicy`, `DisclosureStressCase`, `break_even_analysis`, `minimum_advance_explanation` | Re-solve assumptions and necessary financial amounts under explicit conditions. |
| Competing explanations | `DisclosureAlternative`, `compare_disclosure_explanations` | Compare accounts across accounting mappings, preserving bounds, witnesses and conflicts. |
| Next evidence | `DisclosureEvidenceRequest`, `evaluate_disclosure_evidence`, `plan_disclosure_evidence` | Test hypothetical answers or search a finite catalog for separating bundles. |
| Portfolio evidence | `PortfolioUniverse`, `PortfolioMetricPolicy`, `compile_portfolio_evidence`, `compile_portfolio_metrics` | Account for each original position and retain per-measure coverage. |
| Portfolio reporting | `portfolio_headline_report`, `joint_portfolio_report`, `plan_portfolio_repairs` | Separate headline truth, summary support, reporting choices and missing evidence. |
| Review and replay | `AnalystDecisionBrief`, `capture_disclosure_snapshot`, `capture_portfolio_snapshot`, `capture_joint_portfolio_snapshot`, `FrozenJointPortfolioContract` | Export review artifacts and replay saved inputs offline. |

These layers share numerical and artifact infrastructure, while their evidence
meanings stay explicit. An accounting mapping is a reviewed relationship;
an explanation is a catalog restriction; an evidence outcome is hypothetical.
Portfolio acquisition plans describe conditional width reductions. None of
these establishes the company's actual cause or supplies a missing observation.

`updatesupport_finance.explanations` retains the same public comparison and
planning imports as the package root. Private modules beginning with `_` are
implementation details. Snapshot formats keep their existing schemas and replay
validation; stored source fingerprints can report a changed implementation.

Use the [disclosure evidence guide](disclosure-evidence.md),
[customer-advance guide](customer-advance-reconciliation.md),
[explanation workflow](disclosure-explanations.md) and
[joint portfolio guide](joint-portfolio-report.md) for complete examples.
