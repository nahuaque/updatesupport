# Changelog

All notable changes to `updatesupport` are documented here.

## Unreleased

### Changed

- Consolidated finance snapshot serialization, runtime provenance, portable Q
  encoding and portfolio evidence restoration; existing formats still replay.
- Separated finance explanation comparison from hypothetical evidence planning
  behind the existing public imports; reused shared report exports and finite
  subset/frontier helpers without changing financial results or search order.

### Added

- Finance explanation comparisons across explicit accounting mappings, with
  checked witnesses, conflicts and mapping-dependent conclusions; bounded
  hypothetical evidence catalogs produce inclusion-minimal separating bundles
  while preserving source facts and rejecting vacuous separation.
- Direct finance evidence-outcome evaluation without an elimination goal,
  retaining mapping/dependency guards, surviving explanations and optional
  financial requirements for competing-accounts analyst briefs.
- Reviewed customer-advance bridges with typed movement mappings, signed
  noncash unknowns, incomplete-mapping residuals and separate indirect cash-flow
  diagnostics; conditional minimum alternative-explanation analyses reuse the
  existing solver and preserve scope, accounting basis, witnesses and conflicts.
- Joint finance portfolio evidence and raw shared-report frontiers, retaining
  per-metric coverage, known nonjoint contributions, missing-evidence floors,
  explicit evaluation scopes, and optional disclosure costs.
- Portable conditional bucket refinements and bounded, target-aware single-drill
  suggestions evaluated across every declared metric and Q.
- Conditional report/evidence acquisition frontiers preserving unavailable
  sources, unknown costs, ties, and fixed nonjoint weight assumptions.
- Offline joint portfolio snapshots and uncalibrated frozen reporting contracts,
  with issuer/share-class, unresolved-security, measurement, and support checks.
- Finance measurement contracts and completeness/freshness ledgers, with
  source-preserving numerical normalization and shared-primitive cash bridges.
- Rounded percentage allocation margins and share-threshold targets with
  explicit, positive amount denominators.
- Dated flow partitions, cumulative subtotals and explicitly selected timing
  schedules, retaining measurement and source-cohort scope.
- Structured evidence roles and constraint policies, named stress cases with
  conflict diagnosis, and break-even decision-variable analyses.
- Analyst decision briefs and snapshot comparison guards for target definitions,
  measurement windows and anonymous cohorts; older snapshots still replay.
- Finance portfolio universe/coverage ledgers and fixed-weight missing-data
  bounds, with provider-neutral holdings/fundamentals compilation, fiscal
  normalization diagnostics, explicit metric policies, and taxonomy provenance.
- Finance headline reports separating observed truth, summary support, and
  upstream coverage; scope-aware refinement results and breaking transfers in
  percentage points and currency amounts.
- Declarative disclosure allocation tables with leaf/subtotal hierarchies,
  reported rounding precision, measure equivalence declarations, preserved
  qualifiers, and structural exclusions.
- Portable portfolio snapshots containing evidence, configuration, coverage,
  results, raw capture references, and runtime provenance, with offline replay.
- Core `q_moment_bounds` and finance issuer-cap/reallocation mandates, plus
  `respect_q=True` minimum-breaking witnesses sharing the forward convex Q.

### Fixed

- Finance headline and coverage table exports no longer recurse through the
  generic exporter. Tables retain evaluation scope, missing positions, original
  universe totals, refinements and breaking transfers.
- Shared design deduplicates equivalent primary/scenario Q presets through
  serialization without collapsing distinct stress or solver settings.
- Finance decision briefs show infeasible interval status and absent endpoints
  separately from genuinely unbounded intervals.
- Decision-only design reports now show their decision-certifying refinement;
  ambiguity frontier candidates are labeled explicitly and shown only when an
  ambiguity criterion was declared.
- CVXPY local intervals preserve ordered, attainable endpoints when nominally
  equal solver values differ by roundoff within the problem tolerance.

## 0.1.6 - 2026-09-15

### Added

- Added vendor-neutral financial disclosure facts, precision-aware compilation,
  evidence-context validation, and explicit source links on constraints.
- Added claim-aware feasible allocation tables, financial reconciliation
  helpers, and versioned disclosure snapshots with offline replay and run diffs.
- Added named-linear assignment residual checks and irreducible conflict
  diagnosis, including variable bounds and financial source provenance.
- Named-linear claims now handle sufficient one-sided bounds and record an
  explicit numerical decision buffer; failures and infeasibility remain distinct
  from unresolved thresholds. Exact comparisons are available with zero buffers.

- Added optional claim-level `max_dropped_weight_share` requirements, with
  retained-weight evidence beside verdicts and in structured exports, and
  coverage-aware repair, rollup, portfolio, and frozen-policy evaluation.
- Added versioned frozen-policy save/load and dictionary/JSON round trips for
  portable column-target contracts, including support and fingerprint
  validation, preserved rollup mappings, and explicit rejection of unsupported
  executable targets or schema versions.

- Added historical total-variation radius calibration from consecutive-period
  within-public recompositions, with public-law restandardization, rolling
  one-step backtests, support-drift diagnostics, structured exports, and direct
  calibrated claim audit/design handoffs.
- Added exact one-column categorical rollup design under saturated Q, including
  claim-aware selection, group-count and public-cell tradeoffs, Pareto exports,
  Bell-number search guardrails, and transformed-data audit handoffs.
- Added exact multi-claim shared representation search through claim portfolios,
  with claim-specific scenario grids, shared schema constraints, portfolio
  certification, best-effort diagnostics, Pareto exports, and selected-schema
  claim audit handoffs.
- Added direct minimum claim-breaking witnesses under fixed public marginals,
  with exact TV/HiGHS solves, CVXPY-backed L2 and Mahalanobis geometries,
  margin-separated threshold semantics, within-fiber transfer ledgers, and
  structured report exports.
- Added calibrated public-report design orchestration combining historical TV
  calibration, optional exact categorical rollup, single- or multi-claim public
  representation search, selected-schema audits, and nearest-breaking-distance
  comparisons against the calibrated radius.
- Added frozen public-report policies for leakage-free future-batch audits and
  ordered holdout backtests, including deterministic policy fingerprints,
  design-period support contracts, realized TV-radius monitoring, support-drift
  diagnostics, operational verdicts, and structured exports.

### Packaging and QA

- Released `updatesupport` `0.1.6` and `updatesupport-finance` `0.1.5`.
- Raised the core `finance` extra to `updatesupport-finance>=0.1.5` and the
  finance plugin core dependency floor to `updatesupport>=0.1.6`.
- Synchronized the finance plugin descriptor and Colab installation requirements
  with the released package versions.

## 0.1.5 - 2026-07-09

### Added

- Added claim-centered public report design and cost-aware repair planning,
  including repair options, recommended public representations, and shared
  report export mixins.
- Added optional ResidOpt-backed endpoint and refinement screening for claims,
  certificates, and frontier search, with docs for exact/conservative
  screening semantics.
- Added named linear feasibility and disclosure-triangulation tooling,
  including named constraints, dual/binding diagnostics, claim audits,
  constraint attribution, and finance disclosure examples.
- Added RevOps funnel stability examples for direct, CSV, and trend workflows.
- Added conformal prediction integration with regression/classification
  adapters, multi-target reporting-stability reports, Sphinx API docs, and
  benchmark examples.

### Changed

- Centralized structured artifact exports through a shared report mixin and
  broadened top-level public API exports.
- Expanded benchmark-gallery coverage across RevOps, conformal prediction,
  causal/ACIC/Folktables, AI/ML evaluation, and product experimentation case
  studies.
- Updated finance plugin documentation and examples around disclosure
  triangulation and analyst-facing feasibility reports.

### Packaging and QA

- Bumped `updatesupport` to `0.1.5` and `updatesupport-finance` to `0.1.4`.
- Raised the core `finance` extra to `updatesupport-finance>=0.1.4` and the
  finance plugin core dependency floor to `updatesupport>=0.1.5`.

## 0.1.4 - 2026-07-05

### Added

- Added Shapley-style refinement attribution for allocating joint
  hidden-composition ambiguity reduction across candidate public refinements,
  with exact enumeration for small candidate sets and permutation sampling for
  larger sets.
- Added nested/hierarchical claim trees for auditing multi-level reporting
  claims, including recursive markdown reports and flat summary/node/edge
  exports for hierarchical Bayesian model-review workflows.
- Added multi-target support-function reports with per-target intervals,
  markdown output, structured tables, dataframe export hooks, and dual
  diagnostics for each endpoint solve.
- Added admissible-set intersection algebra via `q_intersection(...)`,
  `QPreset.__and__`, and `CvxpyAdmissibleSetSpec.intersect(...)`.
- Extended support-function frontier oracle search to accept convex
  `q_intersection(...)` composites and `bounded_shift` presets.

### Packaging and QA

- Bumped `updatesupport` to `0.1.4` and `updatesupport-finance` to `0.1.3`.
- Synchronized finance plugin metadata with the package version and raised its
  core dependency floor to `updatesupport>=0.1.4`.
- Updated finance Colab install cells to require the current core and finance
  package versions.

## 0.1.3 - 2026-07-04

### Added

- Added interaction-aware refinement reports for finding refinement sets whose
  combined ambiguity reduction is stronger than one-column rankings alone.
- Added robust comparison and ranking report exports for leaderboards,
  benchmark comparisons, and other pairwise-margin review workflows.
- Added breakdown-point report exports for threshold claims, including
  radius-level decision stability tables.
- Added AI/ML evaluation and product experimentation examples showing how to
  audit benchmark and A/B testing claims for hidden segment recomposition.
- Added a Sphinx documentation site with a Furo theme, custom landing-page
  styling, API reference pages, and a GitHub Pages deployment workflow.

### Changed

- Consolidated the high-level claim workflow around audit language:
  `us.claim(...)`, `ClaimSpec.audit(...)`, `us.audit_claim(...)`, and
  `ClaimAudit`.
- Reworked README and Sphinx documentation organization around the current API,
  positioning, mathematical boundaries, transport presets, refinement
  intelligence, and case-study links.
- Tightened documentation language around "hidden" meaning retained but not
  publicly reported, and around ambiguity being conditional on the chosen
  refinement and Q preset.
- Compressed repeated Sphinx framing so conceptual explanation lives in the
  dedicated representation, positioning, and mathematical-soundness pages.
- Hardened Colab notebooks and tutorial prose, including replacing the fragile
  EconML notebook path with a DoWhy downstream reporting audit.

### Packaging and QA

- Added docs build coverage to CI and included notebook examples in the source
  distribution manifest.
- Fixed Bandit false positives around claim label names and removed a raw
  `assert` from the interaction-refinement tests.

### Finance Plugin

- Bumped `updatesupport-finance` to `0.1.2`.
- Improved finance Colab notebooks with clearer exposition, safer plotting, and
  CVXPY stress-analysis walkthroughs.
- Refined finance documentation around the plugin value proposition and
  model-risk portfolio examples.

## 0.1.2 - 2026-07-03

### Added

- Added serializable audit specs and structured report exports for JSON, table,
  and dataframe workflows.
- Added data diagnostics, representation-stability certificates, model-assisted
  joint analysis, hidden-composition bootstrap/posterior summaries, and
  decision-invariance claim checks.
- Added target-contract guardrails and supported target families including
  linear, ratio, moment-transform, procedure, and uncertain linear targets.
- Added CVXPY, support-function, SOCP, and SCIP/MIP-backed solver paths,
  including covariate-balance, L2, Mahalanobis, fiber-support-floor, scalarized
  frontier, MIP frontier, and exact minimum-representation search modes.
- Added analyst-facing witness reports, finance concentration-stress helpers,
  and model-assisted portfolio uncertainty.

### Fixed

- Fixed saturated fixed-public-law adequacy witnesses so zero-mass public fibers
  are ignored and nonzero fixed public masses scale the witness gap correctly.
- Fixed frontier observed-value calculations to use the compiled target
  functional directly, preserving ratio/procedure target semantics.

## 0.1.1 - 2026-07-02

### Added

- Added `public_representation_frontier(...)` for searching public reporting
  designs that trade off public-cell count, added refinement columns, and
  hidden-composition ambiguity.
- Added frontier search modes: exhaustive, greedy, and beam search, with trace
  metadata for evaluated candidates, exactness, pruning, and stopping reasons.
- Added selected-representation explanations for frontier reports, including
  baseline-vs-selected ambiguity, scenario-level reductions, screened-out
  refinements, and close dominated alternatives.
- Added sensitivity-aware frontier grids over Q presets, sparse-cell thresholds,
  and alternate hidden-state definitions.
- Added core plugin infrastructure with `UpdateSupportPlugin`,
  `PluginRegistry`, entry-point discovery, and lookup helpers for plugin
  metrics, Q presets, compilers, and report profiles.
- Added `RowMetric` / `row_metric(...)` so domain packages can define
  row-level targets without changing the tabular compiler.
- Added a core `finance` optional extra that depends on the separate
  `updatesupport-finance` package.

### Changed

- Integrated public-representation frontier output into the Folktables ACS
  example so case studies can show both ambiguity diagnostics and candidate
  public reporting designs.
- Expanded the public API exports for frontier search, plugin registration, and
  row-metric support.
- Reworked the top-level README around the main value proposition:
  hidden-composition ambiguity as a reporting-stability audit.
- Moved detailed theory and backend material out of the README into
  `docs/theory-and-backends.md`.
- Linked causal-inference integration docs to EconML, DoWhy, DoubleML, and
  CausalML project documentation.
- Updated install guidance now that `updatesupport` is available from PyPI via
  `pip install updatesupport` or `uv add updatesupport`.

### Packaging and Release

- Added CI coverage for Python 3.10 through 3.13, package builds, distribution
  checks, and the finance workspace package.
- Added a separate finance publish workflow for the monorepo plugin package.
- Hardened the core PyPI publish workflow with locked dependency sync, Ruff,
  pytest, Bandit, distribution builds, `twine check`, and disabled PyPI
  attestations for the core publish path.
- Included the changelog, docs, and examples in the core source distribution.
- Added release documentation covering the core and plugin tag conventions.

## 0.1.0 - 2026-07-01

- Initial PyPI release.
