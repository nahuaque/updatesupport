# Changelog

All notable changes to `updatesupport` are documented here.

## Unreleased

- No unreleased changes.

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
