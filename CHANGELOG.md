# Changelog

All notable changes to `updatesupport` are documented here.

## Unreleased

### Added

- Added `AuditSpec`, `QSpec`, `AuditRun`, and `run_audit(...)` for
  JSON-serializable public-descent, sensitivity, and public-representation
  frontier/certificate audit configurations.
- Added `RepresentationStabilityCertificate` and
  `certify_public_representation(...)` to turn frontier search into a
  pass/fail/inconclusive review artifact with selected representation,
  assumptions, limitations, Markdown output, and structured exports.
- Added `BatchedCvxpyEnvironments` plus `backend="batched_cvxpy"` for
  CVXPY-backed Q presets, enabling compatible sensitivity-grid rows to solve
  local intervals with scenario-by-state CVXPY variables.
- Added `as_dict()` methods for public-descent and sensitivity reports so
  executed audit specs can emit structured report payloads.
- Added structured export helpers and report methods: `to_json()`,
  `to_tables()`, `to_dataframes()`, `report_to_json(...)`,
  `report_tables(...)`, and `report_dataframes(...)`.
- Added pre-solve data diagnostics for retained/dropped support, sparse-cell
  filtering, missing category encoding, singleton public fibers, constant-target
  fibers, and skipped refinement candidates.
- Added plugin SDK polish with `PluginMetadata`, plugin validation reports,
  duplicate-name protection, `validate_plugin(...)`, `assert_valid_plugin(...)`,
  and plugin descriptor serialization.
- Added a mathematical/statistical soundness note documenting the fixed linear
  target contract, nonlinear target boundary, finite optimization model, Q
  preset semantics, backend guarantees, statistical assumptions, and
  limitations.
- Added `LinearTarget` and `TargetContract` as the first internal target
  functional layer, with target-contract metadata in public-descent structured
  exports and Markdown reports.
- Added `UnsupportedTarget` and `UnsupportedTargetError` guardrails so nonlinear
  or representation-dependent target objects fail explicitly instead of being
  silently interpreted as fixed linear targets.
- Added `RatioTarget` for fixed linear-fractional targets, with exact saturated
  public-fiber intervals, finite-environment evaluation, CVXPY DQCP support for
  local/fixed-public-law ratio intervals, and explicit guardrails for ratio
  cases that still need dedicated constrained solvers.
- Added `TargetCapabilities` and `MomentTransformTarget` for fixed transforms
  of linear moments, with affine transforms reduced to linear targets,
  convex/concave transforms exposing exact CVXPY-compatible one-sided
  endpoints, monotone transforms exposing conservative interval bounds, and
  non-additive decomposition APIs gated by capability flags.
- Added `ProcedureTarget` and `ProcedureTargetContext` for
  representation-dependent reporting procedures that compile to a column or
  row metric per public representation before solving.
- Added optional SCIP solver support for CVXPY-backed presets via
  `updatesupport[scip]`, `solver="SCIP"` on CVXPY environments, solver
  metadata on `QPreset`/`QSpec`, and clearer missing-solver diagnostics.
- Added `q_fiber_support_floor(...)`, the first SCIP-backed mixed-integer Q
  preset, to require each public fiber to keep a minimum number of hidden cells
  active above a minimum share; `QSpec` now carries preset-specific `settings`.
- Added `WitnessReport` and `witness_report(...)` for analyst-facing
  lower-vs-upper adversarial witness reports that show which hidden cells move
  between interval endpoints while the public distribution stays fixed.
- Added `ConvexAdmissibleSet`, `SupportFunctionResult`, and
  `SupportFunctionBackend`, a CVXPY `SuppFunc`-based backend for evaluating
  fixed-linear transport intervals as support functions of the admissible
  hidden-distribution set.
- Added `CvxpyAdmissibleSetSpec` and `cvxpy_admissible_set_spec(...)` so
  compatible CVXPY Q presets expose reusable admissible-set constraint builders
  for standard, parameterized, batched, and support-function backends.
- Added `SupportFunctionIntervalResult` and `support_interval(...)` methods for
  direct support-function interval evaluation from a convex admissible set,
  support-function backend, or CVXPY admissible-set spec.

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
