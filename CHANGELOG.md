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
- Added `q_covariate_balance(...)`, a CVXPY/SOCP-compatible stress preset for
  causal and model-review workflows that bounds standardized hidden
  covariate-moment drift while preserving the observed public law, including
  parameterized sensitivity and support-function frontier support.
- Added finance-plugin portfolio concentration helpers:
  `q_factor_exposure_shift(...)` and `q_regional_concentration_shift(...)`,
  backed by exposure-weighted hidden-cell moments and the core
  covariate-balance preset.
- Added finance-plugin `finance_sensitivity_grid(...)` and
  `certify_portfolio_segmentation(...)` to produce model-risk Q profiles and
  pass/fail/inconclusive segmentation certificates backed by core frontier
  certification.
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
- Added scalarized public-representation frontier scoring and
  `search="scalarized"` so reviewers can rank or greedily search candidates by
  explicit weighted ambiguity/complexity tradeoffs.
- Added `search="mip"` for SCIP-powered public-representation column selection
  under saturated Q presets, including ambiguity limits, scalarized objectives,
  public-cell counting, hard bucket-budget constraints, and solver metadata in
  the frontier trace.
- Added `search="mip_oracle"` for budgeted stable reporting design: SCIP acts
  as a discrete public-representation master, compatible convex Q presets are
  evaluated through support-function oracles, and failed proposals receive
  no-good cuts.
- Added `search="mip_minimum"` / `search="mip_exact"` for exact minimum public
  representation under convex Q presets, with support-function oracle
  verification, `minimum_objective` selection, hard bucket constraints, and
  certificate/report trace metadata.
- Added claim-level verification with `ReportingClaim`,
  `ClaimVerificationReport`, and `verify_claim(...)`, composing primary
  public-descent evidence, statistical uncertainty, counterexample witnesses,
  repair/certification search, Markdown output, and structured exports.
- Added model-assisted joint analysis with `fit_joint_distribution(...)`,
  `NonparametricJointDistribution`, Bayesian-bootstrap joint-cell draws, and
  `verify_claim(..., joint_model=..., joint_draws=...)` summaries.
- Added `hidden_composition_uncertainty(...)` and
  `HiddenCompositionUncertaintyReport` for posterior/bootstrap uncertainty over
  hidden composition, including Bayesian-bootstrap and multinomial bootstrap
  draw methods, public-law-preserving within-fiber draws, quantile summaries,
  Markdown output, and structured exports.
- Added SOCP-compatible `q_l2_budget(...)` and
  `q_mahalanobis_budget(...)` presets for Euclidean and covariance-aware
  ellipsoidal hidden-composition stress tests through the CVXPY backends.
- Added `UncertainLinearTarget`, `target_standard_error=...`, and
  `effect_standard_error=...` so public-descent and causal-effect reports can
  widen hidden-composition ambiguity with supplied hidden-cell estimator
  standard errors while keeping the base point-estimate interval separate.
- Added an SOCP confidence-core diagnostic for `UncertainLinearTarget` under
  CVXPY-compatible Q sets, solving the common-overlap interval of admissible
  composition-specific estimator confidence bands and exporting it in reports.
- Hardened the finance plugin `ModelRiskReport` with explicit Markdown and
  structured export sections for the reported portfolio risk estimate, supplied
  statistical/model uncertainty, hidden-composition ambiguity,
  concentration-stress ambiguity, refinement recommendations, dual diagnostics,
  data diagnostics, limitations, and reviewer notes.
- Added finance-plugin model-assisted portfolio uncertainty via
  `model_assisted_portfolio_uncertainty(...)`, optional
  `composition_uncertainty_draws=...` on `model_risk_report(...)`, and
  `expected_loss_standard_error(...)` for PD/LGD delta-method hidden-cell
  estimator uncertainty.

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
