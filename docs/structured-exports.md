# Structured Exports

All major report objects can emit structured payloads in addition to Markdown.

```python
report = us.public_descent_report(...)

payload = report.as_dict()
json_text = report.to_json()
tables = report.to_tables()
frames = report.to_dataframes()
```

`to_tables()` is dependency-free. It returns a dictionary of named tables, where
each table is a tuple of row dictionaries:

```python
tables["summary"]
tables["worst_fibers"]
tables["refinements"]
```

`to_dataframes()` converts those same tables to pandas DataFrames. Pandas is an
optional dependency; install it directly or use the examples extra:

```bash
pip install "updatesupport[examples]"
# or
uv add "updatesupport[examples]"
```

The same helpers are also available as functions:

```python
us.report_to_json(report)
us.report_tables(report)
us.report_dataframes(report)
```

## AuditRun Exports

When a report is produced by `AuditSpec`, the executed `AuditRun` keeps the spec
and report together:

```python
run = spec.run(rows_or_frame)

run.to_json()
run.to_tables()
run.to_dataframes()
```

The table export includes the serialized spec plus prefixed report tables:

```python
tables["spec"]
tables["report_summary"]
tables["report_refinements"]
```

## Common Table Names

Public-descent reports expose:

- `summary`
- `worst_fibers`
- `refinements`
- `data_diagnostics`
- `dual_diagnostics`
- `estimator_uncertainty` when hidden-cell target standard errors were supplied

The `summary` table and JSON payload include `target_contract` metadata so
review systems can tell whether the report used a linear target, a supported
ratio target, or another compiled target contract. Procedure-aware reports also
include `target_procedure`, `target_procedure_context`, and `compiled_target`
fields so consumers can tell which reporting procedure produced the compiled
target values.

When `target_standard_error=...` or `effect_standard_error=...` is supplied,
public-descent exports include an `estimator_uncertainty` table. The summary
table also includes `has_estimator_uncertainty` and conservative adjusted
lower/upper/diameter fields. The table records the base point-estimate
transport interval, endpoint-adjusted margins when witness distributions are
available, and the conservative fixed-public-law outer interval.

If the selected Q backend can solve the SOCP confidence-core diagnostic, exports
also include `estimator_uncertainty_confidence_core`. That table records the
common-overlap interval, whether it is empty, the empty-core gap, endpoint
witness distributions, and available dual diagnostics.

Public-descent exports also include `fiber_decomposition_available` and
`fiber_diagnostic_kind`. When decomposition is unavailable, for example for a
variable-denominator ratio target, fiber rows report point ranges and set
`contribution` to `null` instead of emitting a misleading zero contribution.

Adversarial witness reports expose:

- `summary`
- `fiber_shifts`
- `cell_shifts`

Use `report.witness_report()` or `us.witness_report(...)` when a model review
needs to inspect the actual lower-vs-upper endpoint distributions behind the
reported ambiguity. The witness tables show which hidden cells gained or lost
mass and whether each public fiber still matches the same public distribution.

Sensitivity reports expose:

- `summary`
- `scenarios`

Refinement-sensitivity reports expose:

- `summary`
- `refinement_candidates`
- `refinement_scenarios`
- `refinement_rows`

Interaction-aware refinement reports expose:

- `summary`
- `interaction_candidates`
- `singletons`

Public-representation frontier reports expose:

- `summary`
- `search_trace`
- `screened_refinements`
- `frontier`
- `dominated`
- `candidates`
- `candidate_scenarios`

When scalarized frontier scoring is requested, `summary` includes
`scalarized_weights` and `best_scalarized`, while each candidate row includes
`scalarized_score` and `scalarized_components`.

When MIP frontier search is used, `search_trace` includes solver metadata such
as `solver`, `solver_status`, `objective_value`, and
`optimization_guarantee`. MIP-oracle search also includes
`oracle_iterations` and `oracle_rejections`. Non-scalarized MIP-oracle and
MIP-minimum runs also include `minimum_objective`, which records whether the
solver enumerated candidates by public-cell count or added-column count.

Representation-stability certificates expose:

- `summary`
- `reasons`
- `limitations`
- `selected_scenarios`
- prefixed frontier evidence such as `frontier_summary`,
  `frontier_candidates`, and `frontier_candidate_scenarios`

Claim audit reports expose:

- `summary`
- `claim`
- `claim_refinement_recommendations`
- `reasons`
- `limitations`
- prefixed primary evidence such as `primary_summary` and `primary_refinements`
- prefixed certificate evidence when a repair/certification was run
- prefixed witness evidence when a counterexample witness was produced
- model-assisted tables, when requested: `model_assisted_summary`,
  `model_assisted_metric_summaries`, `model_assisted_draws`, and
  `model_assisted_cells`

Hidden-composition uncertainty reports expose:

- `summary`
- `metric_summaries`
- `draws`
- `joint_cells`

Causal reporting suites prefix the component tables, for example:

- `primary_summary`
- `primary_refinements`
- `sensitivity_scenarios`
- `refinement_sensitivity_refinement_candidates`

## JSON Payloads

`to_json()` serializes the report's structured dictionary payload, converting
tuples and other sequence-like values into JSON-compatible arrays. Use this for
review artifacts, model cards, CI outputs, and systems that need stable report
metadata without parsing Markdown.
