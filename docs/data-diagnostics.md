# Data Diagnostics

`updatesupport` runs lightweight data diagnostics while compiling tabular data
and before solving the transport problem.

Diagnostics do not replace schema validation. Hard errors, such as missing
required columns, non-finite targets, or negative weights, still raise
exceptions. Diagnostics are for review-relevant conditions where the audit can
continue but the retained support deserves interpretation.

## What Is Reported

Compiled `GroupedProblem` objects carry a `diagnostics` object:

```python
grouped = us.from_dataframe(...)

grouped.diagnostics.as_dict()
grouped.diagnostics.diagnostics
```

Public reports include those diagnostics and any report-level candidate
refinement diagnostics:

```python
report = us.public_descent_report(
    rows_or_frame,
    public=["segment"],
    hidden=["segment", "driver", "region"],
    target="outcome_rate",
    candidate_refinements=["driver", "missing_column"],
)

report.diagnostics
report.to_tables()["data_diagnostics"]
```

## Current Checks

The current pre-solve diagnostics include:

- hidden cells dropped by `min_cell_weight`
- dropped weight and dropped weight share
- zero-weight rows
- missing category values encoded as `NA`
- public cells with only one retained hidden cell
- public cells whose retained hidden-cell target values are constant
- candidate refinements that are already public
- candidate refinements not present in the hidden state space

Hard data errors remain hard errors:

- public columns not included in hidden columns
- missing required public, hidden, target, or weight columns
- non-finite targets or weights
- negative weights
- no retained hidden cells after sparse-cell filtering

## Interpretation

Singleton public fibers and constant-target fibers are not wrong. They mean
those public cells cannot contribute hidden-composition ambiguity under the
retained hidden state space.

Dropped hidden cells are more consequential. Raising `min_cell_weight` can make
the state space less noisy, but it changes both the retained support and the
observed public law used in the stress test.

## Require Sufficient Retained Weight

Claims can make coverage an explicit requirement:

```python
claim = us.claim(
    "Reported rate is stable",
    public=["segment"],
    hidden=["segment", "channel"],
    target="rate",
    weight="count",
    min_cell_weight=10,
    ambiguity_limit=0.02,
    max_dropped_weight_share=0.05,
)
audit = claim.audit(rows)
```

The claim is `inconclusive` if filtering discards more than 5% of input weight
in the primary report or any evaluated certificate scenario. Equality is
allowed. Without a weight column, this measures the share of rows discarded.
The limit must be finite and between zero and one; `None` (the default) keeps
the existing verdict behavior.

This guard is separate from the ambiguity calculation: an interval can remain
exact for the retained population while coverage is insufficient for the
claim. A public refinement cannot restore discarded weight, so an insufficient
coverage result also prevents certifying repair recommendations, rollup
candidates, and shared representation candidates. The lower-level
representation certificate continues to describe ambiguity on retained support.

`audit.coverage` exposes the primary retained and dropped weight shares, the
worst evaluated dropped share, the declared limit, and
`coverage_requirement_met`. These fields appear in JSON and summary tables;
Markdown shows retained input weight beside the verdict. With a coverage
requirement, unavailable diagnostics produce an inconclusive verdict, and
experimental endpoint screening is disabled so evaluated scenarios retain
their coverage diagnostics. An input with no surviving cells still raises a
data-validation error.

The requirement is preserved when a calibrated design is frozen and saved.
Each future audit checks coverage against the weight in that future batch.

Missing category values are encoded as `NA` so the audit can proceed. If the
amount of missingness is material, treat `NA` as an explicit category in the
review rather than as harmless noise.
