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

Missing category values are encoded as `NA` so the audit can proceed. If the
amount of missingness is material, treat `NA` as an explicit category in the
review rather than as harmless noise.
