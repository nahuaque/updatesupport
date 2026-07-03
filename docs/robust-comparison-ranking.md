# Robust Comparison And Ranking

Robust comparison asks:

> If the public mix is fixed but the hidden composition inside public buckets
> changes, does the observed winner or ranking still hold?

This is the comparison version of public-descent analysis. Instead of auditing a
single scalar target, `updatesupport` audits several alternatives in long-form
data and checks pairwise margins under the same admissible hidden recomposition.

## Why Pairwise Margins Matter

Do not certify a ranking by comparing independent item intervals. Independent
intervals allow each alternative to move under a different hidden mix, which is
usually not the comparison being reported.

For alternatives `A` and `B`, the robust comparison target is the cellwise
margin:

```text
score(A, hidden cell) - score(B, hidden cell)
```

For “lower is better” metrics such as loss, the sign is reversed. The pairwise
ordering is certified when the lower endpoint of that margin interval stays
above the chosen margin threshold, usually zero.

## Minimal Example

```python
import updatesupport as us

report = us.robust_comparison_report(
    rows,
    item="model",
    public=["task_family", "difficulty"],
    hidden=["task_family", "difficulty", "source", "prompt_template"],
    target="score",
    weight="n",
    q="saturated",
    higher_is_better=True,
)

print(report.to_markdown())
```

The input should be long-form: one row per alternative and hidden cell, or raw
rows that aggregate to that shape.

## Report Status

The report status has three levels:

- `full_ranking_stable`: every observed pairwise ordering is certified,
- `winner_stable`: the observed winner is certified, but some lower-ranked
  pairwise ordering is not,
- `ambiguous_winner`: at least one challenger can become competitive with the
  observed winner under the selected stress test.

## Lower-Is-Better Metrics

For losses, error rates, latency, default rates, or other metrics where smaller
values are better:

```python
report = us.robust_ranking_report(
    rows,
    item="model",
    public=["segment"],
    hidden=["segment", "scenario"],
    target="loss",
    weight="n",
    higher_is_better=False,
)
```

`robust_ranking_report()` is an alias for `robust_comparison_report()`.

## Structured Exports

Robust comparison reports support the same artifact paths as other reports:

```python
payload = report.to_json()
tables = report.to_tables()
frames = report.to_dataframes()
```

The main tables are:

- `summary`: winner, certified winner, status, Q, and representation metadata,
- `items`: observed value and hidden-composition interval per alternative,
- `pairwise_margins`: margin interval and certification status for each observed
  pairwise ordering.

## Current Scope

The first API slice assumes a balanced comparison design: every alternative has
the same retained hidden-cell support and the same hidden-cell weights. That is
the clean setting for model leaderboards, benchmark slices, A/B variants scored
on the same segment cells, and other common comparison reports.

As elsewhere in `updatesupport`, `hidden` means not publicly reported in the
coarse comparison. It does not mean unobserved by the analyst.
