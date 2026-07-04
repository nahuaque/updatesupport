# Interaction-Aware Refinement Search

The ordinary refinement table asks:

> If we add one hidden column to the public representation, how much ambiguity
> goes away?

Interaction-aware refinement search asks a broader question:

> Are there small sets of hidden columns that are weak alone but strong together?

This matters when the instability is carried by an interaction. For example,
`channel` may not help much by itself, and `tenure_band` may not help much by
itself, but `channel × tenure_band` may split the hidden cells that actually
drive the aggregate.

## Minimal Example

```python
import updatesupport as us

report = us.recommend_refinement_interactions(
    rows,
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "channel", "tenure_band", "device"],
    target="uplift",
    weight="n",
    candidate_refinements=["channel", "tenure_band", "device"],
    max_order=2,
)

print(report.to_markdown())
```

The report evaluates single columns and combinations up to `max_order`, then
ranks the candidate sets by ambiguity reduction.

## Key Columns

The candidate table includes:

- `reduction`: baseline ambiguity minus ambiguity after adding the column set,
- `interaction_gain`: additional reduction beyond the best one-column member of
  the set,
- `additive_synergy`: reduction beyond the sum of the one-column reductions in
  the set.

Positive `interaction_gain` is the main analyst signal. It means the combined
refinement does something that the best single column does not.

Positive `additive_synergy` is stronger. It means the combined refinement beats
the sum of the individual one-column reductions. Negative values are common and
usually indicate overlapping or redundant refinements.

## Search Bounds

The default search uses `max_order=2` and `max_evaluations=128`. This keeps the
first-cut API usable on wide tables. Increase `max_order` or set
`max_evaluations=None` when the candidate list is small enough for exhaustive
search.

```python
report = us.recommend_refinement_interactions(
    rows,
    public=public,
    hidden=hidden,
    target=target,
    candidate_refinements=candidates,
    max_order=3,
    max_evaluations=None,
)
```

If the search hits the cap, `report.truncated` is `True`. In that case, treat the
result as a screened search rather than an exhaustive ranking.

## Relation To Frontier Search

Interaction-aware refinement search is a diagnostic ranking tool. It answers:

> Which small refinement sets remove hidden-composition ambiguity, and do any of
> them work only jointly?

Public-representation frontier search is a design tool. It answers:

> Which public representations are Pareto-efficient under one or more stress
> scenarios and complexity constraints?

Use interaction-aware refinement search when you want a fast, readable table of
candidate interactions. Use frontier search when you need a broader design-space
optimization.

## Structured Exports

```python
payload = report.to_json()
tables = report.to_tables()
frames = report.to_dataframes()
```

The main tables are:

- `summary`
- `interaction_candidates`
- `singletons`

As elsewhere in `updatesupport`, the result is relative to the chosen hidden
refinement, target, and Q preset. It does not certify that no useful interaction
exists outside the supplied candidate columns or beyond the searched order.
