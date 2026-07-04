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

## Shapley Attribution

Use `attribute_refinement_ambiguity(...)` when you want to allocate joint
ambiguity reduction back to the candidate columns:

```python
attribution = us.attribute_refinement_ambiguity(
    rows,
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "channel", "tenure_band", "device"],
    target="uplift",
    weight="n",
    candidate_refinements=["channel", "tenure_band", "device"],
)

print(attribution.to_markdown())
```

The value function is:

> ambiguity reduction after adding a set of hidden columns to the public
> representation.

The Shapley value for a column is its average marginal reduction across the
coalitions in which it could be added. This answers a different question from
the interaction table:

- `recommend_refinements(...)`: which column helps most by itself?
- `recommend_refinement_interactions(...)`: which small column sets work well
  together?
- `attribute_refinement_ambiguity(...)`: how should the joint reduction be
  attributed across the supplied columns?

For small candidate sets, the attribution is exact and enumerates all
coalitions. For larger sets, the function uses permutation sampling unless you
raise `max_exact_columns`.

The attribution table includes:

- `shapley_value`: allocated ambiguity reduction,
- `shapley_percent`: share of the full candidate-set reduction,
- `singleton_reduction`: reduction from adding the column alone,
- `interaction_lift`: Shapley value minus singleton reduction.

Positive `interaction_lift` means a column matters more in combination than it
appears to matter alone. Negative `interaction_lift` usually means the column's
solo effect overlaps with other refinements.

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

Refinement attribution reports expose:

- `summary`
- `attributions`
- `coalitions`

As elsewhere in `updatesupport`, the result is relative to the chosen hidden
refinement, target, and Q preset. It does not certify that no useful interaction
exists outside the supplied candidate columns or beyond the searched order.
