# Categorical Rollup Design

`design_categorical_rollup(...)` searches for an exact global grouping of one
retained categorical column under saturated Q. It answers a more specific
question than column-subset refinement search:

> Can we publish a useful rollup of this category instead of either suppressing
> it completely or exposing every retained level?

The first slice exhaustively evaluates every set partition of the retained
category levels up to a declared maximum group count. It returns the smallest
budget-feasible grouping that certifies the claim when one exists.

## Basic Workflow

```python
import updatesupport as us

claim = us.claim(
    "Conversion remains above the reporting floor",
    public=["segment", "region"],
    hidden=["segment", "region", "acquisition_channel", "tenure_band"],
    target="conversion_rate",
    weight="users",
    ambiguity_limit=0.015,
    decision=us.threshold_decision(">=", 0.18),
    q_presets=["saturated"],
)

design = claim.design_categorical_rollup(
    rows,
    column="acquisition_channel",
    max_groups=4,
    bucket_budget=40,
    output_column="acquisition_channel_group",
)

print(design.to_markdown())
```

The functional spelling is equivalent:

```python
design = us.design_categorical_rollup(
    rows,
    claim,
    column="acquisition_channel",
    max_groups=4,
)
```

The report includes the selected mapping, ambiguity before and after the
rollup, the best grouping for each group count, the Pareto frontier over group
count/public cells/ambiguity, claim status, and exact-search metadata.

## Exact Saturated Evaluation

For a candidate global mapping `g(c)` from category level `c` to rollup group,
the proposed public representation is:

```text
(base public columns, g(c))
```

Within each resulting public fiber, saturated Q may place the fiber's mass on
any retained hidden cell in that fiber. Candidate ambiguity is therefore:

```text
sum_(o, group) p(o, group)
    * [max h(d) - min h(d)] over d in that public fiber
```

The implementation evaluates this closed form directly. It does not launch an
LP or CVXPY solve for every grouping.

The category mapping is global: a category belongs to the same rollup group in
every base public cell. This produces one reportable schema rather than a
different category definition for every segment or region.

## Selection Semantics

When the claim has an ambiguity limit or decision rule, selection proceeds as
follows:

1. discard candidates that exceed the public-cell budget;
2. find candidates satisfying the supplied ambiguity and decision requirements;
3. choose the certifying candidate with the fewest category groups, then the
   fewest realized public cells, then the smallest ambiguity;
4. if no candidate certifies the claim, return the lowest-ambiguity
   budget-feasible candidate and label it as non-certifying.

Without a claim threshold, the selected design minimizes ambiguity within the
declared group and public-cell budgets. The frontier remains available when an
analyst wants a different complexity/stability tradeoff.

## Applying The Design

There are three possible outputs:

- **One group**: retain the base public representation.
- **One group per category**: add the original categorical column.
- **Intermediate rollup**: add a generated categorical group column.

Use the report to transform rows or immediately rerun the claim:

```python
transformed = design.transform(rows)
audit = design.audit(rows)
```

`design.recommended_public` gives the selected public columns, and
`design.selected_claim` gives the corresponding saturated `ClaimSpec`.

Applying an intermediate design to a previously unseen category raises an
error. New categories are support drift and are not silently assigned to an
arbitrary group.

## Search Guard

The number of set partitions grows by Bell numbers. Exact search therefore has
`max_categories=9` by default. Nine retained levels produce 21,147 possible
partitions before applying `max_groups`.

Raise `max_categories` only when that combinatorial cost is acceptable. Larger
domains will need a later constrained, dynamic-programming, MIP, or heuristic
search mode.

## Structured Exports

```python
design.to_json()
design.to_tables()
design.to_dataframes()
```

Named tables include `summary`, `claim`, `selected_groups`,
`best_by_group_count`, `frontier`, and `limitations`.

## Scope Boundary

This feature searches one retained categorical column under saturated Q. It
does not yet learn ordered numeric cutpoints, category hierarchies, different
groupings by public fiber, or rollups jointly optimized across several columns.
