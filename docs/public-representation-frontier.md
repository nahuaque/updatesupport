# Public Representation Frontier

`public_representation_frontier(...)` searches over candidate public
representations and keeps the Pareto frontier of reporting choices.

Use it when the review question is not only:

> Which one hidden variable would reduce ambiguity?

but:

> What is the smallest public representation that keeps hidden-composition
> ambiguity acceptably low across a set of stress tests?

The first implementation searches over subsets of named candidate refinement
columns. It does not learn arbitrary partitions. Every candidate is a concrete
public representation:

```text
base_public + zero or more candidate_refinements
```

That keeps the result explainable to an analyst or model reviewer.

## Example

```python
import updatesupport as us

frontier = us.public_representation_frontier(
    rows_or_frame,
    base_public=["product", "region"],
    hidden=[
        "product",
        "region",
        "credit_score_band",
        "ltv_band",
        "broker_channel",
        "vintage",
    ],
    target="expected_loss",
    weight="ead",
    candidate_refinements=[
        "credit_score_band",
        "ltv_band",
        "broker_channel",
        "vintage",
    ],
    q_presets=[
        "saturated",
        us.q_bounded_shift(0.5),
        "observed",
    ],
    ambiguity_limit=0.005,
    bucket_budget=40,
)

print(frontier.to_markdown())
```

## What The Frontier Means

A candidate is Pareto-frontier if no other evaluated representation has:

- no more public cells,
- no more added public columns,
- no larger ambiguity under every stress test,
- and at least one strict improvement.

The report also exposes two scalar conveniences:

- `frontier.minimal_stable`: the smallest representation whose worst-case
  ambiguity is below `ambiguity_limit`.
- `frontier.best_under_bucket_budget(...)`: the most stable representation
  with no more than the supplied number of public cells.

`max_ambiguity` is the conservative summary across Q presets. `mean_ambiguity`
is useful for ranking, but the Pareto test itself compares each stress-test
scenario.

## How To Use It

Start with the public categories that are already in the report. Add candidate
refinements that are plausible to publish or operationalize. Then run a small
stress grid, usually including:

- `saturated` as a conservative benchmark,
- `bounded_shift` for a practical observed-mix perturbation,
- `observed` as the zero-shift baseline.

For CVXPY-backed presets, keep the candidate set small on the first pass. The
search is combinatorial in the number of candidate refinement columns.

## Interpretation

This is a reporting-design tool. It does not decide causal adjustment sets,
train a model, or estimate statistical uncertainty. It asks which public
representation best trades off public-cell complexity against
hidden-composition stability for a supplied target.
