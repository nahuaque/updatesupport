# Public Representation Frontier

`public_representation_frontier(...)` searches over candidate public
representations and keeps the Pareto frontier of reporting choices.

Use it when the review question is not only:

> Which one hidden variable would reduce ambiguity?

but:

> What is the smallest public representation that keeps hidden-composition
> ambiguity acceptably low across a set of stress tests?

The search runs over subsets of named candidate refinement columns. It does not
learn arbitrary partitions. Every candidate is a concrete public representation:

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
    search="beam",
    beam_width=12,
    max_added_columns=4,
    max_evaluations=500,
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

## Search Modes

Use `search="exhaustive"` when the candidate set is small. This is the default
and evaluates every allowed subset up to `max_added_columns`.

Use `search="greedy"` for a fast first pass. It starts from the base public
representation and repeatedly adds the best next refinement until it reaches the
ambiguity limit, runs out of improving refinements, or hits `max_added_columns`.

Use `search="beam"` when the candidate set is larger. Beam search keeps the
best `beam_width` partial representations at each depth and expands only those.
It is heuristic, but it preserves the same representation semantics: every
evaluated candidate is still a named public-column refinement set.

The returned report includes `frontier.search_trace` with:

- `search`: the selected search mode.
- `exact`: whether the reported frontier is exhaustive over the requested
  candidate space.
- `evaluated_candidates` and `candidate_space_size`.
- `stopping_reason`, such as `completed`, `ambiguity_limit reached`, or
  `max_evaluations reached`.
- pruning counts for beam and optional bucket-budget enforcement.

Useful constraints:

- `max_added_columns`: maximum number of hidden columns promoted into public.
- `max_evaluations`: hard cap on candidate evaluations.
- `must_include`: columns that must appear in every evaluated representation.
- `must_exclude`: columns to remove from the search space.
- `enforce_bucket_budget=True`: treat `bucket_budget` as a hard pruning rule.

By default, `bucket_budget` is a recommendation/reporting budget used by
`best_under_bucket_budget(...)`; set `enforce_bucket_budget=True` when it should
also prune the search.

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
