# Public Representation Frontier

`public_representation_frontier(...)` searches over candidate public
representations and keeps the Pareto frontier of reporting choices.

Use `certify_public_representation(...)` instead when you want a pass/fail
representation-stability certificate built from the same frontier machinery.
See [Representation stability certificates](representation-stability-certificates.md).

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
    min_cell_weights=[1, 10, 25],
    hidden_sets=[
        [
            "product",
            "region",
            "credit_score_band",
            "ltv_band",
            "broker_channel",
            "vintage",
        ],
        [
            "product",
            "region",
            "credit_score_band",
            "ltv_band",
            "broker_channel",
        ],
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
- `frontier.best_scalarized`: the lowest weighted score when
  `scalarized_weights` is supplied.

`max_ambiguity` is the conservative summary across Q presets. `mean_ambiguity`
is useful for ranking, but the Pareto test itself compares each stress-test
scenario.

## Scalarized Selection

Use `scalarized_weights` when the review has an explicit utility tradeoff, such
as "one extra public bucket is acceptable only if it buys at least this much
ambiguity reduction." The scalarized score is a weighted sum of named
candidate-level components:

- `max_ambiguity`
- `mean_ambiguity`
- `public_cells`
- `hidden_cells`
- `added_columns`

Lower scores are better:

```python
frontier = us.public_representation_frontier(
    rows_or_frame,
    base_public=["product", "region"],
    hidden=[
        "product",
        "region",
        "credit_score_band",
        "ltv_band",
        "broker_channel",
    ],
    target="expected_loss",
    weight="ead",
    candidate_refinements=[
        "credit_score_band",
        "ltv_band",
        "broker_channel",
    ],
    q_presets=["saturated", us.q_bounded_shift(0.5)],
    scalarized_weights={
        "max_ambiguity": 1.0,
        "public_cells": 0.0001,
        "added_columns": 0.001,
    },
)

print(frontier.best_scalarized)
```

This does not change Pareto dominance. The report still shows the full frontier
and scenario evidence. The scalar score is an explicit decision aid for choosing
one representation from the evaluated candidates.

## Explaining The Selected Representation

Use `explain_minimal_stable()` when an ambiguity limit is supplied:

```python
explanation = frontier.explain_minimal_stable()
if explanation is not None:
    print(explanation.to_markdown())
```

Use `explain(...)` for any evaluated representation:

```python
print(frontier.explain(["credit_score_band", "ltv_band"]).to_markdown())
```

The explanation separates:

- baseline ambiguity versus selected ambiguity,
- ambiguity reduction by scenario,
- added public cells and added public columns,
- scenarios where the selected representation still fails the ambiguity limit,
- close dominated alternatives,
- requested refinements screened out of the search,
- whether the search result was exact or heuristic.

The full `frontier.to_markdown()` report includes a selected-representation
explanation. It uses the minimal stable representation when available, then the
best candidate within `bucket_budget`, then the first Pareto-frontier candidate.

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

Use `search="scalarized"` when you want a greedy search guided by
`scalarized_weights` instead of ambiguity alone. If no weights are supplied,
the search defaults to `{"max_ambiguity": 1.0}`. This is useful when public-cell
or added-column penalties should affect the search path, not only the final
report ranking.

Use `search="mip"` when the stress grid uses saturated Q presets and you want
SCIP to solve the public-column selection problem directly instead of using a
greedy or beam heuristic:

```python
frontier = us.public_representation_frontier(
    rows_or_frame,
    base_public=["product", "region"],
    hidden=["product", "region", "score_band", "ltv_band", "channel"],
    target="expected_loss",
    weight="ead",
    candidate_refinements=["score_band", "ltv_band", "channel"],
    q_presets=["saturated"],
    ambiguity_limit=0.005,
    search="mip",
    max_added_columns=2,
)
```

The MIP mode optimizes the saturated ambiguity objective over the declared
candidate columns. It supports `ambiguity_limit`, `max_added_columns`,
`must_include`, `must_exclude`, scalarized weights over `max_ambiguity`,
`mean_ambiguity`, `public_cells`, `hidden_cells`, and `added_columns`, and hard
bucket budgets when `enforce_bucket_budget=True`.

Limitations:

- MIP search currently supports saturated Q presets only.
- Representation-dependent `ProcedureTarget` objects are not supported because
  the target changes with the selected public representation.
- The returned candidate table contains evaluated evidence candidates, not an
  exhaustively enumerated Pareto frontier.
- If `bucket_budget` is supplied without `enforce_bucket_budget=True`, the
  budget is still reporting-only and the MIP guarantee is not a full
  budget-constrained certificate.

Use `search="mip_oracle"` when you want SCIP to act as a discrete master
problem and a support-function oracle to evaluate the actual convex transport
stress test:

```python
frontier = us.public_representation_frontier(
    rows_or_frame,
    base_public=["product", "region"],
    hidden=["product", "region", "score_band", "ltv_band", "channel"],
    target="expected_loss",
    weight="ead",
    candidate_refinements=["score_band", "ltv_band", "channel"],
    q_presets=[
        us.q_intersection(
            us.q_tv_budget(0.10),
            us.q_covariate_balance(0.25, hidden_moments),
        )
    ],
    ambiguity_limit=0.005,
    bucket_budget=40,
    search="mip_oracle",
    max_added_columns=3,
)
```

The master MIP proposes candidate public representations under the declared
column and bucket constraints. Each proposed candidate is then evaluated against
the declared Q grid. Compatible convex presets are routed through
`backend="support_function"` automatically. If a candidate fails the oracle
ambiguity limit, the master receives a no-good cut and proposes the next
candidate.

`search="mip_oracle"` currently requires `ambiguity_limit`. It supports
`saturated`, `observed`, `bounded_shift`, convex divergence/norm presets,
`covariate_balance`, `mahalanobis_budget`, `wasserstein`, and convex
`q_intersection(...)` composites. The public-cell `bucket_budget`, when
supplied, is a hard master constraint in this mode.

Use `search="mip_minimum"` when the review question is:

> What is the minimum public representation that satisfies this ambiguity
> limit under the declared convex Q stress tests?

```python
frontier = us.public_representation_frontier(
    rows_or_frame,
    base_public=["product", "region"],
    hidden=["product", "region", "score_band", "ltv_band", "channel"],
    target="expected_loss",
    weight="ead",
    candidate_refinements=["score_band", "ltv_band", "channel"],
    q_presets=[us.q_tv_budget(0.10), us.q_kl_budget(0.05)],
    ambiguity_limit=0.005,
    bucket_budget=40,
    search="mip_minimum",
    minimum_objective="public_cells",
    max_added_columns=3,
)
```

This mode uses the same SCIP master plus support-function oracle as
`search="mip_oracle"`, but gives it an exact-minimum contract. SCIP enumerates
candidate representations in increasing `minimum_objective` order and the
support-function oracle evaluates each candidate against the declared Q grid.
The first oracle-stable candidate is the exact minimum under the declared
objective, search bounds, and hard bucket constraints.

Supported minimum objectives are:

- `minimum_objective="public_cells"`: minimize the maximum public-cell count
  across the stress grid, tie by added-column count, then by saturated proxy
  ambiguity.
- `minimum_objective="added_columns"`: minimize the number of added public
  columns, tie by maximum public-cell count, then by saturated proxy ambiguity.

`search="mip_minimum"` requires `ambiguity_limit`, does not accept
`scalarized_weights`, and supports the same named Q presets as
`search="mip_oracle"`. Use `search="mip_oracle"` if you want a scalarized proxy
search instead of an exact minimum under one of the supported objectives.

The returned report includes `frontier.search_trace` with:

- `search`: the selected search mode.
- `exact`: whether the reported frontier is exhaustive over the requested
  candidate space, or whether MIP mode solved its supported selection objective
  exactly.
- `evaluated_candidates` and `candidate_space_size`.
- `scenario_count`: the number of Q / min-cell / hidden-set scenarios evaluated
  for each representation.
- `stopping_reason`, such as `completed`, `ambiguity_limit reached`, or
  `max_evaluations reached`.
- pruning counts for beam and optional bucket-budget enforcement.
- MIP-specific solver metadata, including `solver`, `solver_status`,
  `objective_value`, and `optimization_guarantee`.
- MIP-oracle counters, including `oracle_iterations` and `oracle_rejections`.
- Exact-minimum metadata, including `minimum_objective`, when
  `search="mip_minimum"` or non-scalarized `search="mip_oracle"` uses a declared
  objective order.

Useful constraints:

- `max_added_columns`: maximum number of hidden columns promoted into public.
- `max_evaluations`: hard cap on candidate evaluations.
- `must_include`: columns that must appear in every evaluated representation.
- `must_exclude`: columns to remove from the search space.
- `enforce_bucket_budget=True`: treat `bucket_budget` as a hard pruning rule.

By default, `bucket_budget` is a recommendation/reporting budget used by
`best_under_bucket_budget(...)`; set `enforce_bucket_budget=True` when it should
also prune the search.

## Sensitivity-Aware Grids

The frontier can score every representation across more than Q presets. Add
`min_cell_weights` to test sparse-cell thresholds, and add `hidden_sets` to test
alternate retained hidden-state definitions:

```python
frontier = us.public_representation_frontier(
    rows_or_frame,
    base_public=["AGE_BAND", "EDU_BAND"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND"],
    target="__target__",
    weight="PWGTP",
    candidate_refinements=["SEX", "OCC_MAJOR", "WKHP_BAND"],
    q_presets=["saturated", us.q_bounded_shift(0.5)],
    min_cell_weights=[1, 10, 25],
    hidden_sets=[
        ["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND"],
        ["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR"],
    ],
)
```

Every candidate must be evaluable in every hidden-set scenario. Candidate
refinements are therefore limited to columns that are present in all supplied
hidden sets. If a hidden column appears only in one scenario, it can still affect
that scenario's hidden state space, but it will not be promoted into the public
representation by the frontier search.

Candidate-level `public_cells` and `hidden_cells` are conservative maxima across
the scenario grid. The Markdown table shows ranges when those counts vary across
min-cell thresholds or hidden-set definitions.

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
