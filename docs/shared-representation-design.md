# Multi-Claim Shared Representation Design

`ClaimPortfolio` searches for one public segmentation that supports several
reported claims at the same time. This matches the way real reports are built:
one dashboard or review pack usually uses a shared set of public dimensions for
many metrics, decisions, and model outputs.

The central question is:

> What is the smallest common public representation that certifies every claim
> under that claim's own stress-test scenarios?

## Basic Workflow

Define ordinary `ClaimSpec` objects with the same base public columns, hidden
columns, and weight column. Targets, ambiguity limits, decisions, Q presets,
sparse-cell thresholds, and hidden-set scenarios may differ.

```python
import updatesupport as us

conversion = us.claim(
    "Conversion remains above the reporting floor",
    public=["segment", "region"],
    hidden=["segment", "region", "channel", "tenure_band", "device"],
    target="conversion_rate",
    weight="users",
    ambiguity_limit=0.015,
    candidate_refinements=["channel", "tenure_band", "device"],
)

uplift = us.claim(
    "Experiment uplift remains launch-positive",
    public=["segment", "region"],
    hidden=["segment", "region", "channel", "tenure_band", "device"],
    target="treatment_lift",
    weight="users",
    decision=us.threshold_decision(">=", 0.0),
    q_presets=[us.q_tv_budget(0.10), "saturated"],
    candidate_refinements=["channel", "tenure_band", "device"],
)

portfolio = us.claim_portfolio(
    conversion,
    uplift,
    name="Experiment KPI report",
    candidate_refinements=["channel", "tenure_band", "device"],
)

design = portfolio.design(
    rows,
    max_added_columns=2,
    bucket_budget=40,
)

print(design.to_markdown())
```

The functional spelling is equivalent:

```python
design = us.design_shared_representation(rows, portfolio)
```

## What Is Shared

Every portfolio claim must use the same:

- base public columns;
- retained hidden columns;
- weight column.

Each candidate is one concrete shared schema:

```text
base public + a subset of candidate refinements
```

That candidate is evaluated separately for every claim. A claim keeps its own:

- target functional;
- ambiguity limit and decision rule;
- Q presets, including a distinct primary Q when supplied;
- hidden-set sensitivity scenarios;
- minimum-cell-weight scenarios.

The design therefore does not collapse heterogeneous targets onto one numeric
scale. It asks whether each claim passes its own declared contract under the
same public representation.

## Exact Search And Selection

The first slice uses exact exhaustive column-subset search. It reuses
`public_representation_frontier(...)` for each claim, then joins exact candidate
rows by their shared added-column set.

A candidate certifies the portfolio when every claim:

- meets its ambiguity limit across all declared scenarios, when supplied; and
- preserves its observed decision across all declared scenarios, when supplied.

Among portfolio-certifying candidates, selection minimizes:

1. the maximum realized public-cell count across claims and scenarios;
2. the number of added columns;
3. total residual ambiguity as a tie-breaker.

If no candidate certifies every claim, the report returns an explicitly labelled
best-effort design. It first maximizes the number of certified claims, then
minimizes the worst normalized threshold violation. It never labels that result
as a shared certificate.

`max_evaluations=4096` guards the exact subset search by default. Set a smaller
value for tighter operational limits or `None` when a larger exact search is
intentional.

## Shared Constraints

Use `bucket_budget` to cap the public-cell count. When omitted, the design uses
the smallest non-null bucket budget declared by a portfolio claim.

Claim-level `must_include` requirements are combined across the portfolio.
Claim-level `must_exclude` requirements are also combined. A column required by
one claim and excluded by another is reported as a contract conflict before
search begins.

Candidate refinements must be present in every claim's hidden-set scenarios.
Otherwise the proposed schema could not be evaluated consistently across the
portfolio.

## Applying The Selected Schema

The selected public columns are available as:

```python
design.recommended_public
design.selected_claims
```

Rerun full claim audits under the selected shared schema with:

```python
audits = design.audit(rows)
```

This returns one ordinary `ClaimAudit` per portfolio claim, preserving the
existing claim report contract.

## Claim Portfolio Versus Claim Tree

`ClaimTree` organizes hierarchical claims and reports which nodes pass or fail.
It audits each node's declared public representation independently.

`ClaimPortfolio` solves a different problem: every claim must use one shared
public representation, and the search chooses that common schema. Use a tree
for hierarchy and a portfolio for shared report design.

## Structured Exports

```python
design.to_json()
design.to_tables()
design.to_dataframes()
```

Named tables include `summary`, `claims`, `candidates`, `frontier`,
`selected_claims`, `selected_scenarios`, `candidate_claims`, and `limitations`.

## Scope Boundary

This first slice searches subsets of existing retained columns. It does not yet
jointly optimize categorical rollups, numeric cutpoints, or different base
public schemas. Those transformations can be prepared upstream; the resulting
columns can then participate in the shared candidate set.

