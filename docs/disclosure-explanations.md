# Compare financial explanations and plan evidence

The unreleased finance API compares economic explanations across explicit
accounting mappings. It answers two questions:

1. Which explanations fit the supplied disclosures under each mapping?
2. Which hypothetical evidence outcomes would exclude a selected explanation
   across the retained mappings?

This helps an analyst decide what to investigate next. A disappearing customer
liability might represent a transfer, refund or credit. The same balance-sheet
and indirect cash-flow totals can support several explanations. Testing them
together shows which additional information would change that conclusion.

## Keep the shared model and disputed mappings separate

Start with a core `NamedLinearFeasibilityProblem`: source facts, variable domains
and relationships common to all alternatives. Keep incomplete bridges open with
signed residuals. A base tier must not contain a disputed mapping that a later
alternative is intended to replace. The comparison is additive; it never removes
or substitutes source facts or shared relationships.

Supply `DisclosureAlternative` objects in two catalogs:

- **Mappings** declare how captions or measurement scopes relate. For example,
  whether an indirect cash-flow line covers deferred revenue alone or also
  includes a separate customer-liability caption.
- **Explanations** restrict economic movements. For example, a refund-only
  explanation can exclude transfers and residual movements. Those exclusions
  are assumptions, not reported zeros.

The API labels each alternative's constraints as analyst policies and rejects
reported or derived fact constraints in alternatives. Put source measurements
in the base problem and pass their `DisclosureFact` evidence separately. Existing
measurement normalization, precision and evidence-link checks still apply.

```python
import updatesupport_finance as f

comparison = f.compare_disclosure_explanations(
    problem,
    base_tier="reported",
    mappings=mappings,
    explanations=explanations,
    evidence=facts,
)
cell = comparison.case("joint", "refund_only")
print(cell.status, cell.intervals, cell.witness, cell.conflict)
print(comparison.to_markdown())
```

Each cell includes financial target bounds and a checked feasible assignment, or
an irreducible conflict when infeasible. Numerical failures remain
`undetermined`. An auxiliary fixed-zero probe provides a feasibility witness
even if all financial targets are unbounded. It has no financial meaning.
Use `targets=()` for compatibility alone, or pass existing target names to
limit the bounds calculated. `include_conflicts=False` skips conflict diagnosis.

The summary reports `possible_in_all_feasible_mappings`, `mapping_dependent`,
`excluded_in_all_feasible_mappings` or `undetermined`. These describe compatibility
across source-compatible mappings. They do not report probability or identify
the company's actual explanation.

## Search a finite catalog of hypothetical evidence outcomes

`DisclosureEvidenceRequest` declares an outcome that might be learned, rather
than a source that has already supplied it. Its constraints remain hypothetical
policies. `mapping_names` represents a scope-review outcome selecting retained
mappings. Such selection does not label discarded mappings infeasible.
`requires` names companion outcomes needed for a valid bundle.

```python
plan = f.plan_disclosure_evidence(
    comparison,
    requests,
    eliminate=["refund_only"],
    max_bundles=64,
)
for bundle in plan.minimal_bundles:
    print(bundle.requests, bundle.retained_mappings)
```

In the [synthetic example](../packages/updatesupport-finance/examples/disclosure_explanations.py),
an approximately 150 million customer-caption exit, 220 million deferred-revenue
increase and 80 million cash-flow adjustment fit both transfer-only and
refund-only explanations under two mappings. A hypothetical cap of 20 million
on other net deferred-revenue additions excludes the refund-only explanation
under the `dr_only` mapping. It does not exclude it under the `joint` mapping.
The planner therefore finds two inclusion-minimal sufficient bundles:

- A gross-refund cap of 20 million by itself.
- An other-deferred-revenue cap of 20 million together with a review confirming
  the `dr_only` mapping.

Neither part of the second bundle suffices alone. This is a conditional research
plan: the actual future outcomes may differ from the supplied hypothetical caps.

Every sufficient bundle must leave at least one feasible catalog explanation
in every source-compatible retained mapping. Contradictory outcomes, empty
mapping selections, missing dependencies, uncovered mappings and unresolved
solver results cannot certify separation. The empty bundle is considered too,
so an already excluded explanation needs no additional evidence.

The search enumerates `2 ** len(requests)` bundles, with a default limit of 64.
Minimal means no sufficient strict subset exists in the supplied catalog. It
does not mean cheapest, likeliest or highest expected information gain. A sole
surviving catalog entry is conditional on the catalog and does not establish
an actual cause. Include other or mixed explanations where relevant.

## Evaluate different possible answers directly

`evaluate_disclosure_evidence` evaluates selected hypothetical outcomes without
an elimination goal. Use it when drafting an analyst brief: a low amount might
exclude one account, a high amount another, and an intermediate amount both.
Keep mixed or unclassified explanations open where appropriate.

```python
outcome = f.evaluate_disclosure_evidence(
    comparison,
    requests,
    selected=["refund_cap"],
    targets=None,  # Financial bounds as well as compatibility.
)
print(outcome.status)
print(outcome.comparison.summary())
```

`selected=None` selects the entire supplied catalog; `selected=()` adds no
evidence. Names must belong to the catalog. Dependencies are checked against the
selected set. The default `targets=()` solves compatibility only; `targets=None`
bounds all financial targets. `include_conflicts=True` adds conflict diagnosis.
The result supports JSON and Markdown exports with its solved comparison.

`compatible` means every source-compatible retained mapping has a surviving
catalog explanation. `uncovered_mapping` means a mapping fits the observations
but none of the supplied accounts does: revisit the catalog. Contradictory
answers remain `inconsistent_with_all_mappings`; missing dependencies, empty
mapping selections and unresolved solver results retain distinct statuses.

### Write a competing-accounts brief

Lead with the economic question, why the distinction matters and two accounts
that fit the same reporting. Define material amounts or shares explicitly:
exact zero exclusions can make trivial movements defeat a useful narrative.
Keep other movements open and check alternative screening definitions.

For each evidence question, declare:

- The measurement, period, caption/cohort and gross/net basis required.
- Current source coverage and a candidate source for the missing answer.
- Hypothetical answers in different directions and their tested consequences.
- Whether an answer excludes an account, requires another movement, or calls
  for a different framing.

Financial stakes and story labels are analyst-authored. Compatibility and
necessary amounts are solver-derived. A sole survivor does not establish an
actual cause. Missing or unmatched scope is not a numerical answer. Examples
of future outcomes need not cover every possible answer or predict its likelihood.

## Export and replay

Comparisons and plans support `to_json()`, `to_markdown()`, `to_tables()` and
`to_dataframes()`. JSON retains source evidence, conditions, full solver reports
and witnesses for both the basis comparison and evaluated bundles.
`bundle.comparison` gives access to an outcome's solved matrix. Planning uses
compatibility-only solves; re-run the comparison on that outcome's
`source_problem`, `base_tier`, mappings and explanations for financial bounds.

Use the existing snapshot API to save the complete comparison matrix with a
selected financial target and tier. Carry source normalization and scope records
into `context` together with `comparison.snapshot_context()`:

```python
snapshot = f.capture_disclosure_snapshot(
    comparison.report.problem,
    facts=facts,
    as_of=review_cutoff,
    target="refund",
    tier=cell.tier,
    context={**source_context, **comparison.snapshot_context()},
)
replayed = f.DisclosureSnapshot.from_json(snapshot.to_json()).replay()
```

Source retrieval, accounting interpretation and candidate evidence selection
remain explicit analyst inputs. This workflow does not infer customer identities,
cash-flow scope or undisclosed movements from caption names.
