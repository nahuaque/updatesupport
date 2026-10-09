# Multi-metric portfolio reporting and repairs

The APIs combine multiple financial measurements, their coverage,
and one reporting schema. They diagnose the ambiguity a summary leaves, suggest
local drills, and compare conditional evidence plans. These APIs are available
in `updatesupport-finance>=0.1.6` with `updatesupport>=0.1.7`.

## Compile a common book without losing known evidence

```python
import updatesupport as us
import updatesupport_finance as uf

evidence = uf.compile_portfolio_metrics(
    universe,
    observations=observations,
    taxonomy=taxonomy,
    policies={
        "negative_ocf": uf.PortfolioMetricPolicy("OCF", "currency"),
        "negative_cash_gap": uf.PortfolioMetricPolicy("OCF_minus_cash_capex", "currency"),
    },
)
# Alternatively, combine existing validated CompiledPortfolio objects:
# evidence = uf.JointPortfolioEvidence({"negative_ocf": ocf, "negative_cash_gap": gap})
```

Every metric must use the same supplied holdings, issuer mapping, currency,
cutoff, and eligibility policy. Joins check per-position weights and descriptors,
shared fact definitions, descriptor provenance, and issuer measurements across
share classes. Original observations retain derivation inputs and fiscal dates;
latest annual evidence does not imply a common calendar-year observation.

The movable book is the positive-value intersection covered by every metric.
For each metric, `metric_scope(name)` retains its complete coverage ledger,
known contributions outside that intersection, missing domains, and eligible
scope evidence floor. For instance, missing capex for one company does not erase
its known OCF from another metric.

Common-book endpoints are lifted as
`(common_value * endpoint + fixed_known_contribution + missing_domain_contribution) / eligible_value`.
Nonjoint holdings weights and known values are fixed. Missing domains range
independently per metric; this does not assume that all joint endpoint outcomes
can occur together. Unbounded missing domains make eligible conclusions
inconclusive. Eligible/supplied denominators remain separate, and no independent
holdings completeness or NAV reconciliation is inferred.

## Inspect a frontier without fabricating a mandate

```python
report = uf.joint_portfolio_report(
    evidence,
    public=("sector",),
    hidden=("sector", "industry", "issuer_id", "listing_age"),
    candidate_refinements=("industry", "listing_age"),
    q_presets=("saturated", us.q_tv_budget(0.01)),
    max_evaluations=64,
)
print(report.to_markdown())
tables = report.to_tables()
```

Without criteria this returns `raw_frontier`, no selected certificate. Optional
`ambiguity_limits={metric: width}` and `decisions={metric: DecisionRule}` apply
in the explicit `scope="eligible"` (default) or `scope="common"`. Precision is in
the metric's units: a binary share width of `0.05` is five percentage points.
Decision criteria require every admissible value to pass the declared rule.
A Q mapping `{metric: (preset, ...)}` supports different scenarios per metric.
Intervals envelope every declared Q, and per-Q results remain inspectable.

The finite search enumerates global descriptor subsets up to
`max_added_columns`, plus individual supplied conditional proposals. The budget
counts representations, each solved for every metric/Q. It raises before
search if the full enumeration exceeds `max_evaluations`. Report-cell counts
refer to the common book, not every supplied holding. Pareto choices retain
ties; optimality is restricted to the declared candidate set.
Optional `disclosure_costs={column: effort}` adds caller-supplied disclosure
effort to frontier comparisons when every evaluated proposal can be priced.
Partial cost information remains visible without treating missing estimates as
zero. Evidence effort and disclosure effort remain separate dimensions.

For a chosen common-book threshold, use
`report.breaking_witness(metric, candidate_index=..., decision=..., q_index=...)`.
This invokes the existing inverse machinery with `respect_q=True`. TV mass
times common holdings value is a transferred-value equivalent, not a forecast
loss or trade recommendation. Eligible-scope unknown evidence is not silently
included in this witness.

## Discover and replay local drills

```python
suggestions = us.suggest_conditional_refinements(
    evidence.rows,
    public=chosen_public_columns,
    targets=tuple(evidence.metrics),
    candidate_columns=("industry", "listing_age"),
    weight="weight", max_candidates=64,
)
report = uf.joint_portfolio_report(
    evidence, public=("sector",), hidden=hidden,
    candidate_refinements=allowed_columns,
    conditional_refinements=suggestions.candidates,
)
```

`ConditionalRefinement(name, column, predicate)` is a data-only bucket split.
Its `as_dict()`/`from_dict()` and `transform(rows)` preserve tagged categories
and leave other buckets intact. A predicate includes every base public field;
additional predicate fields become part of that proposal's reporting schema.
This lets a local drill of an already refined report compete against global
schemas in one artifact. Each proposal is evaluated alone; arbitrary combinations
of local drills are outside this search. All generated columns are appended to
the declared hidden support; state-keyed Q matrices must use that full support.

Suggestions exclude identity and direct target columns by default, inspect
mixed-target buckets, and enumerate separable descriptors. `possible_count`
and `exact` disclose candidate truncation. Evaluation always solves the entire
book under Q; local improvements are not added together. Suggestions are
target-aware and in-sample, with no causal or out-of-sample classification claim.

## Plan evidence acquisition alongside reporting

```python
actions = [uf.EvidenceAcquisition(
    "company cash package", ("security_id",),
    residual_widths={"negative_ocf": 0, "negative_cash_gap": 0},
    availability="unknown",
    source_requirements=("annual OCF", "compatible cash capex"),
)]
plans = uf.plan_portfolio_repairs(
    report, actions,
    ambiguity_limits={"negative_ocf": .05, "negative_cash_gap": .05},
    max_evaluations=4096,
)
```

Each package declares attainable residual widths conditional on acquisition,
not the eventual metric value or a successful API call. Width zero means exact
measurement if research succeeds, without assuming its sign. Bounded nonbinary
domains can be narrowed to explicit positive residual widths. Actions must
affect missing nonjoint evidence. Unavailable packages are omitted from
combinations; unknown availability stays visible. Optional effort costs enter
the frontier only with complete estimates; absent estimates do not become zero.
The planner retains insufficient plans and ties and labels all feasible plans
as conditional. It performs no retrieval.

**If newly covered companies enter the movable book, rebuild and rerun.**
The prospective width guarantee holds only when resolved nonjoint values and
weights remain fixed. This API does not optimize outcome-dependent new support.

## Save and monitor without historical calibration

```python
snapshot = uf.capture_joint_portfolio_snapshot(report, capture_references=hashes)
restored = uf.JointPortfolioSnapshot.from_json(snapshot.to_json())
replayed = restored.replay()  # No provider calls.
restored.replay_matches()
restored.runtime_differences()
uf.compare_joint_portfolio_snapshots(snapshot, restored)

contract = uf.FrozenJointPortfolioContract.freeze(report, candidate_index=chosen_index)
future = contract.audit(new_evidence)
contract.audit(None)  # Inconclusive: no_snapshot.
```

Snapshots archive normalized facts, derivations, descriptor provenance, coverage,
domains, schemas, predicates, Q, scope, results, capture references, and versions.
They recompile supplied evidence and validate result scope/configuration.
Hashes detect corruption; they do not authenticate source completeness.
Snapshots do not include credentials, executable callbacks, or provider adapters.

The contract fixes schema and measurement policies; it does not redesign or
recalibrate. Q is analyst-declared and applied to each fresh batch. Reference TV
movement is diagnostic, not a historically calibrated coverage guarantee.
New issuer support, new unresolved securities, lost reference public buckets,
or changed measurement/scope definitions yield inconclusive results. A new
share class of an existing issuer retains its facts and separate value.
Hidden support may contract when reference public fibers survive. Changed
classifications are detected as support changes, not silently rewritten.

Generic single-claim contracts use
`us.FrozenReportContract.freeze(claim, rows)` with portable JSON round-trip and
the same support diagnostics, independent of `FrozenPublicReportPolicy`'s
historical calibration factory.

Run the [provider-free example](../packages/updatesupport-finance/examples/joint_portfolio_workflow.py)
to exercise diagnosis, local repairs, conditional research, replay, and freezing.
