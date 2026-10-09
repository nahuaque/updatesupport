# Disclosure evidence and reproducible claim audits

`updatesupport-finance` can preserve financial facts, compile them to explicit
constraints, and show feasible allocations supporting or opposing a claim.
The APIs accept vendor-neutral records; retrieval and filing-version selection
remain the responsibility of an adapter or analyst.

## Analyst workflow

The APIs extend the low-level solver with five optional layers.
They use the same linear feasibility engine, precision checks, claim audits,
conflict reports and offline snapshots. Existing hand-built problems still work.

### Measurement contracts and numerical units

`MeasurementDefinition(name, kind, unit, currency, scope, sign)` declares the
meaning of an observation. Kinds are `stock`, `flow`, `schedule` and `lifetime`.
Wrap facts in `MeasurementObservation`, including an explicit completeness flag,
component fact IDs and source-to-cash multiplier when needed. Supply
`DisclosureRequirement` contracts with requested periods, maximum age and
expected components to `compile_disclosure_evidence`.

The requirement ledger records satisfied, missing, unavailable, incompatible,
ambiguous, stale and incomplete inputs. Missing values never become zero. A
derived total with an incomplete or unavailable dependency is quarantined.
`.require_valid()` rejects an incomplete contract; `.fact_for(name)` returns a
satisfied modeled fact. Coverage is only as strong as the supplied requirements.
No helper discovers undocumented provider omissions from a field name.

`unit_scales={"USD": ("USD million", 1e6)}` normalizes numerical units while
retaining each original `DisclosureFact` in `normalizations`. Powers of ten keep
the original decimals exactly expressible; currency is unchanged. An explicit
`cash_multiplier=-1` records a verified sign conversion. Neither conversion
loosens the solver's feasibility tolerance. Save `.snapshot_context()` in the
snapshot `context` to validate that the original and modeled facts still agree.

`CashMeasure(name, components, scope)` declares a signed cash measure over named
primitives or other cash measures. `cash_measure_constraints` expands the shared
primitives, rejecting cycles and unknown inputs. Alternative measures therefore
reuse the same advance or financing movement. This prevents separate copies of
the same modeled input from drifting; it cannot infer whether a declared cash
definition is economically appropriate. `cash_bridge_constraint` requires cash
scopes at both endpoints; differing scopes need an explicit signed adjustment
(closing scope minus opening scope). Declare the adjustment's domain yourself.

### Percentage margins and threshold claims

Use `AllocationShareMargin` beside amount margins:

```python
share_margin = usf.AllocationShareMargin(
    "customer_share", percentage_fact, denominator="company_revenue",
    columns=["selected_direct_buyer"], bounds=(.155, .165),
    policy="inclusive half-percentage-point allowance for displayed 16%",
)
```

The denominator names an amount margin, not an interchangeable revenue label.
The compiler checks entity, period and numerator containment. Share facts use
`ratio` or `percent` units without currency. Explicit bounds are ratios and
require a named policy; otherwise source decimals or explicit `exact=True` are
required. Qualitative words such as *approximately* do not select numeric bounds.

`table.share_target(..., denominator="compute_revenue", threshold=1/3)` produces
the excess `numerator - denominator/3`, together with a source-supported,
strictly positive denominator floor. `table.problem` retains the floor in every
tier, including custom tiers. Audit a `lower_at_least=0` claim on the excess.
This tests a share threshold; it does not optimize a general ratio.

The standalone `percentage_constraints` supports explicitly declared stock to
future-recognition relationships. Its `denominator_nonnegative=True` argument
is a caller assertion; the caller must establish that domain in the model.
`varying_fields` records intentional linked evidence context differences.

### Dated windows and selected timing

`time_window_allocation` partitions a `DisclosureWindow` into contiguous,
disjoint inclusive-date buckets. It retains the source cohort, its as-of date,
measurement contract and schedule evidence role. Each bucket is nonnegative;
without a timing policy it may contain any share of the total. Subtotal helpers
sum unique buckets for cumulative windows.

Call `.even_schedule_constraints(basis="months", policy=...)` to explicitly
choose even monthly conversion. Whole calendar months are required: a twelve
month September–August schedule split at May yields nine and three months, with
75% allocated before the fiscal year end **only in the selected policy tier**.
Daily schedules use inclusive day counts. Lifetime commitments and instant book
liabilities are rejected as flows; first supply a separately declared schedule.
Year-end balances do not certify intrayear liquidity.

### Policies, stresses and break-even targets

`ConstraintPolicy(name, EvidenceRole, constraints)` attaches a structured role
and group name to constraints. Roles distinguish reported facts, declared
accounting relationships, management expectations and analyst policies.
`disclosure_support_basis(problem, tier)` reports whether support uses only
reported evidence and accounting declarations or is conditional on other roles.
Unclassified constraints make the result conditional.

```python
case = usf.DisclosureStressCase(
    "prepayment_delay", "cash_plan",
    {"prepayment_assumption": revised_prepayment_constraint},
    "Delay a declared amount while retaining the other cash-plan constraints",
)
analysis = usf.run_disclosure_scenarios(problem, [case])
```

Cases replace or remove named active constraints and re-solve every shared
identity. Inconsistent combinations return infeasible intervals and conflict
reports. Changes to reported history require `historical_counterfactual=True`;
changes to accounting relationships require `model_counterfactual=True`.
Counterfactual observations are labeled analyst policies and retain original
fact IDs as lineage, rather than pretending to be reported observations.

`break_even_analysis` bounds an existing target under explicit conditions. For
required issuance, release the fixed issuance policy and require ending cash to
equal opening cash. A cash-floor inequality instead often gives a finite minimum
and an unbounded maximum. It only releases policies/expectations. The helper
does not invent a cash limit or assign probabilities to stresses.

Reuse `attribute_disclosure_constraints(..., groups={policy.name:
[c.name for c in policy.constraints]})` for leave-one-policy-out bound changes.
This measures dependence on supplied constraints, not causal effects or expected
information value. Minimal combined policy changes require a separately chosen
perturbation metric and are outside this API.

### Decision briefs and comparison guards

`AnalystFinding` combines a question, `DisclosureAuditPack`, `DisclosureScope`,
optional `AnalystBaseline`, decision impacts, missing evidence and unanswered
scope. Baselines must use the target's display unit. `AnalystDecisionBrief`
renders compact Markdown, JSON and tables with readable target/claim labels,
bounded results and tier support. It contains no issuer-specific interpretation.
Keep different cash, revenue, customer, geography and cohort meanings explicit.

Save `scope.as_dict()` under snapshot `context["scope"]`.
`compare_disclosure_snapshots` checks target expression, unit and scale alongside
`compare_disclosure_scopes`. Different window kinds or declared cohorts are
flagged; anonymous customer identities cannot be linked across periods by label.
Missing scope metadata remains *unassessed*, including older snapshots, which
continue to replay. A comparable result is descriptive, not a causal trend.

The complete provider-neutral example is
[`analyst_decision_workflow.py`](../packages/updatesupport-finance/examples/analyst_decision_workflow.py).

## Record the financial context

```python
import updatesupport_finance as usf

fact = usf.DisclosureFact(
    fact_id="example-revenue-v1",
    entity="example-cik",
    concept="revenue",
    value=123,
    scale=1_000_000,
    unit="USD",
    currency="USD",
    decimals=-6,
    period_start="2025-01-01",
    period_end="2025-12-31",
    available_at="2026-02-01T12:00:00Z",
    source_id="example-filing",
    source_url="https://example.com/filing",
)

constraint = usf.disclosure_fact_constraint("reported_revenue", "revenue", fact)
```

The amount is `value * scale` in base units. `decimals=-6` means nearest-million
rounding in those base units; this example compiles to `[122500000, 123500000]`.
Inclusive rounding bounds deliberately retain both endpoints. An unknown
precision (`decimals=None`) requires an explicit `exact=True` choice before
compilation. Variable units must match the fact's base unit; no automatic
currency conversion occurs.

`period_start=None` denotes an instant rather than a duration. Availability is
an explicit timestamp with a timezone, never inferred from the accounting
period. Derived facts require `reported=False`, a description in `derivation`,
and stable `derivation_inputs` IDs. The derivation text records the recipe;
this API does not execute or independently validate its arithmetic.

## Validate evidence links

```python
spec = usf.disclosure_triangulation_spec(
    variables=[usf.disclosure_variable("revenue", unit="USD")],
    constraints=[constraint],
    targets=[usf.disclosure_target("revenue", "revenue", unit="USD")],
    tiers=[usf.disclosure_tier("reported", [constraint.name])],
)

diagnostics = usf.validate_disclosure_evidence(
    [fact], problem=spec, as_of="2026-02-02T00:00:00Z"
)
```

Diagnostics identify duplicate IDs and contexts, simultaneously active filing
versions, evidence unavailable at the selected time, missing or cyclic lineage,
stale fact-derived bounds, and incompatible linked contexts. Each result has a
code, fact IDs, a message, and an optional constraint name.

For hand-built relationships, use
`link_disclosure_evidence(constraint, facts, varying_fields=...)`. By default,
linked facts must share entity, unit, currency, period, and dimensions.
Explicitly permit `period` for a cross-period bridge or `dimensions` for a
segment rollup. This declaration documents a relationship; it does not prove
that the accounting definitions are compatible. The validator checks only
declared evidence links, so an unlinked relationship still needs analyst review.

Old versions may remain as derivation inputs. Version conflicts are checked
within each active tier; mutually exclusive tiers may contain different
versions. The as-of cutoff still applies to every fact in the snapshot.

## Financial relationship helpers

- `rounded_amount_constraint(..., increment=...)` encodes a declared rounding
  convention in the same units as its expression.
- `reconciliation_constraint(total=..., components=..., residual=...)` declares
  that signed components and an optional residual exhaust the total.
- `stock_flow_constraint(opening=..., closing=..., movements=..., residual=...)`
  declares a stock-flow bridge with signed movement coefficients.
- Existing `containment_constraint(...)` expresses a declared component capacity.

Residual variables must use `lower=None` when they can be negative. The finance
variable helper otherwise defaults to nonnegative. These constructors do not
infer accounting identities or classify unexplained residuals as data errors.

## Show what supports or opposes a claim

```python
report = usf.triangulate_disclosure(spec)
claim = usf.disclosure_claim(
    target="revenue", tier="reported", lower_at_least=120_000_000
)
pack = usf.disclosure_audit_pack(report, claim=claim, evidence=[fact])
print(pack.to_markdown())
```

Audit packs include structured evidence and feasible-allocation tables by
default. `include_allocations=False` avoids the additional solves. Without a
claim, the table uses finite endpoint assignments. With a claim, supporting
allocations satisfy the complete claim, including both bounds of a two-sided
claim. Existing extrema are reused when suitable, with additional solves for
interior or unbounded cases. Opposing allocations cross a bound by at least a
recorded positive separation.
An interior supporting allocation can exist even when both extrema fail.

The `disclosure_allocation_checks` table includes original constraints and
variable-bound sides, their values, violations, and numerical tolerances.
`disclosure_allocation_attempts` records failed and infeasible witness solves;
an unavailable witness is not a claim verdict. Witnesses are numerical feasible
examples, not predictions about the undisclosed allocation.

## Numerical claims and conflicting disclosures

One finite endpoint can establish a one-sided claim: `[10, infinity)` supports
`x >= 5`. Infeasibility, solver failure, and failed residual checks produce
inconclusive claims, with the solve status preserved separately.

`feasibility_tolerance` on the problem controls HiGHS primal/dual feasibility
settings and post-solve residual checks in original expression units. It
defaults to `1e-7` and must be at least `1e-10`. Scale mixed-magnitude models
deliberately. Assignment checks include variable bounds as well as constraints.

Claims use a numerical buffer of
`absolute_tolerance + relative_tolerance * abs(threshold)`, with defaults
`1e-7` and `1e-9`. Bounds within that buffer do not resolve the claim. Setting
both to zero opts into raw comparisons, including equality. These settings
and solver diagnostics are numerical evidence, not exact-arithmetic certificates
or financial reporting tolerances. Rounded disclosures must still be modeled
as intervals separately.

```python
conflict = usf.find_disclosure_conflict(spec, tier="reported", max_checks=1000)
print(conflict.to_markdown())
```

The deletion filter includes variable-bound sides and preserves source
provenance. `irreducible` means no single member can be removed while retaining
infeasibility; it does not promise the globally smallest conflict. Other statuses
are `feasible`, `budget_exhausted`, and `inconclusive`. No constraints are silently
relaxed. The core equivalents are `find_named_linear_conflict(...)` and
`check_named_linear_assignment(...)`.

## Capture and replay a snapshot

```python
snapshot = usf.capture_disclosure_snapshot(
    spec, facts=[fact], as_of="2026-02-02T00:00:00Z",
    target="revenue", tier="reported", claim=claim,
    assumptions=["Nearest-million rounding is represented by inclusive bounds."],
)
saved_json = snapshot.to_json(indent=2)
restored = usf.DisclosureSnapshot.from_json(saved_json)
replayed = restored.replay()
print(restored.fingerprint)
print(restored.runtime_differences())
```

A versioned bundle includes the input facts, full problem, selected claim,
assumptions, results, Python/package versions, and source-code hashes for both
packages (including editable checkouts). Canonical JSON provides a
stable SHA-256 content fingerprint and detaches records from mutable caller
objects. The fingerprint identifies contents; it does not authenticate sources.
Loading validates the schema, evidence, and agreement between stored input
records and the report. Replay recomputes using the installed runtime; exact
reproduction of solver assignments or duals is not promised across versions
or multiple optimal solutions.

`compare_disclosure_snapshots(before, after)` separates fact changes, constraint
changes, other model changes, assumptions, claim selection, runtime changes,
intervals, and verdicts. Records match on stable fact IDs; an amendment assigned
a new ID appears as removal/addition. These differences do not establish a unique
causal explanation for the changed result.

The full invented-data workflow is executable without feed access:

```bash
uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/disclosure_evidence_workflow.py \
  --output /tmp/disclosure-evidence-demo
```

It demonstrates a supported allocation claim becoming inconclusive after a
revision, with saved original/revised snapshots and their comparison.
