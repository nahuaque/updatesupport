# Disclosure evidence and reproducible claim audits

`updatesupport-finance` can preserve financial facts, compile them to explicit
constraints, and show feasible allocations supporting or opposing a claim.
The APIs accept vendor-neutral records; retrieval and filing-version selection
remain the responsibility of an adapter or analyst.

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
