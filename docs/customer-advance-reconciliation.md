# Customer-advance reconciliation

The development API combines reviewed liability, revenue, receivables and cash
flow disclosures to answer two questions:

- What net advance cash is compatible with these disclosures?
- How much net noncash movement would an alternative cash interpretation require?

`CustomerAdvanceBridge` compiles a provider-neutral mapping into the existing
named-linear solver. Source retrieval, unit normalization, accounting review and
the choice of a hypothetical cash ceiling remain explicit inputs. These APIs are
unreleased; use the finance package from this checkout.

## Declare the accounting mapping

```python
import updatesupport_finance as f

scope = f.DisclosureScope(
    "ExampleCo", "advance liabilities", "USD million", "2026-06-30", "H1",
    "all customers", period_start="2026-01-01", currency="USD",
)
bridge = f.CustomerAdvanceBridge(
    name="advances", scope=scope, opening="opening", closing="closing",
    movements=[
        f.AdvanceMovement("cash", "net_cash", "Net cash including old unpaid collections"),
        f.AdvanceMovement("recognized", "recognition", "Recognition from advance liabilities"),
        f.AdvanceMovement("unpaid_open", "unpaid_opening", "Opening unpaid advances"),
        f.AdvanceMovement("unpaid_close", "unpaid_closing", "Closing unpaid advances"),
        f.AdvanceMovement("other", "noncash", "All other signed noncash movements"),
    ],
    accounting_regime="Reviewed source-specific policy",
    recognition_basis="All recognition from advances, including current-period cohorts",
    complete=True,
    completeness_basis="Reviewed exhaustive movement classes; noncash amounts unknown",
)
```

The identity is:

```text
closing = opening + net cash + unpaid change + net noncash - recognition
```

Net cash includes collections of old unpaid advance bills and subtracts refunds.
It is distinct from gross receipts and from the indirect cash-flow statement's
liability adjustment. Recognition from an opening cohort is only one component
of total recognition; map other cohorts separately or retain a bounded unknown.
Do not substitute total company revenue without reviewing its relationship to
this liability and cohort.

| Movement kind | Primitive domain | Effect on the liability |
| --- | --- | --- |
| `net_cash` | Signed | Addition |
| `cash_receipt` | Nonnegative magnitude | Addition |
| `cash_refund` | Nonnegative magnitude | Reduction |
| `recognition` | Nonnegative magnitude | Reduction |
| `unpaid_opening` | Nonnegative stock | Reduction |
| `unpaid_closing` | Nonnegative stock | Addition |
| `unpaid_change` | Signed change | Addition |
| `noncash` | Signed | Addition; `effect=-1` reverses a source sign |

Multiple primitives may belong to a category. Net cash cannot be mixed with
gross receipts/refunds. Use both unpaid endpoints, including a constrained zero
when appropriate, or a signed change; mixing the representations is rejected.
A source-positive noncash reduction uses `effect=-1`; its amount is still signed
unless a separate source or policy constraint establishes a magnitude domain.

`complete=True` is the caller's assertion that the mapping covers every movement
class. Unresolved *amounts* remain unknown: an unrestricted `noncash` primitive
can keep the cash interval unbounded. Absent categories in a complete mapping
have zero contribution because of this declaration. It requires a written
`completeness_basis`; the package does not verify an issuer's accounting policy.

Use `complete=False` if movement classes may be omitted. The compiler adds a
signed `bridge.unclassified` residual that may include cash, recognition or
noncash movements. Zero totals for omitted categories refer only to supplied
primitives. The residual is never automatically attributed to noncash, and the
minimum-noncash explanation helper rejects an incomplete mapping.

## Add evidence and separate the cash-flow diagnostic

Supply source bounds with `disclosure_fact_constraint`, after normalizing to the
scope's numerical unit. This retains source precision and provenance. Each
primitive is created once; other cash measures can reuse its name. Arbitrary
amount caps belong in explicit evidence or `ConstraintPolicy` constraints.
For example, gross AR can constrain unpaid advance bills only after reviewing
entity, cohort, date, allowance treatment and whether those bills are included.
Net AR alone is not automatically a gross-billing cap.

```python
problem = bridge.problem(
    constraints=source_constraints,
    variables=[f.disclosure_variable("cfo", lower=None, unit=scope.unit)],
)
```

The default tier is `reported`. Custom `tiers` retain the bridge's structural
identities in every scenario. Additional variables must use the same numerical
unit; variable names cannot replace generated primitives. The caller must review
currency, dates, cohort and definitions of those additional inputs. Normalize
currencies separately; the package does not infer FX or unit conversions here.

An optional `cash_flow_adjustment="cf_adjustment"` creates a signed input and a
separate target, `bridge.cash_flow_gap`:

```text
gap = closing - opening - indirect cash-flow liability adjustment
```

The gap is diagnostic. The compiler never equates it to receipts or to a
particular noncash cause. Keep ASC 606 financing, acquisition, FX and
reclassification mappings separate from ASC 842 straight-line adjustments;
the `accounting_regime` and `recognition_basis` record the reviewed meanings.

## Solve the alternative interpretation

```python
analysis = f.minimum_advance_explanation(
    bridge, problem,
    cash_ceiling={"cfo": .8},
    ceiling_description="Hypothetical net cash at most 80% of reported CFO",
)
required = analysis.report.interval(
    target=bridge.noncash, scenario="advance_explanation",
)
print(required.lower, required.upper, required.status)
```

The ceiling can be an amount, a variable name, a coefficient mapping or a
`NamedLinearExpression` with a constant. All use the bridge's numerical unit.
A fractional CFO expression imposes a linear condition; it does not compute a
ratio or certify a positive denominator. Establish positive CFO separately if
interpreting the result as a share.

The lower endpoint is the minimum signed net noncash addition needed under the
condition. A negative minimum is retained. A missing endpoint with `unbounded`
status means the encoded evidence imposes no finite bound; `infeasible` means
the alternative contradicts active constraints and yields a conflict report.
Endpoint assignments are feasible partial accounting explanations, not complete
financial statements or claims about economic plausibility.

The condition is labeled `analyst_policy`, so support is explicitly conditional.
`release_constraints` can remove active policies or management expectations,
following `break_even_analysis`; it cannot remove reported facts or accounting
identities. The helper also checks that its bridge variables, targets and
identities remain intact in the selected tier.

An acquisition-only amount cap cannot bound all noncash effects. Interest
expense net of capitalized amounts cannot automatically cap customer financing
accretion. Qualitative words such as “primarily” do not generate numeric bounds.
The workflow reports necessary alternative explanations without assigning their
probabilities or estimating recurring CFO.

## Audit and replay

Use the ordinary claim, allocation witness, constraint attribution and conflict
APIs on the compiled problem. Save `bridge.snapshot_context()` in an offline
`DisclosureSnapshot`, merged with normalization records. The saved mapping
records accounting regime, recognition basis, cash basis and completeness;
the snapshot preserves source facts and all compiled relationships for replay.

The complete [synthetic offline example](../packages/updatesupport-finance/examples/customer_advance_reconciliation.py)
shows an unidentified cash interval, a minimum alternative explanation and a
separate exhaustive table that reconstructs cash. Regression controls also
reproduce disclosed Core Scientific cash amounts from public SEC filings and
check conditional noncash requirements using public Nebius figures. They test
accounting mapping and arithmetic; they do not establish novel discovery or superiority
over a spreadsheet solver.
