# updatesupport-finance

Financial model-risk extensions for
[`updatesupport`](https://pypi.org/project/updatesupport/).

`updatesupport-finance` audits what financial summaries and disclosures support.
It bounds hidden allocations, tests claims against supplied evidence, and shows
how conclusions depend on measurement definitions and assumptions. It also
audits whether a public risk segmentation supports a reported portfolio metric.

The workflows below are available in `updatesupport-finance>=0.1.6` with
`updatesupport>=0.1.7`. Source retrieval, accounting interpretation and policy selection
remain explicit inputs. No provider credentials or proprietary captures are
needed to use these APIs.

## Disclosure analysis for analysts

The disclosure workflow supports questions such as how much segment revenue
must come from a disclosed customer group, whether cash measures treat advances
consistently, and what funding is required under an explicitly chosen plan.

1. Declare `MeasurementDefinition` and `DisclosureRequirement` contracts; use
   `compile_disclosure_evidence` to quarantine missing, stale or incomplete inputs.
   Numerical normalization preserves original facts and source precision.
2. Compile amount and percentage disclosures with `allocation_table`,
   `AllocationMargin` and `AllocationShareMargin`. `share_target` tests a threshold
   with a positive, explicitly named denominator.
3. Use `CashMeasure` and `cash_bridge_constraint` for declared cash definitions;
   use `time_window_allocation` for dated flow partitions. Choose conversion
   schedules explicitly; stocks and lifetime commitments are not annual flows.
4. Label `ConstraintPolicy` groups as reported evidence, accounting relationships,
   management expectations or analyst policies. Named `DisclosureStressCase`
   replacements re-solve the complete model. `break_even_analysis` bounds a
   decision variable under supplied conditions and reports infeasible cases.
5. Save `capture_disclosure_snapshot(..., context=...)` with normalization maps
   and a `DisclosureScope`. Create an `AnalystDecisionBrief` from audit packs,
   arithmetic baselines, decision impacts and concrete missing evidence.
   Comparison guards flag different windows and anonymous customer cohorts.

See the [synthetic analyst example](examples/analyst_decision_workflow.py) and
[workflow documentation](../../docs/disclosure-evidence.md).

### Customer advances and alternative funding explanations

`AdvanceMovement` and `CustomerAdvanceBridge` compile reviewed customer-advance
roll-forwards, separating net cash, recognition, unpaid bills and signed noncash
movements. Incomplete mappings retain an unclassified residual. An optional
liability / indirect cash-flow gap remains a separate diagnostic.

`minimum_advance_explanation` calculates the least net noncash addition required
under an explicit hypothetical cash ceiling. It preserves unbounded and
infeasible results and labels the condition as an analyst policy. Accounting
regime, recognition basis, completeness and measurement scope remain explicit.
See the [workflow guide](../../docs/customer-advance-reconciliation.md) and
[offline example](examples/customer_advance_reconciliation.py).

### Compare explanations and choose the next evidence

`compare_disclosure_explanations` crosses named economic explanations with
explicit accounting mappings. Its matrix retains financial bounds, checked
witnesses, conflicts and conclusions that depend on the mapping.
`plan_disclosure_evidence` finds inclusion-minimal bundles of hypothetical
measurement or scope-review outcomes that exclude selected explanations across
the retained mappings. Contradictory outcomes and empty or uncovered mappings
cannot certify separation. JSON exports preserve the basis and outcome witnesses.
`evaluate_disclosure_evidence` shows what selected hypothetical answers leave
standing, without an elimination goal, including optional financial requirements.
This supports competing-accounts briefs with explicit answer implications.
See the [workflow guide](../../docs/disclosure-explanations.md) and
[synthetic offline example](examples/disclosure_explanations.py).
Compatibility does not identify an actual cause or assign probabilities to the
supplied explanation catalog.

## Multi-metric portfolio reports

`compile_portfolio_metrics` validates named measurements on one supplied
universe. `joint_portfolio_report` exposes a shared complexity frontier with
common-book and eligible-scope uncertainty, retaining known facts outside the
intersection. Criteria are optional. Core `ConditionalRefinement` objects and
`suggest_conditional_refinements` support portable local bucket drills.

`plan_portfolio_repairs` compares finite report/evidence combinations under
fixed nonjoint weights, preserving source availability and cost uncertainty.
`JointPortfolioSnapshot` replays offline; `FrozenJointPortfolioContract` monitors
a chosen schema/Q/measurement contract without inventing historical calibration.
See the [workflow guide](../../docs/joint-portfolio-report.md) and
[offline example](examples/joint_portfolio_workflow.py).

## API organization

Import the supported API from `updatesupport_finance`. The
[finance API map](../../docs/api-surface.md#finance-api) groups the APIs
by evidence, accounting models, explanation analysis, portfolio reporting and
offline replay. Imports from `updatesupport_finance.explanations` remain supported.

Report types with tables share the core JSON/DataFrame exporter and define their
own tables. `PortfolioHeadlineReport.to_tables()` includes headline truth and
summary support alongside the full coverage ledger, refinements and transfers;
`PortfolioCoverageReport.to_tables()` retains the supplied and eligible totals.

Internally, explanation comparison and evidence planning have separate engines.
Snapshot formats share serialization and provenance helpers, and finite planning
workflows share subset enumeration and frontier selection. Files beginning with
`_` are implementation details; snapshot schemas remain owned by each workflow.

## Segmentation and model review

The core question is:

> If a model report only shows risk by coarse public buckets such as
> `product x region x FICO band x LTV band`, could the reported expected-loss
> estimate materially change if the hidden mix inside those buckets shifted?

This is a segmentation adequacy check for reported risk metrics. It is designed
for model-review and portfolio-monitoring artifacts, not as a replacement for
model validation, calibration, backtesting, or statistical uncertainty analysis.

The strongest use case is retail-credit or insurance model review where the
institution has richer internal cells but reports risk at a coarser governance
segmentation. In that setting, "hidden" means internally retained but not shown
in the public or validation-pack segmentation.

Install directly:

```bash
pip install updatesupport-finance
uv add updatesupport-finance
```

Or through the core package extra:

```bash
pip install "updatesupport[finance]"
uv add "updatesupport[finance]"
```

The package provides finance-oriented row metrics, Q preset aliases, portfolio
compilation, and a model-risk report profile while keeping financial vocabulary
out of the core `updatesupport` package.

It also provides a disclosure-triangulation front end over the core
`updatesupport` named-linear feasibility solver. That surface is generic:
unknown variables, linear constraints, target expressions, and tiered
assumption sets are user supplied. The finance package only adds disclosure
vocabulary, provenance, declared measurement contracts and modeling helpers.

Conic concentration presets require the core CVXPY extra when solved:

```bash
pip install "updatesupport[cvxpy]" updatesupport-finance
uv add "updatesupport[cvxpy]" updatesupport-finance
```

## Portfolio Evidence and Headline Audits

These APIs are available in `updatesupport-finance>=0.1.6` with
`updatesupport>=0.1.7`.

The provider-neutral workflow is:

1. Declare a `PortfolioUniverse`: original positions, currency, holdings cutoff,
   and eligible security types. Position values must be nonnegative; gross/net
   short-portfolio conventions and currency conversion need an explicit adapter.
2. Supply `FundamentalObservation` objects wrapping `DisclosureFact`, plus
   `TaxonomyAssignment` labels and a `PortfolioMetricPolicy`.
3. Call `compile_portfolio_evidence`. It selects the latest compatible fiscal
   period and latest version available by the cutoff, quarantines ambiguous
   versions, and records every position as covered, missing, unmapped, or excluded.
4. Call `portfolio_headline_report` with a declared decision and scope. Inspect
   observed truth, summary support, coverage, refinements, and breaking transfers.
5. Save a `PortfolioSnapshot` for offline recompilation/replay and comparison.

See the complete [synthetic example](examples/portfolio_headline.py). It uses a
portfolio with 80 units of covered equity, 20 of missing equity, and 10 of cash:

```python
compiled = usf.compile_portfolio_evidence(
    universe, observations=observations, taxonomy=taxonomy,
    policy=usf.PortfolioMetricPolicy(
        concept="OperatingCashFlow", unit="currency", period_kind="FY",
        transform="below", threshold=0,
    ),
)
hidden = ["sector", "industry", "issuer_id", "target_status"]
report = usf.portfolio_headline_report(
    compiled, headline="Less than 30% negative annual operating cash flow",
    decision=us.threshold_decision("<", .30),
    public=["sector"], hidden=hidden, scope="eligible",
    candidate_refinements=["industry", "target_status"],
    direct_target_refinements=["target_status"],
)
print(report.to_markdown())
```

The example's covered-book share is 25%; the **eligible observed share is bounded
by 20%–40%**. Cash is explicitly excluded from that denominator. Missing equity
is retained as unknown rather than dropped or assigned zero. `scope="covered"`
asks only about the observed book; `scope="eligible"` includes missing eligible
positions. Core provided-row retention and upstream universe coverage remain
separate fields.

`actual_headline` and `summary_support` each return `supported`, `contradicted`,
or `inconclusive`. They describe whether the declared pass condition holds,
rather than core's `pass` status, which means a decision is invariant even when
the invariant decision is failure. Eligible summary bounds hold unknown weights
fixed and let their missing metrics vary over the policy's `target_bounds`.
Binary `below` metrics use [0, 1]; continuous `value` metrics require an explicit
currency and optional domain bounds. Without domain bounds, unknown eligible
values make the headline inconclusive. An empty covered book is supported as
an evidence/coverage artifact without a core solve.

The `below` transform tests the **reported base-unit value**, not an inferred
unrounded economic amount. Fact units, dimensions, period kind, and optional
currency must match the policy. `max_age_days` can reject stale periods.
Fact IDs and source metadata, derivation inputs, original fiscal ends, and
normalized fiscal labels remain in evidence. An adapter must explicitly provide
issuer mappings and period/source semantics; the compiler does not guess them.

Taxonomy labels carry a scheme, optional version, availability, and effective
timestamp. Undated current labels produce `taxonomy_as_of_unknown` diagnostics.
Set `strict_taxonomy_as_of=True` to require all three historical fields for
required labels. Future classifications are unavailable. Availability fields do
not independently establish the integrity of a provider's historical archive.

Refinements are marked `independent` or `direct_target`. `target_status` is a
direct disclosure of the tested classification. Other direct-target labels
must be declared explicitly. Each refinement is evaluated in the headline's
scope: disclosing covered status cannot remove missing whole-book uncertainty.

Breaking witnesses refer to the **covered book**, preserve its public-bucket
totals, and show issuer/state transfers in covered percentage points, eligible
percentage points, and currency amounts. They do not impute missing values or
certify an eligible-scope breaking distance. The same forward Q is respected by
default (`respect_q=True`); set it false only for an explicitly unrestricted
inverse comparison.

### Mandate Constraints

```python
q = usf.q_portfolio_mandate(
    compiled, hidden=hidden,
    mandate=usf.PortfolioMandate(
        issuer_caps={"issuer-B": .28},
        locked_issuers=["issuer-A"],
        preserve_columns=["industry"],
    ),
    base_q=us.q_tv_budget(.05),
)
```

Caps are shares of the **normalized covered book**. Locked issuers fix their
total weights; preserved columns fix every category total and permit transfers
within those groups. Restrictions must be represented in `hidden`, and the
observed book must satisfy its caps. Unknown issuers are errors. These policies
compile to the core `q_moment_bounds` preset and can intersect other convex Q
presets. Forward bounds and inverse witnesses use the same constraints; no
second mandate solver is introduced. Non-saturated constrained witnesses need
CVXPY. Trade execution and liquidity constraints are outside this model.

### Offline Portfolio Snapshots

```python
snapshot = usf.capture_portfolio_snapshot(
    report, capture_references={"holdings-response": raw_sha256},
)
saved = snapshot.to_json(indent=2)
restored = usf.PortfolioSnapshot.from_json(saved)
replayed = restored.replay()
changes = usf.compare_portfolio_snapshots(snapshot, restored)
runtime_changes = restored.runtime_differences()
```

Bundles embed normalized input evidence, the scope policy and ledger, compiled
rows, claim/Q configuration (including tuple-keyed moments), result digest,
optional raw capture hashes, and runtime/source versions. Loading checks the
result digest, input/result consistency, and deterministic recompilation.
Replay makes no provider calls. Built-in portable Q values are required;
callable environments are not serialized. SHA256 checks detect corruption and
identify supplied captures; they do not authenticate a vendor archive. Compare
periods only after reviewing scope, policy, evidence, and taxonomy changes.

## Declarative Allocation Tables

`allocation_table` compiles declared leaf/subtotal axes and disclosed margins
into the existing named-linear feasibility engine:

```python
table = usf.allocation_table(
    rows=[usf.AllocationMember("Phone"), usf.AllocationMember("Other")],
    columns=[usf.AllocationMember("US"), usf.AllocationMember("Outside US")],
    margins=[
        usf.AllocationMargin("company", total_fact),
        usf.AllocationMargin("phone", phone_fact, rows=["Phone"]),
        usf.AllocationMargin("us", us_fact, columns=["US"]),
    ],
    measure="Revenue", unit="USD", as_of="2026-10-06T00:00:00Z",
    exhaustive=True,
)
target = table.target("phone_outside_us", rows=["Phone"], columns=["Outside US"])
problem = table.problem([target])
result = usf.triangulate_disclosure(problem)
snapshot = usf.capture_disclosure_snapshot(
    problem, facts=table.facts, as_of=table.as_of,
    target=target.name, tier="reported", assumptions=table.assumptions,
)
```

Leaves form caller-declared exhaustive, nonnegative partitions. Add explicit
`Other` leaves when needed. A subtotal declares its children and creates no
extra cells; overlapping selections and cyclic hierarchies raise errors.
`structural_zeros=[("Phone", "US")]` is an explicit exclusion assumption.
Literal qualifiers are preserved in the compiled problem; jurisdiction meaning
is never inferred from a label. Margins must share entity, period, unit,
currency, and measure. A margin whose source concept differs from the declared
measure requires `measure="Revenue"` on that `AllocationMargin` to assert
equivalence while retaining the original fact concept.

Fact precision drives rounding intervals. Missing precision requires an
explicit `exact=True`; it is never silently treated as exact. Existing
triangulation, conflict detection, evidence validation, attribution, attainable
allocations, and disclosure snapshots consume the compiled problem directly.
For signed financial reconciliations use `reconciliation_constraint` with
explicit signed residual variables instead of a nonnegative allocation table.

## Why This Is Useful

Financial analysts already monitor model performance, population drift,
calibration, overrides, and scenario sensitivity. Those checks usually ask
whether the model or portfolio changed.

`updatesupport-finance` asks a different question:

> Is the reporting segmentation itself adequate for the metric being reported?

For example, a validation pack may report expected loss by:

- `product`
- `region`
- `fico_band`
- `ltv_band`

But inside those public buckets, hidden composition may vary by:

- broker channel
- employment type
- vintage
- hardship history
- documentation type
- local housing market
- borrower cashflow pattern

If those hidden subgroups have different expected-loss rates, the public
segmentation may not fully support the reported aggregate. The report quantifies
that hidden-composition ambiguity and identifies which hidden variables would
most improve the public segmentation.

That is narrower than a full model-risk-management workflow, but it maps to a
real review artifact: a validation pack can state whether the reported
segmentation pins down the expected-loss, default-rate, LGD, delinquency, or
approval-benefit metric under a declared composition stress test.

## Recommended Positioning

Lead with the saturated fixed-public-law audit:

```python
q = "saturated"
```

Saturated stress keeps the reported public segment masses fixed and allows any
retained hidden cell inside each public segment to receive that segment's mass.
It is conservative, easy to explain, and does not require defending a
hand-chosen radius. It answers the first control question:

> Does this public segmentation pin down the metric at all on the retained
> support?

Radius-based presets are still useful, but they should be treated as secondary
sensitivity scenarios:

```python
q = usf.q_portfolio_mix_shift(radius=0.25)
q = usf.q_exposure_weighted_tv(radius=0.10)
q = usf.q_factor_exposure_shift(...)
q = usf.q_regional_concentration_shift(...)
```

Those scenarios require a governance rationale: historical drift, challenger
monitoring, an approved sensitivity grid, or another documented benchmark. A
validator can reasonably ask why a radius is acceptable, so the report should
state that rationale in reviewer notes or model-review documentation.

## Model-Risk Boundaries

This plugin is evidence for one control: **reported-segmentation adequacy**. It
does not validate the underlying PD, LGD, EAD, prepayment, delinquency, capital,
or approval-benefit model. It also does not replace:

- calibration and backtesting;
- discrimination or ranking performance;
- population stability / drift monitoring;
- override, policy, or use-test review;
- documentation of assumptions and limitations;
- independent validation of model design, implementation, and governance.

The report is meant to sit beside those controls. It says whether the
segmentation used to communicate or govern a supplied metric is stable under
declared hidden-composition stress.

## What The Report Separates

The package is intentionally narrow. It separates:

- reported risk estimate: the supplied metric, such as expected loss or default
  rate
- statistical uncertainty: confidence intervals or model uncertainty supplied by
  other workflows, plus optional hidden-cell metric standard errors
- hidden-composition ambiguity: how far the reported metric can move when hidden
  mix shifts inside fixed public buckets
- concentration-stress ambiguity: the same ambiguity translated into
  factor-exposure or regional-concentration stress language when those Q presets
  are used
- refinement recommendations: hidden fields that would make the public
  representation more stable
- dual diagnostics and data diagnostics: solver-sensitivity signals and
  pre-solve data warnings that reviewers can attach to validation evidence
- limitations and reviewer notes: explicit boundaries around what the report
  does and does not validate

This is not a confidence interval and not a full model-risk-management system.
It is a reviewable control for one practical question: whether the reporting
representation is stable enough for the risk metric.

## Disclosure Triangulation

Some finance questions are not hidden-composition audits. They are feasibility
questions over overlapping disclosures:

> Given several reported totals, component containments, rounded growth rates,
> anchors, and analyst assumptions, what interval remains possible for an
> undisclosed scalar quantity?

Use `triangulate_disclosure(...)` for that shape. It delegates to the core
`updatesupport` named-linear feasibility API, so there is no separate finance
solver and no issuer-specific logic.

```python
import updatesupport_finance as usf

growth_constraints = usf.rounded_growth_constraints(
    "component_growth",
    current="component_current",
    previous="component_previous",
    growth_percent=55.0,
    rounding=5.0,
    provenance="Example rounded growth disclosure",
    verified=True,
)

spec = usf.disclosure_triangulation_spec(
    variables=[
        usf.disclosure_variable("component_previous"),
        usf.disclosure_variable("component_current"),
        usf.disclosure_variable("total_previous"),
        usf.disclosure_variable("total_current"),
    ],
    constraints=[
        usf.exact_disclosure_constraint(
            "reported_total_previous",
            "total_previous",
            100.0,
        ),
        usf.exact_disclosure_constraint(
            "reported_total_current",
            "total_current",
            200.0,
        ),
        usf.containment_constraint(
            "previous_containment",
            child="component_previous",
            parent="total_previous",
        ),
        usf.containment_constraint(
            "current_containment",
            child="component_current",
            parent="total_current",
        ),
        *growth_constraints,
        usf.interval_disclosure_constraint(
            "current_anchor",
            "component_current",
            lower=90.0,
            category="assumption",
        ),
    ],
    targets=[
        usf.disclosure_target(
            "previous_component",
            "component_previous",
            label="Previous-period component",
        )
    ],
    tiers=[
        usf.disclosure_tier(
            "T0 containment",
            [
                "reported_total_previous",
                "reported_total_current",
                "previous_containment",
                "current_containment",
            ],
        ),
        usf.disclosure_tier(
            "T1 + growth + anchor",
            [
                "reported_total_previous",
                "reported_total_current",
                "previous_containment",
                "current_containment",
                "component_growth_lower",
                "component_growth_upper",
                "current_anchor",
            ],
        ),
    ],
)

report = usf.triangulate_disclosure(spec)
print(report.to_markdown())
```

The output is a tiered feasibility interval report with active-constraint
tables, endpoint assignments, binding constraints, provenance metadata, JSON
exports, and DataFrame exports. Rounded-growth helpers assume the previous
period variable is nonnegative and encode rounded percentages as inclusive
linear relaxations.

Use `attribute_disclosure_constraints(...)` to rank which active disclosures
actually narrow a target interval. It removes one constraint or constraint group
at a time, re-solves the interval, and reports the resulting width increase.
The underlying named-linear report also includes side-specific binding
constraints and HiGHS marginal diagnostics for each solved endpoint.

Use `disclosure_claim(...)` when the review question is an assertion rather
than an interval:

```python
claim = usf.disclosure_claim(
    target="component_previous",
    tier="T2 + anchor disclosure",
    lower_at_least=50.0,
)

audit = claim.audit(report)
print(audit.to_markdown())
```

The audit returns `pass`, `fail`, or `inconclusive`, plus the feasible interval,
margin to failure when certified, relevant attribution rows, endpoint dual
diagnostics, and structured exports.

For an analyst-facing artifact, use `disclosure_audit_pack(...)` to bundle the
source disclosures, active constraints, headline interval, claim verdict,
constraint attribution, endpoint diagnostics, assumptions, reviewer notes, and
limitations:

```python
pack = usf.disclosure_audit_pack(
    report,
    claim=claim,
    sources=[
        {
            "label": "Reported total",
            "value": "200.0",
            "url": "https://example.com/filing",
            "description": "Source disclosure used as an equality constraint.",
        }
    ],
    assumptions=[
        "The undisclosed component is nonnegative.",
        "The component cannot exceed the disclosed containing total.",
    ],
)

print(pack.to_markdown())
```

The pack is the preferred review shape when the output needs to be attached to
an analyst note, disclosure QA memo, model-review pack, or evidence archive.

A complete generic example is available in
`examples/disclosure_triangulation.py`:

```bash
uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/disclosure_triangulation.py
```

The Exxon Mobil examples demonstrate the same API on public SEC disclosures:

```bash
uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/exxon_revenue_recognition_triangulation.py

uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/exxon_capex_capacity_triangulation.py

uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/exxon_debt_bridge_triangulation.py
```

The capex example also runs a public-representation frontier over candidate
disclosure refinements, showing that segment disclosure stabilizes the
Upstream-share claim while geography alone does not.

## Analyst Workflow

For feed-backed disclosure analysis, the package also provides:

- `DisclosureFact`, precision-aware fact constraints, and explicit evidence links;
- compatibility, filing-version, availability, and derivation-lineage diagnostics;
- claim-aware supporting/opposing allocation tables with residual checks;
- rounded-amount, reconciliation, and signed stock-flow helpers;
- source-linked irreducible conflict diagnosis;
- versioned offline snapshots, replay, and structured before/after comparisons.

See the [disclosure evidence guide](https://github.com/nahuaque/updatesupport/blob/main/docs/disclosure-evidence.md)
and `examples/disclosure_evidence_workflow.py` for a complete offline example.
Vendor retrieval and historical filing-version selection belong in the adapter.

### Portfolio review

1. Choose public buckets from the model report.
2. Choose hidden refinements that are available internally but not shown in the
   public segmentation.
3. Choose the target risk metric.
4. Start with saturated fixed-public-law stress.
5. Add radius-based mix, TV, factor, or concentration scenarios only when the
   radius has a documented review rationale.
6. Set a review threshold for hidden-composition ambiguity.
7. Attach the generated Markdown report to a model-review or monitoring pack.

The review status is deliberately simple:

- `pass`: ambiguity and public adequacy checks are within the chosen thresholds
- `attention required`: the public segmentation may need refinement or explicit
  acceptance of the ambiguity band

## Example

```python
import updatesupport_finance as usf

report = usf.model_risk_report(
    portfolio,
    public=["product", "region", "fico_band", "ltv_band"],
    hidden=[
        "product",
        "region",
        "fico_band",
        "ltv_band",
        "broker_channel",
        "employment_type",
        "vintage",
    ],
    metric=usf.expected_loss(pd="pd", lgd="lgd"),
    exposure="ead",
    metric_standard_error=usf.expected_loss_standard_error(
        pd="pd",
        lgd="lgd",
        pd_standard_error="pd_se",
        lgd_standard_error="lgd_se",
    ),
    q="saturated",
    model_id="EL_RETAIL_2026Q2",
    portfolio_name="Retail credit portfolio",
    as_of_date="2026-06-30",
    intended_use="Expected-loss segmentation model review",
    ambiguity_limit=0.0025,
    public_adequacy_required=False,
    statistical_interval=(0.018, 0.024),
    statistical_confidence_level=0.95,
    statistical_method="validation bootstrap",
    composition_uncertainty_draws=500,
    composition_uncertainty_seed=123,
    composition_uncertainty_confidence_level=0.90,
    reviewer_notes=[
        "Review saturated segmentation adequacy with the portfolio monitoring owner.",
    ],
)

print(report.to_markdown())
```

This keeps four uncertainty notions separate:

- observed estimate: the exposure-weighted portfolio metric from the retained
  data
- supplied statistical/model uncertainty: an external interval or standard
  error from validation, bootstrap, survey, or model-estimation workflows
- hidden-composition ambiguity: the fixed-public-law transport interval under
  the selected Q stress test
- hidden-cell estimation uncertainty: optional hidden-cell metric standard
  errors, such as delta-method PD/LGD uncertainty from
  `expected_loss_standard_error(...)`

`composition_uncertainty_draws=...` adds a model-assisted posterior/bootstrap
summary over hidden composition. It uses the core
`hidden_composition_uncertainty(...)` layer, preserving public bucket masses by
default and resampling hidden composition inside each public fiber.

You can also run that layer directly:

```python
uncertainty = usf.model_assisted_portfolio_uncertainty(
    portfolio,
    public=["product", "region", "fico_band", "ltv_band"],
    hidden=[
        "product",
        "region",
        "fico_band",
        "ltv_band",
        "broker_channel",
        "employment_type",
        "vintage",
    ],
    metric=usf.expected_loss(pd="pd", lgd="lgd"),
    exposure="ead",
    draws=500,
    seed=123,
    q="saturated",
    ambiguity_limit=0.0025,
)
```

Structured exports are available for downstream model-risk systems:

```python
json_payload = report.to_json()
tables = report.to_tables()
frames = report.to_dataframes()  # Requires pandas.
```

The finance wrapper exposes finance-named tables that are intended to feed
validation packs, governance dashboards, model inventory systems, and evidence
archives:

- `finance_model_risk`: one-row review summary with metadata, status, reported
  estimate, ambiguity, adequacy flag, and Q preset
- `finance_review_reasons`: threshold breaches or adequacy failures
- `finance_concentration_stress`: concentration-stress interpretation of the
  active Q preset
- `finance_statistical_uncertainty`: supplied statistical/model uncertainty,
  when provided
- `finance_estimator_uncertainty`: hidden-cell standard-error adjustment, when
  provided
- `finance_model_assisted_summary`,
  `finance_model_assisted_metric_summaries`,
  `finance_model_assisted_draws`, and
  `finance_model_assisted_joint_cells`: posterior/bootstrap hidden-composition
  uncertainty outputs, when requested
- `finance_refinement_recommendations`: candidate public refinements ranked by
  ambiguity reduction
- `finance_dual_diagnostics`: largest CVXPY dual multipliers, when available
- `finance_data_diagnostics`: pre-solve data diagnostics
- `finance_limitations` and `finance_reviewer_notes`: review boundaries and
  analyst notes

Core `updatesupport` tables are also included with a `core_` prefix, such as
`core_summary`, `core_worst_fibers`, and `core_refinements`, so finance users
can keep both the domain summary and the underlying audit evidence.

The report answers:

- What is the reported portfolio risk estimate?
- What statistical or model uncertainty was supplied separately?
- What range is still possible under hidden mix shifts?
- How should the ambiguity be interpreted under concentration-stress presets?
- Does the ambiguity exceed the review threshold?
- Which public buckets drive the instability?
- Which hidden fields are most valuable as public refinements?
- Which solver duals and data diagnostics should reviewers inspect?
- Which small public segmentation sits on the stability frontier, and why did
  it beat nearby alternatives?

A synthetic portfolio example is available in `examples/model_risk_portfolio.py`
in the source repository:

```bash
uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/model_risk_portfolio.py
```

The example prints both the finance model-risk report and a core
`public_representation_frontier(...)` report for the same expected-loss metric.
The frontier section compares baseline versus selected ambiguity, close
dominated alternatives, and any screened-out refinement fields.

## Colab Demo Notebooks

Interactive Colab demos are available under `examples/notebooks`:

- [Portfolio model-risk walkthrough](https://colab.research.google.com/github/nahuaque/updatesupport/blob/main/packages/updatesupport-finance/examples/notebooks/model_risk_portfolio_colab.ipynb):
  expected-loss segmentation audit, saturated and radius-based stress
  scenarios, hidden-cell risk plots, refinement recommendations, and
  public-representation frontier search.
- [Model-assisted portfolio uncertainty](https://colab.research.google.com/github/nahuaque/updatesupport/blob/main/packages/updatesupport-finance/examples/notebooks/model_assisted_portfolio_uncertainty_colab.ipynb):
  PD/LGD estimator uncertainty, posterior/bootstrap hidden-composition draws,
  and decision-threshold invariance.

Both notebooks use seaborn plus `ipywidgets` controls so analysts can inspect
the flagship saturated audit and adjust sensitivity radii, ambiguity limits,
draw counts, and decision thresholds in the browser.

## Finance Sensitivity Profiles

`finance_sensitivity_grid(...)` builds an opinionated Q grid for portfolio
model-risk review:

```python
q_presets = usf.finance_sensitivity_grid(
    portfolio,
    hidden=[
        "product",
        "region",
        "fico_band",
        "ltv_band",
        "broker_channel",
        "employment_type",
        "vintage",
    ],
    exposure="ead",
    factors={
        "macro_beta": "macro_beta",
        "rate_sensitivity": "rate_sensitivity",
    },
)
```

The default `credit_expected_loss` profile includes:

- saturated hidden-composition stress as the conservative benchmark
- bounded portfolio-mix shift as a documented sensitivity scenario
- exposure-weighted total-variation shift
- factor-exposure shift, when `factors=...` is supplied
- regional concentration shift
- observed no-shift baseline

Use the grid to show whether a conclusion depends only on the saturated
worst-case benchmark or also appears under narrower governance-approved
scenarios.

## Portfolio Concentration Stress Presets

Use concentration presets when independent hidden-bucket movement is too
coarse for a model-risk review. These helpers constrain portfolio-level exposure
drift while preserving the observed public segmentation.

Factor exposure drift:

```python
q = usf.q_factor_exposure_shift(
    0.20,
    portfolio,
    hidden=[
        "product",
        "region",
        "fico_band",
        "ltv_band",
        "broker_channel",
        "employment_type",
    ],
    factors={
        "macro_beta": "macro_beta",
        "rate_sensitivity": "rate_sensitivity",
        "house_price_beta": "house_price_beta",
    },
    exposure="ead",
)
```

Regional concentration drift:

```python
q = usf.q_regional_concentration_shift(
    0.10,
    portfolio,
    hidden=[
        "product",
        "region",
        "fico_band",
        "ltv_band",
        "broker_channel",
        "employment_type",
    ],
    region="region",
    exposure="ead",
)
```

Both helpers compile exposure-weighted hidden-cell moments and route through the
core `q_covariate_balance(...)` preset:

```text
|| standardized_factor_or_concentration_shift ||_2 <= radius
```

In model-review language, this asks:

> If the public risk buckets stay fixed, but hidden portfolio factor exposure or
> regional concentration can drift within this L2 tolerance, how much can the
> reported risk metric move?

This maps naturally to expected loss, default rate, LGD, delinquency, approval
benefit, and capital review where shifts are governed by portfolio exposure
profiles rather than arbitrary independent hidden-cell movement.

## Portfolio Segmentation Certificate

Use `certify_portfolio_segmentation(...)` when the output should be a
pass/fail/inconclusive artifact for a model-review pack:

```python
certificate = usf.certify_portfolio_segmentation(
    portfolio,
    public=["product", "region", "fico_band", "ltv_band"],
    hidden=[
        "product",
        "region",
        "fico_band",
        "ltv_band",
        "broker_channel",
        "employment_type",
        "vintage",
    ],
    metric=usf.expected_loss(pd="pd", lgd="lgd"),
    exposure="ead",
    candidate_refinements=["broker_channel", "employment_type", "vintage"],
    factors={"macro_beta": "macro_beta"},
    ambiguity_limit=0.0025,
    bucket_budget=80,
    search="exhaustive",
    model_id="EL_RETAIL_2026Q2",
    portfolio_name="Retail credit portfolio",
    intended_use="Expected-loss segmentation review",
)

print(certificate.to_markdown())
```

The returned `FinanceStabilityCertificate` keeps the underlying core
`RepresentationStabilityCertificate` at `certificate.core`, while adding
finance metadata and model-risk interpretation language.

To write the Markdown report:

```bash
uv run --package updatesupport-finance python \
  packages/updatesupport-finance/examples/model_risk_portfolio.py \
  --output data/finance_model_risk_report.md
```
