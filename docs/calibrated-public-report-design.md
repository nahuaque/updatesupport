# Calibrated Public-Report Design

`design_calibrated_public_report(...)` combines four pieces of refinement
intelligence into one review workflow:

1. calibrate a total-variation stress radius from historical recompositions,
2. optionally replace one high-cardinality categorical refinement with an exact
   global rollup,
3. choose one public representation for a claim or claim portfolio under the
   calibrated stress tests,
4. show the nearest fixed-public composition that would break each threshold
   claim.

The result answers a practical design question:

> Given the composition changes we have historically observed, what public
> buckets support these claims now, and how much larger would a
> decision-breaking recomposition have to be?

## Portfolio Example

```python
import updatesupport as us

portfolio = us.claim_portfolio(
    conversion_claim,
    retention_claim,
    margin_claim,
    name="Quarterly KPI report",
    candidate_refinements=["channel", "tenure", "plan"],
)

design = portfolio.design_calibrated(
    historical_rows,
    current_rows,
    period="quarter",
    coverage=0.90,
    min_train_transitions=3,
    rollup_column="channel",
    rollup_max_groups=4,
    rollup_output_column="channel_group",
    max_added_columns=2,
    bucket_budget=40,
)

print(design.to_markdown())
```

The functional entry point is equivalent:

```python
design = us.design_calibrated_public_report(
    historical_rows,
    current_rows,
    portfolio,
    period="quarter",
)
```

A single claim has the same method:

```python
design = claim.design_calibrated(
    historical_rows,
    current_rows,
    period="quarter",
)
```

Calibrated TV audits use CVXPY when the calibrated radius is nonzero:

```bash
pip install "updatesupport[cvxpy]"
```

## Execution Order

### 1. Optional categorical rollup

When `rollup_column` is supplied, the designated anchor claim performs exact
one-column partition search under saturated Q. The search is fitted on the
pooled historical and current rows so every known category can be transformed.
`rollup_claim_index` selects that anchor with zero-based Python indexing.

If an intermediate rollup is selected, the generated group column replaces the
original column in the candidate-refinement set. The same mapping is then
applied to every historical and current row and every claim.

The rollup is only a proposal generator. It does not certify the other claims.
The later calibrated shared search must validate the resulting mapping against
the full portfolio.

### 2. Historical TV calibration

Each claim receives a `HistoricalTVCalibrationReport`. The radius is the
requested higher empirical quantile of support-compatible consecutive-period
TV distances after the later hidden mix is restandardized to the earlier public
law.

The radius is calibrated at the claim's declared base public representation and
then held fixed across candidate refinements. This gives every candidate the
same historical stress severity instead of making the stress budget easier or
harder after seeing a candidate's result. It is not a candidate-specific
recalibration and can be conservative for a finer public representation.

The composition radius may coincide across claims sharing the same state-space
contract, but each report retains claim-specific target and decision backtests.

### 3. Public representation search

For one claim, the workflow delegates to `PublicReportDesign`. For two or more
claims, it delegates to exact `SharedRepresentationDesign` search. Every
candidate is evaluated under the corresponding calibrated TV preset rather than
an arbitrary radius.

The selected shared representation is one schema used by every claim. If no
candidate certifies all claims, the report returns the existing shared-search
best-effort design and marks the combined status accordingly.

### 4. Direct breaking witnesses

After selecting the public representation, every threshold claim receives a
minimum-TV breaking witness. The report compares that distance with the
historically calibrated TV radius:

```text
calibrated TV radius: 0.05
nearest claim break:  0.15
radius multiple:      3.0x
```

This says the closest retained-support decision flip is three times the
historical stress budget. It does not say the claim has a particular probability
of failing. The witness solve deliberately ranges beyond the calibrated ball so
the report can measure the distance to failure.

## Report Contract

`CalibratedPublicReportDesign` provides:

- the recommended public representation and combined status,
- one calibration report per claim,
- the optional exact categorical rollup and selected mapping,
- either a single-claim `PublicReportDesign` or portfolio
  `SharedRepresentationDesign`,
- selected-representation claim audits,
- minimum-TV breaking witnesses and radius multiples for threshold claims,
- Markdown, JSON, named-table, and DataFrame exports.

Important tables include:

- `summary`,
- `claim_outcomes`,
- `calibration_transitions`,
- `calibration_backtests`,
- `breaking_witnesses`,
- `breaking_transfers`,
- prefixed `rollup_*`, `public_design_*`, or `shared_design_*` tables.

## Scope

The combined report remains conditional on the retained refinement and observed
support. Historical calibration cannot protect against a future regime or new
hidden cell that is absent from history. The optional rollup is anchor-driven
and global across public fibers; it is not a jointly optimized portfolio rollup.
Its safety comes from subsequent portfolio validation, not from the anchor
search alone.

Current rows participate in fitting an optional rollup mapping, so that step is
report design rather than out-of-sample validation. Historical rolling
backtests remain time-ordered and use only prior transitions for each evaluated
radius.

As elsewhere, `hidden` means retained by the analyst but omitted from the public
reporting representation. None of these artifacts is a confidence interval or
an absolute robustness guarantee.
