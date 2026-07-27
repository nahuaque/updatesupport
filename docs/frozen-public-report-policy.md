# Frozen Public-Report Policies

A calibrated design can be frozen into an out-of-sample reporting contract:

```python
design = portfolio.design_calibrated(
    historical_rows,
    design_period_rows,
    period="quarter",
    coverage=0.90,
    rollup_column="channel",
    max_added_columns=2,
)

policy = design.freeze()
```

The policy retains:

- the selected public representation,
- each selected `ClaimSpec` and decision threshold,
- each claim's historically calibrated TV radius,
- the optional categorical rollup mapping,
- design-period public and retained hidden support,
- normalized design-period hidden-cell weights,
- a deterministic policy fingerprint.

It removes remaining candidate-refinement search from the selected claims.
Auditing and backtesting therefore evaluate the frozen reporting contract; they
do not redesign it.

## Audit One Future Batch

```python
audit = policy.audit(
    next_quarter_rows,
    period="2026-Q3",
)

print(audit.status)
print(audit.to_markdown())
```

The audit recompiles the supplied target values for the future batch, applies
the frozen rollup, and runs every selected claim under its frozen calibrated TV
radius.

It also measures the realized hidden-composition shift relative to the design
period. For each claim, the future composition is restandardized to the
design-period public law before total variation is measured. This isolates
within-public-bucket recomposition from changes in the reported public bucket
shares.

For threshold claims, `policy.audit(...)` computes the current minimum
claim-breaking witness by default. Set `include_breaking_witness=False` when
only the policy verdict and drift diagnostics are needed.

## Verdicts

The policy layer uses operational verdicts:

- `pass`: every claim passes, support is compatible, and the realized
  recomposition is inside each calibrated TV radius;
- `review`: support remains compatible, but at least one claim fails or one
  realized recomposition exceeds its calibrated radius;
- `inconclusive`: a claim audit is inconclusive or the future batch cannot be
  evaluated under the frozen support contract.

The underlying claim audit still retains its ordinary `pass`, `fail`, or
`inconclusive` status. `review` is a policy-monitoring action, not a new
mathematical claim verdict.

## Support Drift

The frozen policy compares retained support before interpreting a future
result.

The following produce an inconclusive policy audit:

- an unseen category in a frozen intermediate rollup,
- a new retained hidden cell,
- a new public cell,
- a missing positive-mass design-period public fiber.

A missing hidden cell inside a surviving public fiber is a support contraction.
It is reported, but the batch remains evaluable: zero current mass on an
existing retained cell is a composition change that the TV comparison can
represent.

This distinction prevents a calibrated radius learned on one finite state space
from being silently applied to a different state space.

## Backtest Holdout Periods

```python
backtest = policy.backtest(
    holdout_rows,
    period="quarter",
    period_order=["2025-Q4", "2026-Q1", "2026-Q2"],
)

print(backtest.to_markdown())
```

Every holdout period is evaluated independently against the same design-period
baseline. The backtest does not roll the baseline forward and does not
recalibrate from holdout outcomes.

The summary includes:

- period and claim pass rates,
- support compatibility rate,
- review and inconclusive counts,
- calibrated-radius breach count,
- per-period and per-claim outcomes.

Minimum breaking witnesses are disabled by default during backtesting to avoid
an extra inverse solve for every threshold claim and period. Pass
`include_breaking_witness=True` when those diagnostics are required.

## Structured Exports

The policy, audit, and backtest all support:

```python
policy.to_json()
policy.to_tables()
policy.to_dataframes()

audit.to_json()
audit.to_tables()
audit.to_dataframes()

backtest.to_json()
backtest.to_tables()
backtest.to_dataframes()
```

Policy tables include the frozen claim contracts, reference support, and
rollup mapping. Audit tables include claim outcomes, support drift, and
breaking witnesses. Backtest tables include period summaries and flattened
claim-period outcomes.

## Scope

Freezing prevents holdout leakage from radius calibration, rollup selection,
and representation search. It does not turn the result into a statistical
guarantee.

Target values are recompiled from each evaluated batch. The policy therefore
audits whether the current reported target survives hidden recomposition under
the frozen reporting contract; it does not attribute changes in target values
or model behavior to composition.

Observed TV remains conditional on the retained refinement and design-period
support. It excludes changes in public bucket shares and does not protect
against a future regime outside the frozen state space. A radius breach is a
review trigger, not a hypothesis test or failure probability.
