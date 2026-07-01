# Representation Adequacy Guide

`updatesupport` is for auditing whether a public representation is adequate for
an estimate. The public representation is the set of categories you are willing
or able to report. The hidden representation is a finer state space that may
affect the estimate even when the public counts are unchanged.

The central question is:

> If the public distribution stayed fixed, how much could the reported estimate
> change because the hidden mix inside public cells changed?

That question is different from causal identification, prediction accuracy, and
sampling uncertainty. It is a stability question about a representation.

## Core Workflow

1. Define the public columns.
2. Define hidden columns that refine the public columns.
3. Define the target, such as a binary outcome rate or another linear estimand.
4. Choose an admissible environment class `Q`.
5. Compute the observed value, stress interval, ambiguity, worst public fibers,
   and useful refinements.

For ACSIncome, a public representation might be:

```text
AGE_BAND x EDU_BAND x SEX
```

and the hidden variables might include:

```text
OCC_MAJOR, COW, WKHP_BAND, RAC1P, MAR, POBP, RELP
```

The report then asks whether age, education, and sex are enough to determine the
income-threshold rate after hidden occupational and household-composition
differences are allowed to move within public cells.

## Report Contract

A useful analyst-facing report should include:

- **Observed value**: the estimate under the observed hidden mix.
- **Lower and upper stress values**: the smallest and largest values allowed by
  the chosen stress test.
- **Transport ambiguity**: the upper value minus the lower value.
- **Public adequacy**: whether the public representation determines the
  estimand under `Q`.
- **Worst public fibers**: public cells where hidden variation contributes most
  to ambiguity.
- **Recommended refinements**: hidden variables that would reduce ambiguity if
  added to the public representation.

This report should be readable without requiring the audience to know the
underlying finite-support theory.

## What `Q` Means

`Q` is the set of admissible hidden distributions considered plausible for the
stress test. It defines how hidden composition is allowed to change.

The current example uses a public-fiber-saturated `Q`: within each public cell,
any retained hidden subgroup can receive the mass assigned to that public cell.
This is deliberately conservative and easy to explain, but it can be too broad
for some analyses.

Near-term library work should add named `Q` presets:

- `saturated`: arbitrary reweighting inside public fibers
- `min_cell_weight`: saturated after dropping tiny hidden cells
- `bounded_shift`: limited movement away from the observed hidden mix
- `tv_budget`: total-variation budget around the observed hidden distribution

Every report should state which `Q` was used. Otherwise the ambiguity number is
not interpretable.

## Interpretation Rules

Transport ambiguity is not a confidence interval. It does not say how much
sampling error is present. It says how much the estimate could move under the
allowed hidden-composition changes.

Public adequacy is estimand-specific. A public representation may be adequate
for one target and inadequate for another.

Small ambiguity does not prove that the public representation is correct. It
means the reported estimate is stable under the chosen hidden state space and
environment class.

Large ambiguity does not prove the estimate is wrong. It means the public
representation leaves room for materially different answers under the stress
test.

## Common Failure Modes

Zero ambiguity can be real, but it can also mean the hidden state space is too
thin or the target is constant inside each public fiber.

Extremely large ambiguity can be real, but it can also be caused by tiny hidden
cells with noisy empirical rates. Reports should expose `min_cell_weight` and
sensitivity checks.

Hidden columns may not be available in production. In that case, `updatesupport`
is still useful during validation on richer data, but the report should state
that the audit depends on a richer reference dataset.

A simple groupby table is not enough. The value is in comparing public stability
against hidden refinements while preserving the public law.

## Near-Term Implementation Slices

The next API surface should make the analyst workflow direct:

```python
report = us.public_descent_report(
    data,
    target="PINCP_GT_50000",
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["OCC_MAJOR", "COW", "WKHP_BAND", "RAC1P", "MAR", "POBP", "RELP"],
    weight="PWGTP",
    min_cell_weight=25,
    q="saturated",
)

print(report.to_markdown())
```

Implementation should land in this order:

1. `from_dataframe(...)`: compile weighted tabular data into a finite problem.
2. `PublicDescentReport`: store observed value, stress interval, ambiguity,
   adequacy, worst fibers, and refinements.
3. `recommend_refinements(...)`: rank candidate hidden variables by ambiguity
   reduction.
4. named `Q` presets: make stress-test assumptions explicit and repeatable.
5. sensitivity reports: vary `min_cell_weight`, hidden columns, and `Q`.

The documentation should keep the ACSIncome case study as the primary example
and put the finite-support theory underneath it.
