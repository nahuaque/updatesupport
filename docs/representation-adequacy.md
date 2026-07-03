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

For causal workflows, use causal inference libraries such as
[EconML](https://www.pywhy.org/EconML/),
[DoWhy](https://www.pywhy.org/dowhy/),
[DoubleML](https://docs.doubleml.org/), or
[CausalML](https://causalml.readthedocs.io/en/latest/) to identify and estimate
effects first, then use `updatesupport` to audit the public reporting
representation. See
[Using `updatesupport` with causal inference libraries](causal-library-integration.md).

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
  added to the public representation, with before ambiguity, after ambiguity,
  absolute reduction, and percentage reduction.
- **Sensitivity-aware refinements**: variables whose ambiguity reduction remains
  useful across Q presets, hidden-state choices, and sparse-cell thresholds.
- **Public representation frontier**: the Pareto frontier of candidate public
  representations when the review asks for the smallest public bucket design
  that remains stable across stress tests.
- **Representation stability certificate**: a pass/fail/inconclusive decision
  artifact built from frontier search, recording the selected public
  representation, stress-test assumptions, search guarantee, and limitations.

This report should be readable without requiring the audience to know the
underlying finite-support theory. In causal or model-based workflows, the report
should also keep four ideas separate: the supplied causal/model estimate,
statistical uncertainty from the estimation workflow, hidden-composition
ambiguity from the update-support stress test, and public refinement
recommendations for the reporting representation.

## What `Q` Means

`Q` is the set of admissible hidden distributions considered plausible for the
stress test. It defines how hidden composition is allowed to change.

The current example uses a public-fiber-saturated `Q`: within each public cell,
any retained hidden subgroup can receive the mass assigned to that public cell.
This is deliberately conservative and easy to explain, but it can be too broad
for some analyses.

The named `Q` presets are:

- `saturated`: arbitrary reweighting inside public fibers
- `observed`: only the observed hidden mix is admissible
- `bounded_shift`: limited per-hidden-cell relative movement away from the
  observed hidden mix
- `tv_budget`: total-variation budget around the observed hidden mix
- `chi_square_budget`: Pearson chi-square divergence budget around the observed
  hidden mix
- `kl_budget`: KL divergence budget around the observed hidden mix
- `wasserstein`: Wasserstein budget using an explicit hidden-cell cost matrix

`min_cell_weight` is a separate robustness knob: it drops tiny hidden cells before
the finite problem is compiled. `Q` then controls how the retained hidden cells
may be reweighted.

Every report should state which `Q` was used. Otherwise the ambiguity number is
not interpretable.

For practical guidance on when to use each preset, see
[Transport presets](transport-presets.md).

For public bucket design, see
[Public representation frontier](public-representation-frontier.md).

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

The first compiler and report slices are now exposed as
`updatesupport.from_dataframe(...)` and `updatesupport.public_descent_report(...)`.
Together, they compile weighted tabular observations into a finite problem,
compute a public-descent audit, and render the result as Markdown.

```python
report = us.public_descent_report(
    data,
    target="PINCP_GT_50000",
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=[
        "AGE_BAND",
        "EDU_BAND",
        "SEX",
        "OCC_MAJOR",
        "COW",
        "WKHP_BAND",
        "RAC1P",
        "MAR",
        "POBP",
        "RELP",
    ],
    weight="PWGTP",
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P", "RELP"],
    min_cell_weight=25,
    q="saturated",
)

print(report.to_markdown())
```

The Q preset surface now supports several levels of conservatism:

- `q="saturated"`: arbitrary hidden reweighting inside each observed public
  cell.
- `q=us.q_bounded_shift(radius)`: cellwise relative mass bands around the
  observed hidden distribution.
- `q=us.q_tv_budget(radius)`: total-variation budget around the observed hidden
  distribution, using the optional CVXPY backend.
- `q=us.q_chi_square_budget(radius)`: Pearson chi-square divergence budget
  around the observed hidden distribution, using the optional CVXPY backend.
- `q=us.q_kl_budget(radius)`: KL divergence budget around the observed hidden
  distribution, using the optional CVXPY backend.
- `q=us.q_wasserstein(cost, radius)`: Wasserstein budget with an explicit
  hidden-cell cost matrix, using the optional CVXPY backend.
- `q="observed"`: no hidden-composition shift.

Install the CVXPY extra before using TV, chi-square, KL, or Wasserstein presets:

```bash
# with pip
pip install "updatesupport[cvxpy]"

# with uv
uv add "updatesupport[cvxpy]"
```

For dense radius sweeps on a fixed compiled state space, pass
`backend="parameterized_cvxpy"` to those presets and update the environment's
`radius` parameter between solves.

The next implementation slices should focus on deeper sensitivity reporting and
experimental relational transport types such as Gromov-Wasserstein only when the
application supplies two comparable hidden-state geometries. For current preset
selection guidance, see [Transport presets](transport-presets.md).

The documentation should keep the ACSIncome case study as the primary example
and put the finite-support theory underneath it.
