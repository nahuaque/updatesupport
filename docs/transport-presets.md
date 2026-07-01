# Transport Presets

`updatesupport` uses a finite admissible environment `Q` to define the
hidden-composition stress test. In the tabular compiler, every built-in preset
keeps the observed public distribution fixed. The preset controls how the
retained hidden cells may be reweighted while those public totals stay fixed.

The choice of `Q` is part of the estimand report. The same data, public columns,
hidden columns, and target can produce different ambiguity widths under
different presets.

## Quick Choice Guide

| Preset | Best first use | What it allows | Main tradeoff |
| --- | --- | --- | --- |
| `q="saturated"` | Conservative first-pass audit | Any reweighting among retained hidden cells inside each public cell | Can be wider than a plausible operational shift |
| `q=us.q_bounded_shift(radius)` | Practical default after the first pass | Each hidden-cell mass stays within a relative band around its observed mass | Radius is cellwise, so it may be too restrictive for rare cells or too loose for large cells |
| `q=us.q_tv_budget(radius)` | Overall churn budget | Total variation distance from the observed hidden distribution is bounded | Needs CVXPY and can concentrate the budget on the most influential cells |
| `q=us.q_chi_square_budget(radius)` | Variance-scaled divergence budget | Pearson chi-square divergence from the observed hidden distribution is bounded | Needs CVXPY and penalizes shifts out of small observed cells strongly |
| `q=us.q_kl_budget(radius)` | Information-divergence budget | KL divergence from the observed hidden distribution is bounded | Needs CVXPY and the radius is less intuitive than direct mass movement |
| `q=us.q_wasserstein(cost, radius)` | Similarity-aware shifts | Hidden mass can move cheaply between similar cells and expensively between dissimilar cells | Requires a defensible cost matrix and CVXPY |
| `q="observed"` | Baseline or sanity check | No hidden-composition shift | Always reports zero hidden-composition ambiguity |

Recommended reporting pattern:

```python
sensitivity = us.sensitivity_report(
    rows,
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "education_band", "region", "occupation"],
    target="target",
    weight="sample_weight",
    min_cell_weights=[1, 10, 25],
    q_presets=[
        "saturated",
        us.q_bounded_shift(0.5),
        "observed",
    ],
)

print(sensitivity.to_markdown())
```

Use `saturated` to understand the worst case on the retained support, use
`bounded_shift` or another constrained preset for a more substantive stress
test, and use `observed` as the zero-shift baseline.

## Shared Semantics

All built-in presets produced by `from_dataframe(...)`,
`public_descent_report(...)`, and `audit_effects(...)` have these semantics:

- The observed public law is fixed.
- Hidden cells are the retained hidden states after `min_cell_weight` filtering.
- No preset invents unseen hidden cells.
- The target value in each hidden cell is the empirical weighted mean supplied
  by the compiled data.
- The reported interval is a sensitivity or partial-identification interval,
  not a sampling confidence interval.

`min_cell_weight` and `Q` answer different questions. `min_cell_weight` decides
which hidden cells are stable enough to include in the state space. `Q` decides
how the retained cells may be reweighted.

## Saturated

Use:

```python
q = "saturated"
# or
q = us.q_saturated()
```

This fixes each public cell's total mass and allows arbitrary reweighting among
the retained hidden cells inside that public cell.

Use `saturated` when:

- You want the most conservative within-public-cell stress test.
- You do not have a defensible bound on how much hidden composition can move.
- You want a clear diagnostic of which public fibers contain meaningful hidden
  target variation.
- You want a fast preset that does not require the CVXPY extra.

Avoid treating `saturated` as realistic when:

- Hidden subgroups cannot plausibly vanish or take over a public cell.
- The retained state space contains noisy small cells.
- Your substantive question is about modest drift around the observed hidden
  distribution.

Interpretation:

> Holding public cell shares fixed, how much could the answer move if each
> public cell were allowed to put its mass on any retained hidden subgroup
> inside that cell?

## Observed

Use:

```python
q = "observed"
# or
q = us.q_observed()
```

This admits only the observed hidden distribution. It is the no-shift baseline,
so the ambiguity width is zero.

Use `observed` when:

- You want a baseline row in a sensitivity table.
- You want to verify that the compiled observed value is being reported.
- You want to compare stress-test intervals against the unperturbed hidden mix.

Do not use `observed` as a representation adequacy stress test. It rules out
the hidden-composition changes that `updatesupport` is designed to audit.

Interpretation:

> What is the target value with the hidden mix exactly as observed?

## Bounded Shift

Use:

```python
q = us.q_bounded_shift(0.5)
```

or, when passing a string preset:

```python
q = "bounded_shift"
q_radius = 0.5
```

For each retained hidden cell with observed mass `m`, the candidate mass must
stay between:

```text
max(0, m * (1 - radius)) and min(1, m * (1 + radius))
```

The public totals are still fixed.

Radius guidance:

- `0.0`: identical to `observed`.
- `0.25`: moderate cellwise drift.
- `0.5`: useful practical default for first sensitivity runs.
- `1.0`: each cell can range from zero to twice its observed mass.
- Larger values increasingly approach saturated behavior, subject to the public
  law and cellwise upper bounds.

Use `bounded_shift` when:

- You want a bounded, easy-to-explain stress test without CVXPY.
- You believe every retained hidden cell should remain near its observed share.
- You want to prevent all of the shift budget from concentrating on one
  high-leverage cell.
- You want a default constrained preset for README examples and routine reports.

Be cautious when:

- Rare hidden cells have unstable empirical target values.
- The same relative radius has very different absolute meaning for small and
  large cells.
- The application is better described by an overall churn budget than by
  cell-by-cell bounds.

Interpretation:

> Holding public cell shares fixed, how much could the answer move if every
> retained hidden subgroup were allowed to move by at most this relative amount
> from its observed mass?

## Total-Variation Budget

Use:

```python
q = us.q_tv_budget(0.15)
```

Install the CVXPY extra first:

```bash
uv sync --extra cvxpy
```

This constrains total variation distance from the observed hidden distribution:

```text
TV(q, p_observed) = 0.5 * sum_s |q(s) - p_observed(s)| <= radius
```

The public totals are still fixed.

Use `tv_budget` when:

- You can justify an overall amount of hidden-composition churn.
- You want the optimizer to allocate that churn wherever it has the largest
  effect on the target.
- You want a single global budget rather than per-cell relative bounds.

Be cautious when:

- Concentrating the shift budget on a small number of cells would be
  substantively implausible.
- You need a dependency-light workflow without CVXPY.
- Stakeholders may interpret the radius less easily than a cellwise percent
  movement.

Interpretation:

> Holding public cell shares fixed, how much could the answer move if the total
> amount of hidden mass moved away from the observed hidden mix were at most
> this TV radius?

## Chi-Square Budget

Use:

```python
q = us.q_chi_square_budget(0.15)
```

Install the CVXPY extra first:

```bash
uv sync --extra cvxpy
```

This constrains Pearson chi-square divergence from the observed hidden
distribution:

```text
chi2(q, p_observed) = sum_s (q(s) - p_observed(s))^2 / p_observed(s) <= radius
```

The public totals are still fixed.

Use `chi_square_budget` when:

- You want a smooth convex alternative to a TV budget.
- You want deviations from small observed cells to be penalized more heavily.
- You want a global divergence budget but do not have a hidden-cell geometry for
  Wasserstein.

Be cautious when:

- Stakeholders expect a direct "percentage of mass moved" interpretation.
- Very small retained cells remain in the state space; the denominator makes
  movement out of them expensive, but their target values may still be noisy.
- You need a dependency-light workflow without CVXPY.

Interpretation:

> Holding public cell shares fixed, how much could the answer move if the
> candidate hidden mix had Pearson chi-square divergence at most this radius
> from the observed hidden mix?

## KL Budget

Use:

```python
q = us.q_kl_budget(0.05)
```

Install the CVXPY extra first:

```bash
uv sync --extra cvxpy
```

This constrains KL divergence from the observed hidden distribution:

```text
KL(q || p_observed) = sum_s q(s) * log(q(s) / p_observed(s)) <= radius
```

The public totals are still fixed. Because the compiler only retains hidden
cells with positive observed weight, the reference distribution is positive on
the retained support.

Use `kl_budget` when:

- You want an information-divergence sensitivity set.
- You want a smooth alternative to TV that is common in robust statistics and
  distributionally robust optimization.
- You want to penalize multiplicative departures from the observed hidden mix.

Be cautious when:

- The radius does not have a simple mass-moved interpretation.
- You need to communicate results to an audience that expects direct cellwise
  movement bounds.
- You need a dependency-light workflow without CVXPY.

Interpretation:

> Holding public cell shares fixed, how much could the answer move if the
> candidate hidden mix had KL divergence at most this radius from the observed
> hidden mix?

## Wasserstein Budget

Use:

```python
states = grouped.problem.states
cost = {
    (left, right): 0.0 if left == right else hidden_cell_distance(left, right)
    for left in states
    for right in states
}

q = us.q_wasserstein(cost, radius=0.15)
```

Install the CVXPY extra first:

```bash
uv sync --extra cvxpy
```

This constrains the minimum transport cost from the observed hidden distribution
to the candidate hidden distribution. The cost can be passed as either a mapping
from `(left_state, right_state)` pairs to nonnegative costs or as a square
matrix in hidden-state order. Missing diagonal entries in a mapping are treated
as zero; missing off-diagonal entries are an error.

Use `wasserstein` when:

- You have a meaningful geometry or similarity measure between hidden cells.
- You want shifts between similar hidden cells to be cheaper than shifts between
  dissimilar cells.
- You want to encode that two hidden subgroups are plausible substitutes even
  when arbitrary TV movement would be too coarse.

Be cautious when:

- The cost matrix is arbitrary or hard to defend.
- The hidden states mix categorical variables with no natural distance.
- You need a simple report for nontechnical stakeholders.

Interpretation:

> Holding public cell shares fixed, how much could the answer move if hidden
> composition were allowed to move away from the observed mix only within this
> transport-cost budget?

## Choosing Radii

There is no universal correct radius. Treat radii as sensitivity assumptions and
report them explicitly.

A practical workflow:

1. Run `observed`, `bounded_shift(0.25)`, `bounded_shift(0.5)`, and `saturated`.
2. If the conclusion changes only under `saturated`, say that the result is
   stable to moderate hidden-composition drift but not to unconstrained
   within-public-cell reweighting.
3. If the conclusion changes under small bounded shifts, refine the public
   representation or report the ambiguity prominently.
4. If you have a substantive churn, divergence, or distance scale, add
   `q_tv_budget(...)`, `q_chi_square_budget(...)`, `q_kl_budget(...)`, or
   `q_wasserstein(...)` as domain-specific robustness checks.

Good reports state:

- the preset name
- the radius, if any
- the hidden columns and public columns
- the `min_cell_weight`
- whether CVXPY-backed presets were used
- the observed value and ambiguity interval under each scenario

## Preset Aliases

The compiler accepts both helper functions and string aliases:

| Canonical preset | Accepted strings |
| --- | --- |
| `saturated` | `"saturated"`, `"public_fiber_saturated"`, `"public-fiber-saturated"` |
| `observed` | `"observed"`, `"point"`, `"observed_only"` |
| `bounded_shift` | `"bounded"`, `"bounded-shift"`, `"bounded_shift"` |
| `tv_budget` | `"tv"`, `"total-variation"`, `"total_variation"`, `"tv_budget"` |
| `chi_square_budget` | `"chi-square"`, `"chi_square"`, `"chi-square-budget"`, `"chi_square_budget"`, `"chi2"`, `"chisquare"` |
| `kl_budget` | `"kl"`, `"kl-budget"`, `"kl_budget"`, `"kullback-leibler"`, `"relative-entropy"`, `"relative_entropy"` |
| `wasserstein` | `"w1"`, `"wasserstein"` |

For named presets with a radius, either call the helper function or pass
`q_radius`:

```python
us.from_dataframe(..., q=us.q_bounded_shift(0.5))
us.from_dataframe(..., q="bounded_shift", q_radius=0.5)
```

Prefer helper functions in reusable code because they make the radius and any
required cost matrix explicit.
