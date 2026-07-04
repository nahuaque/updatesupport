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
| `q=us.q_l2_budget(radius)` | Smooth Euclidean shift budget | L2 distance from the observed hidden distribution is bounded | Needs CVXPY and is scale-sensitive across cells |
| `q=us.q_covariate_balance(radius, moments)` | Causal/model-review balance stress test | Standardized hidden covariate-moment drift is bounded | Needs CVXPY and a defensible moment map |
| `q=us.q_mahalanobis_budget(radius, covariance=...)` | Covariance-aware ellipsoidal shifts | Covariance-standardized distance from the observed hidden distribution is bounded | Needs a defensible positive definite covariance matrix and CVXPY |
| `q=us.q_wasserstein(cost, radius)` | Similarity-aware shifts | Hidden mass can move cheaply between similar cells and expensively between dissimilar cells | Requires a defensible cost matrix and CVXPY |
| `q=us.q_fiber_support_floor(min_active, min_share=...)` | MIP support-diversity floor | Each public bucket must keep several active hidden cells above a minimum share | Needs SCIP or another MIP-capable CVXPY solver |
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
test, and use `observed` as the zero-shift baseline. The generated Markdown
includes a scenario summary and interpretation section before the full grid, so
the widest scenario and any mixed public-adequacy conclusion are visible without
reading the whole table first.

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

## Intersecting Presets

Use `q_intersection(...)` when a stress test needs several admissibility
requirements at once:

```python
q = us.q_intersection(
    us.q_tv_budget(0.10),
    us.q_covariate_balance(0.25, hidden_moments),
    backend="support_function",
)
```

This means "allow only hidden distributions that satisfy every listed preset."
It is an AND operation on admissible sets, not a sensitivity grid. The resulting
ambiguity interval is no wider than any component interval because the feasible
set is smaller.

The first algebra slice supports convex CVXPY-compatible components, plus
`observed`, `saturated`, and `bounded_shift`. Mixed-integer components such as
`q_fiber_support_floor(...)` are intentionally rejected by this convex
admissible-set compiler.

For lower-level workflows, `CvxpyAdmissibleSetSpec` also supports intersection:

```python
combined = tv_spec.intersect(balance_spec)
interval = combined.support_interval(grouped.problem)
```

## CVXPY Solver Choice

CVXPY-backed presets use CVXPY's default solver unless a solver is named
explicitly. The TV, chi-square, KL, L2, covariate-balance, Mahalanobis, and
Wasserstein preset helpers accept `solver` and `solver_options`:

```python
q = us.q_tv_budget(0.15, solver="SCIP")
```

Install the SCIP extra first:

```bash
pip install "updatesupport[scip]"
# or
uv add "updatesupport[scip]"
```

Use this when you want the existing convex transport model to route through
SCIP, or when you use mixed-integer presets such as `q_fiber_support_floor(...)`.
The stress-test semantics do not change: `tv_budget(radius=0.15)` remains the
reported Q preset.

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

## Fiber Support Floor

Use:

```python
q = us.q_fiber_support_floor(2, min_share=0.10)
```

Install a MIP-capable CVXPY solver first. The default solver for this preset is
SCIP:

```bash
pip install "updatesupport[scip]"
# or
uv add "updatesupport[scip]"
```

This fixes the observed public law and adds binary active-cell indicators
inside each public fiber. At least `min_active` hidden cells in every retained
public fiber must carry at least `min_share` of that public fiber's mass.
`max_active` can also cap the number of active hidden cells:

```python
q = us.q_fiber_support_floor(2, min_share=0.10, max_active=5)
```

Use `fiber_support_floor` when:

- A saturated stress test unrealistically moves all public-bucket mass into a
  single hidden subgroup.
- You need an operational support-diversity rule: hidden subgroups may shrink,
  but retained public buckets cannot collapse below a minimum number of
  credible active cells.
- You are comfortable using a mixed-integer solver such as SCIP.

Be cautious when:

- Retained public fibers are sparse after `min_cell_weight` filtering; every
  positive-mass public fiber needs at least `min_active` retained hidden cells.
- `min_share * min_active` is close to one; the feasible set may become very
  narrow.
- You rely on dual diagnostics. Mixed-integer solves generally do not expose
  the same useful dual variables as continuous convex solves.

Interpretation:

> Holding public cell shares fixed, how much could the answer move if each
> public bucket had to keep at least this many hidden subgroups meaningfully
> active?

## Total-Variation Budget

Use:

```python
q = us.q_tv_budget(0.15)
```

Install the CVXPY extra first:

```bash
# with pip
pip install "updatesupport[cvxpy]"

# with uv
uv add "updatesupport[cvxpy]"
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
# with pip
pip install "updatesupport[cvxpy]"

# with uv
uv add "updatesupport[cvxpy]"
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
# with pip
pip install "updatesupport[cvxpy]"

# with uv
uv add "updatesupport[cvxpy]"
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

## L2 Budget

Use:

```python
q = us.q_l2_budget(0.10)
```

Install the CVXPY extra first:

```bash
# with pip
pip install "updatesupport[cvxpy]"

# with uv
uv add "updatesupport[cvxpy]"
```

This constrains the Euclidean distance from the observed hidden distribution:

```text
||q - p_observed||_2 <= radius
```

The public totals are still fixed. This is an SOCP-compatible stress test.

Use `l2_budget` when:

- You want a smooth norm budget rather than a TV, KL, or chi-square divergence.
- You want hidden-composition changes to spread more naturally than a pure TV
  budget, which can concentrate movement on a few cells.
- You need a simple conic stress test before specifying a richer covariance or
  cost structure.

Be cautious when:

- Hidden cells have very different natural scales or uncertainty levels; plain
  L2 treats one unit of mass movement the same across cells.
- You need a direct mass-moved interpretation. TV is more direct for that.
- You need a dependency-light workflow without CVXPY.

Interpretation:

> Holding public cell shares fixed, how much could the answer move if the
> candidate hidden mix stayed within this Euclidean radius of the observed
> hidden mix?

## Covariate-Balance Budget

Use:

```python
moments = {
    "standardized_prior_spend": {
        ("young", "low_income", "urban"): -0.40,
        ("young", "low_income", "rural"): -0.15,
        ("older", "high_income", "urban"): 0.55,
    },
    "standardized_baseline_risk": {
        ("young", "low_income", "urban"): 0.80,
        ("young", "low_income", "rural"): 0.30,
        ("older", "high_income", "urban"): -0.25,
    },
}

q = us.q_covariate_balance(0.25, moments)
```

Install the CVXPY extra first:

```bash
# with pip
pip install "updatesupport[cvxpy]"

# with uv
uv add "updatesupport[cvxpy]"
```

This constrains hidden covariate balance rather than cellwise distributional
distance:

```text
|| standardize(M q - baseline) ||_2 <= radius
```

Here `M` is a moment-by-hidden-cell matrix. By default, `baseline` is the
observed hidden-distribution moment vector, and each moment is standardized by
its weighted observed hidden-cell standard deviation. You can override either
with `baseline=...` or `scale=...`.

Use `covariate_balance` when:

- You are auditing a causal, uplift, or model-review report and the natural
  stress condition is balance drift on hidden covariates.
- You want to preserve the public report while allowing hidden composition to
  change within an interpretable L2 tolerance on standardized moments.
- You have hidden-cell summaries from an estimator, causal adapter, or grouped
  dataframe and want a conic stress test that speaks the language of balance.

Be cautious when:

- The selected moments do not cover the hidden mechanisms that matter for the
  target.
- A zero balance radius is misread as no hidden shift. It only means exact
  balance on the supplied moments; multiple hidden distributions may still be
  admissible.
- The default observed standard deviations are not the right scale for the
  review question. In that case, pass an explicit `scale`.

Interpretation:

> Holding public cell shares fixed, how much could the reported target move if
> hidden covariate balance were allowed to drift by at most this standardized
> L2 tolerance?

## Mahalanobis Budget

Use:

```python
covariance = [
    [1.0, 0.2, 0.0],
    [0.2, 1.0, 0.1],
    [0.0, 0.1, 1.0],
]

q = us.q_mahalanobis_budget(0.10, covariance=covariance)
```

Install the CVXPY extra first:

```bash
# with pip
pip install "updatesupport[cvxpy]"

# with uv
uv add "updatesupport[cvxpy]"
```

This constrains covariance-standardized distance from the observed hidden
distribution:

```text
sqrt((q - p_observed)' Sigma^-1 (q - p_observed)) <= radius
```

The public totals are still fixed. `covariance` must be a symmetric positive
definite matrix in retained hidden-state order, or a mapping from
`(left_state, right_state)` pairs to covariance entries.

Use `mahalanobis_budget` when:

- You have a defensible covariance model for hidden-composition shifts.
- You want correlated hidden-cell movements to be treated differently from
  independent movements.
- You want an ellipsoidal stress test that maps naturally to model-risk,
  survey-weight, or portfolio-composition uncertainty language.

Be cautious when:

- The covariance matrix is estimated from sparse cells or is not positive
  definite.
- The covariance model is harder to explain than the underlying report.
- You need a dependency-light workflow without CVXPY.

Interpretation:

> Holding public cell shares fixed, how much could the answer move if hidden
> composition shifted within this covariance-standardized ellipsoid around the
> observed hidden mix?

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
# with pip
pip install "updatesupport[cvxpy]"

# with uv
uv add "updatesupport[cvxpy]"
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
4. If you have a substantive churn, divergence, norm, covariance, or distance
   scale, add `q_tv_budget(...)`, `q_chi_square_budget(...)`,
   `q_kl_budget(...)`, `q_l2_budget(...)`,
   `q_covariate_balance(...)`, `q_mahalanobis_budget(...)`, or
   `q_wasserstein(...)` as domain-specific robustness checks.

Good reports state:

- the preset name
- the radius, if any
- the hidden columns and public columns
- the `min_cell_weight`
- whether CVXPY-backed presets were used
- the observed value and ambiguity interval under each scenario
- the largest CVXPY dual multipliers, when available, as local diagnostics for
  which public-law, Q-budget, or lower-bound constraints are most influential

## Parameterized CVXPY Sweeps

The CVXPY-backed TV, chi-square, KL, L2, covariate-balance, Mahalanobis, and
Wasserstein presets can use an opt-in parameterized backend:

```python
grouped = us.from_dataframe(
    rows,
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "occupation"],
    target="target",
    q=us.q_tv_budget(0.10, backend="parameterized_cvxpy"),
)

interval_010 = grouped.problem.global_transport_modulus()

grouped.problem.environments.set_parameter("radius", 0.20)
interval_020 = grouped.problem.global_transport_modulus()
```

This reuses cached CVXPY problem objects for the fixed state space and updates
CVXPY parameters for the objective, public law, and radius. Use it for dense
radius sweeps where rebuilding the CVXPY problem for each radius would dominate
the runtime. If the hidden state space, public projection, or cost matrix
changes, compile a new problem.

The sensitivity-grid helpers apply this optimization automatically for
compatible TV, chi-square, KL, L2, Mahalanobis, and Wasserstein rows. They still
recompile when a scenario changes the hidden state space, minimum retained cell
weight, public projection, Mahalanobis covariance matrix, or Wasserstein cost
matrix.

## Batched CVXPY Sweeps

For sensitivity grids where adjacent scenarios share the same retained state
space and Q family, use the batched backend:

```python
report = us.sensitivity_report(
    rows,
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "occupation"],
    target="target",
    q_presets=[
        us.q_tv_budget(0.10, backend="batched_cvxpy"),
        us.q_tv_budget(0.20, backend="batched_cvxpy"),
    ],
)
```

This uses CVXPY variables shaped like `q[scenario, state]` to solve contiguous
compatible scenarios in one optimization problem. The first slice reuses the
existing one-dimensional Q builders per scenario, so it is conservative and
compatible with the current TV, chi-square, KL, L2, Mahalanobis, and
Wasserstein presets.

## Preset Aliases

The compiler accepts both helper functions and string aliases:

| Canonical preset | Accepted strings |
| --- | --- |
| `saturated` | `"saturated"`, `"public_fiber_saturated"`, `"public-fiber-saturated"` |
| `observed` | `"observed"`, `"point"`, `"observed_only"` |
| `intersection` | `"intersection"`, `"intersect"`, `"and"`, `"meet"`, `"q_intersection"` |
| `bounded_shift` | `"bounded"`, `"bounded-shift"`, `"bounded_shift"` |
| `tv_budget` | `"tv"`, `"total-variation"`, `"total_variation"`, `"tv_budget"` |
| `chi_square_budget` | `"chi-square"`, `"chi_square"`, `"chi-square-budget"`, `"chi_square_budget"`, `"chi2"`, `"chisquare"` |
| `kl_budget` | `"kl"`, `"kl-budget"`, `"kl_budget"`, `"kullback-leibler"`, `"relative-entropy"`, `"relative_entropy"` |
| `l2_budget` | `"l2"`, `"l2-budget"`, `"l2_budget"`, `"euclidean"`, `"euclidean_budget"` |
| `mahalanobis_budget` | `"mahalanobis"`, `"mahalanobis-budget"`, `"mahalanobis_budget"`, `"ellipsoid"`, `"ellipsoidal"` |
| `wasserstein` | `"w1"`, `"wasserstein"` |

For named presets with a radius, either call the helper function or pass
`q_radius`:

```python
us.from_dataframe(..., q=us.q_bounded_shift(0.5))
us.from_dataframe(..., q="bounded_shift", q_radius=0.5)
```

Prefer helper functions in reusable code because they make the radius and any
required cost matrix explicit.
