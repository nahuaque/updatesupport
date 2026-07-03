# Mathematical and Statistical Soundness

`updatesupport` is mathematically sound for a specific and explicit class of
audits:

> finite hidden state spaces, fixed public projections, fixed hidden-state
> target values, and explicit admissible hidden-distribution classes.

In that setting the reported ambiguity interval is a well-defined
partial-identification or sensitivity interval. It is not a confidence
interval, not a causal identification result, and not a generic nonlinear
functional optimizer.

The most important boundary is the aggregate target. The current core library
assumes a fixed linear plug-in target:

```text
psi(q) = sum_d h(d) q(d)
```

where `d` ranges over retained hidden cells, `q(d)` is hidden-cell mass, and
`h(d)` is the supplied target value for hidden cell `d`.

If the aggregate target is nonlinear in `q`, changes when the public
representation changes, or depends on a model refit under each representation,
then it is not automatically covered by the current soundness guarantee. Such a
target needs an explicit reformulation or a future target-functional backend.

## Core Object

The current mathematical object is finite.

- `D` is a finite hidden state space.
- `pi: D -> O` maps hidden states to public report buckets.
- `h: D -> R` is a fixed hidden-state target value.
- `q` is a probability distribution over `D`.
- `Q` is the admissible class of hidden distributions.
- `psi(q) = sum_d h(d) q(d)` is the aggregate being audited.

The public law is the pushforward `pi#q`. For a fixed observed public law `p`,
the primary interval is:

```text
lower = inf { psi(q) : q in Q, pi#q = p }
upper = sup { psi(q) : q in Q, pi#q = p }
ambiguity = upper - lower
```

This is the hidden-composition ambiguity conditional on:

- the retained finite hidden support,
- the supplied cell-level target values,
- the public projection,
- the chosen admissible set `Q`.

## Target Contract

The current target contract is:

```text
hidden cell d  -> fixed scalar h(d)
aggregate      -> q-weighted average of h(d)
```

This covers many common review metrics when expressed at the right unit:

- rates,
- means,
- average scores,
- expected loss rates,
- default rates,
- loss-given-default averages,
- average treatment effects or uplift scores supplied by an external estimator.

The phrase "fixed scalar" matters. Once `from_dataframe(...)` compiles the
problem, `updatesupport` treats the hidden-cell target values as inputs. It
does not update them when `q` changes, when a public bucket is refined, or when
a model would be refit.

## What Is Not Covered Automatically

Many useful targets are not fixed linear functionals in their native form.
Those targets are not invalid, but they need explicit handling.

### Ratio Targets

Some ratios can be made linear by choosing the right weight.

For example, expected loss rate can be written as an exposure-weighted mean:

```text
EL rate = sum_i exposure_i * pd_i * lgd_i / sum_i exposure_i
```

If exposure defines the row or cell weight, the compiled target is the
exposure-weighted mean of `pd * lgd`, which is linear in the normalized
exposure distribution.

But an arbitrary ratio:

```text
T(q) = numerator(q) / denominator(q)
```

is not the same thing unless the denominator is fixed by the chosen measure or
the ratio has been deliberately reformulated. If the denominator can move under
the hidden-composition stress test, the current linear interval is not a bound
for the native ratio without additional math.

### Moment Transforms

Targets like:

```text
T(q) = g(E_q[a], E_q[b], E_q[c])
```

are nonlinear unless `g` is affine. Examples include squared means, calibrated
indices, nonlinear risk scores, and transformed moments.

The current library can audit a supplied scalar cell value such as
`h(d) = g_d`, but that is different from optimizing the global nonlinear
functional `g(E_q[a], E_q[b], E_q[c])`.

### Distributional Targets

Quantiles, tail probabilities at endogenous thresholds, Gini coefficients,
AUC, KS statistics, top-decile shares, and approval-rate-at-threshold metrics
are not fixed linear averages in general.

Some distributional targets can be bounded with specialized linear, convex, or
mixed-integer formulations. The current public API does not claim those bounds.

### Representation-Dependent Targets

The trickiest case is a target that changes when the public representation
changes. Examples:

- the analyst refits a model after adding a public bucket,
- subgroup estimates are shrinkage-adjusted differently under a finer public
  table,
- bins are recalibrated after refinement,
- a public-cell estimator uses within-cell sample size or regularization,
- a causal estimator recomputes effects after changing reported covariates.

Then the object is not one fixed `psi(q)`. It is a reporting procedure:

```text
public representation -> compiled target values -> reported aggregate
```

That can still be useful to audit, but the claim changes. The result is a
procedure-comparison sensitivity analysis, not the current fixed-target
transport interval. Reports should say that the target was recomputed under
each representation.

### Composition-Dependent Structural Targets

If the target value itself depends on the transported hidden distribution:

```text
h = h(d, q)
```

then the aggregate is a nonlinear or equilibrium functional `T(q)`, not
`sum_d h(d) q(d)`. Examples include interference, capacity effects, market
equilibrium, feedback policies, or model outputs that change when portfolio
composition changes.

The current adequacy, fiber-range, witness, and refinement formulas do not
apply directly to that setting.

## Public Adequacy

For the current fixed linear target, a support or reporting partition `S` is
adequate when:

```text
S#q1 = S#q2  implies  psi(q1) = psi(q2)
```

for every admissible `q1, q2 in Q`.

Public adequacy is the special case where `S` is the public partition induced by
`pi`. If public adequacy is false, then at least one admissible hidden
composition shift preserves the public representation while changing the
linear aggregate target.

For a general nonlinear target `T(q)`, the analogous condition would be:

```text
S#q1 = S#q2  implies  T(q1) = T(q2)
```

That is a different mathematical problem. The current least-support and
fiber-range shortcuts are not valid for arbitrary `T`.

## Saturated Reweighting

The `saturated` preset fixes the observed public law and allows arbitrary
reweighting among retained hidden cells inside each public fiber.

For the current fixed linear target, let:

```text
range(o) = max_{d: pi(d)=o} h(d) - min_{d: pi(d)=o} h(d)
```

Then the observed-law saturated ambiguity is exactly:

```text
sum_o p(o) range(o)
```

The lower endpoint puts each public bucket's mass on the lowest-target retained
hidden cell in that bucket. The upper endpoint puts the same public mass on the
highest-target retained hidden cell. This is why the report can decompose the
ambiguity into public-fiber contributions.

For saturated environments with an explicitly fixed public law, public fibers
with zero admissible mass are irrelevant to adequacy. Hidden target variation in
a zero-mass public fiber cannot move the aggregate because no admissible
distribution can place mass there.

## Linear and Convex Backends

The finite linear backend uses `scipy.optimize.linprog`.

- The simplex constraints are implicit: `q(d) >= 0` and `sum_d q(d) = 1`.
- Public-law constraints are linear equalities.
- User-supplied bounds and linear constraints are added directly.
- The objective is the fixed linear target `sum_d h(d) q(d)`.
- Linear objectives over polytopes attain extrema at feasible optima, so the
  reported lower and upper values are the correct LP interval endpoints when
  the LP solves successfully.

The CVXPY backend currently generalizes the admissible set `Q`, not the target
functional.

- The target objective remains linear in `q`.
- Built-in and custom CVXPY constraints define a convex feasible set.
- CVXPY solves the lower and upper optimization problems separately.
- Dual diagnostics are KKT-style local sensitivity diagnostics for the solved
  problem; they are not causal explanations or global feature importance.

This distinction matters: CVXPY support does not mean that arbitrary nonlinear
aggregate targets are currently supported. It means convex constraints on the
admissible hidden distribution are supported for the same fixed linear target.

Parameterized CVXPY reuses the same symbolic problem while changing parameters
such as a radius. Tests compare parameterized and standard CVXPY backends on
the built-in divergence presets.

## Meaning of the Q Presets

Each Q preset defines an explicit admissible hidden-composition stress test.

| preset | admissible set |
| --- | --- |
| `observed` | only the observed retained hidden distribution |
| `saturated` | fixed public law, arbitrary hidden reweighting inside public fibers |
| `bounded_shift(r)` | fixed public law and `(1-r) q0(d) <= q(d) <= (1+r) q0(d)` |
| `tv_budget(r)` | fixed public law and `0.5 * ||q - q0||_1 <= r` |
| `chi_square_budget(r)` | fixed public law and `sum_d (q(d)-q0(d))^2 / q0(d) <= r` |
| `kl_budget(r)` | fixed public law and `sum_d q(d) log(q(d)/q0(d)) <= r` |
| `wasserstein(cost, r)` | fixed public law and transport cost from `q0` to `q` no larger than `r` |

Here `q0` is the observed retained hidden distribution. The presets do not
estimate which stress test is true. They make the stress test explicit and then
solve it exactly or numerically under that definition.

## Tabular Statistical Semantics

`from_dataframe(...)` compiles rows into the finite linear target contract.

- Hidden states are retained hidden cells.
- The hidden-state target `h(d)` is the weighted empirical mean of the target
  within hidden cell `d`.
- The observed hidden distribution `q0` is the normalized retained cell weight.
- The observed public law is the pushforward of `q0`.

This is statistically sound as a plug-in finite empirical analysis: the
optimization treats the compiled cell means and weights as the inputs being
audited.

It deliberately does not claim that those inputs are known without error. If
cell means are estimated, modeled, survey-weighted, or causal-effect estimates,
their standard errors, bootstrap intervals, survey-design uncertainty, or model
uncertainty should be reported separately. `updatesupport` separates that
uncertainty from hidden-composition ambiguity.

The `min_cell_weight` option changes the retained finite support. That can make
reports less sensitive to one-off sparse cells, but it also changes the
estimand to the retained support. Data diagnostics report dropped cells,
dropped mass, missing category encoding, singleton public fibers, and
constant-target fibers so that this choice is visible.

## Causal Inference Semantics

In causal workflows, `updatesupport` audits reported effects after a causal
estimator has produced row-level, subgroup-level, or hidden-cell-level effect
values.

The library does not:

- identify causal graphs,
- choose adjustment sets,
- estimate treatment effects,
- prove ignorability,
- correct interference, positivity, or measurement problems.

It does:

- take supplied effect estimates as fixed `h(d)` values,
- hold the public reporting distribution fixed,
- stress hidden composition under the selected `Q`,
- report how much the aggregate fixed-effect summary could move,
- keep causal estimate, statistical uncertainty, hidden-composition ambiguity,
  and refinement recommendations separate.

If the causal estimator is refit under each representation, or if the effect
function changes with the transported distribution, the current fixed-target
interpretation no longer applies. That workflow should be documented as a
procedure-level sensitivity analysis.

## Refinements, Sensitivity, and Frontiers

Refinement recommendations are deterministic re-runs of the same fixed-target
interval after promoting a candidate hidden column into the public
representation. A candidate is ranked by:

```text
baseline ambiguity - refined ambiguity
```

This is a reporting-stability recommendation. It is not a claim that the
variable is causal, a confounder, or operationally worth collecting.

Sensitivity reports evaluate the same public representation across a grid of
Q presets, sparse-cell thresholds, or hidden-state definitions. They summarize
how conclusions change across declared scenarios. The grid is not a posterior
distribution and does not assign probabilities to scenarios.

Public-representation frontier search compares candidate public bucket designs
by public-cell count and ambiguity. Exhaustive search is exact over the
declared candidate set. Greedy and beam search are heuristics; their reports
include trace metadata so the user can see when a frontier is approximate.

If target values are recomputed for each candidate representation, frontier
results compare reporting procedures rather than one fixed target transported
over different public projections. That is a useful but different claim.

## What Future Nonlinear Support Would Need

A sound nonlinear extension should make the target contract explicit, for
example:

```python
LinearTarget(h)                    # current behavior
RatioTarget(numerator, denominator)
MomentTransformTarget(moments, transform)
CvxpyTarget(objective_builder)
ProcedureTarget(compiler_callback)
```

Each target type would need its own adequacy condition, interval solver, witness
construction, report language, and tests. Some nonlinear targets are convex or
linear-fractional and can be solved cleanly. Some require mixed-integer
optimization or only admit conservative bounds. Some representation-dependent
procedures are best treated as scenario comparisons rather than transport
intervals.

Until that target-functional layer exists, nonlinear targets should either be:

- reduced explicitly to a fixed linear plug-in target,
- audited through externally computed scalar hidden-cell values with the
  limitation stated,
- kept out of the core soundness claim.

## Implementation Checks

The test suite exercises the current mathematical contract directly:

- saturated closed-form fiber-range formulas,
- fixed public-law saturated witnesses and zero-mass public fibers,
- finite-environment witness construction,
- linear-program local and global transport intervals,
- the line-segment no-least-support example,
- CVXPY custom convex constraints,
- TV, chi-square, KL, and Wasserstein budget presets,
- parameterized CVXPY equivalence to standard CVXPY on the preset grid,
- tabular compiler validation and diagnostics,
- report/refinement/frontier structured outputs.

The code also validates distribution normalization, nonnegative weights,
finite targets, public-law normalization, valid partition refinements, and
nonnegative divergence radii/costs.

## What Would Make a Claim Unsound

The reported interval can be misleading if:

- a nonlinear target is silently interpreted as a fixed linear target,
- a representation-dependent estimator is refit but described as the same
  transported target,
- the hidden state space omits important unseen states,
- hidden-cell target values are noisy but treated as exact without separate
  uncertainty reporting,
- the chosen Q preset is not a plausible stress test for the application,
- `min_cell_weight` drops substantively important cells,
- public buckets encode post-treatment or otherwise inappropriate variables in
  a causal workflow,
- greedy or beam frontier search is read as exhaustive.

These are modeling and interpretation risks, not hidden behavior in the
optimizer. The library is designed to make the finite support, target, public
projection, Q preset, diagnostics, and limitations inspectable.

## Bottom Line

`updatesupport` is sound for the question it currently asks:

> Conditional on the finite retained support, fixed hidden-cell target values,
> selected public representation, and explicit admissible hidden-composition
> class, how much can the fixed linear aggregate move while the public
> distribution is held fixed?

That is a well-defined finite optimization problem. The core library solves it
by closed form, linear programming, or convex optimization depending on `Q`, and
reports the result separately from statistical and causal uncertainty.

If the aggregate target is nonlinear or representation-dependent, the project
should say so explicitly and either reformulate it as a fixed linear target or
add a dedicated target-functional solver mode.
