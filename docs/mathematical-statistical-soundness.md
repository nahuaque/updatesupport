# Mathematical and Statistical Soundness

`updatesupport` is mathematically sound for a specific and explicit class of
audits:

> finite retained refinement cells, fixed public projections, target
> functionals fixed after workflow compilation, and explicit admissible
> distribution classes.

In that setting the reported ambiguity interval is a well-defined
partial-identification or sensitivity interval. It is not a confidence
interval, not a causal identification result, and not a generic nonlinear
functional optimizer.

For product positioning and lineage, see
[Positioning and lineage](positioning-and-lineage.md).

## Two Boundaries Up Front

First, "hidden" means **not publicly reported at the chosen reporting level**.
It does not mean unobserved, latent, unknowable, or missing from the analyst's
data. In the usual tabular workflow, hidden cells are retained finer cells that
the analyst has observed or constructed but has chosen not to publish in the
coarse public report. A better mnemonic is:

```text
public = reported buckets
hidden = retained finer refinement inside those buckets
```

Second, every ambiguity number is relative to a declared refinement and a
declared admissible class `Q`. It is not an absolute bound on all ways the world
could differ, all omitted variables, or all possible unmodeled populations.
Changing the retained refinement columns, sparse-cell rule, target values, public
projection, or `Q` changes the mathematical problem and can change the bound.

The most important technical boundary is the aggregate target. The primary tabular
contract compiles to a fixed linear plug-in target:

```text
psi(q) = sum_d h(d) q(d)
```

where `d` ranges over retained finer cells, `q(d)` is retained-cell mass, and
`h(d)` is the supplied target value for retained cell `d`.

`RatioTarget` adds an explicit fixed linear-fractional contract for supported
solver backends. `MomentTransformTarget` adds fixed transforms of linear
moments, with current solver support for affine transforms. `ProcedureTarget`
adds a workflow contract for representation-dependent reporting procedures: the
procedure is compiled to a column or row metric for each representation, and the
finite problem then solves the compiled fixed target.

`UncertainLinearTarget` keeps the same fixed linear point-estimate target and
adds retained-cell estimator standard errors. Those standard errors do not
change the base transport optimization. They are used in the report layer to
widen the hidden-composition interval with endpoint-specific and conservative
estimator-uncertainty margins.

If the aggregate target is nonlinear in `q` or depends on the transported
distribution itself, it is not automatically covered by the current soundness
guarantee. Such a target needs an explicit reformulation or a dedicated
target-functional backend.

## Core Object

The current mathematical object is finite.

- `D` is a finite retained refinement state space. These are the cells not
  shown in the public report, not necessarily unobserved cells.
- `pi: D -> O` maps retained cells to public report buckets.
- `h: D -> R` is a fixed retained-cell target value.
- `q` is a probability distribution over `D`.
- `Q` is the admissible class of retained-cell distributions.
- `psi(q) = sum_d h(d) q(d)` is the aggregate being audited.

The public law is the pushforward `pi#q`. For a fixed observed public law `p`,
the primary interval is:

```text
lower = inf { psi(q) : q in Q, pi#q = p }
upper = sup { psi(q) : q in Q, pi#q = p }
ambiguity = upper - lower
```

This is the hidden-composition ambiguity conditional on:

- the retained finite refinement support,
- the supplied cell-level target values,
- the public projection,
- the chosen admissible set `Q`.

Nothing in this definition says that `D` is the true complete state space. It
is the state space the audit has chosen to retain and stress. If substantively
important refinements are absent from `D`, the interval does not protect
against them.

## Target Contract

The default target contract is:

```text
retained cell d -> fixed scalar h(d)
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

The phrase "fixed scalar" matters. Once `from_dataframe(...)` compiles one
finite problem, `updatesupport` treats the retained-cell target values as
inputs.
It does not update them when `q` changes. If a `ProcedureTarget` is supplied,
procedure-aware workflows compile a fresh target for each public representation
or scenario before constructing that finite problem.

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

`MomentTransformTarget` makes this boundary explicit. If `g` is affine and the
target declares `affine_coefficients`, then:

```text
g(E_q[m_1], ..., E_q[m_k])
  = c + sum_j a_j E_q[m_j]
  = sum_d (c + sum_j a_j m_j(d)) q(d)
```

so the target is a fixed linear target in equivalent retained-cell form. Those
affine moment transforms support adequacy checks, interval solving, and
public-fiber decomposition.

Non-affine moment transforms are supported only when the declared mathematical
shape justifies the requested operation:

- if `g` is convex and a DCP-valid `cvxpy_transform` is supplied, minimization is
  a convex problem and the lower endpoint can be exact;
- if `g` is concave and a DCP-valid `cvxpy_transform` is supplied, maximization
  is a convex-compatible problem and the upper endpoint can be exact;
- if `g` is monotone in every bounded moment, the library can produce a
  conservative two-sided interval by optimizing each moment separately and
  applying `g` to the monotone box corners.

Those monotone box endpoints may not be jointly attainable. The resulting
interval is therefore valid as a conservative bound, not necessarily a sharp
identified interval. Non-affine moment transforms also do not claim public
adequacy or additive public-fiber decomposition support, because retained-cell
point ranges of `g(m(d))` are not, in general, the same object as variation in
`g(E_q[m])`.

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

`ProcedureTarget` supports this as a workflow-level object. The compiler
receives a `ProcedureTargetContext` containing the public columns, hidden
columns, weights, sparse-cell threshold, and Q preset, then returns a concrete
column name or `RowMetric`. Each compiled finite problem is still a fixed-target
transport problem. Comparisons across refinements, sensitivity scenarios, or
frontier candidates are therefore procedure-comparison sensitivity analyses,
not transport intervals for one unchanged target.

### Composition-Dependent Structural Targets

If the target value itself depends on the transported retained-cell distribution:

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
reweighting among retained cells inside each public fiber.

For the current fixed linear target, let:

```text
range(o) = max_{d: pi(d)=o} h(d) - min_{d: pi(d)=o} h(d)
```

Then the observed-law saturated ambiguity is exactly:

```text
sum_o p(o) range(o)
```

The lower endpoint puts each public bucket's mass on the lowest-target retained
cell in that bucket. The upper endpoint puts the same public mass on the
highest-target retained cell. Because public buckets are disjoint and the
public law is fixed, the bucket-level contributions add. This is why the report
can decompose saturated ambiguity into public-fiber contributions:

```text
contribution(o) = p(o) range(o)
ambiguity = sum_o contribution(o)
```

For saturated environments with an explicitly fixed public law, public fibers
with zero admissible mass are irrelevant to adequacy. Target variation in a
zero-mass public fiber cannot move the aggregate because no admissible
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
admissible retained-cell distribution are supported for the same fixed linear
target.

Parameterized CVXPY reuses the same symbolic problem while changing parameters
such as a radius. Tests compare parameterized and standard CVXPY backends on
the built-in divergence presets.

## Meaning of the Q Presets

Each Q preset defines an explicit admissible retained-cell composition stress
test.

| preset | admissible set |
| --- | --- |
| `observed` | only the observed retained-cell distribution |
| `saturated` | fixed public law, arbitrary retained-cell reweighting inside public fibers |
| `bounded_shift(r)` | fixed public law and `(1-r) q0(d) <= q(d) <= (1+r) q0(d)` |
| `tv_budget(r)` | fixed public law and `0.5 * ||q - q0||_1 <= r` |
| `chi_square_budget(r)` | fixed public law and `sum_d (q(d)-q0(d))^2 / q0(d) <= r` |
| `kl_budget(r)` | fixed public law and `sum_d q(d) log(q(d)/q0(d)) <= r` |
| `wasserstein(cost, r)` | fixed public law and transport cost from `q0` to `q` no larger than `r` |

Here `q0` is the observed retained-cell distribution. The presets do not
estimate which stress test is true. They make the stress test explicit and then
solve it exactly or numerically under that definition.

## Tabular Statistical Semantics

`from_dataframe(...)` compiles rows into the finite linear target contract.

- Hidden states are retained finer cells, meaning cells not shown in the public
  report.
- The retained-cell target `h(d)` is the weighted empirical mean of the target
  within retained cell `d`.
- The observed retained-cell distribution `q0` is the normalized retained cell
  weight.
- The observed public law is the pushforward of `q0`.

This is statistically sound as a plug-in finite empirical analysis: the
optimization treats the compiled cell means and weights as the inputs being
audited.

It deliberately does not claim that those inputs are known without error. If
cell means are estimated, modeled, survey-weighted, or causal-effect estimates,
their standard errors, bootstrap intervals, survey-design uncertainty, or model
uncertainty should be accounted for explicitly.

When retained-cell estimator standard errors are supplied through
`target_standard_error=...` or `effect_standard_error=...`, `updatesupport`
constructs an `UncertainLinearTarget`. The base interval remains:

```text
lower = inf { sum_d mu(d) q(d) : q in Q, pi#q = p }
upper = sup { sum_d mu(d) q(d) : q in Q, pi#q = p }
```

where `mu(d)` is the retained-cell point estimate. The report then adds:

```text
se(q) = sqrt(sum_d (se(d) q(d))^2)
```

assuming independent retained-cell target-estimation errors after compilation.
If lower and upper witness distributions are available, the report evaluates
`se(q)` at those endpoint witnesses and widens each endpoint by
`confidence_multiplier * se(q)`.

The report also emits a conservative fixed-public-law outer interval using:

```text
se_conservative = sqrt(sum_o (p(o) max_{d: pi(d)=o} se(d))^2)
```

This bound is deliberately wider when the retained cell that maximizes
estimator standard error is not the same retained cell that maximizes or
minimizes the point estimate. It is not an exact joint nonconvex solve over both
retained-cell composition and target-estimation error. It is a transparent
reporting adjustment that keeps statistical uncertainty and hidden-composition
ambiguity visible as separate quantities.

When the selected Q backend is CVXPY-compatible, the report may also include an
SOCP confidence-core diagnostic:

```text
core_lower = sup { mu(q) - z se(q) : q in Q, pi#q = p }
core_upper = inf { mu(q) + z se(q) : q in Q, pi#q = p }
```

The lower-core problem is a concave maximization and the upper-core problem is a
convex minimization, so both are disciplined convex/SOCP-compatible. This
diagnostic computes the intersection of all composition-specific
estimator-adjusted confidence bands. It is not an outer partial-identification
interval. A nonempty core means every admissible retained-cell composition
shares a common estimator-adjusted value range. An empty core means
retained-cell composition shift can separate the estimator-adjusted bands.

The `min_cell_weight` option changes the retained finite support. That can make
reports less sensitive to one-off sparse cells, but it also changes the
estimand to the retained support. Data diagnostics report dropped cells,
dropped mass, missing category encoding, singleton public fibers, and
constant-target fibers so that this choice is visible.

## Causal Inference Semantics

In causal workflows, `updatesupport` audits reported effects after a causal
estimator has produced row-level, subgroup-level, or retained-cell-level effect
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
- stress retained-cell composition under the selected `Q`,
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
Q presets, sparse-cell thresholds, or retained refinement definitions. They summarize
how conclusions change across declared scenarios. The grid is not a posterior
distribution and does not assign probabilities to scenarios.

Public-representation frontier search compares candidate public bucket designs
by public-cell count and ambiguity. Exhaustive search is exact over the
declared candidate set. Greedy and beam search are heuristics; their reports
include trace metadata so the user can see when a frontier is approximate.

If target values are recomputed for each candidate representation, frontier
results compare reporting procedures rather than one fixed target transported
over different public projections. `ProcedureTarget` makes that claim explicit
in report metadata.

## Ratio Targets

`RatioTarget(numerator, denominator)` represents the fixed linear-fractional
functional:

```text
psi(q) = sum_d n(d) q(d) / sum_d w(d) q(d)
```

The current ratio slice is intentionally narrow and explicit:

- denominator values must be strictly positive on retained cells;
- `FiniteEnvironments` can evaluate ratios directly on enumerated admissible
  distributions;
- `PublicFiberSaturated` solves fixed-public-law ratio extrema exactly with a
  Charnes-Cooper linear-program transform;
- `CvxpyEnvironments` solves local and fixed-public-law ratio extrema with
  CVXPY disciplined quasiconvex programming (`solve(qcp=True)`);
- if the denominator is constant over the state space, the ratio reduces to a
  fixed linear target and can use the existing linear LP/CVXPY backends;
- variable-denominator ratios under the finite-linear LP backend,
  parameterized CVXPY backend, or pairwise same-support-law global CVXPY
  problems without a fixed public law still raise `UnsupportedTargetError`.

For a fixed public law `p`, the saturated ratio interval solves:

```text
lower = inf { (n . q) / (w . q) : q in Q_sat, pi#q = p }
upper = sup { (n . q) / (w . q) : q in Q_sat, pi#q = p }
```

The Charnes-Cooper transform sets `y = q / (w . q)` and
`tau = 1 / (w . q)`. The ratio objective becomes linear:

```text
n . y
```

and the constraints become:

```text
w . y = 1
sum_{d in pi^{-1}(o)} y_d = p_o tau
y >= 0, tau >= 0
```

This is a linear program, so the fixed-public saturated ratio interval is an
exact partial-identification interval for the declared finite problem.

Ratio targets do not have the same additive public-fiber decomposition as
linear targets. Reports therefore treat public-fiber tables for ratio targets
as point-range diagnostics, not as additive shares of total ambiguity.

## What Future Nonlinear Support Would Need

A sound nonlinear extension should make the target contract explicit, for
example:

```python
LinearTarget(h)
UncertainLinearTarget(mu, se)  # linear point target with report-level SE widening
RatioTarget(numerator, denominator)  # supported for finite/saturated Q
MomentTransformTarget(moments, transform)  # affine, one-sided CVXPY, monotone boxes
CvxpyTarget(objective_builder)
```

Each target type would need its own adequacy condition, interval solver, witness
construction, report language, and tests. Affine moment transforms are already
reduced to fixed linear retained-cell target values; convex/concave moment
transforms expose only the DCP-compatible endpoint; monotone moment transforms
can expose conservative box bounds. Some remaining nonlinear targets are
convex or linear-fractional and can be solved cleanly. Some require
mixed-integer optimization or only admit conservative bounds.
Representation-dependent procedures are currently treated as scenario
comparisons through `ProcedureTarget`, not as a single transported nonlinear
functional.

Until a broader target-functional layer exists, unsupported nonlinear targets
should either be:

- reduced explicitly to a fixed linear plug-in target,
- audited through externally computed scalar retained-cell values with the
  limitation stated,
- kept out of the core soundness claim.

The current API includes guardrails for this boundary. `UnsupportedTarget` can
be used as an explicit marker for quantile, distributional, or other nonlinear
target objects that are not supported. Passing one into `FiniteProblem` or
`from_dataframe(...)` raises `UnsupportedTargetError` instead of silently
treating it as a linear target. Passing an uncompiled `ProcedureTarget` directly
to `FiniteProblem` also raises; pass it to `from_dataframe(...)` or a
procedure-aware workflow instead.

## Implementation Checks

The test suite exercises the current mathematical contract directly:

- saturated closed-form fiber-range formulas,
- saturated ratio-target intervals via Charnes-Cooper LP,
- affine moment-transform targets and capability guardrails for nonlinear
  moment transforms,
- uncertain linear targets, including retained-cell standard-error aggregation
  estimator-uncertainty-aware report intervals, and SOCP confidence-core
  diagnostics,
- procedure-target compilation and recompilation across tabular, report, and
  frontier workflows,
- fixed public-law saturated witnesses and zero-mass public fibers,
- finite-environment witness construction,
- linear-program local and global transport intervals,
- the line-segment no-least-support example,
- CVXPY DQCP local/fixed-public-law ratio intervals,
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
- a representation-dependent estimator is refit but described as one unchanged
  transported target rather than a `ProcedureTarget` workflow,
- the retained refinement space omits important relevant cells,
- retained-cell target values are noisy but treated as exact without separate
  uncertainty reporting,
- the chosen Q preset is not a plausible stress test for the application,
- `min_cell_weight` drops substantively important cells,
- public buckets encode post-treatment or otherwise inappropriate variables in
  a causal workflow,
- greedy or beam frontier search is read as exhaustive.

These are modeling and interpretation risks, not undocumented behavior in the
optimizer. The library is designed to make the finite support, target, public
projection, Q preset, diagnostics, and limitations inspectable.

## Bottom Line

`updatesupport` is sound for the question it currently asks:

> Conditional on the finite retained support, fixed retained-cell target values,
> selected public representation, and explicit admissible retained-cell composition
> class, how much can the fixed linear aggregate move while the public
> distribution is held fixed?

That is a well-defined finite optimization problem. The core library solves it
by closed form, linear programming, or convex optimization depending on `Q`, and
reports the result separately from statistical and causal uncertainty.

If the aggregate target is nonlinear or representation-dependent, the project
should say so explicitly and either reformulate it as a fixed supported target,
use `ProcedureTarget` for procedure-comparison workflows, or add a dedicated
target-functional solver mode.
