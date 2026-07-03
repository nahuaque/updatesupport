# Theory and Backends

This page collects details that are useful once you want to understand or
extend the computational model behind `updatesupport`.

For a quick product overview, installation instructions, and first report, see
the top-level [README](../README.md).

For a compact audit of assumptions, guarantees, and limitations, see
[Mathematical and statistical soundness](mathematical-statistical-soundness.md).

## Core Model

`updatesupport` implements a finite-state computational version of the
update-relevant support machinery from
[Update-Relevant Support: Hume's Missing Descent](https://philpapers.org/go.pl?id=BRUUSH&proxyId=&u=https%3A%2F%2Fphilpapers.org%2Farchive%2FBRUUSH.pdf).

It models:

- a finite hidden state space `D`
- a public projection `pi: D -> O`
- a fixed target functional, currently either a linear target
  `psi(q) = <h, q>` or a ratio target
  `psi(q) = <n, q> / <w, q>`
- optional procedure-target workflows that compile a representation-dependent
  target into one of those fixed target functionals before solving
- an admissible environment class `Q`

The library checks whether a public or refined representation is adequate and
quantifies remaining ambiguity among admissible hidden distributions that share
the same public law.

## Main Python Surface

Core finite objects:

- `FiniteProblem`
- `LinearTarget`
- `MomentTransformTarget`
- `ProcedureTarget`
- `ProcedureTargetContext`
- `RatioTarget`
- `TargetCapabilities`
- `TargetContract`
- `UnsupportedTarget`
- `UnsupportedTargetError`
- `Partition`
- `PublicFiberSaturated`
- `FiniteEnvironments`
- `LineSegment`
- `PolytopeEnvironments`
- `CvxpyEnvironments`
- `ConvexAdmissibleSet`
- `CvxpyAdmissibleSetSpec`
- `SupportFunctionBackend`
- `SupportFunctionIntervalResult`
- `SupportFunctionResult`
- `ParameterizedCvxpyEnvironments`

Tabular and reporting helpers:

- `from_dataframe(...)`
- `DataDiagnostic`
- `DataDiagnostics`
- `AuditSpec`
- `AuditRun`
- `QSpec`
- `run_audit(...)`
- `report_to_json(...)`
- `report_tables(...)`
- `report_dataframes(...)`
- `public_descent_report(...)`
- `sensitivity_report(...)`
- `recommend_refinements(...)`
- `recommend_refinements_sensitivity(...)`
- `public_representation_frontier(...)`

Causal and estimator helpers:

- `audit_effects(...)`
- `causal_reporting_stability(...)`
- `adapt_dataframe_effects(...)`
- `adapt_econml_effects(...)`
- `adapt_dowhy_effects(...)`
- `adapt_doubleml_effects(...)`
- `audit_dowhy_effects(...)`
- `dowhy_refutation_from_report(...)`

Built-in Q presets:

- `saturated`
- `observed`
- `bounded_shift`
- `tv_budget`
- `chi_square_budget`
- `kl_budget`
- `wasserstein`

## Finite Linear Backend

`PolytopeEnvironments` uses `scipy.optimize.linprog` for finite-state
environment classes described by linear equality and inequality constraints:

```python
import updatesupport as us

problem = us.FiniteProblem(
    states=["a", "b"],
    public={"a": "o", "b": "o"},
    estimand={"a": 0.0, "b": 4.0},
    environments=us.PolytopeEnvironments(
        constraints=[
            us.geq({"a": 1.0}, 0.25),
            us.geq({"b": 1.0}, 0.25),
        ]
    ),
)

result = problem.global_transport_modulus()

print(result.lower, result.upper, result.diameter)
# 1.0 3.0 2.0
```

The simplex constraints are implicit. Additional constraints can be supplied
with `us.leq(...)`, `us.geq(...)`, `us.eq(...)`, or
`us.linear_constraint(...)`.

Ratio targets with a constant denominator over the finite state space reduce to
linear targets and can use this backend. Variable-denominator `RatioTarget`
objects intentionally raise `UnsupportedTargetError` here until a
linear-fractional LP transform is added for general finite-linear constraints.

## Ratio Targets

`RatioTarget` represents a fixed linear-fractional estimand:

```python
import updatesupport as us

target = us.RatioTarget(
    numerator={"low": 1.0, "high": 4.0, "anchor": 10.0},
    denominator={"low": 1.0, "high": 2.0, "anchor": 5.0},
    name="loss_ratio",
)

problem = us.FiniteProblem(
    states=["low", "high", "anchor"],
    public={"low": "A", "high": "A", "anchor": "B"},
    estimand=target,
    environments=us.PublicFiberSaturated.fixed({"A": 0.5, "B": 0.5}),
)

interval = problem.global_transport_modulus()
print(interval.lower, interval.upper, interval.diameter)
```

For `PublicFiberSaturated` with fixed public marginals, ratio intervals are
solved exactly using a Charnes-Cooper linear-program transform. With unfixed
public marginals, the global saturated result checks each one-public-fiber law
and returns the worst ratio diameter. `FiniteEnvironments` also supports ratio
targets because it evaluates the target directly on enumerated distributions.

`CvxpyEnvironments` supports variable-denominator ratio targets for local
transport and for global transport when `fixed_public_law` is set. This uses
CVXPY's disciplined quasiconvex programming path (`solve(qcp=True)`) for
affine-over-positive-affine ratio objectives.

Current ratio limitations:

- denominator values must be strictly positive on retained states;
- public-fiber contribution tables are point-range diagnostics, not additive
  decompositions of the ratio interval;
- `PolytopeEnvironments`, `LineSegment`, and
  `ParameterizedCvxpyEnvironments` require a fixed linear target unless the
  ratio denominator is constant over the state space;
- `CvxpyEnvironments` ratio support does not yet cover pairwise
  same-support-law global optimization without a fixed public law.

## Procedure Targets

`ProcedureTarget` is for reporting procedures whose row-level target depends on
the chosen public representation. The procedure is not passed directly to
`FiniteProblem`; `from_dataframe(...)` calls the compiler with a
`ProcedureTargetContext` and expects a column name or `RowMetric` back.

```python
import updatesupport as us

def compile_target(context: us.ProcedureTargetContext) -> us.RowMetric:
    scale = len(context.public)
    return us.row_metric(
        f"scaled_score_x{scale}",
        lambda row: scale * float(row["score"]),
        columns=("score",),
        description="score scaled by public representation size",
    )

target = us.ProcedureTarget(
    "representation_scaled_score",
    compile_target,
    description="representation-dependent score procedure",
)

report = us.public_descent_report(
    rows,
    public=["segment"],
    hidden=["segment", "driver"],
    target=target,
)
```

Procedure-aware workflows such as `recommend_refinements(...)`,
`sensitivity_report(...)`, and `public_representation_frontier(...)` re-run the
compiler for each candidate representation or scenario. Read those results as
procedure-comparison sensitivity analyses. Inside each solved finite problem,
the compiled target is still fixed after compilation.

## Moment Transform Targets

`MomentTransformTarget` represents targets of the form:

```text
psi(q) = g(E_q[m_1], ..., E_q[m_k])
```

where each `m_j` is a fixed hidden-state moment. The current solver support is
deliberately capability-based:

- affine transforms reduce to a fixed linear target and support the usual
  adequacy, interval, and public-fiber decomposition APIs;
- convex transforms with `cvxpy_transform` support exact minimization through a
  CVXPY-backed environment, but maximization is generally nonconvex;
- concave transforms with `cvxpy_transform` support exact maximization through a
  CVXPY-backed environment, but minimization is generally nonconvex;
- monotone transforms with every moment marked `increasing` or `decreasing`
  support conservative interval bounds by optimizing each moment separately and
  applying the transform to the resulting moment box.

```python
import updatesupport as us

target = us.MomentTransformTarget(
    moments={
        "score": {"low": 0.0, "high": 2.0},
        "risk": {"low": 1.0, "high": 3.0},
    },
    transform=lambda m: 1.0 + 3.0 * m["score"] - 0.5 * m["risk"],
    affine_coefficients={"score": 3.0, "risk": -0.5},
    intercept=1.0,
    name="affine_score_risk",
)

problem = us.FiniteProblem(
    states=["low", "high"],
    public={"low": "bucket", "high": "bucket"},
    estimand=target,
)
```

The target contract keeps `kind="moment_transform"` and exposes capability
flags through `TargetCapabilities`. Affine moment transforms default to
adequacy, interval, and public-fiber decomposition support. Non-affine moment
transforms can support interval bounds, but they do not support public-adequacy
or additive public-fiber decomposition diagnostics unless a future target
backend supplies a valid decomposition.

For nonlinear moment transforms, use `partial_identification_interval(public_law)`
or an environment with a fixed public law. One-sided exact endpoints are exposed
through `moment_transform_endpoint(minimize=True/False)`. If only one exact
endpoint is convex-compatible, add monotonicity declarations to get a
two-sided conservative interval.

Use `RatioTarget` for linear-fractional targets, `ProcedureTarget` for
representation-dependent procedures, and a future nonlinear backend for
distributional statistics such as quantiles, Gini coefficients, calibrated
indices without valid curvature/monotonicity declarations, or threshold metrics.

## Convex CVXPY Backend

`CvxpyEnvironments` supports the same finite-state simplex and linear
constraints, plus custom convex constraints over the state-probability vector:

```python
import updatesupport as us

def cap_b(_cp, q, _states, state_index):
    return (q[state_index["b"]] <= 0.75,)

problem = us.FiniteProblem(
    states=["a", "b"],
    public={"a": "o", "b": "o"},
    estimand={"a": 0.0, "b": 1.0},
    environments=us.CvxpyEnvironments(
        fixed_public_law={"o": 1.0},
        constraint_builders=(cap_b,),
    ),
)
```

The TV, chi-square, KL, and Wasserstein Q presets are wrappers around this
backend. Use CVXPY when admissible hidden shifts are convex but not just
finite-linear constraints.

Solved CVXPY transport intervals expose dual diagnostics:

```python
interval = grouped.problem.global_transport_modulus()
for row in interval.dual_summary(top=5):
    print(row.solve, row.name, row.kind, row.magnitude)
```

These rows are CVXPY/KKT sensitivity diagnostics. Large multipliers identify
constraints that are locally influential for the solved interval, such as
public-law equalities, Q-budget constraints, or active state lower bounds.
Custom constraint builders can return `us.cvxpy_constraint(...)` to attach a
readable name and kind to their dual rows.

### Support-Function Backend

`SupportFunctionBackend` exposes the admissible hidden-distribution set as a
CVXPY support function. For a fixed linear target `psi(q) = <h, q>`, the
transport interval can be computed as:

```text
upper = sigma_Q(h)
lower = -sigma_Q(-h)
```

where `sigma_Q(y) = sup_{q in Q} <y, q>`.

```python
env = us.SupportFunctionBackend(
    fixed_public_law={"North": 1.0},
)

problem = us.FiniteProblem(
    states=["low", "high"],
    public={"low": "North", "high": "North"},
    estimand={"low": 0.0, "high": 1.0},
    environments=env,
)

interval = problem.global_transport_modulus()
```

You can also inspect the set directly:

```python
q_set = env.convex_admissible_set(problem, public_law={"North": 1.0})
result = q_set.support_value([0.0, 1.0])
interval = q_set.support_interval([0.0, 1.0])
```

Compatible built-in convex presets can opt into this backend:

```python
q = us.q_tv_budget(0.10, backend="support_function")
```

Those same presets also expose their admissible-set constraints before a backend
is chosen:

```python
spec = us.cvxpy_admissible_set_spec(
    us.q_tv_budget(0.10),
    public_law=grouped.public_law,
    public_map=grouped.problem.public_map,
    cell_weights=grouped.cell_weights,
)

q_set = spec.convex_admissible_set(grouped.problem)
upper = q_set.support_value(
    [grouped.problem.estimand_map[state] for state in grouped.problem.states]
)
interval = spec.support_interval(grouped.problem)
```

`CvxpyAdmissibleSetSpec` can materialize the same constraints as a standard
CVXPY environment, parameterized CVXPY environment, batched CVXPY environment,
or support-function backend. This keeps the mathematical admissible set shared
across solver modes.

`support_interval(...)` returns a `SupportFunctionIntervalResult` with the
lower value, upper value, diameter, and the lower/upper optimizer vectors. For a
linear target direction `h`, it evaluates `sigma_Q(h)` and `sigma_Q(-h)`.

This first support-function slice is for continuous convex Q sets and fixed
linear targets. Mixed-integer presets such as `q_fiber_support_floor(...)` and
variable-denominator `RatioTarget` problems continue to use their existing
solver paths.

## SCIP Solver Selection

The CVXPY backend uses CVXPY's default installed solver unless a solver is
named explicitly. To route a CVXPY-backed environment through SCIP, install the
SCIP extra:

```bash
pip install "updatesupport[scip]"
# or
uv add "updatesupport[scip]"
```

Then pass `solver="SCIP"` through a CVXPY-backed Q preset:

```python
grouped = us.from_dataframe(
    rows,
    public=["region"],
    hidden=["region", "occupation", "channel"],
    target="outcome",
    q=us.q_tv_budget(0.10, solver="SCIP"),
)
```

For hand-built problems, pass the same solver name to any CVXPY environment:

```python
env = us.CvxpyEnvironments(
    fixed_public_law={"North": 0.4, "South": 0.6},
    solver="SCIP",
)
```

`solver_options` are forwarded to `cvxpy.Problem.solve(...)`, so any
CVXPY-supported SCIP options can be supplied there.

The first built-in mixed-integer preset is `q_fiber_support_floor(...)`:

```python
q = us.q_fiber_support_floor(2, min_share=0.10)
```

It keeps the observed public law fixed but adds binary active-cell indicators
inside each public fiber. Each positive-mass public fiber must keep at least
`min_active` hidden cells carrying at least `min_share` of that public fiber's
mass. This is useful when the saturated stress test is too permissive because it
allows a public bucket to collapse onto a single hidden subgroup. Mixed-integer
solves generally do not produce the same dual diagnostics as continuous convex
solves; use their primal intervals and witness distributions as the main
diagnostic evidence.

SCIP can also power public-representation search:

```python
frontier = us.public_representation_frontier(
    rows,
    base_public=["region"],
    hidden=["region", "occupation", "channel"],
    target="outcome",
    candidate_refinements=["occupation", "channel"],
    q_presets=["saturated"],
    search="mip",
)
```

This MIP mode is a column-selection optimizer for saturated fixed-public-law
ambiguity. It is not a generic mixed-integer wrapper around every Q preset. Use
it when the design question is which named hidden columns to promote into the
public representation under saturated stress tests.

For convex Q presets, use the MIP master / support-function oracle mode:

```python
frontier = us.public_representation_frontier(
    rows,
    base_public=["region"],
    hidden=["region", "occupation", "channel"],
    target="outcome",
    candidate_refinements=["occupation", "channel"],
    q_presets=[us.q_tv_budget(0.10)],
    ambiguity_limit=0.02,
    bucket_budget=20,
    search="mip_oracle",
)
```

Here SCIP proposes public representations under the discrete search constraints,
then each proposal is evaluated by the support-function oracle for the declared
convex Q grid. Failing proposals receive no-good cuts. This is useful when a
saturated master is too conservative, but you still want a solver-guided search
over public-report designs.

When the design question is the exact minimum representation satisfying an
ambiguity limit, use `search="mip_minimum"`:

```python
frontier = us.public_representation_frontier(
    rows,
    base_public=["region"],
    hidden=["region", "occupation", "channel"],
    target="outcome",
    candidate_refinements=["occupation", "channel"],
    q_presets=[us.q_tv_budget(0.10), us.q_chi_square_budget(0.25)],
    ambiguity_limit=0.02,
    bucket_budget=20,
    search="mip_minimum",
    minimum_objective="public_cells",
)
```

This mode keeps the convex Q evaluation in the support-function oracle, while
the SCIP master enumerates public representations in increasing objective
order. The first oracle-stable proposal certifies the exact minimum under the
declared objective (`public_cells` or `added_columns`), search bounds, and hard
bucket constraints.

## Parameterized CVXPY Sweeps

For repeated radius sweeps on the same compiled finite problem, use the
parameterized backend:

```python
grouped = us.from_dataframe(
    rows_or_frame,
    public=["AGE_BAND", "SEX"],
    hidden=["AGE_BAND", "SEX", "OCC_MAJOR"],
    target="__target__",
    q=us.q_tv_budget(0.10, backend="parameterized_cvxpy"),
)

first = grouped.problem.global_transport_modulus()
grouped.problem.environments.set_parameter("radius", 0.20)
second = grouped.problem.global_transport_modulus()
```

`ParameterizedCvxpyEnvironments` caches the CVXPY problem and updates CVXPY
parameters for the objective, public law, and preset radius.

## Batched CVXPY Local Intervals

`BatchedCvxpyEnvironments` solves multiple fixed-public-law local interval
problems together with variables shaped like:

```text
q[scenario, state]
```

This is useful for sensitivity-grid rows that share the same retained hidden
state space and public projection. The public API is:

```python
intervals = env.batched_local_transport(problem, public_laws)
```

The sensitivity-report workflow opts in when adjacent Q presets use
`backend="batched_cvxpy"`. Current batched solves reuse existing one-dimensional
CVXPY constraint builders per scenario; future tensor-specialized builders can
make Wasserstein-style couplings use shapes such as
`gamma[scenario, source_state, target_state]`.

## Theory Example: No Least Support

The finite poset of adequate supports need not have a least element.

```python
import updatesupport as us

problem = us.FiniteProblem(
    states=["a", "b", "c"],
    public={"a": "o", "b": "o", "c": "o"},
    estimand={"a": 0.0, "b": 1.0, "c": 2.0},
    environments=us.LineSegment(
        center={"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
        direction={"a": 0.0, "b": 1.0, "c": -1.0},
        radius=1 / 3,
    ),
)

least = problem.least_support()

print(least.exists)
# False

for support in least.minimal_supports:
    print(support.format())
# {{a, c}, {b}}
# {{a, b}, {c}}
```

## Future Transport Types

Experimental relational transport types such as Gromov-Wasserstein make sense
only when the application supplies comparable hidden-state geometries. For most
current use cases, start with the built-in Q presets described in
[Transport presets](transport-presets.md).
