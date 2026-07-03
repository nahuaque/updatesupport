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
- `ProcedureTarget`
- `ProcedureTargetContext`
- `RatioTarget`
- `TargetContract`
- `UnsupportedTarget`
- `UnsupportedTargetError`
- `Partition`
- `PublicFiberSaturated`
- `FiniteEnvironments`
- `LineSegment`
- `PolytopeEnvironments`
- `CvxpyEnvironments`
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
