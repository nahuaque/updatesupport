# updatesupport

Are your observed categories good enough for the estimate you are reporting?

`updatesupport` is a Python library for representation adequacy and
transport-stability auditing. It asks whether a coarse public representation,
such as age band, education band, and sex, is enough to determine an aggregate
estimate once hidden composition inside those public cells is allowed to vary.

The motivating workflow is simple:

1. Choose the public categories you would report.
2. Choose hidden variables that refine those public categories.
3. Choose the target rate or linear estimand you care about.
4. Stress test the estimate while holding the public distribution fixed.
5. Report how much the answer could move, which public cells drive the movement,
   and which extra variables would reduce the ambiguity.

This is useful when a table, dashboard, policy analysis, or model evaluation
reports aggregates over coarse categories and you want to know whether those
categories are stable enough for the estimate being reported.

## Plain-English Example

In the Folktables ACSIncome demo, the public categories are:

```text
AGE_BAND x EDU_BAND x SEX
```

The hidden variables include occupation, class of worker, weekly-hours band,
race, marital status, birthplace, and relationship status. The observed target
rate is `12.37%`: the share of sampled people exceeding the ACSIncome income
threshold.

The stress test keeps the public mix fixed but allows hidden composition inside
each public cell to change. Under that stress test, the target rate could range
from:

```text
11.79% to 13.44%
```

The width, `1.65` percentage points, is the transport ambiguity. It means that
hidden composition changes could move the aggregate rate by up to about `1.65`
percentage points even when the public demographic mix is held fixed.

That interval is not a confidence interval. It does not measure sampling error.
It measures sensitivity to hidden composition under the chosen stress test.

See [docs/folktables-acs-income-interpretation.md](docs/folktables-acs-income-interpretation.md)
for the analyst-facing interpretation of this result.

## What This Is

`updatesupport` is an audit layer for finite representations. It helps answer:

- Are the public categories adequate for this estimand?
- If not, how large is the remaining ambiguity?
- Which public cells contribute most to the ambiguity?
- Which hidden variables would make the public representation more stable?

It is not a causal inference package, a sampling-uncertainty estimator, or a
replacement for substantive modeling. It can complement those workflows by
checking whether the categories used to report an estimate are too coarse.

For causal workflows, use DoWhy, EconML, CausalML, or DoubleML to estimate or
validate causal effects, then use `updatesupport` to audit whether the public
categories used to report those effects are stable to hidden composition changes.
See [docs/causal-library-integration.md](docs/causal-library-integration.md).

```python
report = us.audit_effects(
    df,
    public=["AGE_BAND", "SEX"],
    hidden=["AGE_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    effect="tau_hat",
    weight="sample_weight",
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    q=us.q_bounded_shift(0.5),
)
```

## Install Locally

```bash
uv sync
uv run python -m unittest
```

For the Folktables examples:

```bash
uv sync --extra examples
```

For the EconML causal example:

```bash
uv sync --extra causal
```

For DoWhy `CausalRefutation` conversion:

```bash
uv sync --extra dowhy
```

## Core Model

The library implements a finite-linear version of the update-support machinery
from `docs/UpdateSupport.pdf`. It models:

- a finite hidden state space `D`
- a public projection `pi: D -> O`
- a linear estimand `psi(q) = <h, q>`
- an admissible environment class `Q`

The library then checks whether a public or refined support is adequate and
quantifies the remaining ambiguity among admissible environments that share the
same public law.

## Tabular Compiler

Use `from_dataframe(...)` to compile a pandas-like dataframe or iterable of row
mappings into a finite problem:

```python
import updatesupport as us

grouped = us.from_dataframe(
    rows_or_frame,
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND"],
    target="__target__",
    weight="PWGTP",
    min_cell_weight=25,
    q="saturated",
)

interval = grouped.problem.global_transport_modulus()

print(grouped.public_law)
print(interval.lower, interval.upper, interval.diameter)
```

Each retained hidden cell becomes one finite state. The estimand value for that
state is the weighted empirical target mean inside the cell, and the environment
fixes the observed public law while allowing saturated reweighting inside public
fibers.

## Q Presets

`Q` is the admissible environment class used for the hidden-composition stress
test. The built-in presets are:

- `q="saturated"` or `us.q_saturated()`: fix the observed public law and allow
  arbitrary reweighting among retained hidden cells inside each public cell.
- `q=us.q_bounded_shift(radius)`: fix the observed public law and constrain each
  hidden-cell mass to stay within `(1 +/- radius)` times its observed mass.
- `q=us.q_tv_budget(radius)`: fix the observed public law and constrain total
  variation distance from the observed hidden distribution. This uses the
  optional CVXPY backend.
- `q=us.q_chi_square_budget(radius)`: fix the observed public law and constrain
  Pearson chi-square divergence from the observed hidden distribution. This uses
  the optional CVXPY backend.
- `q=us.q_kl_budget(radius)`: fix the observed public law and constrain KL
  divergence from the observed hidden distribution. This uses the optional CVXPY
  backend.
- `q=us.q_wasserstein(cost, radius)`: fix the observed public law and constrain
  Wasserstein distance from the observed hidden distribution using an explicit
  hidden-cell cost matrix. This uses the optional CVXPY backend.
- `q="observed"` or `us.q_observed()`: use only the observed hidden distribution,
  giving zero hidden-composition ambiguity.

Install the CVXPY extra before using TV, chi-square, KL, or Wasserstein presets:

```bash
uv sync --extra cvxpy
```

See [docs/transport-presets.md](docs/transport-presets.md) for guidance on
which preset to use, how to choose radii, and how to interpret sensitivity
tables.

## Public Descent Report

Use `public_descent_report(...)` to produce a structured report and render it as
Markdown:

```python
report = us.public_descent_report(
    rows_or_frame,
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND"],
    target="__target__",
    weight="PWGTP",
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND"],
    min_cell_weight=25,
    q="saturated",
    title="ACSIncome Representation Adequacy Report",
)

print(report.to_markdown())
```

The report includes the observed value, stress interval, transport ambiguity,
public adequacy flag, worst public fibers, and one-column refinement candidates
with before ambiguity, after ambiguity, absolute reduction, and percentage
reduction.

## Sensitivity Checks

Use `sensitivity_report(...)` to rerun the audit across Q presets,
`min_cell_weight` thresholds, and alternative hidden-column sets:

```python
sensitivity = us.sensitivity_report(
    rows_or_frame,
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND"],
    target="__target__",
    weight="PWGTP",
    min_cell_weights=[1, 10, 25],
    q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
)

print(sensitivity.to_markdown())
```

This is the recommended way to check whether the headline ambiguity is sensitive
to sparse hidden cells or to the chosen admissible-environment preset. The
Markdown output starts with a scenario summary, highlights the lowest- and
highest-ambiguity scenarios, flags mixed public-adequacy conclusions, and then
renders the full scenario table.

When the grid contains repeated TV, chi-square, KL, or Wasserstein presets that
differ only by radius, the sensitivity routines automatically route those rows
through the parameterized CVXPY backend and reuse the compiled problem for the
fixed hidden state space.

Use `recommend_refinements_sensitivity(...)` to rank candidate public
refinements across the same kind of grid:

```python
refinements = us.recommend_refinements_sensitivity(
    rows_or_frame,
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="__target__",
    weight="PWGTP",
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    min_cell_weights=[1, 10, 25],
    q_presets=["saturated", us.q_bounded_shift(0.5), "observed"],
)

print(refinements.to_markdown())
```

The aggregate ranking reports mean reduction, worst-case reduction, rank
stability, and the number of scenarios where each refinement ranked first.

## Public-Fiber-Saturated Example

When all reweightings inside public fibers are admissible, the transport
modulus has the closed form:

```text
Omega(p; psi) = sum_o p(o) * (max_{u in pi^-1(o)} h(u) - min_{u in pi^-1(o)} h(u))
```

```python
import updatesupport as us

problem = us.FiniteProblem(
    states=["a", "b", "c", "d"],
    public={"a": "x", "b": "x", "c": "y", "d": "y"},
    estimand={"a": 0.0, "b": 1.0, "c": 0.0, "d": 3.0},
    environments=us.PublicFiberSaturated(),
)

print(problem.is_public_adequate())
# False

print(problem.fiber_ranges())
# {"x": 1.0, "y": 3.0}

print(problem.global_transport_modulus().diameter)
# 3.0

print(problem.local_transport_modulus({"x": 0.25, "y": 0.75}).diameter)
# 2.5
```

## How To Read A Report

A typical report should separate four ideas:

- **Observed value**: the estimate under the observed hidden composition.
- **Stress interval**: the possible estimate range after hidden composition is
  varied within the chosen environment class `Q`.
- **Transport ambiguity**: the width of that interval.
- **Refinement value**: how much ambiguity would shrink if another hidden
  variable were added to the public representation.

The stress interval is a partial-identification or stability interval, not a
statistical confidence interval. If the interval is wide, the public categories
do not determine the estimate very tightly under the chosen stress test. If the
interval is narrow, the estimate is comparatively stable to the modeled hidden
composition changes.

## Folktables ACS Worked Example

The Folktables example turns ACSIncome or ACSEmployment into an update-support
stress test:

- public cells are coarse observed categories such as age band, education band,
  and sex
- hidden cells refine those categories with occupation, race, work hours, and
  other task-specific ACS fields
- the estimand is the observed label rate in each hidden cell
- the environment class allows arbitrary reweighting inside the observed public
  cells while preserving the observed public law

Run the real Folktables ACSIncome example:

```bash
uv run --extra examples python examples/folktables_acs.py \
  --task income \
  --states CA \
  --year 2018 \
  --sample 50000 \
  --min-cell-weight 25
```

Run ACSEmployment instead:

```bash
uv run --extra examples python examples/folktables_acs.py \
  --task employment \
  --states CA TX \
  --year 2018
```

The script prints:

- the observed target rate
- the partial-identification interval under hidden reweighting
- the observed-law transport ambiguity
- a statistical interpretation of the interval and ambiguity
- worst public fibers by ambiguity contribution
- one-column refinements ranked by ambiguity reduction, including before/after
  ambiguity and percentage reduction

There is also a no-download smoke demo:

```bash
uv run python examples/folktables_acs.py --synthetic
```

There is also a causal-effect reporting example. It fits an EconML CATE
estimator, computes `tau_hat = estimator.effect(X)`, then audits whether that
effect is stable when reported by coarse public categories:

```bash
uv run --extra examples --extra causal python examples/folktables_acs_causal.py \
  --task income \
  --states CA \
  --year 2018 \
  --sample 50000
```

The no-download version is:

```bash
uv run --extra causal python examples/folktables_acs_causal.py --synthetic
```

The built-in first stage uses EconML `CausalForestDML`. In a real causal
workflow, swap in the DoWhy, EconML, CausalML, or DoubleML estimator that fits
your identification strategy and produces a `tau_hat` effect target; the
`updatesupport` stage is the same.

For DoWhy workflows, use `audit_dowhy_effects(...)` to package the
representation audit with the DoWhy estimate, then call `audit.to_refutation()`
to produce a DoWhy `CausalRefutation` object when the optional DoWhy dependency
is installed.

## Current Python Surface

Implemented now:

- `FiniteProblem`
- `Partition`
- `PublicFiberSaturated`
- `FiniteEnvironments`
- `LineSegment`
- `PolytopeEnvironments` via SciPy `linprog`
- `CvxpyEnvironments` for convex finite-environment restrictions
- `ParameterizedCvxpyEnvironments` for cached CVXPY radius sweeps
- `from_dataframe(...)` for compiling grouped tabular data into a finite problem
- Q presets: `saturated`, `observed`, `bounded_shift`, `tv_budget`,
  `chi_square_budget`, `kl_budget`, and `wasserstein`
- `PublicDescentReport` with Markdown output
- `public_descent_report(...)` for analyst-facing report objects
- `audit_effects(...)` for causal/uplift effect-reporting stability audits
- `audit_dowhy_effects(...)` and `dowhy_refutation_from_report(...)` for DoWhy
  workflows
- `recommend_refinements(...)` for ranking candidate hidden variables
- `recommend_refinements_sensitivity(...)` for aggregating refinement value
  across Q, hidden-set, and sparsity scenarios
- `sensitivity_report(...)` for robustness grids over Q, hidden sets, and
  `min_cell_weight`
- adequacy checks with witnesses
- adequate, minimal, and least support enumeration for small finite problems
- local and global transport moduli
- partial-identification intervals
- cardinal gaps when a least support exists
- simple Markdown reports

Planned next slices:

- experimental transport types such as Gromov-Wasserstein once the comparison
  object is a pair of relational hidden-state geometries rather than one fixed
  hidden-cell cost matrix

## Linear-Polytope Backend

`PolytopeEnvironments` uses `scipy.optimize.linprog` for finite-linear
environment classes:

```python
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
with `us.leq(...)`, `us.geq(...)`, `us.eq(...)`, or `us.linear_constraint(...)`.

## CVXPY Backend

`CvxpyEnvironments` supports the same finite-state simplex and linear
constraints, plus custom convex constraints over the state-probability vector:

```python
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

The TV, chi-square, KL, and Wasserstein Q presets are thin wrappers around this
backend.

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
parameters for the objective, public law, and preset radius. It is useful when
you are sweeping radii for TV, chi-square, KL, or Wasserstein budgets on a fixed
state space.

## Theory Example: No Least Support

The finite poset of adequate supports need not have a least element.

```python
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

## More Documentation

- [Representation adequacy guide](docs/representation-adequacy.md)
- [Transport preset guide](docs/transport-presets.md)
- [Using `updatesupport` with causal inference libraries](docs/causal-library-integration.md)
- [Folktables ACSIncome result interpretation](docs/folktables-acs-income-interpretation.md)
- [UpdateSupport PDF](docs/UpdateSupport.pdf)
