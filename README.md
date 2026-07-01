# updatesupport

Finite causal support adequacy and transport ambiguity in Python.

`updatesupport` is a small first cut of the finite-linear machinery from
`docs/UpdateSupport.pdf`. It models:

- a finite hidden state space `D`
- a public projection `pi: D -> O`
- a linear estimand `psi(q) = <h, q>`
- an admissible environment class `Q`

The library then checks whether a public or refined support is adequate and
quantifies the remaining ambiguity among admissible environments that share the
same public law.

## Install locally

```bash
uv sync
uv run python -m unittest
```

## Public-fiber-saturated example

When all reweightings inside public fibers are admissible, the transport modulus
has the closed form:

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

## No-least-support example

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

## Linear-polytope backend

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

## Folktables ACS worked example

The Folktables example turns ACSIncome or ACSEmployment into an update-support
stress test:

- public cells are coarse observed categories such as age band, education band,
  and sex
- hidden cells refine those categories with occupation, race, work hours, and
  other task-specific ACS fields
- the estimand is the observed label rate in each hidden cell
- the environment class allows arbitrary reweighting inside the observed public
  cells while preserving the observed public law

Install the optional example dependencies:

```bash
uv sync --extra examples
```

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
- worst public fibers by ambiguity contribution
- one-column refinements ranked by ambiguity reduction

There is also a no-download smoke demo:

```bash
uv run python examples/folktables_acs.py --synthetic
```

## MVP scope

Implemented now:

- `FiniteProblem`
- `Partition`
- `PublicFiberSaturated`
- `FiniteEnvironments`
- `LineSegment`
- `PolytopeEnvironments` via SciPy `linprog`
- adequacy checks with witnesses
- adequate/minimal/least support enumeration for small finite problems
- local/global transport moduli
- partial-identification intervals
- cardinal gaps when a least support exists
- simple Markdown reports

Not implemented yet:

- CVXPY backend
- pandas/YAML loaders
- plotting
