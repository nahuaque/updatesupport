# Experimental ResidOpt Backend

`updatesupport` can optionally call
[`residopt`](https://github.com/nahuaque/residopt) as an endpoint compiler for
some repeated robust-optimization subproblems. This is an experimental
integration, not a default dependency.

The first supported path is an L2 hidden-composition stress test:

```python
import updatesupport as us

grouped = us.from_dataframe(
    rows,
    public=["region"],
    hidden=["region", "channel", "tenure_band"],
    target="metric",
    weight="weight",
    q=us.q_l2_budget(0.05),
)

report = us.residopt_l2_support_interval(grouped)
print(report.to_markdown())
```

For repeated endpoint evaluations, build a reusable compiler once:

```python
compiler = us.ResidOptL2EndpointCompiler.from_grouped(grouped)

primary = compiler.interval()
channel_direction = {
    state: 1.0 if state[-1] == "paid" else 0.0
    for state in grouped.problem.states
}
channel = compiler.interval(direction=channel_direction)

print(compiler.compiled_template_count)
print(compiler.support_solve_count)
```

The compiler caches the public-incidence nullspace and a parameterized residopt
SOCP template. The first interval builds the template; later intervals update
the projected direction parameter and solve the same compiled problem.

During local development with a sibling checkout, run with:

```bash
PYTHONPATH=../residopt/src python your_script.py
```

or check availability directly:

```python
import updatesupport as us

print(us.residopt_available().as_dict())
```

## What It Compiles

For a fixed public law and observed hidden distribution `q0`, the adapter writes
hidden-distribution shifts as

```text
q = q0 + delta
```

and enforces public-law equality by projecting `delta` into the nullspace of the
public-incidence matrix. For an L2 radius, the endpoint support calculation is:

```text
sup <direction, delta>
subject to A delta = 0
           ||delta||_2 <= radius
```

After the nullspace projection this becomes an ellipsoidal support-function
problem. `residopt` compiles that support atom to an SOCP certificate.

## Certificate Semantics

The current adapter preserves public-law equality and the L2 radius, but it
drops hidden-cell nonnegativity. That means:

- the `residopt` certificate is exact for the compiled ellipsoidal support atom;
- the returned interval is conservative for the original simplex-constrained
  `updatesupport` endpoint;
- if a claim survives this conservative interval, it also survives the tighter
  exact endpoint under the same L2 radius;
- if the interval is too wide, use the standard CVXPY backend to solve the exact
  simplex-constrained problem.

This distinction is visible in the report:

```python
report.exact_for_updatesupport_q
report.conservative_for_updatesupport_q
report.to_tables()["certificates"]
```

## Why This Matters

The most promising use is repeated endpoint evaluation: refinement ranking,
frontier search, claim repair, and other workflows that evaluate many candidate
public representations. `residopt` gives `updatesupport` a place to route
subproblems through compiled support-function certificates, oracle decisions,
and timing metadata without making the core package depend on a separate
compiler. Use `ResidOptL2EndpointCompiler` when many target directions share the
same fixed public representation and L2 radius.

## Current Limits

This first slice supports fixed linear and uncertain-linear point-estimate
targets. Nonlinear targets, ratio targets, procedure targets, nonnegativity-exact
certificates, and cached compiled models are future integration points.
