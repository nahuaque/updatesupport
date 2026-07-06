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

## Claim Screening

Claim audits can opt into a screen-then-certify path:

```python
claim = us.claim(
    "reported lift remains positive",
    public=["segment"],
    hidden=["segment", "channel", "tenure_band"],
    target="lift",
    weight="users",
    q=us.q_l2_budget(0.05),
    decision=us.threshold_decision(">=", 0.0),
    screening_backend="residopt",
)

audit = claim.audit(rows)
print(audit.screening.as_dict())
```

When the conservative residopt interval already proves the decision rule or
ambiguity limit, the audit returns without running the exact endpoint solve. If
the screen is unavailable or inconclusive, the audit falls back to the ordinary
exact primary solver and records the screening attempt in `audit.screening`.

This mode is intentionally conservative: it can certify passes early, but it
does not use a wide conservative interval to fail a claim without the exact
fallback.

During local development with a sibling checkout, run with:

```bash
PYTHONPATH=../residopt/src python your_script.py
```

or check availability directly:

```python
import updatesupport as us

print(us.residopt_available().as_dict())
```

## Cached Refinement Screening

Refinement and frontier workflows evaluate many candidate public
representations. The experimental refinement-screening context keeps that
workflow explicit:

```python
context = us.ResidOptRefinementScreenContext(
    rows,
    public=["region"],
    hidden=["region", "channel", "tenure_band"],
    target="metric",
    weight="weight",
    q=us.q_l2_budget(0.05),
)

screen = context.screen(
    candidate_refinements=["channel", "tenure_band"],
    ambiguity_limit=0.01,
    exact_fallback=True,
)

print(screen.to_markdown())
```

For each candidate representation, the context:

- compiles or reuses a `GroupedProblem`;
- compiles or reuses one `ResidOptL2EndpointCompiler`;
- evaluates the conservative residopt interval;
- skips the exact CVXPY endpoint if the conservative interval is already within
  the supplied ambiguity limit;
- otherwise runs the exact `updatesupport` endpoint when `exact_fallback=True`.

The report separates these outcomes:

```python
screen.certified_count
screen.exact_solve_count
screen.exact_solve_avoided_count
screen.to_tables()["candidates"]
```

Use `us.residopt_refinement_screen(...)` for a one-shot helper, or keep a
`ResidOptRefinementScreenContext` alive when running several related screens
over the same dataset.

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
certificates, and exact-support certificates with hidden-cell nonnegativity are
future integration points.
