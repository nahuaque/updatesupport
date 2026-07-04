# Breakdown Point Analysis

Breakdown point analysis scans a nested family `Q(radius)` and finds the
smallest radius where a threshold-style claim stops being certified.

This is useful when the reported number supports a concrete action:

- a fairness metric is acceptable only if disparity is below a threshold,
- an experiment ships only if uplift exceeds a threshold,
- a model-risk estimate passes review only if expected loss stays below a
  tolerance,
- an evaluation benchmark is acceptable only if a score clears a bar.

## What The Radius Means

The radius has the meaning assigned by the chosen Q family. For example:

- `bounded_shift`: per-fiber movement away from the observed hidden-cell mix,
- `tv_budget`: total-variation budget around the observed hidden-cell law,
- `chi_square_budget`: chi-square divergence budget,
- `kl_budget`: KL-divergence budget,
- `l2_budget`: Euclidean budget.

The search assumes the family is nested: larger radii permit at least the shifts
allowed by smaller radii. The result is always relative to the chosen hidden
refinement and chosen Q family. It is not an absolute robustness guarantee.

## Minimal Example

```python
import updatesupport as us

report = us.breakdown_point(
    rows,
    public=["age_band", "sex"],
    hidden=["age_band", "sex", "channel", "tenure_band"],
    target="uplift",
    weight="n",
    decision=us.threshold_decision(">=", 0.01, label="ship if uplift >= 1pp"),
    q_family="bounded_shift",
    radius_max=0.5,
    tolerance=1e-4,
)

print(report.to_markdown())
```

The report status has three possible values:

- `found`: a finite breakdown radius was found inside the search range,
- `not_found`: the decision stayed stable through `radius_max`,
- `already_broken`: the decision was not stable even at `radius_min`.

## Interpretation

If the observed estimate is `0.014`, the decision rule is `value >= 0.01`, and
the breakdown radius is `0.22`, the plain-English interpretation is:

> The reported decision survives hidden recomposition up to roughly radius 0.22
> under this Q family. By radius 0.22, at least one admissible hidden mix can move
> the interval across the decision threshold, so the coarse public report no
> longer certifies the decision.

This is a deterministic composition-stability diagnostic, not a confidence
interval.

## Structured Exports

Breakdown reports support the same downstream artifact patterns as the rest of
the framework:

```python
payload = report.to_json()
tables = report.to_tables()
frames = report.to_dataframes()
```

The curve table contains one row per grid radius with the lower endpoint, upper
endpoint, ambiguity width, endpoint decisions, and a `decision_stable` flag.

## Caveats

- The result depends on the chosen finer refinement. If the relevant
  composition variable is absent from the data, breakdown analysis cannot bound
  its effect.
- The result depends on the chosen Q family. A large breakdown radius under one
  family does not imply robustness under every plausible stress test.
- For non-linear targets, the same target-capability guardrails used by
  `public_descent_report()` still apply.
