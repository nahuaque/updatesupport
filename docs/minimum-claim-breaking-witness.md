# Minimum Claim-Breaking Witnesses

A threshold claim often needs a more direct answer than an endpoint interval:

> What is the smallest hidden-cell recomposition that would make this claim
> false while leaving every reported public-bucket share unchanged?

`minimum_claim_breaking_witness(...)` solves that inverse problem directly. It
starts at the observed hidden-cell law, preserves the observed public law, adds
a constraint that places the target on the failing side of the decision rule,
and minimizes the chosen distance from the observed law.

```python
import updatesupport as us

claim = us.claim(
    "experiment lift remains positive",
    public=["market", "plan"],
    hidden=["market", "plan", "channel", "tenure_band", "device"],
    target="treatment_minus_control_lift",
    weight="users",
    decision=us.threshold_decision(">=", 0.0),
)

witness = claim.breaking_witness(rows, distance="tv")
print(witness.to_markdown())
```

An existing audit can reuse its compiled state space:

```python
audit = claim.audit(rows)
witness = audit.breaking_witness(distance="tv")
```

The functional form is equivalent:

```python
witness = us.minimum_claim_breaking_witness(rows, claim, distance="tv")
```

## Distance Geometries

The first implementation supports:

- `tv`: exact linear programming through SciPy/HiGHS. The optimum is the total
  probability mass that must be reassigned. This is usually the clearest
  analyst-facing result.
- `l2`: a convex Euclidean-distance solve through CVXPY.
- `mahalanobis`: a convex CVXPY solve using a supplied positive-definite
  covariance matrix. Correlated or unusually scaled cell shifts can therefore
  receive an explicit geometry.

Install the optional backend for the latter two:

```bash
pip install "updatesupport[cvxpy]"
```

## What The Report Shows

The report separates the mathematical optimizer from its operational reading:

- the observed and witness target values and decisions,
- the globally minimum distance under the selected convex geometry,
- the witness's TV distance even when another geometry is optimized,
- public-law preservation error and solver status,
- every observed-to-witness hidden-cell mass change,
- a within-public-fiber transfer ledger pairing mass sources and destinations.

For TV distance, the transfer ledger has a direct interpretation. A row such as

```text
market=North: channel=direct -> channel=broker, mass=0.032
```

means that moving 3.2% of total retained mass between those hidden cells, while
holding the North public share fixed, is part of the closest claim-breaking
recomposition.

## Statuses

- `found`: a minimum margin-separated decision-breaking witness exists.
- `infeasible`: no recomposition over the retained hidden cells can break the
  claim while preserving the public law.
- `already_broken`: the observed value already fails the declared pass rule, so
  the minimum breaking distance is zero.

## Threshold Margin

Optimization solvers cannot represent strict inequalities directly. The solve
therefore targets a small positive separation from the threshold:

```python
witness = claim.breaking_witness(rows, threshold_margin=1e-6)
```

For a `value >= threshold` claim, the witness is constrained to
`value <= threshold - threshold_margin`. For a `value <= threshold` claim, it
is constrained to `value >= threshold + threshold_margin`. The report records
both the margin and the effective breaking boundary.

## Relation To Breakdown Points

`breakdown_point(...)` evaluates a nested forward family `Q(radius)` and searches
for the first radius whose target interval crosses a decision threshold. It is
the right tool when a radius already has a substantive stress-test meaning.

The minimum witness solves the inverse problem once. It asks for the closest
decision-flipping law in TV, L2, or Mahalanobis geometry and returns that law.
For TV and L2 radius families built from the same observed law and fixed-public
constraints, the optimum is the corresponding exact breakdown distance up to
the declared threshold margin.

## Scope And Limitations

The result is conditional on the chosen finer refinement and the retained
empirical support. `hidden` means observed by the analyst but not publicly
reported; it does not mean statistically unobserved.

The first implementation imposes fixed public marginals but does not intersect
the inverse problem with the claim's forward `Q` preset. That choice keeps the
question precise: how far away is the nearest breaking composition in the
selected geometry? Additional admissibility constraints can be layered into a
later generalization.

The result is deterministic composition sensitivity, not a confidence interval
and not the probability that the claim is false. An infeasible result means the
claim cannot be broken on the retained support under fixed public marginals; it
does not establish robustness to omitted variables, new hidden cells, target
estimation error, or every conceivable population shift.

