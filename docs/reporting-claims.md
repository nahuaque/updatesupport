# Reporting Claims

Claims are the highest-level review artifact in `updatesupport`. Instead of
choosing individual diagnostics up front, declare the aggregate claim you want
to defend and call `.design(...)` or `.audit(...)`.

`us.claim(...)` is the preferred constructor. `ClaimSpec` is the underlying
dataclass when you want to instantiate or serialize the spec directly, and
`PublicReportDesign` is the report object returned by `.design(...)`.

```python
import updatesupport as us

claim = us.claim(
    estimate_name="Income-threshold target rate",
    public=["AGE_BAND", "EDU_BAND", "SEX"],
    hidden=["AGE_BAND", "EDU_BAND", "SEX", "OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    target="income_over_threshold",
    weight="sample_weight",
    q_presets=[us.q_tv_budget(0.10), us.q_chi_square_budget(0.25)],
    candidate_refinements=["OCC_MAJOR", "WKHP_BAND", "RAC1P"],
    ambiguity_limit=0.015,
    decision=us.threshold_decision(
        ">=",
        0.12,
        label="income-threshold rate clears reporting floor",
    ),
    bucket_budget=40,
    statistical_interval=(0.119, 0.128),
)

design = claim.design(rows_or_frame)
print(design.to_markdown())
```

The audit-only form is still available when you just need the claim verdict:

```python
verdict = us.audit_claim(rows_or_frame, claim)
```

The auditor produces one report that separates:

- the reported causal or statistical estimate,
- statistical uncertainty, if supplied,
- hidden-composition ambiguity,
- decision invariance, if a threshold decision rule is supplied,
- a pass/fail/inconclusive verdict,
- a counterexample witness when the public representation is unstable,
- a repair representation when candidate refinements can stabilize the claim,
- claim-centered refinement recommendations and limitations.

## Public Report Design

Use `.design(...)` when the practical question is:

> What public representation should we publish so the claim is defensible?

```python
design = claim.design(
    rows_or_frame,
    action_costs={"OCC_MAJOR": 1.0, "WKHP_BAND": 0.5, "RAC1P": 2.0},
    include_attribution=True,
)
```

`PublicReportDesign` bundles:

- the current `ClaimAudit`,
- the cost-aware `ClaimRepairPlan`,
- the embedded representation certificate and frontier search, when available,
- optional Shapley-style refinement attribution,
- structured tables and Markdown.

Use `us.design_public_report(claim, rows_or_frame)` when a functional style is
more convenient. Use `.audit(...)` when you only need pass/fail/inconclusive
evidence; use `.design(...)` when the report needs a recommended public
representation.

## Verdict Semantics

The current public representation **passes** when its primary
hidden-composition ambiguity is no larger than `ambiguity_limit`, any supplied
decision rule is invariant over the hidden-composition interval, and any
requested representation certificate also passes.

It **fails** when the current representation exceeds the ambiguity limit or
when the hidden-composition interval crosses the supplied decision threshold.
If a repair representation is found, the report still marks the original claim
as failed and shows the repair separately. This distinction is intentional: the
original public representation did not support the claim, but the report tells
you how to make a defensible version.

It is **inconclusive** when neither an `ambiguity_limit` nor a decision rule is
supplied, or when the repair search is heuristic and `exact_required=True`.

## Decision Invariance

Use `decision=...` when the practical claim is a conclusion or action rather
than a raw interval width:

```python
claim = us.claim(
    estimate_name="Expected-loss estimate",
    public=["product", "region", "score_band"],
    hidden=["product", "region", "score_band", "channel", "vintage"],
    target="expected_loss",
    weight="ead",
    q_presets=[us.q_tv_budget(0.10)],
    decision=us.threshold_decision(
        "<=",
        0.025,
        label="expected loss within model-review tolerance",
    ),
    candidate_refinements=["channel", "vintage"],
)
```

For `pass_if <= threshold`, the decision is certified when the upper endpoint
of the hidden-composition interval is still at or below the threshold. It is
certified failed when the lower endpoint is already above the threshold. It is
not invariant when the interval crosses the threshold.

The auditor reports:

- the observed decision,
- lower- and upper-endpoint decisions,
- whether the decision is invariant,
- a decision-specific repair representation, if candidate refinements can make
  the observed decision invariant across all evaluated stress-test scenarios.

This is intentionally different from `ambiguity_limit`. A wide interval can
still support a decision if it stays on one side of the threshold, and a narrow
interval can fail decision invariance if it straddles the threshold.

## Minimum Claim-Breaking Witness

For a threshold claim, use `.breaking_witness(...)` to solve the inverse
question directly:

```python
witness = claim.breaking_witness(rows_or_frame, distance="tv")
same_witness = verdict.breaking_witness(distance="tv")
```

The result is the closest retained hidden-cell law that fails the decision while
preserving every public-bucket mass. Its transfer ledger turns the optimizer
into concrete within-fiber movements. TV gives minimum reassigned probability
mass; optional CVXPY-backed L2 and Mahalanobis modes provide alternate
geometries. See [Minimum claim-breaking witnesses](minimum-claim-breaking-witness.md)
for threshold-margin semantics and scope limitations.

## Claim-Centered Refinements

Use `verdict.recommend_refinements()` when you want the recommendation table in
Python:

```python
for row in verdict.recommend_refinements(top=5):
    print(row.label, row.after_ambiguity, row.reason)
```

These rows are not just the lower-level one-column ambiguity rankings. They
annotate whether a refinement:

- is the selected repair for the claim,
- makes a supplied decision rule invariant,
- satisfies the declared ambiguity limit,
- merely reduces ambiguity without repairing the claim.

This is the main consolidation step: refinement search is interpreted relative
to the claim being reviewed, not as a detached optimizer output.

## Claim Repair Plans

Use `verdict.repair_plan(...)` when you want the refinement evidence packaged
as an action list:

```python
plan = verdict.repair_plan(
    action_costs={"OCC_MAJOR": 1.0, "WKHP_BAND": 0.5, "RAC1P": 2.0},
    top=5,
)

print(plan.to_markdown())
```

The plan ranks candidate public-representation repairs by whether they certify
the claim, then by supplied action cost, resulting public-cell count, and
remaining ambiguity. It is useful when the report needs to answer the reviewer
question, "What should we publish differently?"

The function form audits first when given a claim spec, or reuses an existing
audit without another solve:

```python
plan = us.plan_claim_repair(claim, rows_or_frame)
same_plan = us.plan_claim_repair(verdict)
```

`ClaimRepairPlan` supports the same artifact shapes as other reports:
`as_dict()`, `to_json()`, `to_tables()`, `to_dataframes()`, and
`to_markdown()`.

## Exact Minimum Repairs

By default, claim audit uses exhaustive certificate search because it is
portable. When SCIP is installed and the Q presets are compatible convex
presets, request the exact-minimum repair path:

```python
claim = us.claim(
    estimate_name="Expected-loss estimate",
    public=["product", "region"],
    hidden=["product", "region", "score_band", "ltv_band", "channel"],
    target="expected_loss",
    weight="ead",
    q_presets=[us.q_tv_budget(0.10), us.q_kl_budget(0.05)],
    candidate_refinements=["score_band", "ltv_band", "channel"],
    ambiguity_limit=0.005,
    bucket_budget=40,
    search="mip_minimum",
)
```

This routes the certificate through the SCIP master plus support-function oracle
path and records the exact-minimum guarantee in the embedded certificate trace.

## Structured Output

Claim audit reports support:

```python
verdict.as_dict()
verdict.to_json()
verdict.to_tables()
verdict.to_dataframes()
```

The tables include a top-level `summary`, the serialized `claim`, decision
`reasons`, `limitations`, prefixed primary public-descent evidence, and prefixed
certificate or witness evidence when those components are present.

## Nested / Hierarchical Claims

Use `claim_tree(...)` when the review object has a natural hierarchy: overall
claim, region claims, site claims, subgroup claims, or posterior-summary claims
from a Bayesian hierarchical model.

```python
overall = us.claim(
    "Overall posterior mean treatment effect is launch-positive",
    public=["region"],
    hidden=["region", "site", "device"],
    target="posterior_mean_tau",
    weight="users",
    q_presets=[us.q_tv_budget(0.10)],
    decision=us.threshold_decision(">=", 0.0, label="effect is nonnegative"),
)

regional = us.claim(
    "Regional posterior means are stable",
    public=["region"],
    hidden=["region", "site", "device"],
    target="posterior_mean_tau",
    weight="users",
    q_presets=[us.q_tv_budget(0.10)],
    ambiguity_limit=0.01,
    candidate_refinements=["site", "device"],
)

tree = us.claim_tree(
    us.ClaimNode(overall, role="overall"),
    children=[us.ClaimNode(regional, role="region")],
    name="Hierarchical Treatment-Effect Claim Audit",
)

report = tree.audit(rows_or_frame)
print(report.to_markdown())
```

Each node is an ordinary `ClaimAudit`; the tree only coordinates and summarizes
the node-level audits. `ClaimTreeAudit` reports root status, node outcome
counts, highest-risk branches, and flat `summary`, `nodes`, `edges`, `reasons`,
and `limitations` tables.

For hierarchical Bayesian workflows, the posterior model stays upstream. Pass
posterior means, posterior-draw summaries, credible intervals, or draw-specific
cell targets into the data you audit. The report separates:

- the supplied posterior or statistical uncertainty,
- the hierarchy or pooling structure implicit in those supplied targets,
- hidden-composition ambiguity under the chosen public representation and Q
  stress test.

This means a root aggregate can pass while a child level fails. That is useful
for model review: the overall claim may be stable, but a regional, site-level,
or subgroup claim may need a finer public representation before it can be
reported defensibly.

## Multi-Claim Shared Representations

Use `claim_portfolio(...)` when several claims must use one common public
segmentation. Unlike `ClaimTree`, which coordinates independent node audits, a
portfolio searches one candidate representation and evaluates it against every
claim's own target, Q grid, ambiguity limit, decision rule, hidden-set grid, and
sparse-cell thresholds.

```python
portfolio = us.claim_portfolio(
    conversion_claim,
    uplift_claim,
    risk_claim,
    candidate_refinements=["channel", "tenure_band", "device"],
)

design = portfolio.design(rows, max_added_columns=2, bucket_budget=40)
```

See [Multi-claim shared representation design](shared-representation-design.md).

## Model-Assisted Joint Draws

For plausibility analysis, fit a nonparametric joint distribution and pass it to
the auditor:

```python
joint = us.fit_joint_distribution(
    rows_or_frame,
    public=claim.public,
    hidden=claim.hidden,
    target=claim.target,
    weight=claim.weight,
)

verdict = us.audit_claim(
    rows_or_frame,
    claim,
    joint_model=joint,
    joint_draws=500,
    joint_seed=123,
)
```

The report adds a model-assisted section summarizing ambiguity ranges, public
adequacy rates, and claim failure rates across fitted-joint draws. This is a
model-assisted plausibility check, not a replacement for the adversarial Q
interval. See [Model-assisted joint analysis](model-assisted-joint-analysis.md).
