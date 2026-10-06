"""Forward/inverse consistency for shared convex Q constraints."""

from dataclasses import replace
import unittest

import pytest
import updatesupport as us

# Keep assertions active under optimized Python, matching the existing suite.
check = unittest.TestCase()


ROWS = [
    {"sector": "A", "issuer": "low", "target": 0, "weight": 75},
    {"sector": "A", "issuer": "high", "target": 1, "weight": 25},
]
STATES = (("A", "low"), ("A", "high"))


def claim(q="saturated", threshold=0.3):
    return us.claim(
        "negative share below limit",
        public=("sector",),
        hidden=("sector", "issuer"),
        target="target",
        weight="weight",
        min_cell_weight=0,
        q=q,
        decision=us.threshold_decision("<=", threshold),
    )


@pytest.mark.parametrize("distance", ["tv", "l2", "mahalanobis"])
def test_shared_q_respects_caps_for_each_distance(distance):
    pytest.importorskip("cvxpy")
    q = us.q_moment_bounds(
        {"high": dict(zip(STATES, (0.0, 1.0), strict=True))},
        upper={"high": 0.4},
        solver="CLARABEL",
    )
    kwargs = {"covariance": [[1, 0], [0, 1]]} if distance == "mahalanobis" else {}
    witness = claim(q).breaking_witness(
        ROWS, respect_q=True, distance=distance, threshold_margin=1e-6, **kwargs
    )
    check.assertTrue(
        witness.found and witness.respect_q and witness.q_name == "moment_bounds"
    )
    check.assertEqual(witness.witness_value, pytest.approx(0.300001, abs=1e-7))
    check.assertEqual(witness.witness_tv_distance, pytest.approx(0.050001, abs=1e-7))
    check.assertLess(witness.public_law_error, 1e-7)
    check.assertTrue(all(c.witness_mass >= 0 for c in witness.cells))
    check.assertIn("primary forward Q is imposed", witness.to_markdown())
    blocked = claim(q, threshold=0.45).breaking_witness(
        ROWS, respect_q=True, distance=distance, threshold_margin=1e-6, **kwargs
    )
    check.assertEqual(blocked.status, "infeasible")
    check.assertTrue(
        claim(q, threshold=0.45).breaking_witness(ROWS, distance="tv").found
    )


def test_tv_budget_and_observed_q_are_not_ignored():
    pytest.importorskip("cvxpy")
    for q in (us.q_tv_budget(0.01), "observed"):
        check.assertEqual(
            claim(q).breaking_witness(ROWS, respect_q=True).status, "infeasible"
        )
        check.assertTrue(claim(q).breaking_witness(ROWS).found)
    saturated = claim().breaking_witness(ROWS, respect_q=True)
    check.assertTrue(saturated.found and "scipy" in saturated.solver)


def test_invalid_baseline_is_rejected_even_if_headline_already_false():
    pytest.importorskip("cvxpy")
    q = us.q_moment_bounds(
        {"high": dict(zip(STATES, (0.0, 1.0), strict=True))}, upper={"high": 0.2}
    )
    with pytest.raises(ValueError, match="outside Q"):
        claim(q, threshold=0.1).breaking_witness(ROWS, respect_q=True)
    check.assertEqual(
        claim("observed", threshold=0.1).breaking_witness(ROWS, respect_q=True).status,
        "already_broken",
    )


def test_nonconvex_q_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="mixed-integer"):
        claim(us.q_fiber_support_floor(min_active=1, min_share=0.1)).breaking_witness(
            ROWS, respect_q=True
        )


def test_moment_bounds_validate_names_bounds_and_state_coverage():
    moments = {"high": dict(zip(STATES, (0.0, 1.0), strict=True))}
    for kwargs in (
        {},
        {"upper": {"typo": 0.4}},
        {"upper": {"high": float("nan")}},
        {"lower": {"high": 0.5}, "upper": {"high": 0.4}},
    ):
        with pytest.raises(ValueError):
            us.q_moment_bounds(moments, **kwargs)
    pytest.importorskip("cvxpy")
    q = us.q_moment_bounds({"missing": {STATES[0]: 1}}, upper={"missing": 0.4})
    with pytest.raises(ValueError, match="missing hidden states"):
        claim(q).breaking_witness(ROWS, respect_q=True)


def test_decision_only_design_labels_actual_decision_repair():
    spec = replace(claim(), candidate_refinements=("issuer",))
    report = us.design_public_report(spec, ROWS)
    markdown = report.to_markdown()
    check.assertIn("## Decision-Certifying Refinement", markdown)
    check.assertNotIn("Minimal Stable Frontier Candidate", markdown)
    check.assertNotIn("Minimal Ambiguity Frontier Candidate", markdown)
    explicit = us.design_public_report(replace(spec, ambiguity_limit=0), ROWS)
    check.assertIn("## Minimal Ambiguity Frontier Candidate", explicit.to_markdown())
