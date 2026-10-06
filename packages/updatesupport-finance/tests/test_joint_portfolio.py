"""Provider-free multi-target coverage, report repairs, and offline contracts."""

import json
from dataclasses import replace
import unittest

import pytest
import updatesupport as us
import updatesupport_finance as f

check = unittest.TestCase()
AS_OF = "2026-10-06T12:00:00Z"


def book(*, positions=None, labels=None, value_metric=False):
    positions = positions or [
        f.PortfolioPosition("a1", 30, "A"),
        f.PortfolioPosition("a2", 10, "A"),
        f.PortfolioPosition("b", 20, "B"),
        f.PortfolioPosition("c", 20, "C"),
        f.PortfolioPosition("u", 20, "U"),
        f.PortfolioPosition("cash", 10, security_type="cash"),
    ]
    universe = f.PortfolioUniverse(positions, AS_OF, "USD")
    obs = []
    for concept, values in (
        ("OCF", {"A": 1, "B": -1, "C": -1}),
        ("GAP", {"A": -1, "B": 1}),
    ):
        for issuer, value in values.items():
            fact = f.DisclosureFact(
                issuer + concept,
                issuer,
                concept,
                value,
                "currency",
                "2025-12-31",
                "2026-02-01T00:00:00Z",
                "filing",
                "https://example.org/filing",
                currency="USD",
            )
            obs.append(f.FundamentalObservation(fact, "FY"))
    taxonomy = [
        f.TaxonomyAssignment(
            i,
            (labels or {}).get(
                i,
                {"sector": "S", "age": "young" if i == "A" else "old", "industry": "I"},
            ),
            "test",
        )
        for i in ("A", "B", "C", "U")
    ]
    policies = {
        "ocf": f.PortfolioMetricPolicy("OCF", "currency"),
        "gap": f.PortfolioMetricPolicy("GAP", "currency"),
    }
    if value_metric:
        policies["ocf"] = f.PortfolioMetricPolicy(
            "OCF", "currency", transform="value", currency="USD", target_bounds=(-2, 2)
        )
    return f.compile_portfolio_metrics(
        universe, observations=obs, taxonomy=taxonomy, policies=policies
    )


def report(evidence=None, **kw):
    config = dict(
        public=("sector",),
        hidden=("sector", "issuer_id", "age", "industry"),
        candidate_refinements=("age",),
    )
    config.update(kw)
    return f.joint_portfolio_report(evidence or book(), **config)


def test_join_keeps_share_classes_and_known_nonjoint_contribution():
    b = book()
    check.assertEqual(len(b.metrics["ocf"].rows), 4)
    check.assertEqual(len(b.metrics["gap"].rows), 3)
    check.assertEqual(len(b.rows), 3)
    check.assertEqual(b.common_value, 60)
    s = b.metric_scope("ocf")
    check.assertEqual(s["fixed_known_contribution"], 20)
    check.assertEqual(s["missing_evidence_floor"], 0.2)
    check.assertEqual(b.metric_scope("gap")["missing_evidence_floor"], 0.4)
    check.assertEqual(b.lift_interval("ocf", (0, 1)), (0.2, 1))
    other = b.metrics["gap"]
    universe = replace(other.universe, currency="EUR")
    changed = replace(
        other, universe=universe, coverage=replace(other.coverage, universe=universe)
    )
    with pytest.raises(ValueError, match="identical universes"):
        f.JointPortfolioEvidence({"ocf": b.metrics["ocf"], "gap": changed})
    changed = replace(other, rows=[dict(r, age="changed") for r in other.rows])
    with pytest.raises(ValueError, match="descriptors"):
        f.JointPortfolioEvidence({"ocf": b.metrics["ocf"], "gap": changed})


def test_raw_frontier_and_scoped_contract_do_not_hide_missing_evidence():
    raw = report()
    check.assertEqual(raw.status, "raw_frontier")
    check.assertIsNone(raw.selected)
    check.assertEqual([c["public_cells"] for c in raw.candidates], [1, 2])
    detailed = raw.candidates[1]
    check.assertAlmostEqual(detailed["metrics"]["ocf"]["common_width"], 0)
    check.assertAlmostEqual(detailed["metrics"]["ocf"]["eligible_width"], 0.2)
    check.assertAlmostEqual(detailed["metrics"]["gap"]["eligible_width"], 0.4)
    common = report(scope="common", ambiguity_limits={"ocf": 0.01, "gap": 0.01})
    check.assertEqual(common.selected["public_cells"], 2)
    eligible = report(ambiguity_limits={"ocf": 0.01, "gap": 0.01})
    check.assertIsNone(eligible.selected)
    check.assertIn("common book", raw.to_markdown())
    check.assertEqual(
        json.loads(raw.to_json())["search"]["cost_universe"],
        "positive-value common book",
    )
    with pytest.raises(ValueError, match="budget"):
        report(max_evaluations=1)
    with pytest.raises(ValueError, match="target"):
        report(hidden=("sector", "ocf"), candidate_refinements=())
    check.assertEqual(
        report(
            scope="common", decisions={"ocf": us.threshold_decision("<", 0.9)}
        ).selected["public_cells"],
        2,
    )


def test_conditional_drill_checks_all_metrics_and_q_and_witness_scope():
    b = book()
    suggestion = us.suggest_conditional_refinements(
        b.rows,
        public=("sector",),
        targets=tuple(b.metrics),
        candidate_columns=("age", "issuer_id"),
        weight="weight",
    )
    r = report(
        b,
        candidate_refinements=(),
        conditional_refinements=suggestion.candidates,
        q_presets={
            "ocf": ("saturated", us.q_tv_budget(0.1)),
            "gap": ("observed", "saturated"),
        },
    )
    check.assertEqual(len(r.candidates), 2)
    check.assertEqual(r.candidates[1]["public_cells"], 2)
    check.assertEqual(len(r.candidates[0]["metrics"]["ocf"]["scenarios"]), 2)
    w = r.breaking_witness("ocf", decision=us.threshold_decision("<", 0.5))
    check.assertEqual(w["scope"], "common")
    check.assertTrue(w["witness"]["respect_q"])
    check.assertAlmostEqual(w["transferred_value"], 10, delta=1e-6)
    w = r.breaking_witness("ocf", decision=us.threshold_decision("<", 0.5), q_index=1)
    check.assertEqual(w["witness"]["status"], "infeasible")


def actions():
    return [
        f.EvidenceAcquisition("C gap", ("c",), {"gap": 0}),
        f.EvidenceAcquisition("U package", ("u",), {"ocf": 0, "gap": 0}),
    ]


def test_research_plans_preserve_ties_availability_and_bounded_metric_narrowing():
    r = report()
    plan = f.plan_portfolio_repairs(
        r, actions(), ambiguity_limits={"ocf": 0.01, "gap": 0.01}
    )
    check.assertEqual(plan.as_dict()["evaluations"], 8)
    check.assertEqual(len(plan.feasible_frontier), 1)
    best = plan.feasible_frontier[0]
    check.assertEqual(best["public_cells"], 2)
    check.assertEqual(best["package_count"], 2)
    check.assertIsNone(best["evidence_cost"])
    unavailable = [actions()[0], replace(actions()[1], availability="unavailable")]
    check.assertEqual(
        f.plan_portfolio_repairs(
            r, unavailable, ambiguity_limits={"ocf": 0.01, "gap": 0.01}
        ).feasible_frontier,
        (),
    )
    frontier = f.plan_portfolio_repairs(
        r,
        [*actions(), replace(actions()[1], name="U alternative")],
        ambiguity_limits={"ocf": 0.01, "gap": 0.01},
    ).feasible_frontier
    check.assertEqual(len(frontier), 2)
    narrowed = f.plan_portfolio_repairs(
        report(book(value_metric=True)),
        [
            f.EvidenceAcquisition(
                "U partial", ("u",), {"ocf": 1}, availability="available", cost=2
            )
        ],
        ambiguity_limits={"ocf": 0.3},
    )
    check.assertAlmostEqual(narrowed.plans[-1]["eligible_widths"]["ocf"], 0.2)
    priced_report = report(disclosure_costs={"age": 5})
    priced = f.plan_portfolio_repairs(
        priced_report,
        [replace(a, cost=2) for a in actions()],
        ambiguity_limits={"ocf": 0.01, "gap": 0.01},
    )
    check.assertEqual(priced.feasible_frontier[0]["disclosure_cost"], 5)
    check.assertEqual(priced.feasible_frontier[0]["evidence_cost"], 4)
    with pytest.raises(ValueError, match="budget"):
        f.plan_portfolio_repairs(
            r, actions(), ambiguity_limits={"ocf": 1}, max_evaluations=1
        )


def test_snapshot_roundtrip_offline_and_definition_guards():
    r = report(scope="common", ambiguity_limits={"ocf": 0.01, "gap": 0.01})
    snap = f.capture_joint_portfolio_snapshot(
        r, capture_references={"holdings": "a" * 64}
    )
    restored = f.JointPortfolioSnapshot.from_json(snap.to_json())
    check.assertEqual(snap.fingerprint, restored.fingerprint)
    check.assertTrue(restored.replay_matches())
    check.assertFalse(any(f.compare_joint_portfolio_snapshots(snap, restored).values()))
    bad = snap.as_dict()
    bad["result"]["status"] = "invented"
    with pytest.raises(ValueError, match="digest"):
        f.JointPortfolioSnapshot.from_json(json.dumps(bad))
    bad = snap.as_dict()
    bad["evidence"]["metrics"]["gap"]["rows"][0]["metric"] = 99
    with pytest.raises(ValueError, match="recompilation"):
        f.JointPortfolioSnapshot.from_json(json.dumps(bad))
    frozen = f.FrozenJointPortfolioContract.from_json(
        f.FrozenJointPortfolioContract.freeze(r).to_json()
    )
    check.assertEqual(frozen.audit(book())["status"], "pass")
    check.assertEqual(frozen.audit(None)["reason"], "no_snapshot")
    check.assertEqual(
        frozen.audit(book(value_metric=True))["reason"],
        "measurement_or_scope_definition_changed",
    )
    check.assertEqual(
        frozen.audit(
            book(
                positions=[
                    *book().universe.positions,
                    f.PortfolioPosition("a3", 10, "A"),
                ]
            )
        )["status"],
        "pass",
    )
    check.assertEqual(
        frozen.audit(
            book(
                positions=[
                    *book().universe.positions,
                    f.PortfolioPosition("new", 10, "NEW"),
                ]
            )
        )["reason"],
        "new_issuers",
    )
    check.assertEqual(
        frozen.audit(
            book(
                positions=[
                    *book().universe.positions,
                    f.PortfolioPosition("unknown", 10),
                ]
            )
        )["reason"],
        "unresolved_securities",
    )
    contracted = [p for p in book().universe.positions if p.issuer_id != "B"]
    check.assertEqual(
        frozen.audit(book(positions=contracted))["status"], "inconclusive"
    )
    base_contract = f.FrozenJointPortfolioContract.freeze(
        report(
            scope="common",
            candidate_refinements=(),
            ambiguity_limits={"ocf": 1, "gap": 1},
        )
    )
    check.assertEqual(base_contract.audit(book(positions=contracted))["status"], "pass")


def test_empty_intersection_is_inconclusive_and_zero_values_do_not_create_support():
    evidence = book(
        positions=[
            f.PortfolioPosition("c", 20, "C"),
            f.PortfolioPosition("a_zero", 0, "A"),
        ]
    )
    check.assertEqual(evidence.common_value, 0)
    result = report(evidence)
    check.assertEqual(result.status, "inconclusive_no_common_book")
    check.assertEqual(result.candidates, ())
    check.assertEqual(evidence.lift_interval("ocf", None), (1, 1))
    check.assertTrue(f.capture_joint_portfolio_snapshot(result).replay_matches())


def test_frozen_eligible_precision_reviews_coverage_growth_and_classification_changes():
    original = report(ambiguity_limits={"ocf": 0.5, "gap": 0.5})
    frozen = f.FrozenJointPortfolioContract.freeze(original)
    more_unknown = [
        replace(p, value=120) if p.position_id == "u" else p
        for p in book().universe.positions
    ]
    check.assertEqual(frozen.audit(book(positions=more_unknown))["status"], "review")
    changed = book(labels={"A": {"sector": "S", "industry": "New", "age": "young"}})
    check.assertEqual(frozen.audit(changed)["reason"], "unsupported_support")
    reassigned = [
        replace(p, issuer_id="B") if p.position_id == "a1" else p
        for p in book().universe.positions
    ]
    check.assertEqual(
        frozen.audit(book(positions=reassigned))["reason"], "issuer_mapping_changed"
    )
    reclassified = [
        replace(p, security_type="cash") if p.position_id == "b" else p
        for p in book().universe.positions
    ]
    check.assertEqual(
        frozen.audit(book(positions=reclassified))["reason"],
        "security_eligibility_changed",
    )


def test_snapshot_preserves_tuple_keyed_q_costs_and_conditional_predicate_replay():
    b = book()
    hidden = ("sector", "issuer_id", "age", "industry")
    states = {tuple(r[c] for c in hidden) for r in b.rows}
    cost = {(a, c): float(a != c) for a in states for c in states}
    result = report(b, q_presets=(us.q_wasserstein(cost, 0.1),))
    snapshot = f.capture_joint_portfolio_snapshot(result)
    check.assertTrue(
        f.JointPortfolioSnapshot.from_json(snapshot.to_json()).replay_matches()
    )
    suggestion = us.suggest_conditional_refinements(
        b.rows,
        public=("sector", "industry"),
        targets=tuple(b.metrics),
        candidate_columns=("age",),
    )
    conditional = report(
        b,
        candidate_refinements=("age", "industry"),
        conditional_refinements=suggestion.candidates,
    )
    local = next(c for c in conditional.candidates if c["kind"] == "conditional")
    check.assertEqual(
        local["public_columns"], ("sector", "industry", suggestion.candidates[0].name)
    )
    check.assertTrue(f.capture_joint_portfolio_snapshot(conditional).replay_matches())
