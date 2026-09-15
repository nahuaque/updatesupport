from __future__ import annotations
from dataclasses import replace
import json
import unittest
import updatesupport_finance as usf


def fact(**changes):
    return usf.DisclosureFact(
        **{
            "fact_id": "revenue-v1",
            "entity": "example-cik",
            "concept": "revenue",
            "value": 100.0,
            "unit": "USD",
            "currency": "USD",
            "decimals": 0,
            "period_start": "2025-01-01",
            "period_end": "2025-12-31",
            "available_at": "2026-02-01T12:00:00Z",
            "source_id": "filing-v1",
            "source_url": "https://example.com/filing-v1",
            **changes,
        }
    )


def problem(constraints=(), variables=None):
    return usf.disclosure_triangulation_spec(
        variables=variables or [usf.disclosure_variable("x", unit="USD")],
        constraints=constraints,
        targets=[usf.disclosure_target("x", "x", unit="USD")],
        tiers=[usf.disclosure_tier("reported", [c.name for c in constraints])],
    )


def codes(diagnostics):
    return {row.code for row in diagnostics}


def snapshot(row=None, **changes):
    row = row or fact()
    spec = problem([usf.disclosure_fact_constraint("amount", "x", row)])
    return usf.capture_disclosure_snapshot(
        spec,
        facts=[row],
        as_of="2026-04-01T00:00:00Z",
        target="x",
        tier="reported",
        claim=usf.disclosure_claim(target="x", tier="reported", lower_at_least=95),
        **changes,
    )


class DisclosureEvidenceTests(unittest.TestCase):
    def test_scaled_fact_and_precision_compile_to_base_units(self):
        row = fact(value=123, scale=1000000.0, decimals=-6)
        constraint = usf.disclosure_fact_constraint("amount", "x", row)
        spec = problem([constraint])
        self.assertEqual(usf.validate_disclosure_evidence([row], problem=spec), ())
        interval = usf.triangulate_disclosure(spec).interval(
            target="x", scenario="reported"
        )
        self.assertEqual((interval.lower, interval.upper), (122500000, 123500000))
        self.assertEqual(constraint.metadata["fact_ids"], [row.fact_id])
        self.assertEqual(usf.DisclosureFact(**row.as_dict()), row)

    def test_unknown_precision_requires_an_explicit_choice(self):
        row = fact(decimals=None)
        with self.assertRaisesRegex(ValueError, "unknown precision"):
            usf.disclosure_fact_constraint("amount", "x", row)
        constraint = usf.disclosure_fact_constraint("amount", "x", row, exact=True)
        self.assertTrue(constraint.lower == constraint.upper == 100)
        self.assertIs(constraint.metadata["exact"], True)

    def test_invalid_fact_context_fails_fast(self):
        for changes, error in [
            ({"value": float("nan")}, "value"),
            ({"scale": 0}, "scale"),
            ({"period_start": "2026-01-01"}, "period_start"),
            ({"available_at": "2026-02-01"}, "timezone"),
            ({"reported": False}, "derived facts"),
            ({"decimals": 0.5}, "decimals"),
        ]:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, error):
                    fact(**changes)

    def test_dimensions_are_detached_from_mutable_input(self):
        dimensions = {"segment": "software"}
        row = fact(dimensions=dimensions)
        dimensions["segment"] = "hardware"
        self.assertEqual(row.dimensions["segment"], "software")
        with self.assertRaises(TypeError):
            row.dimensions["segment"] = "hardware"

    def test_context_checks_require_explicit_period_and_dimension_differences(self):
        first = fact()
        second = fact(
            fact_id="segment",
            concept="segment_revenue",
            period_start="2025-10-01",
            currency="EUR",
            dimensions={"segment": "software"},
        )
        relation = usf.disclosure_constraint("relationship", "x", lower=0)
        linked = usf.link_disclosure_evidence(relation, [first, second])
        found = codes(
            usf.validate_disclosure_evidence([first, second], problem=problem([linked]))
        )
        self.assertLessEqual(
            {"incompatible_period", "incompatible_currency", "incompatible_dimensions"},
            found,
        )
        linked = usf.link_disclosure_evidence(
            relation, [first, second], varying_fields=["period", "dimensions"]
        )
        found = codes(
            usf.validate_disclosure_evidence([first, second], problem=problem([linked]))
        )
        self.assertEqual(found, {"incompatible_currency"})

    def test_duplicate_versions_missing_links_and_future_evidence_are_detected(self):
        first = fact()
        revision = fact(
            fact_id="revenue-v2",
            source_id="filing-v2",
            value=90,
            available_at="2026-03-01T12:00:00Z",
        )
        found = codes(
            usf.validate_disclosure_evidence(
                [first, revision], as_of="2026-02-10T00:00:00Z"
            )
        )
        self.assertEqual(found, {"mixed_versions", "future_evidence"})
        self.assertIn(
            "duplicate_id", codes(usf.validate_disclosure_evidence([first, first]))
        )
        linked = usf.disclosure_fact_constraint("amount", "x", revision)
        self.assertIn(
            "missing_fact",
            codes(usf.validate_disclosure_evidence([first], problem=problem([linked]))),
        )

    def test_mutually_exclusive_tiers_can_retain_different_versions(self):
        first = fact()
        revision = fact(fact_id="revenue-v2", source_id="filing-v2", value=90)
        old = usf.disclosure_fact_constraint("old", "x", first)
        new = usf.disclosure_fact_constraint("new", "x", revision)
        spec = replace(
            problem([old, new]),
            scenarios=[
                usf.disclosure_tier("old", ["old"]),
                usf.disclosure_tier("new", ["new"]),
            ],
        )
        self.assertEqual(
            usf.validate_disclosure_evidence([first, revision], problem=spec), ()
        )
        self.assertIn(
            "mixed_versions",
            codes(
                usf.validate_disclosure_evidence(
                    [first, revision], problem=problem([old, new])
                )
            ),
        )

    def test_conflict_report_preserves_financial_provenance(self):
        spec = problem(
            [
                usf.exact_disclosure_constraint(
                    "reported_negative", "x", -1, provenance="source filing"
                )
            ]
        )
        conflict = usf.find_disclosure_conflict(spec, tier="reported")
        self.assertEqual(conflict.status, "irreducible")
        self.assertTrue(
            any((row.provenance == "source filing" for row in conflict.members))
        )
        self.assertTrue(any((row.kind == "variable_bound" for row in conflict.members)))

    def test_derived_lineage_checks_missing_inputs_future_inputs_and_cycles(self):
        parent = fact()
        child = fact(
            fact_id="derived",
            concept="derived_revenue",
            reported=False,
            derivation="source adjustment",
            derivation_inputs=[parent.fact_id],
            available_at="2026-01-01T00:00:00Z",
        )
        self.assertIn(
            "future_derivation_input",
            codes(usf.validate_disclosure_evidence([parent, child])),
        )
        self.assertIn(
            "missing_derivation_input", codes(usf.validate_disclosure_evidence([child]))
        )
        parent = replace(
            parent,
            reported=False,
            derivation="bad cycle",
            derivation_inputs=[child.fact_id],
        )
        self.assertIn(
            "cyclic_derivation",
            codes(usf.validate_disclosure_evidence([parent, child])),
        )

    def test_stale_compiled_values_and_wrong_variable_units_are_detected(self):
        row = fact()
        constraint = usf.disclosure_fact_constraint("amount", "x", row)
        found = codes(
            usf.validate_disclosure_evidence(
                [replace(row, value=120)], problem=problem([constraint])
            )
        )
        self.assertIn("stale_fact_constraint", found)
        spec = problem(
            [constraint], variables=[usf.disclosure_variable("x", unit="USD millions")]
        )
        self.assertIn(
            "incompatible_variable_unit",
            codes(usf.validate_disclosure_evidence([row], problem=spec)),
        )

    def test_two_sided_claim_gets_an_interior_supporting_allocation(self):
        spec = problem(variables=[usf.disclosure_variable("x", lower=0, upper=10)])
        report = usf.triangulate_disclosure(spec)
        claim = usf.disclosure_claim(
            target="x", tier="reported", lower_at_least=3, upper_at_most=7
        )
        pack = usf.disclosure_audit_pack(report, claim=claim)
        self.assertEqual(pack.claim_audit.verdict, "inconclusive")
        allocations = {row.role: row for row in pack.allocations.allocations}
        self.assertTrue(3 <= allocations["supporting"].target_value <= 7)
        self.assertLess(allocations["opposing_lower"].target_value, 3)
        self.assertGreater(allocations["opposing_upper"].target_value, 7)
        self.assertTrue(
            all((check.passed for row in allocations.values() for check in row.checks))
        )
        self.assertIn("Feasible Allocations", pack.to_markdown())
        self.assertIn("disclosure_allocation_checks", pack.to_tables())
        self.assertEqual(
            len(json.loads(pack.to_json())["allocations"]["allocations"]), 3
        )

    def test_one_sided_unbounded_claim_and_infeasible_source_have_honest_witnesses(
        self,
    ):
        report = usf.triangulate_disclosure(
            problem(variables=[usf.disclosure_variable("x", lower=10)])
        )
        claim = usf.disclosure_claim(target="x", tier="reported", lower_at_least=5)
        pack = usf.disclosure_audit_pack(report, claim=claim)
        self.assertEqual(pack.claim_audit.verdict, "pass")
        self.assertEqual(
            [row.role for row in pack.allocations.allocations], ["supporting"]
        )
        bad = problem([usf.disclosure_constraint("negative", "x", upper=-1)])
        pack = usf.disclosure_audit_pack(usf.triangulate_disclosure(bad), claim=claim)
        self.assertEqual(pack.claim_audit.verdict, "inconclusive")
        self.assertEqual(pack.allocations.allocations, ())
        self.assertEqual(pack.allocations.attempts[0]["status"], "infeasible")

    def test_rounding_allows_reconciliation_without_silent_exactness(self):
        constraints = [
            usf.rounded_amount_constraint("reported_total", "x", 100, increment=1),
            usf.rounded_amount_constraint("reported_a", "a", 50, increment=1),
            usf.rounded_amount_constraint("reported_b", "b", 51, increment=1),
            usf.reconciliation_constraint("total", total="x", components=["a", "b"]),
        ]
        spec = problem(
            constraints, variables=[usf.disclosure_variable(v) for v in ("x", "a", "b")]
        )
        interval = usf.triangulate_disclosure(spec).interval(
            target="x", scenario="reported"
        )
        self.assertEqual(interval.status, "bounded")
        self.assertEqual((interval.lower, interval.upper), (100.0, 100.5))

    def test_stock_flow_residual_can_be_negative(self):
        constraints = [
            usf.exact_disclosure_constraint("opening", "opening", 100),
            usf.exact_disclosure_constraint("closing", "closing", 90),
            usf.exact_disclosure_constraint("borrowing", "borrowing", 5),
            usf.stock_flow_constraint(
                "bridge",
                opening="opening",
                closing="closing",
                movements={"borrowing": 1},
                residual="x",
            ),
        ]
        variables = [
            usf.disclosure_variable(v, lower=None)
            for v in ("opening", "closing", "borrowing", "x")
        ]
        interval = usf.triangulate_disclosure(problem(constraints, variables)).interval(
            target="x", scenario="reported"
        )
        self.assertEqual((interval.lower, interval.upper), (-15, -15))

    def test_snapshot_roundtrip_replays_and_is_detached_from_mutation(self):
        captured = snapshot()
        restored = usf.DisclosureSnapshot.from_json(captured.to_json(indent=2))
        self.assertEqual(restored.fingerprint, captured.fingerprint)
        self.assertEqual(restored.replay().claim_audit.verdict, "pass")
        self.assertEqual(
            json.loads(json.dumps(restored.replay().interval.as_dict())),
            captured.as_dict()["result"]["interval"],
        )
        self.assertEqual(restored.runtime_differences(), {})
        payload = restored.as_dict()
        payload["facts"][0]["value"] = 0
        self.assertEqual(restored.as_dict()["facts"][0]["value"], 100)
        with self.assertRaisesRegex(ValueError, "stale_fact_constraint"):
            usf.DisclosureSnapshot.from_json(json.dumps(payload))

    def test_snapshot_differences_separate_evidence_assumptions_and_verdict(self):
        before = snapshot()
        after = snapshot(fact(value=90), assumptions=["Revised evidence snapshot"])
        delta = usf.compare_disclosure_snapshots(before, after)
        self.assertEqual(delta["verdict"], {"before": "pass", "after": "fail"})
        self.assertEqual(delta["fact_changes"]["revenue-v1"]["after"]["value"], 90)
        self.assertIn("amount", delta["constraint_changes"])
        self.assertEqual(delta["assumptions"]["after"], ["Revised evidence snapshot"])
        self.assertEqual(delta["runtime_changes"], {})

    def test_snapshot_rejects_future_evidence_unknown_schema_and_claim_mismatch(self):
        row = fact(available_at="2027-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "future_evidence"):
            snapshot(row)
        payload = snapshot().as_dict()
        payload["schema_version"] = 999
        with self.assertRaisesRegex(ValueError, "schema_version"):
            usf.DisclosureSnapshot.from_json(json.dumps(payload))
        payload = snapshot().as_dict()
        payload["claim"]["target"] = "another"
        with self.assertRaisesRegex(ValueError, "must match"):
            usf.DisclosureSnapshot.from_json(json.dumps(payload))
