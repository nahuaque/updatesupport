from __future__ import annotations

import json
import unittest

import updatesupport as us


def _rows() -> list[dict[str, object]]:
    return [
        {
            "segment": "A",
            "driver": "low",
            "noise": "n",
            "target": 0.0,
            "weight": 30,
        },
        {
            "segment": "A",
            "driver": "high",
            "noise": "n",
            "target": 1.0,
            "weight": 30,
        },
        {
            "segment": "B",
            "driver": "flat",
            "noise": "n",
            "target": 0.5,
            "weight": 40,
        },
    ]


class AuditSpecTests(unittest.TestCase):
    def test_public_descent_spec_round_trips_and_runs(self):
        spec = us.AuditSpec(
            public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            weight="weight",
            candidate_refinements=["driver", "noise"],
            q={"name": "bounded_shift", "radius": 0.5},
            min_cell_weight=1,
            top=2,
            title="Configured Audit",
        )

        restored = us.AuditSpec.from_json(spec.to_json())
        run = restored.run(_rows())
        payload = run.as_dict()

        self.assertEqual(restored, spec)
        self.assertIsInstance(restored.q, us.QSpec)
        self.assertIsInstance(run, us.AuditRun)
        self.assertIsInstance(run.report, us.PublicDescentReport)
        self.assertAlmostEqual(run.report.interval.diameter, 0.3)
        self.assertEqual(run.report.refinements[0].column, "driver")
        self.assertEqual(payload["spec"]["q"], {"name": "bounded_shift", "radius": 0.5})
        self.assertEqual(payload["report_type"], "public_descent")
        self.assertEqual(payload["report"]["title"], "Configured Audit")
        self.assertIn("# Configured Audit", run.to_markdown())
        self.assertEqual(json.loads(run.to_json())["report_type"], "public_descent")

    def test_run_audit_accepts_mapping_spec(self):
        run = us.run_audit(
            {
                "kind": "public",
                "public": ["segment"],
                "hidden": ["segment", "driver"],
                "target": "target",
                "candidate_refinements": ["driver"],
            },
            _rows(),
        )

        self.assertIsInstance(run.report, us.PublicDescentReport)
        self.assertEqual(run.spec.kind, "public_descent")
        self.assertEqual(run.spec.public, ("segment",))

    def test_sensitivity_spec_runs_grid(self):
        spec = us.AuditSpec(
            kind="sensitivity",
            public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            weight="weight",
            q_presets=[
                "saturated",
                {"name": "bounded_shift", "radius": 0.5},
                "observed",
            ],
            min_cell_weights=[1, 35],
            title="Configured Sensitivity",
        )

        run = us.run_audit(spec, _rows())
        payload = run.as_dict()

        self.assertIsInstance(run.report, us.SensitivityReport)
        self.assertEqual(len(run.report.rows), 6)
        self.assertEqual(run.report.summary.scenario_count, 6)
        self.assertEqual(payload["report"]["summary"]["scenario_count"], 6)
        self.assertIn("# Configured Sensitivity", run.to_markdown())

    def test_frontier_spec_runs_search(self):
        spec = us.AuditSpec(
            kind="frontier",
            public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            weight="weight",
            candidate_refinements=["noise", "driver"],
            q_presets=["saturated", {"name": "bounded_shift", "radius": 0.5}],
            ambiguity_limit=0.05,
            bucket_budget=3,
            title="Configured Frontier",
        )

        run = spec.run(_rows())
        payload = run.as_dict()

        self.assertIsInstance(run.report, us.PublicRepresentationFrontier)
        self.assertEqual(run.report.minimal_stable.added_columns, ("driver",))
        self.assertEqual(
            payload["report"]["minimal_stable"]["added_columns"], ("driver",)
        )
        self.assertIn("# Configured Frontier", run.to_markdown())

    def test_certificate_spec_runs_search(self):
        spec = us.AuditSpec(
            kind="certificate",
            public=["segment"],
            hidden=["segment", "driver", "noise"],
            target="target",
            weight="weight",
            candidate_refinements=["noise", "driver"],
            q_presets=["saturated", {"name": "bounded_shift", "radius": 0.5}],
            ambiguity_limit=0.05,
            bucket_budget=3,
            title="Configured Certificate",
        )

        run = spec.run(_rows())
        payload = run.as_dict()

        self.assertIsInstance(run.report, us.RepresentationStabilityCertificate)
        self.assertTrue(run.report.passed)
        self.assertEqual(run.report.certified_candidate.added_columns, ("driver",))
        self.assertEqual(payload["report_type"], "certificate")
        self.assertEqual(payload["report"]["status"], "pass")
        self.assertIn("# Configured Certificate", run.to_markdown())

    def test_q_spec_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            us.QSpec.from_value({"name": "saturated", "unexpected": True})

    def test_q_spec_round_trips_solver_metadata(self):
        spec = us.QSpec.from_value(
            {
                "name": "tv_budget",
                "radius": 0.1,
                "backend": "cvxpy",
                "solver": "SCIP",
                "solver_options": {"limits/time": 5},
            }
        )

        preset = spec.to_preset()

        self.assertEqual(
            spec.as_dict(),
            {
                "name": "tv_budget",
                "radius": 0.1,
                "backend": "cvxpy",
                "solver": "SCIP",
                "solver_options": {"limits/time": 5},
            },
        )
        self.assertEqual(preset.solver, "SCIP")
        self.assertEqual(preset.solver_options, {"limits/time": 5})

    def test_audit_spec_rejects_string_column_sequence(self):
        with self.assertRaises(TypeError):
            us.AuditSpec(public="segment", hidden=["segment"], target="target")


if __name__ == "__main__":
    unittest.main()
