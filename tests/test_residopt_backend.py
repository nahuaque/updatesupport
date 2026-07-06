import unittest

import numpy as np

import updatesupport as us
from updatesupport.residopt_backend import (
    _orthonormal_nullspace,
    _public_incidence_matrix,
)


def _sample_grouped():
    rows = _sample_rows()
    return us.from_dataframe(
        rows,
        public=["region"],
        hidden=["region", "segment"],
        target="metric",
        weight="weight",
        min_cell_weight=0.0,
        q=us.q_l2_budget(0.05),
    )


def _sample_rows():
    rows = [
        {"region": "north", "segment": "new", "metric": 0.10, "weight": 10.0},
        {
            "region": "north",
            "segment": "returning",
            "metric": 0.25,
            "weight": 15.0,
        },
        {"region": "south", "segment": "new", "metric": 0.30, "weight": 20.0},
        {
            "region": "south",
            "segment": "returning",
            "metric": 0.55,
            "weight": 5.0,
        },
    ]
    return rows


class ResidOptBackendTests(unittest.TestCase):
    def test_exports_are_available(self):
        self.assertIn("residopt_available", us.__all__)
        self.assertIn("residopt_l2_support_interval", us.__all__)
        self.assertIn("ResidOptEndpointReport", us.__all__)
        self.assertIn("ResidOptL2EndpointCompiler", us.__all__)
        self.assertIn("residopt_refinement_screen", us.__all__)
        self.assertIn("ResidOptRefinementScreenContext", us.__all__)
        self.assertIn("ResidOptRefinementScreenReport", us.__all__)

    def test_availability_is_structured(self):
        availability = us.residopt_available()
        self.assertIsInstance(availability.available, bool)
        self.assertIn("available", availability.as_dict())

    def test_public_nullspace_preserves_public_law(self):
        grouped = _sample_grouped()
        incidence = _public_incidence_matrix(grouped.problem)
        nullspace = _orthonormal_nullspace(incidence)

        self.assertEqual(nullspace.shape, (4, 2))
        np.testing.assert_allclose(incidence @ nullspace, 0.0, atol=1e-10)
        np.testing.assert_allclose(nullspace.T @ nullspace, np.eye(2), atol=1e-10)

    def test_residopt_l2_interval_runs_when_residopt_is_importable(self):
        try:
            import residopt  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("residopt is not importable") from exc

        grouped = _sample_grouped()
        report = us.residopt_l2_support_interval(grouped, solver="CLARABEL")

        self.assertIsInstance(report, us.ResidOptEndpointReport)
        self.assertLessEqual(report.lower, report.observed_value)
        self.assertGreaterEqual(report.upper, report.observed_value)
        self.assertGreater(report.ambiguity, 0.0)
        self.assertFalse(report.exact_for_updatesupport_q)
        self.assertTrue(report.conservative_for_updatesupport_q)
        self.assertIsNotNone(report.upper_certificate)
        self.assertEqual(report.upper_certificate.template, "ellipsoid_support_socp")
        self.assertEqual(report.compiled_templates_built, 1)
        self.assertEqual(report.support_solves, 2)
        self.assertIn("certificates", report.to_tables())

    def test_residopt_l2_compiler_reuses_parameterized_template(self):
        try:
            import residopt  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("residopt is not importable") from exc

        grouped = _sample_grouped()
        compiler = us.ResidOptL2EndpointCompiler.from_grouped(
            grouped,
            solver="CLARABEL",
        )
        first = compiler.interval()
        second = compiler.interval(direction=[1.0, 0.0, 0.0, -1.0])

        self.assertEqual(first.compiled_templates_built, 1)
        self.assertEqual(first.support_solves, 2)
        self.assertEqual(second.compiled_templates_built, 0)
        self.assertEqual(second.support_solves, 2)
        self.assertEqual(compiler.compiled_template_count, 1)
        self.assertEqual(compiler.support_solve_count, 4)
        self.assertIsNotNone(second.upper_certificate)
        self.assertTrue(second.upper_certificate.metadata["parameterized_template"])

    def test_refinement_screen_certifies_and_avoids_exact_fallback(self):
        try:
            import residopt  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("residopt is not importable") from exc

        report = us.residopt_refinement_screen(
            _sample_rows(),
            public=["region"],
            hidden=["region", "segment"],
            target="metric",
            candidate_refinements=["segment"],
            weight="weight",
            min_cell_weight=0.0,
            q=us.q_l2_budget(0.05, solver="CLARABEL"),
            ambiguity_limit=1.0,
            solver="CLARABEL",
        )

        self.assertIsInstance(report, us.ResidOptRefinementScreenReport)
        self.assertEqual(report.screened_count, 2)
        self.assertEqual(report.certified_count, 2)
        self.assertEqual(report.exact_solve_count, 0)
        self.assertEqual(report.exact_solve_avoided_count, 2)
        self.assertGreaterEqual(report.compiler_cache_size, 2)
        self.assertIn("candidates", report.to_tables())
        self.assertIn("screen_certified", report.to_markdown())

    def test_refinement_screen_falls_back_to_exact_when_inconclusive(self):
        try:
            import residopt  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("residopt is not importable") from exc

        context = us.ResidOptRefinementScreenContext(
            _sample_rows(),
            public=["region"],
            hidden=["region", "segment"],
            target="metric",
            weight="weight",
            min_cell_weight=0.0,
            q=us.q_l2_budget(0.05, solver="CLARABEL"),
            solver="CLARABEL",
        )
        report = context.screen(
            candidate_refinements=["segment"],
            ambiguity_limit=0.0,
        )
        second = context.screen(
            candidate_refinements=["segment"],
            ambiguity_limit=0.0,
        )

        self.assertEqual(report.certified_count, 1)
        self.assertEqual(report.exact_solve_count, 1)
        self.assertEqual(report.exact_solve_avoided_count, 1)
        self.assertTrue(
            any(row.exact_ambiguity is not None for row in report.candidates)
        )
        self.assertGreaterEqual(report.support_solve_count, 2)
        self.assertGreaterEqual(second.screened_count, 2)
        self.assertTrue(all(row.compiler_cache_hit for row in second.candidates))


if __name__ == "__main__":
    unittest.main()
