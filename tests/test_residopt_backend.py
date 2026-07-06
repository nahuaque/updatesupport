import unittest

import numpy as np

import updatesupport as us
from updatesupport.residopt_backend import (
    _orthonormal_nullspace,
    _public_incidence_matrix,
)


def _sample_grouped():
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
    return us.from_dataframe(
        rows,
        public=["region"],
        hidden=["region", "segment"],
        target="metric",
        weight="weight",
        min_cell_weight=0.0,
        q=us.q_l2_budget(0.05),
    )


class ResidOptBackendTests(unittest.TestCase):
    def test_exports_are_available(self):
        self.assertIn("residopt_available", us.__all__)
        self.assertIn("residopt_l2_support_interval", us.__all__)
        self.assertIn("ResidOptEndpointReport", us.__all__)
        self.assertIn("ResidOptL2EndpointCompiler", us.__all__)

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


if __name__ == "__main__":
    unittest.main()
