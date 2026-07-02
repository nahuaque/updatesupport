from __future__ import annotations

import unittest
from unittest import mock

import updatesupport as us
import updatesupport.report as report_module


def _require_cvxpy() -> None:
    try:
        import cvxpy  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("cvxpy extra is not installed") from exc


def _rows():
    return [
        {"PUBLIC": "A", "HIDDEN": "x", "Y": 0.0, "W": 3.0},
        {"PUBLIC": "A", "HIDDEN": "y", "Y": 1.0, "W": 3.0},
        {"PUBLIC": "B", "HIDDEN": "z", "Y": 0.5, "W": 4.0},
    ]


class CvxpyBackendTests(unittest.TestCase):
    def test_custom_convex_constraint_backend_solves_interval(self):
        _require_cvxpy()

        def cap_b(_cp, q, _states, state_index):
            return (q[state_index["b"]] <= 0.75,)

        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand={"a": 0.0, "b": 1.0},
            environments=us.CvxpyEnvironments(
                fixed_public_law={"o": 1.0},
                constraint_builders=(cap_b,),
            ),
        )

        interval = problem.global_transport_modulus()

        self.assertAlmostEqual(interval.lower, 0.0, places=6)
        self.assertAlmostEqual(interval.upper, 0.75, places=6)
        self.assertAlmostEqual(interval.diameter, 0.75, places=6)

    def test_tv_budget_preset_limits_hidden_mass_shift(self):
        _require_cvxpy()

        grouped = us.from_dataframe(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_tv_budget(0.15),
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertEqual(grouped.q_name, "tv_budget(radius=0.15)")
        self.assertAlmostEqual(interval.lower, 0.35, places=6)
        self.assertAlmostEqual(interval.upper, 0.65, places=6)
        self.assertAlmostEqual(interval.diameter, 0.30, places=6)

    def test_tv_budget_dual_diagnostics_are_attached_to_interval(self):
        _require_cvxpy()

        grouped = us.from_dataframe(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_tv_budget(0.15),
        )

        interval = grouped.problem.global_transport_modulus()
        kinds = {row.kind for row in interval.duals}
        names = {row.name for row in interval.duals}
        summary = interval.dual_summary(top=3)

        self.assertTrue(interval.duals)
        self.assertIsInstance(summary[0], us.ConstraintDual)
        self.assertIn("tv_budget", kinds)
        self.assertIn("public_law", kinds)
        self.assertIn("lower_bound", kinds)
        self.assertIn("total-variation budget", names)
        self.assertEqual(summary[0].kind, "tv_budget")
        self.assertGreater(summary[0].magnitude, 0.0)
        self.assertIn("magnitude", summary[0].as_dict())

    def test_public_descent_report_includes_cvxpy_dual_section(self):
        _require_cvxpy()

        report = us.public_descent_report(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_tv_budget(0.15),
            top=0,
        )

        markdown = report.to_markdown()

        self.assertIn("## CVXPY Dual Diagnostics", markdown)
        self.assertIn("total-variation budget", markdown)
        self.assertIn("public-law equality", markdown)

    def test_parameterized_tv_budget_reuses_problem_across_radius_sweep(self):
        _require_cvxpy()

        grouped = us.from_dataframe(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_tv_budget(0.15, backend="parameterized_cvxpy"),
        )
        env = grouped.problem.environments

        first = grouped.problem.global_transport_modulus()
        env.set_parameter("radius", 0.30)
        second = grouped.problem.global_transport_modulus()

        self.assertIsInstance(env, us.ParameterizedCvxpyEnvironments)
        self.assertEqual(env.cache_info()["single_problems"], 1)
        self.assertEqual(grouped.q_name, "tv_budget(radius=0.15)")
        self.assertAlmostEqual(first.lower, 0.35, places=6)
        self.assertAlmostEqual(first.upper, 0.65, places=6)
        self.assertAlmostEqual(second.lower, 0.20, places=6)
        self.assertAlmostEqual(second.upper, 0.80, places=6)
        self.assertIn("tv_budget", {row.kind for row in second.duals})

    def test_parameterized_divergence_presets_match_cvxpy_backend(self):
        _require_cvxpy()

        states = (("A", "x"), ("A", "y"), ("B", "z"))
        cost = {
            (left, right): (0.0 if left == right else 1.0)
            for left in states
            for right in states
        }
        cost[(("A", "x"), ("B", "z"))] = 100.0
        cost[(("B", "z"), ("A", "x"))] = 100.0
        cost[(("A", "y"), ("B", "z"))] = 100.0
        cost[(("B", "z"), ("A", "y"))] = 100.0
        parameterized = [
            us.q_chi_square_budget(0.15, backend="parameterized_cvxpy"),
            us.q_kl_budget(0.08, backend="parameterized_cvxpy"),
            us.q_wasserstein(cost, radius=0.15, backend="parameterized_cvxpy"),
        ]
        standard = [
            us.q_chi_square_budget(0.15),
            us.q_kl_budget(0.08),
            us.q_wasserstein(cost, radius=0.15),
        ]

        for parameterized_q, standard_q in zip(parameterized, standard, strict=True):
            with self.subTest(q=parameterized_q.name):
                parameterized_grouped = us.from_dataframe(
                    _rows(),
                    public=["PUBLIC"],
                    hidden=["PUBLIC", "HIDDEN"],
                    target="Y",
                    weight="W",
                    q=parameterized_q,
                )
                standard_grouped = us.from_dataframe(
                    _rows(),
                    public=["PUBLIC"],
                    hidden=["PUBLIC", "HIDDEN"],
                    target="Y",
                    weight="W",
                    q=standard_q,
                )

                parameterized_interval = (
                    parameterized_grouped.problem.global_transport_modulus()
                )
                standard_interval = standard_grouped.problem.global_transport_modulus()

                self.assertIsInstance(
                    parameterized_grouped.problem.environments,
                    us.ParameterizedCvxpyEnvironments,
                )
                self.assertAlmostEqual(
                    parameterized_interval.lower,
                    standard_interval.lower,
                    places=5,
                )
                self.assertAlmostEqual(
                    parameterized_interval.upper,
                    standard_interval.upper,
                    places=5,
                )

    def test_sensitivity_report_reuses_parameterized_backend_for_radius_grid(self):
        _require_cvxpy()

        with mock.patch(
            "updatesupport.report.from_dataframe",
            wraps=report_module.from_dataframe,
        ) as from_dataframe:
            report = us.sensitivity_report(
                _rows(),
                public=["PUBLIC"],
                hidden=["PUBLIC", "HIDDEN"],
                target="Y",
                weight="W",
                q_presets=[
                    us.q_tv_budget(0.15),
                    us.q_tv_budget(0.30),
                ],
            )

        self.assertEqual(from_dataframe.call_count, 1)
        self.assertEqual(
            [row.q_name for row in report.rows],
            [
                "tv_budget(radius=0.15)",
                "tv_budget(radius=0.3)",
            ],
        )
        self.assertAlmostEqual(report.rows[0].lower, 0.35, places=6)
        self.assertAlmostEqual(report.rows[0].upper, 0.65, places=6)
        self.assertAlmostEqual(report.rows[1].lower, 0.20, places=6)
        self.assertAlmostEqual(report.rows[1].upper, 0.80, places=6)

    def test_refinement_sensitivity_reuses_parameterized_backend_for_radius_grid(
        self,
    ):
        _require_cvxpy()

        with mock.patch(
            "updatesupport.report.from_dataframe",
            wraps=report_module.from_dataframe,
        ) as from_dataframe:
            report = us.recommend_refinements_sensitivity(
                _rows(),
                public=["PUBLIC"],
                hidden=["PUBLIC", "HIDDEN"],
                target="Y",
                candidate_refinements=["HIDDEN"],
                weight="W",
                q_presets=[
                    us.q_tv_budget(0.15),
                    us.q_tv_budget(0.30),
                ],
            )

        self.assertEqual(from_dataframe.call_count, 2)
        self.assertEqual(len(report.scenarios), 2)
        self.assertEqual(
            [row.q_name for row in report.scenarios],
            [
                "tv_budget(radius=0.15)",
                "tv_budget(radius=0.3)",
            ],
        )
        self.assertEqual(report.scenarios[0].best_column, "HIDDEN")
        self.assertEqual(report.scenarios[1].best_column, "HIDDEN")

    def test_chi_square_budget_preset_limits_divergence_from_observed(self):
        _require_cvxpy()

        grouped = us.from_dataframe(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_chi_square_budget(0.15),
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertEqual(grouped.q_name, "chi_square_budget(radius=0.15)")
        self.assertAlmostEqual(interval.lower, 0.35, places=5)
        self.assertAlmostEqual(interval.upper, 0.65, places=5)
        self.assertAlmostEqual(interval.diameter, 0.30, places=5)

    def test_kl_budget_preset_limits_divergence_from_observed(self):
        _require_cvxpy()

        grouped = us.from_dataframe(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_kl_budget(0.08),
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertEqual(grouped.q_name, "kl_budget(radius=0.08)")
        self.assertLess(interval.lower, 0.5)
        self.assertGreater(interval.upper, 0.5)
        self.assertLess(interval.diameter, 0.6)

    def test_wasserstein_budget_preset_uses_explicit_hidden_cost(self):
        _require_cvxpy()

        states = (("A", "x"), ("A", "y"), ("B", "z"))
        cost = {
            (left, right): (0.0 if left == right else 1.0)
            for left in states
            for right in states
        }
        cost[(("A", "x"), ("B", "z"))] = 100.0
        cost[(("B", "z"), ("A", "x"))] = 100.0
        cost[(("A", "y"), ("B", "z"))] = 100.0
        cost[(("B", "z"), ("A", "y"))] = 100.0

        grouped = us.from_dataframe(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_wasserstein(cost, radius=0.15),
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertEqual(grouped.q_name, "wasserstein(radius=0.15)")
        self.assertAlmostEqual(interval.lower, 0.35, places=6)
        self.assertAlmostEqual(interval.upper, 0.65, places=6)
        self.assertAlmostEqual(interval.diameter, 0.30, places=6)

    def test_zero_budget_presets_collapse_to_observed_distribution(self):
        _require_cvxpy()

        for q in (
            us.q_chi_square_budget(0.0),
            us.q_kl_budget(0.0),
            us.q_tv_budget(0.0),
        ):
            with self.subTest(q=q):
                grouped = us.from_dataframe(
                    _rows(),
                    public=["PUBLIC"],
                    hidden=["PUBLIC", "HIDDEN"],
                    target="Y",
                    weight="W",
                    q=q,
                )
                interval = grouped.problem.global_transport_modulus()

                self.assertAlmostEqual(interval.lower, 0.5, places=5)
                self.assertAlmostEqual(interval.upper, 0.5, places=5)
                self.assertAlmostEqual(interval.diameter, 0.0, places=5)
