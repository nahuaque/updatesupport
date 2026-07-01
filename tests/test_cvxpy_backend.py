from __future__ import annotations

import unittest

import updatesupport as us


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

        tv_grouped = us.from_dataframe(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_tv_budget(0.0),
        )
        interval = tv_grouped.problem.global_transport_modulus()

        self.assertAlmostEqual(interval.lower, 0.5, places=6)
        self.assertAlmostEqual(interval.upper, 0.5, places=6)
        self.assertAlmostEqual(interval.diameter, 0.0, places=6)
