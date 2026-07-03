from __future__ import annotations

import unittest

import updatesupport as us


class PackageMetadataTests(unittest.TestCase):
    def test_package_exposes_version(self):
        self.assertRegex(us.__version__, r"^\d+\.\d+\.\d+")


class LinearTargetTests(unittest.TestCase):
    def test_finite_problem_accepts_linear_target(self):
        target = us.LinearTarget(
            {"a": 0.0, "b": 2.0},
            name="demo_target",
            description="demo fixed linear target",
        )

        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand=target,
        )

        self.assertIs(problem.target_functional, target)
        self.assertEqual(problem.estimand_map, {"a": 0.0, "b": 2.0})
        self.assertEqual(problem.target_contract.kind, "linear")
        self.assertEqual(problem.target_contract.formula, "psi(q) = sum_d h(d) q(d)")
        self.assertAlmostEqual(problem.psi({"a": 0.25, "b": 0.75}), 1.5)

    def test_finite_problem_rejects_unsupported_nonlinear_target(self):
        target = us.UnsupportedTarget(
            name="loss_ratio",
            kind="ratio",
            formula="sum loss / sum exposure",
            reason="Ratio targets need a dedicated target-functional backend.",
        )

        with self.assertRaisesRegex(
            us.UnsupportedTargetError,
            "supports only fixed linear plug-in targets",
        ):
            us.FiniteProblem(
                states=["a", "b"],
                public={"a": "o", "b": "o"},
                estimand=target,
            )

    def test_finite_problem_rejects_future_linear_target_shape(self):
        class FutureLinearTarget:
            @property
            def contract(self):
                return us.TargetContract(
                    kind="linear",
                    name="future_linear",
                    formula="psi(q) = sum_d h(d) q(d)",
                    description="future linear target object",
                    fixed_after_compilation=True,
                    supports_adequacy=True,
                    supports_interval=True,
                    supports_fiber_decomposition=True,
                )

        with self.assertRaisesRegex(
            us.UnsupportedTargetError,
            "is not a `LinearTarget`",
        ):
            us.FiniteProblem(
                states=["a", "b"],
                public={"a": "o", "b": "o"},
                estimand=FutureLinearTarget(),
            )


class RatioTargetTests(unittest.TestCase):
    def test_fixed_public_saturated_transport_solves_ratio_target(self):
        target = us.RatioTarget(
            numerator={"a": 1.0, "b": 4.0, "c": 10.0},
            denominator={"a": 1.0, "b": 2.0, "c": 5.0},
            name="loss_ratio",
            description="loss divided by exposure",
        )
        problem = us.FiniteProblem(
            states=["a", "b", "c"],
            public={"a": "x", "b": "x", "c": "y"},
            estimand=target,
            environments=us.PublicFiberSaturated.fixed({"x": 0.5, "y": 0.5}),
        )

        interval = problem.global_transport_modulus()

        self.assertEqual(problem.target_contract.kind, "ratio")
        self.assertFalse(problem.target_contract.supports_fiber_decomposition)
        self.assertAlmostEqual(problem.psi({"a": 0.5, "c": 0.5}), 11 / 6)
        self.assertAlmostEqual(interval.lower, 11 / 6)
        self.assertAlmostEqual(interval.upper, 2.0)
        self.assertAlmostEqual(interval.diameter, 1 / 6)
        self.assertEqual(interval.public_law, {"x": 0.5, "y": 0.5})
        self.assertFalse(problem.is_public_adequate())

    def test_saturated_global_ratio_target_uses_worst_public_fiber(self):
        target = us.RatioTarget(
            numerator={"a": 1.0, "b": 4.0, "c": 1.0, "d": 3.0},
            denominator={"a": 1.0, "b": 2.0, "c": 2.0, "d": 2.0},
        )
        problem = us.FiniteProblem(
            states=["a", "b", "c", "d"],
            public={"a": "x", "b": "x", "c": "y", "d": "y"},
            estimand=target,
        )

        interval = problem.global_transport_modulus()

        self.assertEqual(interval.public_law, {"x": 1.0, "y": 0.0})
        self.assertAlmostEqual(interval.lower, 1.0)
        self.assertAlmostEqual(interval.upper, 2.0)
        self.assertAlmostEqual(interval.diameter, 1.0)

    def test_constrained_backends_reject_variable_denominator_ratio_target(self):
        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "x", "b": "x"},
            estimand=us.RatioTarget(
                numerator={"a": 1.0, "b": 4.0},
                denominator={"a": 1.0, "b": 2.0},
            ),
            environments=us.PolytopeEnvironments(),
        )

        with self.assertRaisesRegex(
            us.UnsupportedTargetError,
            "requires a fixed linear target",
        ):
            problem.global_transport_modulus()

    def test_constant_denominator_ratio_target_uses_linear_backends(self):
        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "x", "b": "x"},
            estimand=us.RatioTarget(
                numerator={"a": 0.0, "b": 2.0},
                denominator={"a": 2.0, "b": 2.0},
            ),
            environments=us.PolytopeEnvironments(),
        )

        interval = problem.global_transport_modulus()

        self.assertAlmostEqual(interval.lower, 0.0)
        self.assertAlmostEqual(interval.upper, 1.0)
        self.assertAlmostEqual(interval.diameter, 1.0)


class SaturatedSupportTests(unittest.TestCase):
    def test_public_descent_succeeds_when_estimand_is_constant_on_fibers(self):
        problem = us.FiniteProblem(
            states=["a", "b", "c", "d"],
            public={"a": "x", "b": "x", "c": "y", "d": "y"},
            estimand={"a": 1.0, "b": 1.0, "c": 2.0, "d": 2.0},
            environments=us.PublicFiberSaturated(),
        )

        self.assertTrue(problem.is_public_adequate())
        self.assertEqual(problem.fiber_ranges(), {"x": 0.0, "y": 0.0})
        self.assertEqual(problem.global_transport_modulus().diameter, 0.0)
        self.assertEqual(problem.least_support().support, problem.public_partition())

    def test_saturated_transport_uses_fiber_range_formula(self):
        problem = us.FiniteProblem(
            states=["a", "b", "c", "d"],
            public={"a": "x", "b": "x", "c": "y", "d": "y"},
            estimand={"a": 0.0, "b": 1.0, "c": 0.0, "d": 3.0},
            environments=us.PublicFiberSaturated(),
        )

        self.assertFalse(problem.is_public_adequate())
        self.assertEqual(problem.fiber_ranges(), {"x": 1.0, "y": 3.0})
        self.assertEqual(problem.global_transport_modulus().diameter, 3.0)

        local = problem.local_transport_modulus({"x": 0.25, "y": 0.75})
        self.assertAlmostEqual(local.lower, 0.0)
        self.assertAlmostEqual(local.upper, 2.5)
        self.assertAlmostEqual(local.diameter, 2.5)

    def test_fixed_public_marginal_global_modulus(self):
        problem = us.FiniteProblem(
            states=["a", "b", "c", "d"],
            public={"a": "x", "b": "x", "c": "y", "d": "y"},
            estimand={"a": 0.0, "b": 1.0, "c": 0.0, "d": 3.0},
            environments=us.PublicFiberSaturated.fixed({"x": 0.4, "y": 0.6}),
        )

        self.assertAlmostEqual(problem.global_transport_modulus().diameter, 2.2)

    def test_fixed_public_zero_mass_fibers_do_not_break_adequacy(self):
        problem = us.FiniteProblem(
            states=["a", "b", "c"],
            public={"a": "x", "b": "x", "c": "y"},
            estimand={"a": 0.0, "b": 10.0, "c": 2.0},
            environments=us.PublicFiberSaturated.fixed({"x": 0.0, "y": 1.0}),
        )

        result = problem.check_public()
        least = problem.least_support()

        self.assertTrue(result.adequate)
        self.assertEqual(least.support, problem.public_partition())
        self.assertAlmostEqual(problem.global_transport_modulus().diameter, 0.0)

    def test_fixed_public_witness_uses_admissible_public_law(self):
        problem = us.FiniteProblem(
            states=["a", "b", "c"],
            public={"a": "x", "b": "x", "c": "y"},
            estimand={"a": 0.0, "b": 10.0, "c": 100.0},
            environments=us.PublicFiberSaturated.fixed({"x": 0.2, "y": 0.8}),
        )

        result = problem.check_public()

        self.assertFalse(result.adequate)
        self.assertIsNotNone(result.witness)
        self.assertAlmostEqual(result.gap, 2.0)
        self.assertEqual(result.witness.public_law, {"x": 0.2, "y": 0.8})
        self.assertAlmostEqual(result.witness.q1["c"], 0.8)
        self.assertAlmostEqual(result.witness.q2["c"], 0.8)
        self.assertAlmostEqual(problem.global_transport_modulus().diameter, 2.0)


class FiniteEnvironmentTests(unittest.TestCase):
    def test_finite_environment_returns_witness_for_inadequate_support(self):
        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand={"a": 0.0, "b": 4.0},
            environments=us.FiniteEnvironments(
                [
                    {"a": 1.0, "b": 0.0},
                    {"a": 0.0, "b": 1.0},
                ]
            ),
        )

        result = problem.check_public()
        self.assertFalse(result.adequate)
        self.assertIsNotNone(result.witness)
        self.assertEqual(result.witness.gap, 4.0)
        self.assertEqual(problem.global_transport_modulus().diameter, 4.0)


class PolytopeEnvironmentTests(unittest.TestCase):
    def test_local_transport_interval_with_fixed_public_law(self):
        problem = us.FiniteProblem(
            states=["a", "b", "c"],
            public={"a": "x", "b": "x", "c": "y"},
            estimand={"a": 0.0, "b": 2.0, "c": 10.0},
            environments=us.PolytopeEnvironments(
                constraints=[
                    us.eq({"c": 1.0}, 0.4),
                ]
            ),
        )

        interval = problem.local_transport_modulus({"x": 0.6, "y": 0.4})

        self.assertAlmostEqual(interval.lower, 4.0)
        self.assertAlmostEqual(interval.upper, 5.2)
        self.assertAlmostEqual(interval.diameter, 1.2)
        self.assertAlmostEqual(interval.q_lower["a"], 0.6)
        self.assertAlmostEqual(interval.q_upper["b"], 0.6)

    def test_global_transport_modulus_respects_polytope_bounds(self):
        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand={"a": 0.0, "b": 4.0},
            environments=us.PolytopeEnvironments(
                constraints=[
                    us.geq({"a": 1.0}, 0.25),
                    us.geq({"b": 1.0}, 0.25),
                ]
            ),
        )

        result = problem.check_public()
        transport = problem.global_transport_modulus()

        self.assertFalse(result.adequate)
        self.assertIsNotNone(result.witness)
        self.assertAlmostEqual(result.witness.gap, 2.0)
        self.assertAlmostEqual(transport.diameter, 2.0)
        self.assertAlmostEqual(transport.lower, 1.0)
        self.assertAlmostEqual(transport.upper, 3.0)

    def test_lp_adequate_supports_can_find_least_support(self):
        problem = us.FiniteProblem(
            states=["a", "b", "c"],
            public={"a": "o", "b": "o", "c": "o"},
            estimand={"a": 0.0, "b": 1.0, "c": 1.0},
            environments=us.PolytopeEnvironments(),
        )

        least = problem.least_support()

        self.assertTrue(least.exists)
        self.assertEqual(
            least.support,
            us.Partition.from_blocks([["a"], ["b", "c"]], universe=problem.states),
        )


class LeastSupportTests(unittest.TestCase):
    def test_line_segment_no_least_example_from_paper(self):
        problem = us.FiniteProblem(
            states=["a", "b", "c"],
            public={"a": "o", "b": "o", "c": "o"},
            estimand={"a": 0.0, "b": 1.0, "c": 2.0},
            environments=us.LineSegment(
                center={"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
                direction={"a": 0.0, "b": 1.0, "c": -1.0},
                radius=1 / 3,
            ),
        )

        least = problem.least_support()

        self.assertFalse(least.exists)
        self.assertEqual(len(least.minimal_supports), 2)
        self.assertEqual(
            {support for support in least.minimal_supports},
            {
                us.Partition.from_blocks([["a", "b"], ["c"]], universe=problem.states),
                us.Partition.from_blocks([["a", "c"], ["b"]], universe=problem.states),
            },
        )
        self.assertEqual(
            least.common_coarsening,
            us.Partition.from_blocks([["a", "b", "c"]], universe=problem.states),
        )
        self.assertAlmostEqual(problem.global_transport_modulus().diameter, 2 / 3)


if __name__ == "__main__":
    unittest.main()
