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


def _require_cvxpy_solver(name: str) -> None:
    _require_cvxpy()
    import cvxpy as cp

    installed = {solver.upper() for solver in cp.installed_solvers()}
    if name.upper() not in installed:
        raise unittest.SkipTest(f"CVXPY solver {name!r} is not installed")


def _rows():
    return [
        {"PUBLIC": "A", "HIDDEN": "x", "Y": 0.0, "W": 3.0},
        {"PUBLIC": "A", "HIDDEN": "y", "Y": 1.0, "W": 3.0},
        {"PUBLIC": "B", "HIDDEN": "z", "Y": 0.5, "W": 4.0},
    ]


class CvxpyBackendTests(unittest.TestCase):
    def test_missing_scip_solver_error_mentions_optional_extra(self):
        _require_cvxpy()

        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand={"a": 0.0, "b": 1.0},
            environments=us.CvxpyEnvironments(
                fixed_public_law={"o": 1.0},
                solver="SCIP",
            ),
        )

        with mock.patch("cvxpy.installed_solvers", return_value=[]):
            with self.assertRaisesRegex(us.CvxpyError, r"updatesupport\[scip\]"):
                problem.global_transport_modulus()

    def test_scip_solver_solves_interval_when_installed(self):
        _require_cvxpy_solver("SCIP")

        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand={"a": 0.0, "b": 1.0},
            environments=us.CvxpyEnvironments(
                fixed_public_law={"o": 1.0},
                solver="scip",
            ),
        )

        interval = problem.global_transport_modulus()

        self.assertAlmostEqual(interval.lower, 0.0, places=5)
        self.assertAlmostEqual(interval.upper, 1.0, places=5)
        self.assertAlmostEqual(interval.diameter, 1.0, places=5)

    def test_fiber_support_floor_uses_scip_mip_when_installed(self):
        _require_cvxpy_solver("SCIP")

        rows = [
            {"PUBLIC": "A", "HIDDEN": "low", "Y": 0.0, "W": 25.0},
            {"PUBLIC": "A", "HIDDEN": "high", "Y": 1.0, "W": 25.0},
            {"PUBLIC": "B", "HIDDEN": "low", "Y": 0.0, "W": 25.0},
            {"PUBLIC": "B", "HIDDEN": "high", "Y": 1.0, "W": 25.0},
        ]

        grouped = us.from_dataframe(
            rows,
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_fiber_support_floor(2, min_share=0.25),
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertEqual(grouped.problem.environments.solver, "SCIP")
        self.assertAlmostEqual(interval.lower, 0.25, places=5)
        self.assertAlmostEqual(interval.upper, 0.75, places=5)
        self.assertAlmostEqual(interval.diameter, 0.5, places=5)

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

    def test_support_function_backend_solves_linear_interval(self):
        _require_cvxpy()

        def cap_b(_cp, q, _states, state_index):
            return (
                us.cvxpy_constraint(
                    q[state_index["b"]] <= 0.75,
                    name="cap hidden cell b",
                    kind="cell_cap",
                    sense="<=",
                    state="b",
                ),
            )

        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand={"a": 0.0, "b": 1.0},
            environments=us.SupportFunctionBackend(
                fixed_public_law={"o": 1.0},
                constraint_builders=(cap_b,),
            ),
        )

        interval = problem.global_transport_modulus()

        self.assertAlmostEqual(interval.lower, 0.0, places=6)
        self.assertAlmostEqual(interval.upper, 0.75, places=6)
        self.assertAlmostEqual(interval.diameter, 0.75, places=6)
        self.assertAlmostEqual(interval.q_lower["a"], 1.0, places=6)
        self.assertAlmostEqual(interval.q_upper["b"], 0.75, places=6)
        self.assertIn("cell_cap", {row.kind for row in interval.duals})

    def test_convex_admissible_set_exposes_support_function(self):
        _require_cvxpy()

        def cap_b(_cp, q, _states, state_index):
            return (q[state_index["b"]] <= 0.75,)

        env = us.SupportFunctionBackend(
            fixed_public_law={"o": 1.0},
            constraint_builders=(cap_b,),
        )
        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand={"a": 0.0, "b": 1.0},
            environments=env,
        )
        admissible_set = env.convex_admissible_set(
            problem,
            public_law={"o": 1.0},
        )

        result = admissible_set.support_value([0.0, 1.0])
        backend_result = env.support_value(
            problem,
            [0.0, 1.0],
            public_law={"o": 1.0},
        )

        self.assertIsInstance(admissible_set, us.ConvexAdmissibleSet)
        self.assertIsInstance(result, us.SupportFunctionResult)
        self.assertAlmostEqual(result.value, 0.75, places=5)
        self.assertAlmostEqual(result.vector[1], 0.75, places=5)
        self.assertAlmostEqual(backend_result.value, 0.75, places=5)

    def test_support_function_backend_can_be_selected_from_q_preset(self):
        _require_cvxpy()

        grouped = us.from_dataframe(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_tv_budget(0.15, backend="support_function"),
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertIsInstance(grouped.problem.environments, us.SupportFunctionBackend)
        self.assertEqual(grouped.q_name, "tv_budget(radius=0.15)")
        self.assertAlmostEqual(interval.lower, 0.35, places=5)
        self.assertAlmostEqual(interval.upper, 0.65, places=5)
        self.assertAlmostEqual(interval.diameter, 0.30, places=5)

    def test_batched_cvxpy_solves_multiple_public_laws(self):
        _require_cvxpy()

        env = us.BatchedCvxpyEnvironments()
        problem = us.FiniteProblem(
            states=["a", "b", "c"],
            public={"a": "x", "b": "x", "c": "y"},
            estimand={"a": 0.0, "b": 1.0, "c": 2.0},
            environments=env,
        )

        intervals = env.batched_local_transport(
            problem,
            [
                {"x": 1.0, "y": 0.0},
                {"x": 0.5, "y": 0.5},
            ],
        )

        self.assertEqual(len(intervals), 2)
        self.assertAlmostEqual(intervals[0].lower, 0.0, places=6)
        self.assertAlmostEqual(intervals[0].upper, 1.0, places=6)
        self.assertAlmostEqual(intervals[1].lower, 1.0, places=6)
        self.assertAlmostEqual(intervals[1].upper, 1.5, places=6)
        self.assertTrue(intervals[0].duals)
        self.assertTrue(
            all(row.index is None or row.index[0] == 0 for row in intervals[0].duals)
        )
        self.assertTrue(
            all(row.index is None or row.index[0] == 1 for row in intervals[1].duals)
        )

    def test_batched_cvxpy_backend_can_be_selected_from_q_preset(self):
        _require_cvxpy()

        grouped = us.from_dataframe(
            _rows(),
            public=["PUBLIC"],
            hidden=["PUBLIC", "HIDDEN"],
            target="Y",
            weight="W",
            q=us.q_tv_budget(0.15, backend="batched_cvxpy"),
        )

        interval = grouped.problem.global_transport_modulus()

        self.assertIsInstance(grouped.problem.environments, us.BatchedCvxpyEnvironments)
        self.assertAlmostEqual(interval.lower, 0.35, places=6)
        self.assertAlmostEqual(interval.upper, 0.65, places=6)

    def test_sensitivity_report_routes_batched_cvxpy_presets_together(self):
        _require_cvxpy()

        calls = []
        original = us.BatchedCvxpyEnvironments.batched_local_transport

        def spy(self, problem, public_laws):
            calls.append(len(public_laws))
            return original(self, problem, public_laws)

        with mock.patch.object(
            us.BatchedCvxpyEnvironments,
            "batched_local_transport",
            spy,
        ):
            report = us.sensitivity_report(
                _rows(),
                public=["PUBLIC"],
                hidden=["PUBLIC", "HIDDEN"],
                target="Y",
                weight="W",
                q_presets=[
                    us.q_tv_budget(0.15, backend="batched_cvxpy"),
                    us.q_tv_budget(0.30, backend="batched_cvxpy"),
                ],
            )

        self.assertEqual(calls, [2])
        self.assertEqual(len(report.rows), 2)
        self.assertAlmostEqual(report.rows[0].ambiguity, 0.30, places=6)
        self.assertAlmostEqual(report.rows[1].ambiguity, 0.60, places=6)

    def test_dqcp_ratio_target_solves_fixed_public_interval(self):
        _require_cvxpy()

        problem = us.FiniteProblem(
            states=["a", "b", "c"],
            public={"a": "x", "b": "x", "c": "y"},
            estimand=us.RatioTarget(
                numerator={"a": 1.0, "b": 4.0, "c": 10.0},
                denominator={"a": 1.0, "b": 2.0, "c": 5.0},
                name="loss_ratio",
            ),
            environments=us.CvxpyEnvironments(
                fixed_public_law={"x": 0.5, "y": 0.5},
            ),
        )

        interval = problem.global_transport_modulus()

        self.assertAlmostEqual(interval.lower, 11 / 6, places=4)
        self.assertAlmostEqual(interval.upper, 2.0, places=4)
        self.assertAlmostEqual(interval.diameter, 1 / 6, places=4)
        self.assertEqual(interval.public_law, {"x": 0.5, "y": 0.5})

    def test_parameterized_cvxpy_rejects_variable_denominator_ratio_target(self):
        _require_cvxpy()

        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "x", "b": "x"},
            estimand=us.RatioTarget(
                numerator={"a": 1.0, "b": 4.0},
                denominator={"a": 1.0, "b": 2.0},
            ),
            environments=us.ParameterizedCvxpyEnvironments(
                fixed_public_law={"x": 1.0},
            ),
        )

        with self.assertRaisesRegex(
            us.UnsupportedTargetError,
            "requires a fixed linear target",
        ):
            problem.global_transport_modulus()

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

    def test_sensitivity_report_recompiles_procedure_targets_for_radius_grid(self):
        _require_cvxpy()

        compiled_radii = []

        def compiler(context):
            compiled_radii.append(context.q.radius)
            return us.row_metric(
                f"Y_radius_{context.q.radius}",
                lambda row: float(row["Y"]),
                columns=("Y",),
            )

        with mock.patch(
            "updatesupport.report.from_dataframe",
            wraps=report_module.from_dataframe,
        ) as from_dataframe:
            report = us.sensitivity_report(
                _rows(),
                public=["PUBLIC"],
                hidden=["PUBLIC", "HIDDEN"],
                target=us.ProcedureTarget("radius_sensitive_target", compiler),
                weight="W",
                q_presets=[
                    us.q_tv_budget(0.15),
                    us.q_tv_budget(0.30),
                ],
            )

        self.assertEqual(from_dataframe.call_count, 2)
        self.assertEqual(compiled_radii, [0.15, 0.30])
        self.assertEqual(len(report.rows), 2)

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

    def test_refinement_sensitivity_recompiles_procedure_targets_for_radius_grid(
        self,
    ):
        _require_cvxpy()
        compiled_contexts = []

        def compiler(context):
            compiled_contexts.append((context.public, context.q.radius))
            return us.row_metric(
                f"Y_radius_{context.q.radius}_public_{len(context.public)}",
                lambda row: float(row["Y"]),
                columns=("Y",),
            )

        with mock.patch(
            "updatesupport.report.from_dataframe",
            wraps=report_module.from_dataframe,
        ) as from_dataframe:
            report = us.recommend_refinements_sensitivity(
                _rows(),
                public=["PUBLIC"],
                hidden=["PUBLIC", "HIDDEN"],
                target=us.ProcedureTarget("radius_sensitive_target", compiler),
                candidate_refinements=["HIDDEN"],
                weight="W",
                q_presets=[
                    us.q_tv_budget(0.15),
                    us.q_tv_budget(0.30),
                ],
            )

        self.assertEqual(from_dataframe.call_count, 4)
        self.assertEqual(
            compiled_contexts,
            [
                (("PUBLIC",), 0.15),
                (("PUBLIC", "HIDDEN"), 0.15),
                (("PUBLIC",), 0.30),
                (("PUBLIC", "HIDDEN"), 0.30),
            ],
        )
        self.assertEqual(len(report.scenarios), 2)

    def test_convex_moment_transform_uses_exact_lower_and_conservative_upper(self):
        _require_cvxpy()

        target = us.MomentTransformTarget(
            moments={"score": {"a": 0.0, "b": 2.0}},
            transform=lambda moments: moments["score"] ** 2,
            curvature="convex",
            cvxpy_transform=lambda cp, moments: cp.square(moments["score"]),
            monotonicity={"score": "increasing"},
            name="squared_mean_score",
        )
        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand=target,
            environments=us.CvxpyEnvironments(fixed_public_law={"o": 1.0}),
        )

        interval = problem.global_transport_modulus()
        endpoint = problem.moment_transform_endpoint(minimize=True)

        self.assertTrue(target.contract.capabilities.supports_exact_lower)
        self.assertFalse(target.contract.capabilities.supports_exact_upper)
        self.assertEqual(interval.bound_type, "conservative")
        self.assertEqual(interval.lower_bound_type, "exact")
        self.assertEqual(interval.upper_bound_type, "conservative")
        self.assertAlmostEqual(interval.lower, 0.0, places=6)
        self.assertAlmostEqual(interval.upper, 4.0, places=6)
        self.assertIsNotNone(interval.q_lower)
        self.assertAlmostEqual(endpoint.lower, 0.0, places=6)

    def test_concave_moment_transform_uses_conservative_lower_and_exact_upper(self):
        _require_cvxpy()

        target = us.MomentTransformTarget(
            moments={"score": {"a": 0.0, "b": 4.0}},
            transform=lambda moments: moments["score"] ** 0.5,
            curvature="concave",
            cvxpy_transform=lambda cp, moments: cp.sqrt(moments["score"]),
            monotonicity={"score": "increasing"},
            name="sqrt_mean_score",
        )
        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand=target,
            environments=us.CvxpyEnvironments(fixed_public_law={"o": 1.0}),
        )

        interval = problem.global_transport_modulus()
        endpoint = problem.moment_transform_endpoint(minimize=False)

        self.assertFalse(target.contract.capabilities.supports_exact_lower)
        self.assertTrue(target.contract.capabilities.supports_exact_upper)
        self.assertEqual(interval.bound_type, "conservative")
        self.assertEqual(interval.lower_bound_type, "conservative")
        self.assertEqual(interval.upper_bound_type, "exact")
        self.assertAlmostEqual(interval.lower, 0.0, places=6)
        self.assertAlmostEqual(interval.upper, 2.0, places=6)
        self.assertIsNotNone(interval.q_upper)
        self.assertAlmostEqual(endpoint.upper, 2.0, places=6)

    def test_convex_moment_transform_endpoint_without_monotonicity_is_one_sided(self):
        _require_cvxpy()

        target = us.MomentTransformTarget(
            moments={"score": {"a": 0.0, "b": 2.0}},
            transform=lambda moments: moments["score"] ** 2,
            curvature="convex",
            cvxpy_transform=lambda cp, moments: cp.square(moments["score"]),
            name="squared_mean_score",
        )
        problem = us.FiniteProblem(
            states=["a", "b"],
            public={"a": "o", "b": "o"},
            estimand=target,
            environments=us.CvxpyEnvironments(fixed_public_law={"o": 1.0}),
        )

        endpoint = problem.moment_transform_endpoint(minimize=True)
        with self.assertRaisesRegex(
            us.UnsupportedTargetError,
            "does not provide both endpoints",
        ):
            problem.global_transport_modulus()

        self.assertFalse(target.contract.supports_interval)
        self.assertTrue(target.contract.capabilities.supports_exact_lower)
        self.assertAlmostEqual(endpoint.lower, 0.0, places=6)

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
