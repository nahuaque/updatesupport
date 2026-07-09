from __future__ import annotations

import json
import unittest

import updatesupport as us


def _rows() -> list[dict[str, object]]:
    return [
        {
            "segment": "all",
            "category": "a",
            "target": 0.0,
            "weight": 100.0,
        },
        {
            "segment": "all",
            "category": "b",
            "target": 0.1,
            "weight": 100.0,
        },
        {
            "segment": "all",
            "category": "c",
            "target": 1.0,
            "weight": 100.0,
        },
    ]


def _claim(*, ambiguity_limit: float = 0.1) -> us.ClaimSpec:
    return us.claim(
        "target rate is stable enough to report",
        public=["segment"],
        hidden=["segment", "category"],
        target="target",
        weight="weight",
        ambiguity_limit=ambiguity_limit,
    )


class CategoricalRollupDesignTests(unittest.TestCase):
    def test_exact_search_finds_smallest_certifying_intermediate_rollup(self):
        design = us.design_categorical_rollup(
            _rows(),
            _claim(),
            column="category",
            max_groups=2,
            output_column="category_group",
        )

        self.assertIsInstance(design, us.CategoricalRollupDesign)
        self.assertEqual(design.status, "certifying_rollup_found")
        self.assertEqual(design.evaluated_partition_count, 4)
        self.assertEqual(design.base.groups, (("a", "b", "c"),))
        self.assertAlmostEqual(design.base.ambiguity, 1.0)
        self.assertEqual(design.selected.groups, (("a", "b"), ("c",)))
        self.assertEqual(design.selected.group_count, 2)
        self.assertEqual(design.selected.public_cells, 2)
        self.assertAlmostEqual(design.selected.ambiguity, 2.0 / 30.0)
        self.assertTrue(design.selected.certifies_claim)
        self.assertTrue(design.uses_rollup_column)
        self.assertEqual(
            design.recommended_public,
            ("segment", "category_group"),
        )

        transformed = design.transform(_rows())
        self.assertEqual(
            [row[design.output_column] for row in transformed],
            [1, 1, 2],
        )
        audit = design.audit(_rows())
        self.assertTrue(audit.passed)
        self.assertEqual(audit.primary.grouped.q_name, "saturated")
        self.assertAlmostEqual(audit.ambiguity, design.selected.ambiguity)

        tables = design.to_tables()
        payload = json.loads(design.to_json())
        markdown = design.to_markdown()
        self.assertIn("selected_groups", tables)
        self.assertIn("frontier", tables)
        self.assertEqual(payload["selected"]["group_count"], 2)
        self.assertIn("Selected Category Groups", markdown)
        self.assertIn("G1={a, b}", markdown)

    def test_claim_method_routes_to_rollup_design(self):
        design = _claim().design_categorical_rollup(
            _rows(),
            column="category",
            max_groups=2,
        )

        self.assertIsInstance(design, us.CategoricalRollupDesign)
        self.assertEqual(design.selected.groups, (("a", "b"), ("c",)))

    def test_bucket_budget_can_leave_only_the_base_representation(self):
        design = us.design_categorical_rollup(
            _rows(),
            _claim(),
            column="category",
            max_groups=2,
            bucket_budget=1,
        )

        self.assertEqual(design.status, "no_certifying_rollup")
        self.assertEqual(design.selected.group_count, 1)
        self.assertFalse(design.selected.certifies_claim)
        self.assertFalse(design.uses_rollup_column)
        self.assertEqual(design.recommended_public, ("segment",))

    def test_decision_rule_can_drive_rollup_selection_without_ambiguity_limit(self):
        claim = us.claim(
            "reported target remains above launch threshold",
            public=["segment"],
            hidden=["segment", "category"],
            target="target",
            weight="weight",
            decision=us.threshold_decision(">=", 0.2),
        )

        design = claim.design_categorical_rollup(
            _rows(),
            column="category",
            max_groups=2,
        )

        self.assertEqual(design.status, "certifying_rollup_found")
        self.assertEqual(design.selected.groups, (("a", "b"), ("c",)))
        self.assertTrue(design.selected.decision_invariant)
        self.assertTrue(design.selected.decision_certified)
        self.assertTrue(design.selected.certifies_claim)

    def test_full_refinement_uses_original_category_column(self):
        design = us.design_categorical_rollup(
            _rows(),
            _claim(ambiguity_limit=0.0),
            column="category",
            max_groups=3,
        )

        self.assertEqual(design.selected.group_count, 3)
        self.assertAlmostEqual(design.selected.ambiguity, 0.0)
        self.assertFalse(design.uses_rollup_column)
        self.assertEqual(design.recommended_public, ("segment", "category"))
        self.assertEqual(design.selected_claim.public, ("segment", "category"))
        self.assertTrue(design.audit(_rows()).passed)

    def test_search_rejects_non_saturated_claim(self):
        claim = us.claim(
            "TV claim",
            public=["segment"],
            hidden=["segment", "category"],
            target="target",
            weight="weight",
            q_presets=[us.q_tv_budget(0.1)],
            ambiguity_limit=0.1,
        )

        with self.assertRaisesRegex(ValueError, "requires.*saturated"):
            us.design_categorical_rollup(
                _rows(),
                claim,
                column="category",
            )

    def test_exact_search_guard_rejects_too_many_categories(self):
        rows = [
            {
                "segment": "all",
                "category": category,
                "target": float(index),
                "weight": 1.0,
            }
            for index, category in enumerate(("a", "b", "c", "d"))
        ]

        with self.assertRaisesRegex(ValueError, "Bell numbers"):
            us.design_categorical_rollup(
                rows,
                _claim(),
                column="category",
                max_categories=3,
            )

    def test_transform_rejects_categories_outside_designed_support(self):
        design = us.design_categorical_rollup(
            _rows(),
            _claim(),
            column="category",
            max_groups=2,
        )
        future = [
            {
                "segment": "all",
                "category": "new",
                "target": 0.5,
                "weight": 10.0,
            }
        ]

        with self.assertRaisesRegex(ValueError, "unseen rollup category"):
            design.transform(future)


if __name__ == "__main__":
    unittest.main()
