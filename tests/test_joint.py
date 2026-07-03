from __future__ import annotations

import unittest

import updatesupport as us


def _rows() -> list[dict[str, object]]:
    return [
        {"segment": "A", "driver": "low", "target": 0.0, "weight": 30},
        {"segment": "A", "driver": "high", "target": 1.0, "weight": 30},
        {"segment": "B", "driver": "flat", "target": 0.5, "weight": 40},
    ]


class JointDistributionTests(unittest.TestCase):
    def test_fit_joint_distribution_draws_weighted_cell_records(self):
        joint = us.fit_joint_distribution(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            method="empirical",
        )

        draw = joint.draw()
        records = draw.records()
        payload = joint.as_dict()

        self.assertIsInstance(joint, us.NonparametricJointDistribution)
        self.assertEqual(joint.cell_count, 3)
        self.assertEqual(payload["method"], "empirical")
        self.assertAlmostEqual(sum(row[draw.weight_column] for row in records), 100.0)
        self.assertEqual({row[draw.target_column] for row in records}, {0.0, 0.5, 1.0})
        self.assertEqual(us.joint_draw_records(draw), records)

    def test_bayesian_bootstrap_draws_are_seeded(self):
        joint = us.fit_joint_distribution(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            effective_sample_size=10,
        )

        first = joint.draw(seed=123)
        second = joint.draw(seed=123)
        third = joint.draw(seed=124)

        self.assertEqual(first.probabilities, second.probabilities)
        self.assertNotEqual(first.probabilities, third.probabilities)
        self.assertAlmostEqual(sum(first.probabilities), 1.0)

    def test_multinomial_bootstrap_draws_are_supported(self):
        joint = us.fit_joint_distribution(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            method="bootstrap",
            effective_sample_size=5,
        )

        draw = joint.draw(seed=123)

        self.assertEqual(joint.method, "bootstrap")
        self.assertAlmostEqual(sum(draw.probabilities), 1.0)
        self.assertTrue(
            all(value in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} for value in draw.probabilities)
        )

    def test_hidden_composition_draw_preserves_public_law(self):
        joint = us.fit_joint_distribution(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            method="bootstrap",
            effective_sample_size=5,
        )

        draw = joint.hidden_composition_draw(seed=123)
        public_masses: dict[tuple[object, ...], float] = {}
        for cell, probability in zip(draw.cells, draw.probabilities, strict=True):
            public_masses[cell.public_value] = (
                public_masses.get(cell.public_value, 0.0) + probability
            )

        self.assertEqual(set(public_masses), set(joint.public_law))
        for public_value, mass in joint.public_law.items():
            self.assertAlmostEqual(public_masses[public_value], mass)

    def test_hidden_composition_uncertainty_summarizes_draws(self):
        report = us.hidden_composition_uncertainty(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            method="bayesian_bootstrap",
            draws=8,
            seed=123,
            ambiguity_limit=0.05,
            confidence_level=0.8,
            min_cell_weight=0,
        )
        markdown = report.to_markdown()
        tables = report.to_tables()

        self.assertIsInstance(report, us.HiddenCompositionUncertaintyReport)
        self.assertEqual(report.draw_count, 8)
        self.assertEqual(report.successful_draws, 8)
        self.assertEqual(report.failed_draws, 8)
        self.assertTrue(report.preserve_public_law)
        self.assertEqual(report.ambiguity_summary.count, 8)
        self.assertEqual(report.ambiguity_summary.confidence_level, 0.8)
        self.assertLessEqual(
            report.ambiguity_summary.lower, report.ambiguity_summary.upper
        )
        self.assertIn("Metric Summaries", markdown)
        self.assertIn("summary", tables)
        self.assertIn("metric_summaries", tables)
        self.assertIn("draws", tables)
        self.assertEqual(len(tables["draws"]), 8)

    def test_hidden_composition_uncertainty_accepts_prefit_joint_model(self):
        joint = us.fit_joint_distribution(
            _rows(),
            public=["segment"],
            hidden=["segment", "driver"],
            target="target",
            weight="weight",
            method="bootstrap",
            effective_sample_size=5,
        )

        report = us.hidden_composition_uncertainty(
            joint,
            draws=4,
            seed=321,
            ambiguity_limit=0.05,
            min_cell_weight=0,
        )

        self.assertEqual(report.joint_model.method, "bootstrap")
        self.assertEqual(report.public_columns, ("segment",))
        self.assertEqual(report.hidden_columns, ("segment", "driver"))
        self.assertEqual(report.draw_count, 4)


if __name__ == "__main__":
    unittest.main()
