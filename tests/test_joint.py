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


if __name__ == "__main__":
    unittest.main()
