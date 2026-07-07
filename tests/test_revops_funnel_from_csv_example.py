from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import updatesupport as us
from examples import revops_funnel_from_csv as recipe
from examples import revops_funnel_stability
from examples import revops_funnel_trend_stability


class RevOpsFunnelFromCsvExampleTests(unittest.TestCase):
    def test_level_csv_recipe_builds_review_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "funnel_cells.csv"
            output_dir = Path(directory) / "review"
            _write_csv(csv_path, revops_funnel_stability.synthetic_funnel_rows())

            rows = recipe.load_csv_rows(csv_path)
            artifacts = recipe.build_level_artifacts(
                rows,
                public=revops_funnel_stability.PUBLIC_COLUMNS,
                hidden=revops_funnel_stability.HIDDEN_COLUMNS,
                target=revops_funnel_stability.CONVERSION_COLUMN,
                weight=revops_funnel_stability.WEIGHT_COLUMN,
                threshold=0.18,
            )
            written = recipe.export_level_review_artifacts(
                output_dir,
                rows,
                public=revops_funnel_stability.PUBLIC_COLUMNS,
                hidden=revops_funnel_stability.HIDDEN_COLUMNS,
                target=revops_funnel_stability.CONVERSION_COLUMN,
                weight=revops_funnel_stability.WEIGHT_COLUMN,
                threshold=0.18,
                include_frontier=False,
            )

            public_report = artifacts["public_report"]
            claim = artifacts["claim_audit"]

            self.assertIsInstance(public_report, us.PublicDescentReport)
            self.assertGreater(public_report.observed_value, 0.18)
            self.assertLess(public_report.interval.lower, 0.18)
            self.assertEqual(claim.status, "fail")
            self.assertGreater(len(written), 5)
            self.assertTrue((output_dir / "revops_csv_level_claim.md").exists())
            self.assertTrue((output_dir / "public_report.json").exists())
            self.assertTrue((output_dir / "claim_audit.json").exists())
            self.assertTrue(
                (output_dir / "tables" / "claim_audit__summary.csv").exists()
            )

    def test_trend_csv_recipe_builds_review_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "paired_funnel_cells.csv"
            output_dir = Path(directory) / "trend_review"
            _write_csv(csv_path, revops_funnel_trend_stability.synthetic_trend_rows())

            rows = recipe.load_csv_rows(csv_path)
            artifacts = recipe.build_trend_artifacts(
                rows,
                public=revops_funnel_trend_stability.PUBLIC_COLUMNS,
                hidden=revops_funnel_trend_stability.HIDDEN_COLUMNS,
                current_target="FY26Q2_sql_conversion_rate",
                prior_target="FY26Q1_sql_conversion_rate",
                current_label="FY26Q2",
                prior_label="FY26Q1",
                weight=revops_funnel_trend_stability.WEIGHT_COLUMN,
            )
            markdown = recipe.render_trend_report(
                rows,
                public=revops_funnel_trend_stability.PUBLIC_COLUMNS,
                hidden=revops_funnel_trend_stability.HIDDEN_COLUMNS,
                current_target="FY26Q2_sql_conversion_rate",
                prior_target="FY26Q1_sql_conversion_rate",
                current_label="FY26Q2",
                prior_label="FY26Q1",
                weight=revops_funnel_trend_stability.WEIGHT_COLUMN,
                include_frontier=False,
            )
            written = recipe.export_trend_review_artifacts(
                output_dir,
                rows,
                public=revops_funnel_trend_stability.PUBLIC_COLUMNS,
                hidden=revops_funnel_trend_stability.HIDDEN_COLUMNS,
                current_target="FY26Q2_sql_conversion_rate",
                prior_target="FY26Q1_sql_conversion_rate",
                current_label="FY26Q2",
                prior_label="FY26Q1",
                weight=revops_funnel_trend_stability.WEIGHT_COLUMN,
                include_frontier=False,
            )

            trend_report = artifacts["trend_report"]
            claim = artifacts["trend_claim"]
            comparison = artifacts["period_comparison"]

            self.assertGreater(trend_report.observed_value, 0.0)
            self.assertLess(trend_report.interval.lower, 0.0)
            self.assertEqual(claim.status, "fail")
            self.assertEqual(comparison.status, "ambiguous_winner")
            self.assertIn("RevOps Funnel CSV Trend Claim Audit", markdown)
            self.assertGreater(len(written), 5)
            self.assertTrue((output_dir / "revops_csv_trend_claim.md").exists())
            self.assertTrue((output_dir / "trend_report.json").exists())
            self.assertTrue((output_dir / "trend_claim.json").exists())
            self.assertTrue((output_dir / "period_comparison.json").exists())

    def test_level_cli_writes_review_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "funnel_cells.csv"
            output_dir = Path(directory) / "review"
            _write_csv(csv_path, revops_funnel_stability.synthetic_funnel_rows())

            recipe.main(
                [
                    "--input",
                    str(csv_path),
                    "--mode",
                    "level",
                    "--public",
                    *revops_funnel_stability.PUBLIC_COLUMNS,
                    "--hidden",
                    *revops_funnel_stability.HIDDEN_COLUMNS,
                    "--target",
                    revops_funnel_stability.CONVERSION_COLUMN,
                    "--weight",
                    revops_funnel_stability.WEIGHT_COLUMN,
                    "--threshold",
                    "0.18",
                    "--output-dir",
                    str(output_dir),
                    "--no-frontier",
                ]
            )

            self.assertTrue((output_dir / "revops_csv_level_claim.md").exists())
            self.assertTrue(
                (output_dir / "tables" / "public_report__summary.csv").exists()
            )

    def test_csv_recipe_rejects_missing_columns(self):
        rows = [{"segment": "smb", "target": "0.1", "weight": "10"}]

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            recipe.build_level_artifacts(
                rows,
                public=("segment",),
                hidden=("segment", "channel"),
                target="target",
                weight="weight",
                threshold=0.0,
            )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
