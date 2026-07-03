from __future__ import annotations

import json
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def test_core_colab_demo_notebooks_are_valid_and_unexecuted(self):
        notebook_dir = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
        notebooks = sorted(notebook_dir.glob("*.ipynb"))

        self.assertEqual(
            {path.name for path in notebooks},
            {"econml_downstream_reporting_colab.ipynb"},
        )
        for path in notebooks:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            self.assertGreaterEqual(len(payload["cells"]), 12)
            source = "\n".join(
                "".join(cell.get("source", ())) for cell in payload["cells"]
            )
            self.assertIn("colab.research.google.com", source)
            self.assertIn("updatesupport[causal,examples]", source)
            self.assertIn("CausalForestDML", source)
            self.assertIn("TwoModelEffectEstimator", source)
            self.assertIn("ECONML_AVAILABLE", source)
            self.assertIn("adapt_econml_effects", source)
            self.assertIn("estimator.effect(X)", source)
            self.assertIn("tau_hat", source)
            self.assertIn("public_representation_frontier", source)
            for cell in payload["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell.get("execution_count"))
                    self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
