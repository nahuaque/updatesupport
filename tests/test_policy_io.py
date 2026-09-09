import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
import json
from unittest.mock import patch
import pytest
import updatesupport as us
from updatesupport.policy import (
    FrozenClaimPolicy,
    FrozenPublicReportPolicy,
    FrozenSupportCell,
)


def _policy(categories=("a", "b")):
    q = us.q_tv_budget(0.2)
    claim = us.claim(
        "Stable rate",
        public=["segment"],
        hidden=["segment", "channel"],
        target="rate",
        weight="weight",
        ambiguity_limit=1.0,
        q=q,
        q_presets=[q],
        min_cell_weight=10,
        max_dropped_weight_share=0.05,
    )
    return FrozenPublicReportPolicy(
        name="Quarterly reporting",
        source_design_status="calibrated_design_found",
        source_period_column="quarter",
        coverage=0.9,
        historical_row_count=12,
        design_row_count=2,
        recommended_public=("segment",),
        claims=(
            FrozenClaimPolicy(
                claim_index=1,
                claim=claim,
                calibrated_radius=0.2,
                reference_support=tuple(
                    (
                        FrozenSupportCell(
                            state=("all", category), public_value=("all",), mass=0.5
                        )
                        for category in categories
                    )
                ),
            ),
        ),
    )


def _rows():
    return [
        {"segment": "all", "channel": "a", "rate": 0.3, "weight": 10},
        {"segment": "all", "channel": "b", "rate": 0.7, "weight": 10},
    ]


class PolicyPersistenceTests(unittest.TestCase):
    def test_policy_round_trip_preserves_category_types_and_contract(self):
        for categories in [("a", "b"), (1, "1"), (("a", 1), ("b", 2)), (False, True)]:
            with self.subTest(categories=categories):
                with tempfile.TemporaryDirectory() as directory:
                    tmp_path = Path(directory)
                    policy = _policy(categories)
                    path = tmp_path / "policy.json"
                    policy.save(path)
                    with (
                        patch.object(
                            us.ClaimSpec,
                            "audit",
                            side_effect=AssertionError("unexpected audit"),
                        ),
                        patch.object(
                            us.ClaimSpec,
                            "calibrate_tv",
                            side_effect=AssertionError("unexpected fit"),
                        ),
                    ):
                        restored = FrozenPublicReportPolicy.load(path)
                    self.assertIsNot(restored, policy)
                    self.assertEqual(restored.fingerprint, policy.fingerprint)
                    self.assertEqual(restored.as_dict(), policy.as_dict())
                    self.assertEqual(
                        restored.claims[0].reference_hidden_cells,
                        policy.claims[0].reference_hidden_cells,
                    )
                    self.assertEqual(
                        restored.claims[0].claim.max_dropped_weight_share, 0.05
                    )
                    self.assertEqual(restored.to_dict()["schema_version"], 1)
                    self.assertEqual(
                        FrozenPublicReportPolicy.from_dict(policy.to_dict()).as_dict(),
                        policy.as_dict(),
                    )

    def test_loaded_policy_reproduces_audits_and_backtests(self):
        pytest.importorskip("cvxpy")
        policy = _policy()
        restored = FrozenPublicReportPolicy.from_json(policy.to_json())
        self.assertEqual(
            restored.audit(_rows()).as_dict(), policy.audit(_rows()).as_dict()
        )
        rows = [dict(row, quarter=period) for period in ("Q1", "Q2") for row in _rows()]
        self.assertEqual(
            restored.backtest(rows, period="quarter").as_dict(),
            policy.backtest(rows, period="quarter").as_dict(),
        )
        sparse = [_rows()[0], dict(_rows()[1], weight=9)]
        self.assertEqual(restored.audit(sparse).status, "inconclusive")

    def test_unknown_schema_versions_are_rejected(self):
        for version in [0, 2, True, "1"]:
            with self.subTest(version=version):
                payload = _policy().to_dict()
                payload["schema_version"] = version
                with pytest.raises(ValueError, match="schema_version"):
                    FrozenPublicReportPolicy.from_dict(payload)

    def test_report_export_without_schema_is_not_silently_treated_as_a_policy(self):
        with pytest.raises(ValueError, match="versioned frozen policy"):
            FrozenPublicReportPolicy.from_dict(_policy().as_dict())

    def test_changed_contract_rejects_saved_fingerprint(self):
        payload = _policy().to_dict()
        payload["claims"][0]["claim"]["max_dropped_weight_share"] = 0.9
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            FrozenPublicReportPolicy.from_dict(payload)

    def test_unknown_missing_and_inconsistent_fields_are_rejected(self):
        for mutation in ["unknown", "missing", "derived"]:
            with self.subTest(mutation=mutation):
                payload = _policy().to_dict()
                if mutation == "unknown":
                    payload["claims"][0]["new_constraint"] = 1
                elif mutation == "missing":
                    del payload["claims"][0]["claim"]["title"]
                else:
                    payload["claim_count"] = 7
                with pytest.raises(
                    ValueError, match="unknown, missing, or inconsistent"
                ):
                    FrozenPublicReportPolicy.from_dict(payload)

    def test_invalid_reference_contract_is_rejected(self):
        for mutation in ["mass", "state", "duplicate", "radius"]:
            with self.subTest(mutation=mutation):
                payload = _policy().to_dict()
                row = payload["claims"][0]
                if mutation == "mass":
                    row["reference_support"][0]["mass"] = -1
                elif mutation == "state":
                    row["reference_support"][0]["public_value"] = ["wrong"]
                elif mutation == "duplicate":
                    row["reference_support"][1] = row["reference_support"][0]
                else:
                    row["calibrated_radius"] = 0.3
                with pytest.raises(ValueError, match="invalid frozen policy contract"):
                    FrozenPublicReportPolicy.from_dict(payload)

    def test_nonportable_target_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            policy = _policy()
            row = policy.claims[0]
            custom = replace(
                policy,
                claims=(
                    replace(
                        row,
                        claim=replace(
                            row.claim,
                            target=us.row_metric("rate", lambda row: row["rate"]),
                        ),
                    ),
                ),
            )
            path = tmp_path / "policy.json"
            with pytest.raises(TypeError, match="column targets"):
                custom.save(path)
            self.assertFalse(path.exists())
            self.assertEqual(custom.as_dict()["claims"][0]["claim"]["target"], "rate")

    def test_duplicate_keys_and_nonfinite_values_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate policy JSON key"):
            FrozenPublicReportPolicy.from_json(
                '{"schema_version": 1, "schema_version": 2}'
            )
        payload = _policy().to_dict()
        payload["coverage"] = float("nan")
        with pytest.raises(ValueError, match="finite numbers"):
            FrozenPublicReportPolicy.from_json(json.dumps(payload))
