import json
from dataclasses import replace
import unittest

import updatesupport as us

check = unittest.TestCase()


def rows():
    return [
        {"sector": "a", "issuer_id": "A", "age": "young", "x": 0, "y": 1, "weight": 30},
        {"sector": "a", "issuer_id": "B", "age": "old", "x": 1, "y": 0, "weight": 20},
        {"sector": "b", "issuer_id": "C", "age": "young", "x": 0, "y": 0, "weight": 50},
    ]


def test_primary_q_roundtrip_keeps_shared_grid_and_distinct_settings():
    def scenarios(claims):
        result = us.ClaimPortfolio(claims).design(rows(), max_added_columns=0)
        return [
            [(s.q_name, s.lower, s.upper) for s in c.scenarios]
            for c in result.base.claim_results
        ]

    for explicit in (False, True):
        qs = (
            us.q_tv_budget(0.1),
            us.q_tv_budget(0.2),
            us.q_tv_budget(0.2, solver="CLARABEL"),
        )
        claims = [
            us.claim(
                n,
                public=("sector",),
                hidden=("sector", "issuer_id", "age"),
                target=n,
                weight="weight",
                min_cell_weight=0,
                ambiguity_limit=0.05,
                q=qs[0] if explicit else None,
                q_presets=qs,
            )
            for n in ("x", "y")
        ]
        restored = [
            us.ClaimSpec.from_dict(json.loads(json.dumps(c.as_dict()))) for c in claims
        ]
        before, after = scenarios(claims), scenarios(restored)
        check.assertEqual(len(before[0]), 3)
        check.assertEqual(before, after)
    defaults = [
        us.claim(
            n,
            public=("sector",),
            hidden=("sector", "issuer_id", "age"),
            target=n,
            weight="weight",
            ambiguity_limit=0.05,
        )
        for n in ("x", "y")
    ]
    check.assertEqual(
        scenarios(defaults),
        scenarios([us.ClaimSpec.from_dict(c.as_dict()) for c in defaults]),
    )


def test_local_drills_are_portable_keep_other_buckets_and_exclude_identity():
    suggested = us.suggest_conditional_refinements(
        rows(),
        public=("sector",),
        targets=("x", "y"),
        candidate_columns=("age", "issuer_id", "x"),
        weight="weight",
    )
    check.assertEqual(len(suggested.candidates), 1)
    check.assertTrue(suggested.exact)
    drill = us.ConditionalRefinement.from_dict(
        json.loads(json.dumps(suggested.candidates[0].as_dict()))
    )
    check.assertEqual(drill.predicate, {"sector": "a"})
    transformed = drill.transform(rows())
    check.assertNotEqual(transformed[0][drill.name], transformed[1][drill.name])
    check.assertEqual(transformed[2][drill.name], ("unsplit",))
    with check.assertRaisesRegex(ValueError, "already exists"):
        drill.transform(transformed)
    bounded = us.suggest_conditional_refinements(
        rows(),
        public=("sector",),
        targets=("x",),
        candidate_columns=("age",),
        max_candidates=0,
    )
    check.assertFalse(bounded.exact)
    check.assertEqual(bounded.possible_count, 1)
    no_separator = [dict(r, age="same") for r in rows()]
    check.assertEqual(
        us.suggest_conditional_refinements(
            no_separator, public=("sector",), targets=("x",), candidate_columns=("age",)
        ).candidates,
        (),
    )


def test_frozen_declared_q_contract_support_contraction_and_new_support():
    claim = us.claim(
        "x",
        public=("sector",),
        hidden=("sector", "issuer_id", "age"),
        target="x",
        weight="weight",
        min_cell_weight=0,
        q="saturated",
        ambiguity_limit=1,
    )
    contract = us.FrozenReportContract.freeze(claim, rows())
    restored = us.FrozenReportContract.from_json(contract.to_json())
    check.assertEqual(contract.as_dict(), restored.as_dict())
    result = restored.audit([rows()[0], rows()[2]])
    check.assertEqual(result["status"], "pass")
    check.assertEqual(result["reason"], "support_contraction")
    check.assertEqual(result["support"]["reference_hidden_cell_count"], 3)
    check.assertNotIn("calibrated_radius", result["support"])
    check.assertEqual(restored.audit([rows()[0]])["status"], "inconclusive")
    check.assertEqual(
        restored.audit([*rows(), dict(rows()[0], issuer_id="D")])["reason"],
        "unsupported_support",
    )
    check.assertEqual(restored.audit([])["reason"], "no_snapshot")
    with check.assertRaisesRegex(ValueError, "mass"):
        replace(
            contract,
            reference_support=(replace(contract.reference_support[0], mass=-1),),
        )
