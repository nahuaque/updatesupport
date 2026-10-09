"""Synthetic cash reconstruction and minimum alternative explanation, offline."""

import updatesupport as us
import updatesupport_finance as f


def run():
    scope = f.DisclosureScope(
        "ExampleCo",
        "advance liabilities",
        "USD million",
        "2026-06-30",
        "H1",
        "all customers",
        period_start="2026-01-01",
        currency="USD",
    )
    bridge = f.CustomerAdvanceBridge(
        name="advances",
        scope=scope,
        opening="opening",
        closing="closing",
        movements=[
            f.AdvanceMovement(
                "cash",
                "net_cash",
                "Net cash including collections of old unpaid advances",
            ),
            f.AdvanceMovement(
                "recognition", "recognition", "Recognition from advance liabilities"
            ),
            f.AdvanceMovement(
                "unpaid_open", "unpaid_opening", "Opening unpaid advances"
            ),
            f.AdvanceMovement(
                "unpaid_close", "unpaid_closing", "Closing unpaid advances"
            ),
            f.AdvanceMovement(
                "other", "noncash", "All other signed noncash liability movements"
            ),
        ],
        accounting_regime="Illustrative reviewed liability policy",
        recognition_basis="Recognition excludes separately mapped noncash adjustments",
        complete=True,
        completeness_basis="Synthetic exhaustive classification; noncash amounts remain unknown",
        cash_flow_adjustment="cf_adjustment",
    )
    # Exact synthetic inputs. For filings, compile normalized DisclosureFacts
    # with disclosure_fact_constraint to retain source rounding and provenance.
    facts = [
        f.exact_disclosure_constraint(
            f"fact:{n}",
            n,
            value,
            category="reported_fact",
            provenance="Synthetic example",
        )
        for n, value in {
            "opening": 10,
            "closing": 100,
            "recognition": 10,
            "unpaid_open": 0,
            "cfo": 100,
            "cf_adjustment": 88,
        }.items()
    ]
    cap = f.interval_disclosure_constraint(
        "unpaid_close_cap",
        "unpaid_close",
        upper=5,
        category="reported_fact",
        provenance="Synthetic gross receivables cap",
    )
    problem = bridge.problem(
        constraints=[*facts, cap],
        variables=[f.disclosure_variable("cfo", lower=None, unit=scope.unit)],
    )
    report = us.solve_named_linear_feasibility(problem)
    cash = report.interval(target=bridge.net_cash, scenario="reported")
    gap = report.interval(target=bridge.cash_flow_gap, scenario="reported")
    print(f"Net cash remains unidentified: [{cash.lower}, {cash.upper}]")
    print(f"Liability / indirect cash-flow diagnostic gap: {gap.lower:g} {scope.unit}")

    analysis = f.minimum_advance_explanation(
        bridge,
        problem,
        cash_ceiling={"cfo": 0.8},
        ceiling_description="Hypothetical net advance cash at most 80% of reported CFO",
    )
    requirement = analysis.report.interval(
        target=bridge.noncash, scenario="advance_explanation"
    )
    print(
        f"Under that condition, net noncash additions must be at least {requirement.lower:g} {scope.unit}"
    )
    print(
        "This is a necessary alternative explanation; its likelihood is not assessed."
    )

    # A separate exhaustive table can identify cash even with an ASC 842
    # straight-line reduction, provided its source-specific mapping is reviewed.
    control = f.CustomerAdvanceBridge(
        "control",
        scope,
        "start",
        "end",
        [
            f.AdvanceMovement(
                "control_cash", "net_cash", "Net cash in exhaustive table"
            ),
            f.AdvanceMovement("earned", "recognition", "Billed/earned recognition"),
            f.AdvanceMovement(
                "straight_line",
                "noncash",
                "Source-positive noncash reduction",
                effect=-1,
            ),
        ],
        "Illustrative ASC 842 table",
        "Billed/earned, separate from straight-line adjustment",
        True,
        "Synthetic exhaustive table with no other movements",
    )
    control_problem = control.problem(
        constraints=[
            f.exact_disclosure_constraint(
                f"control:{n}", n, value, category="reported_fact"
            )
            for n, value in {
                "start": 10,
                "end": 30,
                "earned": 4,
                "straight_line": 3,
            }.items()
        ]
    )
    reconstructed = us.solve_named_linear_feasibility(control_problem).interval(
        target=control.net_cash,
        scenario="reported",
    )
    print(
        f"Exhaustive table reconstructs net cash: {reconstructed.lower:g} {scope.unit}"
    )
    return bridge, problem, analysis, control_problem


if __name__ == "__main__":
    run()
