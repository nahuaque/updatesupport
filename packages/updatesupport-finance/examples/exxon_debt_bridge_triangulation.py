"""Exxon Mobil debt-bridge disclosure-triangulation worked example.

Run from the repository root with:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/exxon_debt_bridge_triangulation.py

Optionally write the Markdown report:

    uv run --package updatesupport-finance python \
        packages/updatesupport-finance/examples/exxon_debt_bridge_triangulation.py \
        --output data/exxon_debt_bridge_triangulation_report.md

The example is a generic stock-flow reconciliation:

    closing stock - opening stock = reported cash-flow movement + residual

Here the stock is book debt, measured as "Notes and loans payable" plus
"Long-term debt." The residual is not missing debt or an accusation. It is the
signed amount that cannot be explained by the selected cash-flow debt financing
lines alone. It can reflect noncash lease obligations, debt assumed or disposed,
foreign-currency translation, current/noncurrent reclassifications, and other
presentation differences.

Values are in USD millions and are stored directly so the example is
reproducible without live SEC access.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import updatesupport as us
import updatesupport_finance as usf


FY2025_FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm"
)
FY2025_BALANCE_SHEET_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/R5.htm"
)
FY2025_CASH_FLOW_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/R7.htm"
)
FY2025_DEBT_DETAIL_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/R88.htm"
)

Q1_2026_FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/xom-20260331.htm"
)
Q1_2026_BALANCE_SHEET_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R4.htm"
)
Q1_2026_CASH_FLOW_URL = (
    "https://www.sec.gov/Archives/edgar/data/34088/000003408826000067/R6.htm"
)

BALANCE_SHEET_TIER = "T0 balance-sheet debt movement"
FULL_BRIDGE_TIER = "T1 balance-sheet + cash-flow debt bridge"
DEFAULT_PERIOD = "fy2025"


@dataclass(frozen=True)
class ExxonDebtBridgePeriod:
    """One scraped Exxon debt-bridge period."""

    key: str
    label: str
    filing_url: str
    balance_sheet_url: str
    cash_flow_url: str
    begin_current_debt: float
    begin_long_term_debt: float
    end_current_debt: float
    end_long_term_debt: float
    cash_flow_debt_lines: Mapping[str, float]
    debt_detail_url: str | None = None
    supplemental_notes: tuple[str, ...] = ()

    @property
    def begin_total_debt(self) -> float:
        return self.begin_current_debt + self.begin_long_term_debt

    @property
    def end_total_debt(self) -> float:
        return self.end_current_debt + self.end_long_term_debt

    @property
    def balance_sheet_debt_change(self) -> float:
        return self.end_total_debt - self.begin_total_debt

    @property
    def net_cash_debt_financing(self) -> float:
        return sum(self.cash_flow_debt_lines.values())

    @property
    def residual_bridge(self) -> float:
        return self.balance_sheet_debt_change - self.net_cash_debt_financing

    @property
    def residual_share_of_debt_change(self) -> float:
        return 100.0 * self.residual_bridge / self.balance_sheet_debt_change


EXXON_DEBT_BRIDGE_PERIODS = {
    "fy2025": ExxonDebtBridgePeriod(
        key="fy2025",
        label="FY2025",
        filing_url=FY2025_FILING_URL,
        balance_sheet_url=FY2025_BALANCE_SHEET_URL,
        cash_flow_url=FY2025_CASH_FLOW_URL,
        debt_detail_url=FY2025_DEBT_DETAIL_URL,
        begin_current_debt=4_955.0,
        begin_long_term_debt=36_755.0,
        end_current_debt=9_296.0,
        end_long_term_debt=34_241.0,
        cash_flow_debt_lines={
            "additions_to_long_term_debt": 2_311.0,
            "reductions_in_long_term_debt": -1_108.0,
            "additions_to_short_term_debt": 2_359.0,
            "reductions_in_short_term_debt": -5_404.0,
            "commercial_paper_and_debt_three_months_or_less": 1_895.0,
        },
        supplemental_notes=(
            "The long-term debt note reports noncurrent finance lease "
            "liability included in long-term debt of 6,313 $M at year end "
            "2025 versus 3,951 $M at year end 2024.",
        ),
    ),
    "q1_2026": ExxonDebtBridgePeriod(
        key="q1_2026",
        label="Q1 2026",
        filing_url=Q1_2026_FILING_URL,
        balance_sheet_url=Q1_2026_BALANCE_SHEET_URL,
        cash_flow_url=Q1_2026_CASH_FLOW_URL,
        begin_current_debt=9_296.0,
        begin_long_term_debt=34_241.0,
        end_current_debt=14_531.0,
        end_long_term_debt=33_130.0,
        cash_flow_debt_lines={
            "additions_to_long_term_debt": 894.0,
            "reductions_in_long_term_debt": -158.0,
            "reductions_in_short_term_debt": -5_402.0,
            "commercial_paper_and_debt_three_months_or_less": 9_075.0,
        },
    ),
}


def build_spec(
    period: str | ExxonDebtBridgePeriod = DEFAULT_PERIOD,
) -> usf.DisclosureTriangulationSpec:
    """Build the Exxon debt-bridge triangulation problem.

    The variables are generic stock-flow reconciliation variables. Exxon
    provides the values through balance-sheet debt captions and cash-flow debt
    financing captions.
    """

    period_row = _period(period)
    variables = [
        usf.disclosure_variable(
            "begin_total_debt",
            lower=0.0,
            unit="$M",
            description="Opening book debt: notes and loans payable plus long-term debt.",
        ),
        usf.disclosure_variable(
            "end_total_debt",
            lower=0.0,
            unit="$M",
            description="Closing book debt: notes and loans payable plus long-term debt.",
        ),
        usf.disclosure_variable(
            "net_cash_debt_financing",
            lower=None,
            unit="$M",
            description="Net cash-flow debt financing movement.",
        ),
        usf.disclosure_variable(
            "debt_bridge_residual",
            lower=None,
            unit="$M",
            description=(
                "Signed stock-flow residual not explained by the selected "
                "cash-flow debt financing lines."
            ),
        ),
    ]
    constraints = _constraints(period_row)
    return usf.disclosure_triangulation_spec(
        title=f"Exxon Mobil {period_row.label} Debt Bridge Triangulation",
        description=(
            "Reconciles Exxon Mobil's balance-sheet book debt movement to "
            "cash-flow debt financing disclosures and isolates the implied "
            "signed residual bridge."
        ),
        variables=variables,
        constraints=constraints,
        targets=_targets(period_row),
        tiers=[
            usf.disclosure_tier(
                BALANCE_SHEET_TIER,
                [
                    "reported_begin_total_debt",
                    "reported_end_total_debt",
                ],
                description=(
                    "Only opening and closing balance-sheet debt totals are "
                    "active."
                ),
            ),
            usf.disclosure_tier(
                FULL_BRIDGE_TIER,
                [constraint.name for constraint in constraints],
                description=(
                    "Adds reported cash-flow debt financing and the stock-flow "
                    "bridge identity."
                ),
            ),
        ],
    )


def build_report(
    period: str | ExxonDebtBridgePeriod = DEFAULT_PERIOD,
) -> usf.DisclosureTriangulationReport:
    """Solve the Exxon debt-bridge triangulation problem."""

    return usf.triangulate_disclosure(build_spec(period))


def build_attribution_report(
    report: us.NamedLinearFeasibilityReport | None = None,
    *,
    target: str = "debt_bridge_residual",
    tier: str = FULL_BRIDGE_TIER,
    group_by: str = "constraint",
) -> us.NamedLinearConstraintAttributionReport:
    """Rank which disclosures identify the bridge residual."""

    if report is None:
        report = build_report()
    return usf.attribute_disclosure_constraints(
        report,
        target=target,
        tier=tier,
        group_by=group_by,
    )


def build_claim_audit(
    report: us.NamedLinearFeasibilityReport | None = None,
    *,
    target: str = "debt_bridge_residual",
    tier: str = FULL_BRIDGE_TIER,
    lower_at_least: float = 1_000.0,
) -> us.NamedLinearClaimAudit:
    """Audit whether the debt bridge residual is at least 1 billion dollars."""

    if report is None:
        report = build_report()
    claim = usf.disclosure_claim(
        target=target,
        tier=tier,
        lower_at_least=lower_at_least,
        label="Debt bridge residual lower-bound claim",
        description=(
            "This claim passes if every feasible reconciliation requires a "
            "signed debt bridge residual at least as large as the threshold."
        ),
        attribution_top=8,
        diagnostic_top=8,
    )
    return claim.audit(report)


def build_audit_pack(
    report: us.NamedLinearFeasibilityReport | None = None,
    *,
    period: str | ExxonDebtBridgePeriod = DEFAULT_PERIOD,
) -> usf.DisclosureAuditPack:
    """Build the analyst-facing debt-bridge audit pack."""

    period_row = _period(period)
    if report is None:
        report = build_report(period_row)
    claim = usf.disclosure_claim(
        target="debt_bridge_residual",
        tier=FULL_BRIDGE_TIER,
        lower_at_least=1_000.0,
        label="Debt bridge residual lower-bound claim",
        description=(
            "This claim asks whether the modeled disclosures require at least "
            "1,000 $M of signed debt bridge residual."
        ),
        attribution_top=8,
        diagnostic_top=8,
    )
    return usf.disclosure_audit_pack(
        report,
        claim=claim,
        title=f"Exxon Mobil {period_row.label} Debt Bridge Disclosure Audit Pack",
        sources=_source_rows(period_row),
        assumptions=[
            "Book debt is modeled as notes and loans payable plus long-term debt.",
            "Cash-flow debt financing is the signed sum of the disclosed "
            "additions, reductions, and net short-duration commercial-paper "
            "or debt financing captions.",
            "The residual is signed. Positive means balance-sheet debt rose "
            "more than cash debt financing explains; negative means cash debt "
            "financing exceeded the balance-sheet debt increase.",
            "Values are in USD millions and are taken from Exxon Mobil's "
            f"rendered SEC XBRL pages for {period_row.label}.",
        ],
        reviewer_notes=_reviewer_notes(period_row),
        attribution_top=8,
        diagnostic_top=8,
    )


def width_reduction_rows(
    report: us.NamedLinearFeasibilityReport,
    *,
    baseline_tier: str = BALANCE_SHEET_TIER,
) -> list[dict[str, Any]]:
    """Summarize interval narrowing relative to the balance-sheet-only tier."""

    rows: list[dict[str, Any]] = []
    for interval in report.intervals:
        baseline = report.interval(target=interval.target, scenario=baseline_tier)
        reduction = None
        reduction_percent = None
        if baseline.width is not None and interval.width is not None:
            reduction = baseline.width - interval.width
            reduction_percent = (
                None if baseline.width <= 0.0 else 100.0 * reduction / baseline.width
            )
        rows.append(
            {
                "target": interval.target,
                "tier": interval.scenario,
                "lower": interval.scaled_lower,
                "upper": interval.scaled_upper,
                "width": interval.scaled_width,
                "reduction_vs_baseline": reduction,
                "reduction_percent_vs_baseline": reduction_percent,
                "status": interval.status,
            }
        )
    return rows


def render_markdown(report: us.NamedLinearFeasibilityReport | None = None) -> str:
    """Render a Markdown audit pack for the default FY2025 example."""

    if report is None:
        report = build_report()
    return build_audit_pack(report).to_markdown()


def render_panel_markdown(
    periods: Iterable[str | ExxonDebtBridgePeriod] = ("fy2025", "q1_2026"),
) -> str:
    """Render a cross-period Exxon debt-bridge panel."""

    period_rows = [_period(period) for period in periods]
    packs = [build_audit_pack(period=period) for period in period_rows]
    lines = [
        "# Exxon Mobil Debt Bridge Disclosure Triangulation Panel",
        "",
        "This panel repeats the same generic stock-flow reconciliation across "
        "Exxon Mobil's FY2025 10-K and Q1 2026 10-Q:",
        "",
        "> Given opening debt, closing debt, and reported cash-flow debt "
        "financing, how much debt movement is left as a signed bridge residual?",
        "",
        "The residual is not a point estimate of a single accounting component. "
        "It is the amount that the selected public disclosures do not explain "
        "through ordinary cash debt financing lines alone.",
        "",
        "## Cross-Period Readout",
        "",
    ]
    lines.extend(_period_readout_table(period_rows, packs))
    lines.extend(
        [
            "",
            "FY2025 is the interesting case: balance-sheet debt rose by "
            "1,827 $M while the selected cash-flow debt financing lines netted "
            "to only 53 $M, leaving a 1,774 $M signed residual bridge. Q1 2026 "
            "looks different: the balance-sheet movement is essentially "
            "explained by cash-flow debt financing.",
        ]
    )
    for pack in packs:
        lines.extend(["", f"## {pack.audit_title}", ""])
        pack_lines = pack.to_markdown().splitlines()
        if pack_lines and pack_lines[0].startswith("# "):
            pack_lines = pack_lines[2:]
        lines.extend(pack_lines)
    return "\n".join(lines)


def _constraints(period: ExxonDebtBridgePeriod) -> list[usf.DisclosureConstraint]:
    return [
        usf.exact_disclosure_constraint(
            "reported_begin_total_debt",
            "begin_total_debt",
            period.begin_total_debt,
            category="balance_sheet_disclosure",
            provenance=(
                f"Exxon Mobil {period.label} balance sheet: "
                f"{period.balance_sheet_url}"
            ),
            description=(
                "Opening notes and loans payable plus opening long-term debt."
            ),
            verified=True,
        ),
        usf.exact_disclosure_constraint(
            "reported_end_total_debt",
            "end_total_debt",
            period.end_total_debt,
            category="balance_sheet_disclosure",
            provenance=(
                f"Exxon Mobil {period.label} balance sheet: "
                f"{period.balance_sheet_url}"
            ),
            description=(
                "Closing notes and loans payable plus closing long-term debt."
            ),
            verified=True,
        ),
        usf.exact_disclosure_constraint(
            "reported_net_cash_debt_financing",
            "net_cash_debt_financing",
            period.net_cash_debt_financing,
            category="cash_flow_disclosure",
            provenance=(
                f"Exxon Mobil {period.label} cash-flow statement: "
                f"{period.cash_flow_url}"
            ),
            description=(
                "Signed sum of disclosed cash-flow debt financing captions."
            ),
            verified=True,
            metadata={"cash_flow_debt_lines": dict(period.cash_flow_debt_lines)},
        ),
        usf.exact_disclosure_constraint(
            "stock_flow_debt_bridge_identity",
            {
                "end_total_debt": 1.0,
                "begin_total_debt": -1.0,
                "net_cash_debt_financing": -1.0,
                "debt_bridge_residual": -1.0,
            },
            0.0,
            category="reconciliation_identity",
            provenance="Analyst stock-flow bridge identity",
            description=(
                "Closing debt minus opening debt equals net cash debt financing "
                "plus the signed bridge residual."
            ),
            verified=True,
        ),
    ]


def _targets(period: ExxonDebtBridgePeriod) -> list[usf.DisclosureTarget]:
    change = period.balance_sheet_debt_change
    targets = [
        usf.disclosure_target(
            "balance_sheet_debt_change",
            {"end_total_debt": 1.0, "begin_total_debt": -1.0},
            label="Balance-sheet debt change",
            unit="$M",
        ),
        usf.disclosure_target(
            "net_cash_debt_financing",
            "net_cash_debt_financing",
            label="Net cash debt financing",
            unit="$M",
        ),
        usf.disclosure_target(
            "debt_bridge_residual",
            "debt_bridge_residual",
            label="Implied debt bridge residual",
            unit="$M",
        ),
    ]
    if change != 0.0:
        targets.extend(
            [
                usf.disclosure_target(
                    "debt_bridge_residual_share_of_balance_sheet_change",
                    {"debt_bridge_residual": 100.0 / change},
                    label="Residual share of balance-sheet debt change",
                    unit="%",
                ),
                usf.disclosure_target(
                    "cash_financing_share_of_balance_sheet_change",
                    {"net_cash_debt_financing": 100.0 / change},
                    label="Cash financing share of balance-sheet debt change",
                    unit="%",
                ),
            ]
        )
    return targets


def _source_rows(period: ExxonDebtBridgePeriod) -> list[dict[str, str]]:
    rows = [
        {
            "label": "Opening book debt",
            "value": f"{period.begin_total_debt:,.0f} $M",
            "url": period.balance_sheet_url,
            "description": (
                "Prior-period notes and loans payable plus long-term debt."
            ),
        },
        {
            "label": "Closing book debt",
            "value": f"{period.end_total_debt:,.0f} $M",
            "url": period.balance_sheet_url,
            "description": (
                "Current-period notes and loans payable plus long-term debt."
            ),
        },
        {
            "label": "Cash-flow debt financing",
            "value": f"{period.net_cash_debt_financing:+,.0f} $M",
            "url": period.cash_flow_url,
            "description": "Signed sum of disclosed cash-flow debt financing lines.",
        },
    ]
    if period.debt_detail_url:
        rows.append(
            {
                "label": "Debt detail note",
                "value": "supplemental",
                "url": period.debt_detail_url,
                "description": (
                    "Debt-detail note that can help interpret bridge components."
                ),
            }
        )
    return rows


def _reviewer_notes(period: ExxonDebtBridgePeriod) -> list[str]:
    notes = [
        "Balance-sheet book debt changed by "
        f"{_format_signed_money(period.balance_sheet_debt_change)}.",
        "Cash-flow debt financing netted to "
        f"{_format_signed_money(period.net_cash_debt_financing)}.",
        "The implied signed residual bridge is "
        f"{_format_signed_money(period.residual_bridge)}, or "
        f"{period.residual_share_of_debt_change:.1f}% of the balance-sheet "
        "debt change.",
        "The residual can reflect noncash lease obligations, debt assumed or "
        "disposed, foreign-currency translation, current/noncurrent "
        "reclassifications, and other presentation differences.",
    ]
    notes.extend(period.supplemental_notes)
    return notes


def _period(period: str | ExxonDebtBridgePeriod) -> ExxonDebtBridgePeriod:
    if isinstance(period, ExxonDebtBridgePeriod):
        return period
    try:
        return EXXON_DEBT_BRIDGE_PERIODS[period]
    except KeyError as exc:
        raise KeyError(f"unknown Exxon debt-bridge period: {period!r}") from exc


def _period_readout_table(
    periods: Iterable[ExxonDebtBridgePeriod],
    packs: Iterable[usf.DisclosureAuditPack],
) -> list[str]:
    lines = [
        "| Period | Balance-sheet debt change | Net cash debt financing | Implied residual bridge | Residual share | $1B residual claim |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for period, pack in zip(periods, packs, strict=True):
        verdict = "n/a" if pack.claim_audit is None else pack.claim_audit.verdict
        lines.append(
            "| "
            + " | ".join(
                [
                    period.label,
                    _format_signed_money(period.balance_sheet_debt_change),
                    _format_signed_money(period.net_cash_debt_financing),
                    _format_signed_money(period.residual_bridge),
                    f"{period.residual_share_of_debt_change:.1f}%",
                    verdict,
                ]
            )
            + " |"
        )
    return lines


def _format_signed_money(value: float) -> str:
    return f"{value:+,.0f} $M"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Exxon debt-bridge triangulation report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown output path.",
    )
    args = parser.parse_args()

    markdown = render_panel_markdown()
    if args.output is None:
        print(markdown)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
