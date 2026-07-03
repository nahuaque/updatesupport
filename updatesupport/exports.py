"""Structured report export helpers."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from typing import Any, Mapping

from .frontier import (
    FrontierCandidateExplanation,
    PublicRepresentationCandidate,
    PublicRepresentationFrontier,
)
from .report import (
    CausalReportingStabilitySuite,
    PublicDescentReport,
    RefinementSensitivityReport,
    SensitivityReport,
)

TableRows = tuple[dict[str, Any], ...]
ReportTables = dict[str, TableRows]


def report_to_json(report: Any, **kwargs: Any) -> str:
    """Serialize a report-like object to structured JSON."""

    options = {"indent": 2, "sort_keys": True}
    options.update(kwargs)
    return json.dumps(_json_ready(_as_payload(report)), **options)


def report_tables(report: Any) -> ReportTables:
    """Return named list-of-dict tables for a report-like object."""

    if _is_audit_run(report):
        tables = _prefix_tables("report", report_tables(report.report))
        tables["spec"] = (_json_ready(report.spec.as_dict()),)
        return tables

    if isinstance(report, PublicDescentReport):
        return _public_descent_tables(report)

    if isinstance(report, SensitivityReport):
        return _sensitivity_tables(report)

    if isinstance(report, RefinementSensitivityReport):
        return _refinement_sensitivity_tables(report)

    if isinstance(report, CausalReportingStabilitySuite):
        return _causal_suite_tables(report)

    if isinstance(report, PublicRepresentationFrontier):
        return _frontier_tables(report)

    if isinstance(report, FrontierCandidateExplanation):
        return _frontier_explanation_tables(report)

    if _is_report_wrapper(report):
        tables = report_tables(report.report)
        wrapper_table = (
            "dowhy_refutation"
            if getattr(report, "refutation_type", None) is not None
            else "wrapper_summary"
        )
        tables[wrapper_table] = (
            {
                "wrapper_type": type(report).__name__,
                "refutation_type": getattr(report, "refutation_type", None),
                "estimated_effect": getattr(report, "estimated_effect", None),
                "new_effect": getattr(report, "new_effect", None),
                "ambiguity": getattr(report, "ambiguity", None),
            },
        )
        return tables

    if hasattr(report, "as_dict"):
        payload = _as_payload(report)
        if isinstance(payload, Mapping):
            return {"summary": (_json_ready(payload),)}

    raise TypeError(f"unsupported report type: {type(report).__name__}")


def report_dataframes(report: Any) -> dict[str, Any]:
    """Return named pandas DataFrames for a report-like object.

    Pandas is optional in core. Install ``updatesupport[examples]`` or install
    pandas directly before calling this helper.
    """

    return tables_to_dataframes(report_tables(report))


def tables_to_dataframes(tables: Mapping[str, TableRows]) -> dict[str, Any]:
    """Convert named list-of-dict tables into pandas DataFrames."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "DataFrame exports require pandas. Install pandas or "
            "`updatesupport[examples]`."
        ) from exc

    return {
        name: pd.DataFrame([_json_ready(row) for row in rows])
        for name, rows in tables.items()
    }


def _public_descent_tables(report: PublicDescentReport) -> ReportTables:
    grouped = report.grouped
    return {
        "summary": (
            {
                "title": report.title,
                "row_count": report.row_count,
                "row_count_label": report.row_count_label,
                "hidden_cells": len(grouped.problem.states),
                "public_cells": len(grouped.problem.public_values),
                "public_columns": grouped.public_columns,
                "hidden_columns": grouped.hidden_columns,
                "target": grouped.target_column,
                "compiled_target": _target_label(grouped.target_column),
                "target_procedure": None
                if grouped.target_procedure is None
                else grouped.target_procedure.name,
                "target_procedure_description": None
                if grouped.target_procedure is None
                else grouped.target_procedure.description,
                "target_procedure_formula": None
                if grouped.target_procedure is None
                else grouped.target_procedure.formula,
                "target_procedure_context": None
                if grouped.target_procedure_context is None
                else grouped.target_procedure_context.as_dict(),
                "target_description": report.target_description,
                "target_contract": grouped.problem.target_contract.as_dict(),
                "target_kind": grouped.problem.target_contract.kind,
                "target_formula": grouped.problem.target_contract.formula,
                "target_fixed_after_compilation": (
                    grouped.problem.target_contract.fixed_after_compilation
                ),
                "q_name": grouped.q_name,
                "q_description": grouped.q_description,
                "min_cell_weight": report.min_cell_weight,
                "observed_label": report.observed_label,
                "observed_value": report.observed_value,
                "lower": report.interval.lower,
                "upper": report.interval.upper,
                "ambiguity": report.interval.diameter,
                "public_adequate": report.public_adequate,
                "interval_contains_observed": report.interval_contains_observed,
                "fiber_decomposition_available": (report.fiber_decomposition_available),
                "fiber_diagnostic_kind": report.fiber_diagnostic_kind,
                "top_fiber_contribution": report.top_fiber_contribution,
                "top_fiber_contribution_share": report.top_fiber_contribution_share,
                "diagnostic_count": len(report.diagnostics),
                "diagnostic_warning_count": sum(
                    1 for row in report.diagnostics if row.severity == "warning"
                ),
            },
        ),
        "worst_fibers": tuple(row.as_dict() for row in report.fibers),
        "refinements": tuple(row.as_dict() for row in report.refinements),
        "data_diagnostics": tuple(row.as_dict() for row in report.diagnostics),
        "dual_diagnostics": tuple(
            row.as_dict() for row in report.interval.dual_summary(top=20)
        ),
    }


def _sensitivity_tables(report: SensitivityReport) -> ReportTables:
    return {
        "summary": (report.summary.as_dict(),),
        "scenarios": tuple(row.as_dict() for row in report.rows),
    }


def _refinement_sensitivity_tables(
    report: RefinementSensitivityReport,
) -> ReportTables:
    return {
        "summary": (
            {
                "title": report.title,
                "row_count": report.row_count,
                "scenarios": len(report.scenarios),
                "successful_scenarios": len(report.successful_scenarios),
                "failed_scenarios": len(report.failed_scenarios),
                "candidate_refinements": len(report.candidates),
                "top_refinement": None
                if not report.candidates
                else report.candidates[0].column,
            },
        ),
        "refinement_candidates": tuple(row.as_dict() for row in report.candidates),
        "refinement_scenarios": tuple(row.as_dict() for row in report.scenarios),
        "refinement_rows": tuple(row.as_dict() for row in report.rows),
    }


def _causal_suite_tables(report: CausalReportingStabilitySuite) -> ReportTables:
    tables: ReportTables = {
        "summary": (
            {
                "title": report.title,
                "observed_value": report.primary.observed_value,
                "lower": report.primary.interval.lower,
                "upper": report.primary.interval.upper,
                "ambiguity": report.primary.interval.diameter,
                "public_adequate": report.primary.public_adequate,
                "q_name": report.primary.grouped.q_name,
                "has_statistical_uncertainty": (
                    report.statistical_uncertainty is not None
                ),
                "has_sensitivity": report.sensitivity is not None,
                "has_refinement_sensitivity": (
                    report.refinement_sensitivity is not None
                ),
            },
        ),
    }
    tables.update(_prefix_tables("primary", _public_descent_tables(report.primary)))
    if report.statistical_uncertainty is not None:
        tables["statistical_uncertainty"] = (report.statistical_uncertainty.as_dict(),)
    if report.sensitivity is not None:
        tables.update(
            _prefix_tables("sensitivity", _sensitivity_tables(report.sensitivity))
        )
    if report.refinement_sensitivity is not None:
        tables.update(
            _prefix_tables(
                "refinement_sensitivity",
                _refinement_sensitivity_tables(report.refinement_sensitivity),
            )
        )
    return tables


def _frontier_tables(report: PublicRepresentationFrontier) -> ReportTables:
    candidate_scenarios: list[dict[str, Any]] = []
    for candidate in report.candidates:
        for scenario in candidate.scenarios:
            row = scenario.as_dict()
            row.update(
                {
                    "candidate_label": candidate.label,
                    "added_columns": candidate.added_columns,
                    "public_columns": candidate.public_columns,
                    "candidate_max_ambiguity": candidate.max_ambiguity,
                    "candidate_mean_ambiguity": candidate.mean_ambiguity,
                    "candidate_public_adequate": candidate.public_adequate,
                }
            )
            candidate_scenarios.append(row)

    return {
        "summary": (
            {
                "title": report.title,
                "row_count": report.row_count,
                "base_public": report.base_public,
                "hidden_columns": report.hidden_columns,
                "hidden_sets": report.hidden_sets,
                "min_cell_weights": report.min_cell_weights,
                "candidate_refinements": report.candidate_refinements,
                "requested_refinements": report.requested_refinements,
                "ambiguity_limit": report.ambiguity_limit,
                "bucket_budget": report.bucket_budget,
                "candidate_count": len(report.candidates),
                "frontier_count": len(report.frontier),
                "dominated_count": len(report.dominated),
                "minimal_stable": None
                if report.minimal_stable is None
                else report.minimal_stable.added_columns,
                "best_under_bucket_budget": None
                if report.best_under_bucket_budget() is None
                else report.best_under_bucket_budget().added_columns,
            },
        ),
        "search_trace": ()
        if report.search_trace is None
        else (report.search_trace.as_dict(),),
        "screened_refinements": tuple(
            row.as_dict() for row in report.screened_refinements
        ),
        "frontier": tuple(_candidate_row(row) for row in report.frontier),
        "dominated": tuple(_candidate_row(row) for row in report.dominated),
        "candidates": tuple(_candidate_row(row) for row in report.candidates),
        "candidate_scenarios": tuple(candidate_scenarios),
    }


def _frontier_explanation_tables(
    explanation: FrontierCandidateExplanation,
) -> ReportTables:
    return {
        "summary": (
            {
                "candidate_label": explanation.candidate.label,
                "added_columns": explanation.candidate.added_columns,
                "public_cells": explanation.candidate.public_cells,
                "baseline_ambiguity": explanation.baseline_ambiguity,
                "selected_ambiguity": explanation.selected_ambiguity,
                "ambiguity_reduction": explanation.ambiguity_reduction,
                "ambiguity_reduction_percent": (
                    explanation.ambiguity_reduction_percent
                ),
                "added_public_cells": explanation.added_public_cells,
                "ambiguity_limit": explanation.ambiguity_limit,
                "bucket_budget": explanation.bucket_budget,
            },
        ),
        "scenario_comparisons": tuple(
            row.as_dict() for row in explanation.scenario_comparisons
        ),
        "close_dominated_alternatives": tuple(
            row.as_dict() for row in explanation.close_dominated_alternatives
        ),
        "screened_refinements": tuple(
            row.as_dict() for row in explanation.screened_refinements
        ),
        "search_trace": ()
        if explanation.search_trace is None
        else (explanation.search_trace.as_dict(),),
    }


def _candidate_row(candidate: PublicRepresentationCandidate) -> dict[str, Any]:
    row = candidate.as_dict().copy()
    row.pop("scenarios", None)
    row["label"] = candidate.label
    return row


def _target_label(target: Any) -> str:
    if isinstance(target, str):
        return target
    return str(getattr(target, "name", type(target).__name__))


def _prefix_tables(prefix: str, tables: ReportTables) -> ReportTables:
    return {f"{prefix}_{name}": rows for name, rows in tables.items()}


def _is_audit_run(value: Any) -> bool:
    return hasattr(value, "spec") and hasattr(value, "report")


def _is_report_wrapper(value: Any) -> bool:
    return hasattr(value, "report") and not hasattr(value, "spec")


def _as_payload(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if is_dataclass(value):
        return {
            field: getattr(value, field)
            for field in getattr(value, "__dataclass_fields__", ())
        }
    return value


def _json_ready(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return _json_ready(value.as_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_ready(item) for item in sorted(value, key=str)]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value
