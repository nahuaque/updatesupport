"""Shared artifact helpers for report-like objects."""

from __future__ import annotations

from typing import Any


class ReportArtifactMixin:
    """Common JSON and DataFrame export methods for structured reports."""

    def to_json(self, **kwargs: Any) -> str:
        from .exports import report_to_json

        return report_to_json(self, **kwargs)

    def to_dataframes(self) -> dict[str, Any]:
        from .exports import tables_to_dataframes

        return tables_to_dataframes(self.to_tables())  # type: ignore[attr-defined]
