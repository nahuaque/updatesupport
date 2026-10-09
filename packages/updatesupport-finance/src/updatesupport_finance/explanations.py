"""Public explanation comparison and hypothetical evidence planning API.

The comparison engine and evidence search are separate internal modules. This
module preserves their shared import path; package-root imports are equivalent.
"""

from ._explanation_comparison import (
    DisclosureAlternative,
    DisclosureExplanationCase,
    DisclosureExplanationComparison,
    compare_disclosure_explanations,
)
from ._explanation_evidence import (
    DisclosureEvidenceRequest,
    DisclosureEvidenceBundle,
    DisclosureEvidenceOutcome,
    DisclosureEvidencePlan,
    evaluate_disclosure_evidence,
    plan_disclosure_evidence,
)

__all__ = [
    "DisclosureAlternative",
    "DisclosureExplanationCase",
    "DisclosureExplanationComparison",
    "compare_disclosure_explanations",
    "DisclosureEvidenceRequest",
    "DisclosureEvidenceBundle",
    "DisclosureEvidenceOutcome",
    "DisclosureEvidencePlan",
    "evaluate_disclosure_evidence",
    "plan_disclosure_evidence",
]
