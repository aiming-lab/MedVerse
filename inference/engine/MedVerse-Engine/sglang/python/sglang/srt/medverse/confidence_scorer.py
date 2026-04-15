"""Clinical confidence scoring for DAG branch outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sglang.srt.medverse.section_types import MedicalSection, MedicalSectionType

_ABNORMAL_LAB_RE = re.compile(
    r"\b(?:elevated|high|low|abnormal|critical|positive|negative(?:\s+for))\b",
    re.IGNORECASE,
)
_NUMERIC_FINDING_RE = re.compile(r"\d+\.?\d*\s*(?:mg|mEq|mmol|g/dL|U/L|%|bpm|mmHg)", re.IGNORECASE)
_CONTRADICTION_RE = re.compile(
    r"\b(?:however|but|contrary|in contrast|although|despite|yet)\b",
    re.IGNORECASE,
)


@dataclass
class ConfidenceBreakdown:
    completeness: float
    consistency: float
    evidence_strength: float
    overall: float


class ClinicalConfidenceScorer:
    """Score a branch's clinical confidence based on its output text."""

    def __init__(
        self,
        completeness_weight: float = 0.35,
        consistency_weight: float = 0.30,
        evidence_weight: float = 0.35,
    ) -> None:
        self._w_completeness = completeness_weight
        self._w_consistency = consistency_weight
        self._w_evidence = evidence_weight

    def score(
        self,
        text: str,
        section: MedicalSection | None = None,
        finished: bool = True,
    ) -> ConfidenceBreakdown:
        completeness = self._score_completeness(text, finished)
        consistency = self._score_consistency(text)
        evidence = self._score_evidence_strength(text, section)
        overall = (
            self._w_completeness * completeness
            + self._w_consistency * consistency
            + self._w_evidence * evidence
        )
        return ConfidenceBreakdown(
            completeness=completeness,
            consistency=consistency,
            evidence_strength=evidence,
            overall=min(1.0, max(0.0, overall)),
        )

    def _score_completeness(self, text: str, finished: bool) -> float:
        if not finished:
            return 0.3
        stripped = text.strip()
        if not stripped:
            return 0.0
        if stripped[-1] in ".!?\n":
            return 1.0
        return 0.75

    def _score_consistency(self, text: str) -> float:
        contradictions = len(_CONTRADICTION_RE.findall(text))
        return max(0.0, 1.0 - contradictions * 0.15)

    def _score_evidence_strength(self, text: str, section: MedicalSection | None) -> float:
        abnormal_hits = len(_ABNORMAL_LAB_RE.findall(text))
        numeric_hits = len(_NUMERIC_FINDING_RE.findall(text))
        if section is not None and section.section_type in (
            MedicalSectionType.LABS,
            MedicalSectionType.IMAGING,
        ):
            base = 0.6
        else:
            base = 0.4
        boost = min(0.4, (abnormal_hits * 0.05 + numeric_hits * 0.04))
        return min(1.0, base + boost)
