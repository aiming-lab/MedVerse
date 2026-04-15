"""Evidence-weighted branch merging for MedVerse DAG decoding."""

from __future__ import annotations

from dataclasses import dataclass, field

from sglang.srt.medverse.confidence_scorer import ClinicalConfidenceScorer
from sglang.srt.medverse.section_types import MedicalSection


@dataclass
class BranchOutput:
    node_id: int
    section: MedicalSection | None
    text: str
    token_ids: list[int] = field(default_factory=list)
    finished: bool = True
    kv_cache_ref: int | None = None


@dataclass
class MergedInput:
    merged_text: str
    token_ids: list[int]
    weighted_kv_refs: list[tuple[int, float]]
    confidence_scores: dict[int, float]
    has_conflicts: bool = False


class EvidenceWeightedMerge:
    """Merge parent branch outputs using clinical confidence weighting."""

    _CONFLICT_MARKER = "\n[CONFLICTING EVIDENCE]\n"
    _SECTION_SEPARATOR = "\n---\n"

    def __init__(self, scorer: ClinicalConfidenceScorer | None = None) -> None:
        self._scorer = scorer or ClinicalConfidenceScorer()

    def merge(self, parent_outputs: list[BranchOutput]) -> MergedInput:
        if not parent_outputs:
            return MergedInput(
                merged_text="",
                token_ids=[],
                weighted_kv_refs=[],
                confidence_scores={},
            )

        scored: list[tuple[BranchOutput, float]] = []
        confidence_scores: dict[int, float] = {}
        for branch in parent_outputs:
            breakdown = self._scorer.score(
                text=branch.text,
                section=branch.section,
                finished=branch.finished,
            )
            scored.append((branch, breakdown.overall))
            confidence_scores[branch.node_id] = breakdown.overall

        total = sum(score for _, score in scored) or 1.0
        weights = [score / total for _, score in scored]

        has_conflicts = self._detect_conflicts([b.text for b, _ in scored])

        parts: list[str] = []
        for (branch, _score), weight in zip(scored, weights):
            header = (
                f"[{branch.section.section_type.value}]"
                if branch.section
                else f"[Node {branch.node_id}]"
            )
            parts.append(f"{header} (confidence={weight:.2f})\n{branch.text}")

        separator = self._CONFLICT_MARKER if has_conflicts else self._SECTION_SEPARATOR
        merged_text = separator.join(parts)

        weighted_kv_refs = [
            (branch.kv_cache_ref, weight)
            for (branch, _), weight in zip(scored, weights)
            if branch.kv_cache_ref is not None
        ]

        ordered = sorted(zip(scored, weights), key=lambda x: x[1], reverse=True)
        merged_token_ids: list[int] = []
        for (branch, _), _ in ordered:
            merged_token_ids.extend(branch.token_ids)

        return MergedInput(
            merged_text=merged_text,
            token_ids=merged_token_ids,
            weighted_kv_refs=weighted_kv_refs,
            confidence_scores=confidence_scores,
            has_conflicts=has_conflicts,
        )

    def _detect_conflicts(self, texts: list[str]) -> bool:
        positive_terms = {"elevated", "high", "positive", "increased", "abnormal"}
        negative_terms = {"normal", "low", "negative", "decreased", "within limits"}
        positive_votes: list[bool] = []
        for text in texts:
            lower = text.lower()
            has_positive = any(t in lower for t in positive_terms)
            has_negative = any(t in lower for t in negative_terms)
            positive_votes.append(has_positive and not has_negative)
        return len(set(positive_votes)) > 1
