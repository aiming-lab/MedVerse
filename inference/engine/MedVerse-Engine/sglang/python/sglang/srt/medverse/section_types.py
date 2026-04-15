"""Medical section type definitions for MedVerse."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class MedicalSectionType(str, Enum):
    """Enum representing clinical note section types."""

    HPI = "HPI"
    VITALS = "VITALS"
    LABS = "LABS"
    IMAGING = "IMAGING"
    PHYSICAL_EXAM = "PHYSICAL_EXAM"
    MEDICATIONS = "MEDICATIONS"
    DIFFERENTIAL_DIAGNOSIS = "DIFFERENTIAL_DIAGNOSIS"
    ASSESSMENT = "ASSESSMENT"
    PLAN = "PLAN"
    UNKNOWN = "UNKNOWN"

    @property
    def is_independent(self) -> bool:
        return self in _INDEPENDENT

    @property
    def is_evidence_dependent(self) -> bool:
        return self in _EVIDENCE_DEPENDENT

    @property
    def is_terminal(self) -> bool:
        return self is MedicalSectionType.PLAN


_INDEPENDENT: frozenset[MedicalSectionType] = frozenset(
    [
        MedicalSectionType.VITALS,
        MedicalSectionType.LABS,
        MedicalSectionType.IMAGING,
        MedicalSectionType.PHYSICAL_EXAM,
        MedicalSectionType.MEDICATIONS,
    ]
)

_EVIDENCE_DEPENDENT: frozenset[MedicalSectionType] = frozenset(
    [
        MedicalSectionType.DIFFERENTIAL_DIAGNOSIS,
        MedicalSectionType.ASSESSMENT,
    ]
)


@dataclass
class MedicalSection:
    """A parsed clinical note section."""

    section_type: MedicalSectionType
    header: str
    content: str
    start_char: int = 0
    end_char: int = 0
    dependency_hints: list[MedicalSectionType] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        if self.header:
            return f"{self.header}: {self.content}"
        return self.content

    def __len__(self) -> int:
        return len(self.content)
