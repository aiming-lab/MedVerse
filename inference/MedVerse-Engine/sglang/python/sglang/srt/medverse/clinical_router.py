"""Route parsed medical sections to appropriate reasoning branches."""

from __future__ import annotations

from sglang.srt.medverse.section_types import MedicalSection, MedicalSectionType


class ClinicalRouter:
    """Determine how each MedicalSection maps to DAG branch roles."""

    def get_root_sections(self, sections: list[MedicalSection]) -> list[MedicalSection]:
        return [s for s in sections if s.section_type == MedicalSectionType.HPI]

    def get_parallel_sections(self, sections: list[MedicalSection]) -> list[MedicalSection]:
        return [s for s in sections if s.section_type.is_independent]

    def get_merge_sections(self, sections: list[MedicalSection]) -> list[MedicalSection]:
        return [
            s for s in sections
            if s.section_type.is_evidence_dependent or s.section_type.is_terminal
        ]

    def get_terminal_sections(self, sections: list[MedicalSection]) -> list[MedicalSection]:
        return [s for s in sections if s.section_type.is_terminal]

    def assign_branch_priorities(
        self, sections: list[MedicalSection]
    ) -> dict[MedicalSectionType, int]:
        priority_map: dict[MedicalSectionType, int] = {}
        for section in sections:
            st = section.section_type
            if st == MedicalSectionType.HPI:
                priority_map[st] = 0
            elif st.is_independent:
                priority_map[st] = 1
            elif st == MedicalSectionType.DIFFERENTIAL_DIAGNOSIS:
                priority_map[st] = 2
            elif st == MedicalSectionType.ASSESSMENT:
                priority_map[st] = 3
            elif st == MedicalSectionType.PLAN:
                priority_map[st] = 4
            else:
                priority_map[st] = 5
        return priority_map
