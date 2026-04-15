"""Clinical text structure parser for MedVerse."""

from __future__ import annotations

import re

from sglang.srt.medverse.section_types import MedicalSection, MedicalSectionType
from sglang.srt.medverse.utils.constants import INDEPENDENT_SECTIONS, SECTION_PATTERNS

_COMPILED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    section: [
        re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        for pattern in patterns
    ]
    for section, patterns in SECTION_PATTERNS.items()
}

_SECTION_SPLIT_RE = re.compile(
    r"(?m)^(?P<header>"
    + "|".join(
        pat
        for pats in SECTION_PATTERNS.values()
        for pat in pats
    )
    + r")\s*[:\-]?(?:\s+(?P<inline>.+))?$",
    re.IGNORECASE,
)


def _classify_header(header_text: str) -> MedicalSectionType:
    for section_name, patterns in _COMPILED_PATTERNS.items():
        for pat in patterns:
            if pat.search(header_text):
                return MedicalSectionType(section_name)
    return MedicalSectionType.UNKNOWN


def _build_dependency_hints(
    section_type: MedicalSectionType,
    all_types: list[MedicalSectionType],
) -> list[MedicalSectionType]:
    if section_type == MedicalSectionType.DIFFERENTIAL_DIAGNOSIS:
        return [t for t in all_types if t.is_independent]
    if section_type == MedicalSectionType.ASSESSMENT:
        deps = [t for t in all_types if t.is_independent]
        if MedicalSectionType.DIFFERENTIAL_DIAGNOSIS in all_types:
            deps.append(MedicalSectionType.DIFFERENTIAL_DIAGNOSIS)
        return deps
    if section_type == MedicalSectionType.PLAN:
        deps = []
        for t in [MedicalSectionType.ASSESSMENT, MedicalSectionType.DIFFERENTIAL_DIAGNOSIS]:
            if t in all_types:
                deps.append(t)
        return deps
    return []


class MedicalSectionParser:
    """Parse structured medical text into typed MedicalSection objects."""

    def parse(self, text: str) -> list[MedicalSection]:
        sections = self._split_into_sections(text)
        if not sections:
            return [
                MedicalSection(
                    section_type=MedicalSectionType.HPI,
                    header="",
                    content=text.strip(),
                )
            ]
        all_types = [s.section_type for s in sections]
        for section in sections:
            section.dependency_hints = _build_dependency_hints(section.section_type, all_types)
        return sections

    def _split_into_sections(self, text: str) -> list[MedicalSection]:
        matches = list(_SECTION_SPLIT_RE.finditer(text))
        if not matches:
            return []

        sections: list[MedicalSection] = []
        for i, match in enumerate(matches):
            header_text = match.group("header")
            inline_text = match.group("inline") or ""
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            body_start = match.end()
            body = text[body_start:end].strip()
            if inline_text:
                body = (inline_text.strip() + "\n" + body).strip()

            section_type = _classify_header(header_text)
            sections.append(
                MedicalSection(
                    section_type=section_type,
                    header=header_text,
                    content=body,
                    start_char=start,
                    end_char=end,
                )
            )

        return sections
