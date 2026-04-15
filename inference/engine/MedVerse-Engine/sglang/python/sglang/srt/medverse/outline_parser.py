"""Parse LLM-generated <Outline> tags into Step dependency structures.

Actual MedVerse14k data format:

    <Plan>
    <Outline> Transient Step 1: A -> B; Dependency: [] </Outline>
    <Outline> Transient Step 2: A -> C; Dependency: [] </Outline>
    <Outline> Transient Step 3: B -> D; Dependency: [1] </Outline>
    <Outline> Transient Step 4: C -> E; Dependency: [2] </Outline>
    <Outline> Transient Step 5: D + E -> Answer; Dependency: [3, 4] </Outline>
    </Plan>
    <Execution>
    <Step> Transient Step 1: A -> B
    ...generated content...
    </Step>
    ...
    </Execution>
    <Conclusion>
    ...final synthesis...
    </Conclusion>

Each <Outline> tag carries:
  - step number  (integer, 1-indexed)
  - description  ("A -> B" style reasoning hop)
  - Dependency   list of upstream step numbers (empty = root step)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepDef:
    """One <Outline> step extracted from the <Plan> block."""

    step_id: str                           # string of the integer, e.g. "1", "3"
    description: str                       # "A -> B" label
    deps: List[str] = field(default_factory=list)  # step_ids of prerequisites

    @property
    def is_root(self) -> bool:
        return len(self.deps) == 0


# ── Patterns ─────────────────────────────────────────────────────────────────

# Format A: XML-tagged  <Outline> Transient Step N: ...; Dependency: [] </Outline>
_OUTLINE_TAG_RE = re.compile(
    r"<Outline>\s*Transient Step\s+(?P<num>\d+)\s*:\s*(?P<desc>[^;]+?)"
    r";\s*Dependency:\s*\[(?P<deps>[^\]]*)\]\s*</Outline>",
    re.IGNORECASE,
)

# Format B: plain-text   Transient Step N: ...; Dependency: []
# (the model sometimes skips <Outline> / </Outline> tags)
_PLAIN_STEP_RE = re.compile(
    r"Transient Step\s+(?P<num>\d+)\s*:\s*(?P<desc>[^;\n]+?)"
    r";\s*Dependency:\s*\[(?P<deps>[^\]]*)\]",
    re.IGNORECASE,
)

# Matches the closing </Plan> tag (Format A)
_PLAN_END_RE = re.compile(r"</Plan>", re.IGNORECASE)

# Format B plan-end: after the LAST Dependency line the model emits 2+ newlines
# (observed: 3 newlines) before starting the Execution section.
# Between plan steps: only 1 newline. So 2+ newlines reliably signals end-of-plan.
_PLAN_END_PLAINTEXT_RE = re.compile(
    r"Dependency:\s*\[[^\]]*\]\s*\n[ \t]*\n",
    re.IGNORECASE,
)


def parse_outline(llm_output: str) -> List[StepDef]:
    """Return all StepDef objects from the plan block in *llm_output*.

    Supports two formats:
      A) XML:        <Outline> Transient Step N: ...; Dependency: [] </Outline>
      B) Plain text:            Transient Step N: ...; Dependency: []

    Returns an empty list if no step definitions are found.
    """
    steps: List[StepDef] = []

    # Try Format A first (XML tagged)
    for m in _OUTLINE_TAG_RE.finditer(llm_output):
        num = m.group("num").strip()
        desc = m.group("desc").strip()
        raw_deps = m.group("deps").strip()
        deps = [d.strip() for d in raw_deps.split(",") if d.strip()]
        steps.append(StepDef(step_id=num, description=desc, deps=deps))

    if steps:
        return steps

    # Fallback: Format B (plain text without XML tags)
    for m in _PLAIN_STEP_RE.finditer(llm_output):
        num = m.group("num").strip()
        desc = m.group("desc").strip()
        raw_deps = m.group("deps").strip()
        deps = [d.strip() for d in raw_deps.split(",") if d.strip()]
        steps.append(StepDef(step_id=num, description=desc, deps=deps))

    return steps


def has_plan_end_tag(text: str) -> bool:
    """Return True if *text* signals that plan generation is complete.

    Supports two formats:
      A) XML: </Plan> closing tag
      B) Plain-text: a Dependency line followed by 2+ newlines
         (between plan steps: only 1 newline; after last step: 2+ newlines)
    """
    if bool(_PLAN_END_RE.search(text)):
        return True
    return bool(_PLAN_END_PLAINTEXT_RE.search(text))


def step_execution_header(step: StepDef) -> str:
    """Return the <Step> opening line the LLM uses during execution.

    Matches the MedVerse14k format exactly:
        <Step> Transient Step N: description
    """
    return f"<Step> Transient Step {step.step_id}: {step.description}\n"


def conclusion_header() -> str:
    """Return the join-transition prompt.

    Closes the <Execution> block and opens <Conclusion> to match training format:
        </Execution>
        <Conclusion>
    """
    return "</Execution>\n<Conclusion>\n"


def execution_header() -> str:
    """Return the header that wraps the <Step> blocks in training format.

    Training format: </Plan>\n<Execution>\n<Step>...
    The model generates \n after </Plan>, so we only need <Execution>\n here.
    """
    return "<Execution>\n"
