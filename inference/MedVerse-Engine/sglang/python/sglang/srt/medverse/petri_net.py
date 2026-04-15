"""Petri Net structure for MedVerse Phase II frontier-based execution.

Notation follows the paper (Sec. 3.3):
  - N = (P, T, F)   places P, transitions T, flow relation F
  - M0              initial token marking
  - Mk              marking after k firings
  - Fk              enabled-transition frontier at step k

One place per step represents "step completed" (has token after firing).
A transition for step S is *enabled* when every dep step's place has a token.
Firing a transition places a token in S's completion place.

Root steps (no deps) are immediately enabled in M0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set

from sglang.srt.medverse.outline_parser import StepDef


@dataclass
class Place:
    """A place in the Petri Net (one per step + implicit source)."""
    place_id: str
    tokens: int = 0          # 0 or 1 in a safe Petri Net

    def has_token(self) -> bool:
        return self.tokens > 0


@dataclass
class Transition:
    """A transition representing one reasoning step."""
    step_id: str
    input_places: List[str]   # place_ids that must have tokens (= dep steps done)
    output_place: str         # place_id that receives a token after firing

    def is_enabled(self, places: Dict[str, Place]) -> bool:
        return all(
            places[pid].has_token()
            for pid in self.input_places
            if pid in places
        )


class PetriNet:
    """Petri Net instantiated from a parsed <Outline> block."""

    # Implicit place that is always marked (represents "parent context ready")
    SOURCE_PLACE = "__source__"

    def __init__(self, steps: List[StepDef]) -> None:
        self.places: Dict[str, Place] = {}
        self.transitions: Dict[str, Transition] = {}
        self._fired: Set[str] = set()       # step_ids that have been fired

        # Source place – always has a token
        self.places[self.SOURCE_PLACE] = Place(self.SOURCE_PLACE, tokens=1)

        # One completion place per step
        for step in steps:
            pid = self._completion_place(step.step_id)
            self.places[pid] = Place(pid, tokens=0)

        # One transition per step
        for step in steps:
            input_pids = (
                [self._completion_place(d) for d in step.deps]
                if step.deps
                else [self.SOURCE_PLACE]
            )
            self.transitions[step.step_id] = Transition(
                step_id=step.step_id,
                input_places=input_pids,
                output_place=self._completion_place(step.step_id),
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    @classmethod
    def from_steps(cls, steps: List[StepDef]) -> "PetriNet":
        return cls(steps)

    def get_enabled_frontier(self) -> List[Transition]:
        """Return currently enabled transitions that have NOT been fired yet."""
        return [
            t for t in self.transitions.values()
            if t.step_id not in self._fired
            and t.is_enabled(self.places)
        ]

    def fire(self, step_id: str) -> None:
        """Fire transition for *step_id*: mark it fired, put token in output place."""
        if step_id not in self.transitions:
            raise KeyError(f"Unknown step '{step_id}'")
        t = self.transitions[step_id]
        self.places[t.output_place].tokens += 1
        self._fired.add(step_id)

    def is_complete(self) -> bool:
        """Return True when all transitions have been fired."""
        return len(self._fired) == len(self.transitions)

    def pending_steps(self) -> List[str]:
        return [sid for sid in self.transitions if sid not in self._fired]

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _completion_place(step_id: str) -> str:
        return f"__done_{step_id}__"
