"""LangGraph state contract for the industry research team.

Domain payloads live in :mod:`contracts`; this module only defines the graph
transport state and keeps the historical model imports stable.
"""

from typing import Literal

from langgraph.graph import MessagesState

from agents.research_team.contracts import (
    AspectCoverage,
    EvidenceCoverage,
    EvidenceStatement,
    ResearchSubject,
    ResearchTask,
    Source,
    SubjectEvidenceCoverage,
)

NextAction = Literal["researcher", "analyst", "writer", "finalize"]


class ResearchState(MessagesState, total=False):
    intent: str
    tasks: list[ResearchTask]
    plan_notes: str
    constraints: list[str]
    subjects: list[ResearchSubject]
    # Sources belong to the active research turn. Researcher explicitly merges
    # results within that turn so Supervisor can replace them at a new turn.
    sources: list[Source]
    analysis: str
    evidence_statements: list[EvidenceStatement]
    evidence_coverage: list[EvidenceCoverage]
    subject_coverage: list[SubjectEvidenceCoverage]
    report: str
    next_action: NextAction
    rounds: int
    max_rounds: int
    active_turn_id: str
    active_context_mode: Literal["fresh", "reuse"]
    turn_attempted: bool
    abort_reason: str


__all__ = [
    "AspectCoverage",
    "EvidenceCoverage",
    "EvidenceStatement",
    "NextAction",
    "ResearchState",
    "ResearchSubject",
    "ResearchTask",
    "Source",
    "SubjectEvidenceCoverage",
]
