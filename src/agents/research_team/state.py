"""LangGraph state contract for the industry research team.

Domain payloads live in :mod:`contracts` and are imported from there directly; this
module owns only the transport state itself and the routing literal.
"""

from typing import Literal

from langgraph.graph import MessagesState

from agents.research_team.contracts import (
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
    # Sources already read by Analyst in this turn, keyed by url_key. Retrieval keeps
    # appending to `sources`, so this is what turns "read the first N" into "read the
    # ones that arrived after the last pass".
    analyzed_source_ids: list[str]
    analysis_batches: int
    # Merged analyst output. Analyst sees one batch per call, so anything it must not
    # forget between batches lives here rather than in the prompt history.
    facts: list[str]
    analysis_patterns: list[str]
    analysis_contradictions: list[str]
    analysis_gaps: list[str]
    analysis: str
    evidence_statements: list[EvidenceStatement]
    evidence_coverage: list[EvidenceCoverage]
    subject_coverage: list[SubjectEvidenceCoverage]
    report: str
    next_action: NextAction
    rounds: int
    active_turn_id: str
    active_context_mode: Literal["fresh", "reuse"]
    turn_attempted: bool
    abort_reason: str
