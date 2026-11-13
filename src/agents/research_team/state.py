"""Shared state contract for the industry research graph.

Every sub-agent reads from and writes to this state; it is the only channel
between them.
"""

import operator
from typing import Annotated, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

NextAction = Literal["researcher", "analyst", "writer", "finalize"]


class ResearchTask(BaseModel):
    id: str = Field(description="Stable identifier, e.g. T1")
    question: str = Field(description="A single answerable research question")
    assignee: Literal["researcher", "analyst", "writer"] = "researcher"
    rationale: str = ""


class Source(BaseModel):
    url: str
    title: str = ""
    snippet: str = Field(default="", description="100-200 字短摘要，用于界面展示")
    summary: str = Field(default="", description="600-800 字长摘要，供模型阅读")
    content: str = Field(default="", description="Fetch 抓取的正文 Markdown")
    site_name: str = ""
    published_at: str = ""
    authority: int = Field(default=-1, ge=-1, le=4, description="-1 未知，3 以上为高权威")
    rerank_score: float = Field(default=0.0, description="0-3，1.5 以上为高质量")
    query: str = Field(default="", description="The query that surfaced this source")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="0-1, derived from authority and rerank score"
    )


class ResearchState(MessagesState, total=False):
    intent: str
    tasks: list[ResearchTask]
    plan_notes: str
    constraints: list[str]
    sources: Annotated[list[Source], operator.add]
    analysis: str
    report: str
    next_action: NextAction
    rounds: int
    max_rounds: int
    turn_attempted: bool
    abort_reason: str
