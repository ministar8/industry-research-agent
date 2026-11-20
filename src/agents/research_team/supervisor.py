"""The supervisor: intent parsing, task decomposition, routing, and result assembly.

It deliberately owns no tools. Everything it does is a decision over state.
"""

from collections.abc import Sequence
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from agents.research_team.contracts import (
    ResearchSubject,
    ResearchTask,
)
from agents.research_team.prompts import load_prompt, render_prompt
from agents.research_team.sources import dedupe_sources, source_appendix
from agents.research_team.state import (
    NextAction,
    ResearchState,
)
from agents.research_team.structured import structured_output
from core import resolve_model, settings
from core.metrics import Degradation, record_degradation

MAX_ROUNDS = 5

# RoutingDecision is a few hundred tokens at most; a tight ceiling makes a degenerate
# runaway generation die cheap instead of burning the full shared LLM_MAX_TOKENS.
ROUTING_MAX_TOKENS = 2048

SUPERVISOR_PROMPT = load_prompt(
    "research_team.supervisor",
    "research_team.supervisor.instructions",
    "research_team.supervisor.status",
)

# Everything a new research turn starts from. Kept as one list because the previous two
# hand-maintained copies had already drifted: rounds, turn_attempted, constraints and
# plan_notes were cleared on one path and kept on the other, which leaks a finished
# turn's scope into the next request.
_TURN_RESET: dict[str, Any] = {
    "analysis": "",
    "report": "",
    "tasks": [],
    "subjects": [],
    "constraints": [],
    "plan_notes": "",
    "evidence_coverage": [],
    "subject_coverage": [],
    "evidence_statements": [],
    "analyzed_source_ids": [],
    "analysis_batches": 0,
    "facts": [],
    "analysis_patterns": [],
    "analysis_contradictions": [],
    "analysis_gaps": [],
    "rounds": 0,
    "turn_attempted": False,
    "abort_reason": "",
}


class RoutingDecision(BaseModel):
    intent: str = Field(description="提炼后的调研目标，一句话，已剥离寒暄与背景")
    is_research: bool = Field(description="是否包含需要外部信息的调研需求")
    sub_questions: list[str] = Field(
        description="拆解出的可独立回答的子问题，调研轮 1-4 个，非调研轮给空列表"
    )
    subjects: list[ResearchSubject] = Field(
        description="本轮研究对象，必须覆盖用户提到的主要对象；非调研轮为空列表"
    )
    constraints: list[str] = Field(description="必须保真的范围限定与交付要求，没有则空列表")
    next: NextAction = Field(description="下一步的路由目标")
    reason: str = Field(description="路由理由，一句话")

    @model_validator(mode="before")
    @classmethod
    def _fill_missing_lists(cls, data: Any) -> Any:
        # Schema-required so constrained decoding must emit them; tolerant here so the
        # json_mode fallback and direct construction keep the old implicit defaults.
        if isinstance(data, dict):
            data.setdefault("sub_questions", [])
            data.setdefault("constraints", [])
            data.setdefault("subjects", [])
        return data


def _status_values(state: ResearchState) -> dict[str, str]:
    """Collect the values the status template declares.

    The wording lives in prompts/research_team/supervisor/status.md; this only gathers
    what it needs, so editing the status block never means editing Python.
    """
    sources = dedupe_sources(state.get("sources"))
    tasks = state.get("tasks") or []
    subjects = state.get("subjects") or []
    coverage = state.get("evidence_coverage") or []
    coverage_summary = (
        "，".join(
            f"{status} {sum(1 for item in coverage if item.status == status)}"
            for status in ("sufficient", "partial", "missing", "conflicting")
            if any(item.status == status for item in coverage)
        )
        or "（无）"
    )
    return {
        "intent": state.get("intent") or "（无）",
        "tasks": "\n".join(f"- [{task.id}] {task.question}" for task in tasks) or "（无）",
        "subjects": "\n".join(
            f"- [{subject.id}] {subject.name}（别名：{'、'.join(subject.aliases) or '无'}）"
            for subject in subjects
        )
        or "（无）",
        "constraints": "\n".join(f"- {item}" for item in state.get("constraints") or [])
        or "（无）",
        "source_count": str(len(sources)),
        "analysis_done": "是" if state.get("analysis") else "否",
        "coverage_summary": coverage_summary,
        "subject_coverage_count": str(len(state.get("subject_coverage") or [])),
        "report_done": "是" if state.get("report") else "否",
        "rounds": str(state.get("rounds") or 0),
        "max_rounds": str(MAX_ROUNDS),
    }


def _recorded_target(rounds: int, decision: RoutingDecision, state: ResearchState) -> NextAction:
    """Apply the round budget, then the state preconditions.

    Capping here rather than in `route` keeps `next_action` truthful: the state never
    records a route the budget forbids, and dispatch needs no policy of its own.
    """
    if rounds > MAX_ROUNDS:
        return "finalize"
    return _enforce(decision, state)


def _constraints_for(
    previous: Sequence[str], decision: RoutingDecision, is_new_turn: bool
) -> list[str]:
    """Keep the verbatim constraint set within a turn; start clean on a new one.

    A later round re-parses the same input and may under-report constraints, so the set
    is merged rather than replaced - but only inside a turn: a new request must not
    inherit the previous one's scope.
    """
    if is_new_turn:
        return list(dict.fromkeys(decision.constraints))
    merged = list(previous)
    merged += [item for item in decision.constraints if item not in merged]
    return merged


def _enforce(decision: RoutingDecision, state: ResearchState) -> NextAction:
    """Downgrade decisions that violate the state preconditions."""
    if not decision.is_research:
        return "finalize"

    has_sources = bool(dedupe_sources(state.get("sources")))
    has_analysis = bool(state.get("analysis"))
    target = decision.next

    if target == "analyst" and not has_sources:
        return "researcher"
    if target == "writer" and not has_analysis:
        return "analyst" if has_sources else "researcher"
    if target == "researcher" and has_sources and not has_analysis:
        # Pre-analysis re-search can loop to the round cap and discard everything collected.
        record_degradation(Degradation.RESEARCHER_FORCED_ANALYST)
        return "analyst"
    return target


def _incoming_turn_id(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    turn_id = config.get("configurable", {}).get("turn_id")
    return str(turn_id) if turn_id else None


def _context_mode(config: RunnableConfig | None) -> str:
    if not config:
        return "fresh"
    return str(config.get("configurable", {}).get("context_mode") or "fresh")


def _is_new_turn(state: ResearchState, config: RunnableConfig | None) -> bool:
    incoming_turn_id = _incoming_turn_id(config)
    if incoming_turn_id is not None:
        return incoming_turn_id != state.get("active_turn_id")
    return bool(state["messages"]) and state["messages"][-1].type == "human"


def _turn_state(state: ResearchState, is_new_turn: bool, context_mode: str) -> ResearchState:
    if not is_new_turn:
        return state
    # _TURN_RESET is a plain table, so the merge needs an explicit cast back to the
    # state contract; every key in it is a real state field.
    return cast(
        ResearchState,
        {
            **state,
            **_TURN_RESET,
            # Reuse keeps the previous ledger, but the ids that track what Analyst read
            # must still restart: a new question gets its own reading pass.
            "sources": list(state.get("sources") or []) if context_mode == "reuse" else [],
        },
    )


def _to_tasks(
    questions: list[str], subjects: list[ResearchSubject] | None = None
) -> list[ResearchTask]:
    subjects = subjects or []
    return [
        ResearchTask(
            id=f"T{index}",
            question=question.strip(),
            assignee="researcher",
            subject_ids=[
                subject.id
                for subject in subjects
                if subject.name in question
                or any(alias and alias in question for alias in subject.aliases)
            ]
            or ([subjects[0].id] if len(subjects) == 1 else []),
        )
        for index, question in enumerate(questions, start=1)
        if question.strip()
    ]


def _fake_decision() -> RoutingDecision:
    return RoutingDecision(
        intent="假模型模式，跳过调研流程",
        is_research=False,
        sub_questions=[],
        constraints=[],
        subjects=[],
        next="finalize",
        reason="USE_FAKE_MODEL 已启用，仅验证链路连通性",
    )


async def supervisor(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    is_new_turn = _is_new_turn(state, config)
    context_mode = _context_mode(config)
    turn_state = _turn_state(state, is_new_turn, context_mode)
    # A bare abort marker can exist on the first request in tests or restored state;
    # only a state with a prior turn may use the human message to clear it.
    has_prior_turn = bool(
        state.get("rounds")
        or state.get("turn_attempted")
        or state.get("report")
        or state.get("analysis")
    )
    if state.get("abort_reason") and not has_prior_turn:
        is_new_turn = False
        turn_state = state
    # A rejected retrieval is already terminal; asking the model to route again just
    # buys another call that must end in finalize.
    if state.get("abort_reason") and not is_new_turn:
        return {"next_action": "finalize"}

    rounds = (turn_state.get("rounds") or 0) + 1
    # rounds is scoped to the active turn; the last message being human is only the
    # compatibility fallback when a caller does not provide a turn_id.

    if settings.USE_FAKE_MODEL:
        decision = _fake_decision()
    else:
        model: Any = resolve_model(config)
        decider = structured_output(
            model, RoutingDecision, "supervisor", max_tokens=ROUTING_MAX_TOKENS
        )
        messages: list[AnyMessage] = render_prompt(SUPERVISOR_PROMPT, **_status_values(turn_state))
        messages.extend(state["messages"][-12:])
        decision = await decider.ainvoke(messages)

    target = _recorded_target(rounds, decision, turn_state)

    update: dict[str, Any] = {
        # Reset first so the explicit values below win: rounds is this turn's count,
        # not the zero a fresh turn starts from.
        **(_TURN_RESET if is_new_turn else {}),
        "intent": decision.intent,
        "next_action": target,
        "rounds": rounds,
        "plan_notes": decision.reason,
        "turn_attempted": False if is_new_turn else bool(turn_state.get("turn_attempted")),
    }

    incoming_turn_id = _incoming_turn_id(config)
    if is_new_turn:
        update["sources"] = list(turn_state.get("sources") or [])
    if incoming_turn_id is not None:
        update["active_turn_id"] = incoming_turn_id
        update["active_context_mode"] = context_mode

    if target in {"researcher", "analyst", "writer"}:
        if is_new_turn:
            update["subjects"] = list(decision.subjects)
            if decision.sub_questions:
                update["tasks"] = _to_tasks(decision.sub_questions, decision.subjects)
            elif target == "researcher":
                record_degradation(Degradation.SUB_QUESTIONS_MISSING)
                update["tasks"] = (
                    _to_tasks([decision.intent], decision.subjects)
                    if decision.intent.strip()
                    else []
                )
        elif not turn_state.get("tasks") and decision.sub_questions:
            update["tasks"] = _to_tasks(decision.sub_questions, turn_state.get("subjects") or [])
        # Any re-run of the pipeline invalidates the previous draft; leaving it in place
        # lets finalize return a report that no longer matches the analysis.
        if turn_state.get("report"):
            update["report"] = ""
        # Previous coverage is deliberately kept: Analyst merges each batch into it and
        # only replaces a judgement that cites evidence. Clearing here would make a
        # retrieval round erase everything the previous batch established.

    update["constraints"] = _constraints_for(
        turn_state.get("constraints") or [], decision, is_new_turn
    )

    return update


def route(state: ResearchState) -> Literal["researcher", "analyst", "writer", "finalize"]:
    # A rejected retrieval (bad key, exhausted balance) cannot be fixed by rescheduling;
    # looping until MAX_ROUNDS just multiplies the failures.
    if state.get("abort_reason"):
        return "finalize"
    # Supervisor already applies MAX_ROUNDS when it records next_action; testing the cap
    # again here only creates a second place for the two decisions to disagree.
    return state.get("next_action") or "finalize"


NO_TARGET_HINT = """没能从这段输入里识别出可调研的对象。

请说明要调研的行业、公司或产品，并尽量带上时间范围、地区与关注的产业链环节。例如：

- 2025 年国内储能电芯厂商的出货量排名与价格走势
- 对比两家头部厂商在欧洲的产能布局
- 工业机器人减速器国产替代的进展与瓶颈

一段话里夹杂寒暄或背景说明没关系，只要其中提到调研对象即可。"""


async def finalize(state: ResearchState) -> dict[str, Any]:
    sources = dedupe_sources(state.get("sources"))
    report = state.get("report")
    analysis = state.get("analysis")
    rounds = state.get("rounds") or 0

    if state.get("abort_reason") == "search_rejected":
        content = (
            "检索服务拒绝了本次请求，本轮未采集到任何来源。\n\n"
            "常见原因是 DeepSeek 账户余额不足或 API Key 失效，重试不会改善。\n"
            "请检查账户状态后重新发起调研。"
        )
        return {"messages": [AIMessage(content=content)]}

    if report:
        content = report + source_appendix(sources)
    elif analysis:
        # Only claim the round cap when it actually fired; otherwise the note misleads
        # whoever reads the output into thinking the graph ran out of budget.
        capped = rounds > MAX_ROUNDS
        note = (
            "\n\n> 本轮达到调度上限，已返回阶段性分析。"
            if capped
            else "\n\n> 本轮未生成成稿报告，已返回阶段性分析。"
        )
        content = "## 调研分析（未生成成稿报告）\n\n" + analysis + note + source_appendix(sources)
    elif not state.get("turn_attempted"):
        # Nothing was researchable in this turn's input; this is guidance, not a failure.
        content = NO_TARGET_HINT
    elif sources:
        content = (
            f"已采集 {len(sources)} 条来源，但本轮未能完成分析，未生成报告。\n\n"
            f"当前状态：调度轮次 {rounds}。\n"
            "建议：稍后重试，或缩小问题范围后重新发起。"
        )
    else:
        content = (
            "本次调研未能采集到可用信息，未生成报告。\n\n"
            f"当前状态：来源 {len(sources)} 条，调度轮次 {rounds}。\n"
            "建议：确认问题范围是否过于宽泛，或稍后重试。"
        )

    return {"messages": [AIMessage(content=content)]}
