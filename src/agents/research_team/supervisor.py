"""The supervisor: intent parsing, task decomposition, routing, and result assembly.

It deliberately owns no tools. Everything it does is a decision over state.
"""

from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from agents.research_team.helpers import (
    dedupe_sources,
    resolve_model,
    source_appendix,
    structured_output,
)
from agents.research_team.state import NextAction, ResearchState, ResearchTask
from core import settings
from core.metrics import record_degradation

MAX_ROUNDS = 5

# RoutingDecision is a few hundred tokens at most; a tight ceiling makes a degenerate
# runaway generation die cheap instead of burning the full shared LLM_MAX_TOKENS.
ROUTING_MAX_TOKENS = 2048

SUPERVISOR_PROMPT = """你是行业与竞品调研团队的调度中枢。你不搜索、不写作、不做分析，只做四件事：

1. 意图解析：从用户的输入中提取真正要调研什么。
2. 任务拆解：把调研目标拆成可独立回答的子问题。
3. 路由调度：决定下一步交给哪个子智能体。
4. 结果整合：调研完成后由系统把产出组装为最终答复（你只需选择 finalize）。

用户通常不会只发一句话，而是发一整段自然描述，其中既有调研需求，也夹杂大量无关内容。
你必须同时做两件事：

【剥离】以下内容不进入任务，也不得影响子问题的措辞：
- 寒暄、致谢、语气词、对上一轮结果的评价
- 个人背景与动机（「我在做投资」「我是做供应链的」），除非它限定了调研范围
- 与调研对象无关的闲谈、情绪表达、重复啰嗦的铺垫

【保真】以下内容必须原样保留，不得概括掉：
- 调研对象：行业、公司、产品、技术路线、产业链环节
- 范围限定：时间区间、地区、统计口径
- 明确的排除项（「不看上游」「排除代工厂」）
- 交付要求：格式（表格 / 清单 / 叙述）、篇幅、面向对象、语言

保真项写入 constraints；没有则给空列表。

可路由的目标：
- researcher：缺少外部信息，或现有来源不足以支撑结论。
- analyst：来源已足够，但还没有形成结构化分析。
- writer：分析已完成，需要产出成稿报告。
- finalize：报告已完成且本轮没有新要求。选择 finalize 时系统返回既有成果，不新生成内容。

硬性规则：
- 来源为空时不得选择 analyst。
- 分析为空时不得选择 writer。
- 子问题必须具体到可直接检索，且必须锚定用户提到的调研对象。
  正面：「2025 年中国换电重卡销量与主要运营商份额」；反面：「了解一下换电」。
- 子问题数量 1-4 个，需覆盖用户提到的全部主要对象，不要遗漏后文才提到的对象。
- 哪怕整段输入只有一句与调研相关，也要抓住它继续调研，不要因为夹杂无关内容就选择 finalize。
- 只有当整段输入确实找不到任何可调研对象时，才选择 finalize。"""


class RoutingDecision(BaseModel):
    intent: str = Field(description="提炼后的调研目标，一句话，已剥离寒暄与背景")
    is_research: bool = Field(description="是否包含需要外部信息的调研需求")
    sub_questions: list[str] = Field(
        description="拆解出的可独立回答的子问题，调研轮 1-4 个，非调研轮给空列表"
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
        return data


def _status_block(state: ResearchState) -> str:
    sources = dedupe_sources(state.get("sources"))
    tasks = state.get("tasks") or []
    task_lines = "\n".join(f"- [{t.id}] {t.question}" for t in tasks) or "（无）"
    constraint_lines = "\n".join(f"- {c}" for c in state.get("constraints") or []) or "（无）"
    return f"""当前调研状态：
- 既有意图：{state.get("intent") or "（无）"}
- 已拆解任务：
{task_lines}
- 交付与范围约束：
{constraint_lines}
- 已采集来源：{len(sources)} 条
- 分析完成：{"是" if state.get("analysis") else "否"}
- 报告完成：{"是" if state.get("report") else "否"}
- 已用调度轮次：{state.get("rounds") or 0}/{state.get("max_rounds") or MAX_ROUNDS}"""


def _enforce(decision: RoutingDecision, state: ResearchState) -> NextAction:
    """Downgrade decisions that violate the state preconditions."""
    has_sources = bool(dedupe_sources(state.get("sources")))
    has_analysis = bool(state.get("analysis"))
    target = decision.next

    if target == "analyst" and not has_sources:
        return "researcher"
    if target == "writer" and not has_analysis:
        return "analyst" if has_sources else "researcher"
    if target == "researcher" and has_sources and not has_analysis:
        # Pre-analysis re-search can loop to the round cap and discard everything collected.
        record_degradation("researcher_forced_analyst")
        return "analyst"
    return target


def _to_tasks(questions: list[str]) -> list[ResearchTask]:
    return [
        ResearchTask(id=f"T{index}", question=question.strip(), assignee="researcher")
        for index, question in enumerate(questions, start=1)
        if question.strip()
    ]


def _fake_decision() -> RoutingDecision:
    return RoutingDecision(
        intent="假模型模式，跳过调研流程",
        is_research=False,
        sub_questions=[],
        constraints=[],
        next="finalize",
        reason="USE_FAKE_MODEL 已启用，仅验证链路连通性",
    )


async def supervisor(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    # A rejected retrieval is already terminal; asking the model to route again just
    # buys another call that must end in finalize.
    if state.get("abort_reason"):
        return {"next_action": "finalize"}

    rounds = (state.get("rounds") or 0) + 1
    max_rounds = state.get("max_rounds") or MAX_ROUNDS
    # rounds is thread-scoped; the last message being human marks the first pass of a
    # new turn, which is where per-turn bookkeeping has to be reset.
    is_new_turn = bool(state["messages"]) and state["messages"][-1].type == "human"

    if settings.USE_FAKE_MODEL:
        decision = _fake_decision()
    else:
        model: Any = resolve_model(config)
        decider = structured_output(
            model, RoutingDecision, "supervisor", max_tokens=ROUTING_MAX_TOKENS
        )
        messages: list[AnyMessage] = [
            SystemMessage(content=SUPERVISOR_PROMPT + "\n\n" + _status_block(state))
        ]
        messages.extend(state["messages"][-12:])
        decision = await decider.ainvoke(messages)

    target = "finalize" if rounds > max_rounds else _enforce(decision, state)

    update: dict[str, Any] = {
        "intent": decision.intent,
        "next_action": target,
        "rounds": rounds,
        "plan_notes": decision.reason,
        "turn_attempted": False if is_new_turn else bool(state.get("turn_attempted")),
    }

    if target in {"researcher", "analyst", "writer"}:
        if decision.sub_questions:
            update["tasks"] = _to_tasks(decision.sub_questions)
        elif target == "researcher" and not state.get("tasks") and decision.intent.strip():
            record_degradation("sub_questions_missing")
            update["tasks"] = _to_tasks([decision.intent])
        # Any re-run of the pipeline invalidates the previous draft; leaving it in place
        # lets finalize return a report that no longer matches the analysis.
        if state.get("report"):
            update["report"] = ""

    # A later round re-parses the same input and may under-report constraints; merging
    # within a turn keeps the verbatim set intact. A new turn starts fresh so the
    # previous turn's scope does not leak into an unrelated request.
    if is_new_turn:
        update["constraints"] = list(dict.fromkeys(decision.constraints))
    else:
        merged = list(state.get("constraints") or [])
        merged += [c for c in decision.constraints if c not in merged]
        update["constraints"] = merged

    return update


def route(state: ResearchState) -> Literal["researcher", "analyst", "writer", "finalize"]:
    # A rejected retrieval (bad key, exhausted balance) cannot be fixed by rescheduling;
    # looping until MAX_ROUNDS just multiplies the failures.
    if state.get("abort_reason"):
        return "finalize"
    rounds = state.get("rounds") or 0
    max_rounds = state.get("max_rounds") or MAX_ROUNDS
    if rounds > max_rounds:
        return "finalize"
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
        capped = rounds > (state.get("max_rounds") or MAX_ROUNDS)
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
