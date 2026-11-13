"""Analyst: clean, deduplicate, cross-validate, and structure the collected sources."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.research_team.helpers import (
    dedupe_sources,
    numbered_sources,
    resolve_model,
    structured_output,
)
from agents.research_team.state import ResearchState
from core import settings
from core.metrics import record_degradation

logger = logging.getLogger(__name__)

ANALYST_PROMPT = """你是行业与竞品调研团队的分析师。输入是检索员采集的原始来源，你负责把它转化为
可信的、可被引用的结构化分析。你不撰写最终报告。

工作方法：
1. 去重：同一事实被多家转载时只保留一手或最权威的那条。
2. 交叉验证：同一事实至少两个独立来源才算确证；只有一个来源时必须标注「单源未验证」。
3. 识别矛盾：指出来源之间的冲突、统计口径差异、时间窗口不一致，不要掩盖。
4. 识别缺口：明确哪些关键信息仍然缺失，以及缺失对结论的影响。
5. 提炼模式：给出 2-5 条对决策有意义的判断，不要复述事实。

硬性要求：
- 每条事实后必须标注来源编号，如 [1][4]。
- 禁止引入来源之外的知识、常识或推测。
- 数据必须保留原始口径（年份、地区、单位）。
- 必须遵守给定的排除项：落在排除范围内的来源不纳入事实基线，并在信息缺口里说明被排除了什么。
- 直接给出结论，每项不超过两句话；不要展示推演过程。"""

MAX_SOURCES_FOR_ANALYSIS = 12

_THIN_ANALYSIS_RETRY = SystemMessage(
    content="上一次分析的「事实基线」不足两条或内容为空。请回到来源编号表，"
    "逐条提取可引用的具体事实（含数字与口径），不要写「无可用事实」。"
)


class AnalysisBundle(BaseModel):
    # Non-empty lists are enforced by the schema: an all-empty bundle is valid JSON but
    # renders as "无可用事实", which then starves the writer and produces an empty report.
    facts: list[str] = Field(min_length=2, description="经去重与交叉验证后的事实，每条带来源编号")
    patterns: list[str] = Field(min_length=1, description="2-5 条关键模式或趋势判断")
    # Contradictions may genuinely be absent; forcing one would invite fabrication.
    contradictions: list[str] = Field(description="来源间的矛盾或口径差异，没有则留空列表")
    gaps: list[str] = Field(min_length=1, description="仍然缺失的关键信息及其影响")


def render_analysis(bundle: AnalysisBundle) -> str:
    def bullets(items: list[str], empty: str) -> str:
        return "\n".join(f"- {item}" for item in items) if items else f"- {empty}"

    return "\n\n".join(
        [
            "## 事实基线\n" + bullets(bundle.facts, "无可用事实"),
            "## 关键模式\n" + bullets(bundle.patterns, "来源不足以提炼模式"),
            "## 交叉验证与矛盾\n" + bullets(bundle.contradictions, "未发现明显矛盾"),
            "## 信息缺口\n" + bullets(bundle.gaps, "未识别出关键缺口"),
        ]
    )


async def analyst(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    sources = dedupe_sources(state.get("sources"))[:MAX_SOURCES_FOR_ANALYSIS]
    if not sources:
        return {"messages": [AIMessage(content="来源为空，无法进入分析阶段。")]}

    if settings.USE_FAKE_MODEL:
        return {
            "analysis": "（假模型模式）无分析内容。",
            "messages": [AIMessage(content="分析完成。")],
        }

    model: Any = resolve_model(config)
    structured = structured_output(model, AnalysisBundle, "analyst")
    constraints = "\n".join(f"- {c}" for c in state.get("constraints") or []) or "（无）"
    messages = [
        SystemMessage(content=ANALYST_PROMPT),
        SystemMessage(
            content=f"调研意图：{state.get('intent') or '（未解析）'}\n\n"
            f"必须遵守的范围与交付约束：\n{constraints}\n\n"
            f"来源列表：\n{numbered_sources(sources)}"
        ),
    ]
    bundle: AnalysisBundle = await structured.ainvoke(messages)
    if len([item for item in bundle.facts if item.strip()]) < 2:
        # Schema already forbids an empty list; this catches a list of blank strings.
        logger.warning("Analyst returned a thin fact base, retrying once")
        record_degradation("analysis_incomplete")
        bundle = await structured.ainvoke([*messages, _THIN_ANALYSIS_RETRY])
    analysis = render_analysis(bundle)

    summary = (
        f"分析完成：事实 {len(bundle.facts)} 条，模式 {len(bundle.patterns)} 条，"
        f"矛盾 {len(bundle.contradictions)} 处，缺口 {len(bundle.gaps)} 处。"
    )
    return {"analysis": analysis, "messages": [AIMessage(content=summary)]}
