"""Writer: turn the structured analysis into a citable report."""

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, create_model

from agents.research_team.helpers import (
    clamp_anchors,
    dedupe_sources,
    numbered_sources,
    resolve_model,
    structured_output,
)
from agents.research_team.state import ResearchState
from core import settings
from core.metrics import record_degradation

logger = logging.getLogger(__name__)

WRITER_PROMPT = """你是行业与竞品调研团队的撰稿人。输入是结构化的分析结果与来源编号表，
你负责产出一份可直接交付的调研报告。

报告结构（固定四段，不得增删）：
1. 摘要：3-5 句话讲清结论，包含最关键的数字。
2. 主要发现：3-6 条，每条一句话结论 + 支撑数据。
3. 分析：解释发现背后的机制、竞争格局或趋势。
4. 结论：给出可执行的判断或建议，并诚实标注不确定性。

硬性要求：
- 每个关键事实后必须保留来源锚点 [n]，n 必须来自给定的来源编号表。
- 不得引入来源编号表之外的 URL 或事实。
- 若给出交付约束（格式、篇幅、面向对象、语言、明确排除项），必须逐条满足。
- 若约束要求「表格」：把对比数据填进 tables 字段（title / headers / rows），由系统渲染。
  不要在 summary / findings / analysis 里手写 Markdown 表格，也不得用项目符号列表代替。
- 若约束明确不要表格：tables 留空，用项目符号呈现。
- 不得使用「据我所知」「一般来说」这类无来源的表述。
- 直接成文，不要先列提纲，也不要复述思考过程。
- 报告正文不得出现元文本：不写自检清单、交付说明、字段说明，也不提及「来源编号表」「给定来源」
  「tables 字段」等内部输入名称；引用来源直接用锚点 [n]，描述信息缺口用「现有来源/已采集来源」。
  约束满足情况由系统校验，无需声明。
- 用中文书面语，不使用感叹号，不使用营销语气。"""

MAX_SOURCES_FOR_REPORT = 12

# Writer turns analysis into prose; it does not need the raw excerpts to do that.
# Measured on the same sources: feeding 12 excerpts cost 13k input tokens and produced
# shorter (sometimes truncated) reports than giving only the numbered index.
INCLUDE_SOURCE_EXCERPTS = False
# Below this, the analysis is too thin to write from, so excerpts come back as material.
MIN_ANALYSIS_CHARS_FOR_INDEX = 1500

# A constraint can ask for a table or explicitly rule one out ("不用表格"). Only the
# former should force the schema, otherwise a bullet-only request gets a stray table.
# The negation must directly modify the table mention within the same clause, so an
# unrelated refusal ("不要分析动力电池") cannot cancel a positive "用表格" request.
_TABLE_NEGATION = re.compile(
    r"(?:不用|不使用|不要|无需|不必|避免|非)(?:使用|用|做|作)?(?:任何|太多)?表格"
)
_CLAUSE_SPLIT = re.compile(r"[，,。;；\n]")


def requires_table(constraints: list[str]) -> bool:
    positive = False
    for constraint in constraints:
        for clause in _CLAUSE_SPLIT.split(constraint):
            if "表格" not in clause:
                continue
            if _TABLE_NEGATION.search(clause):
                return False
            positive = True
    return positive


class TableBlock(BaseModel):
    title: str = Field(default="", description="表格标题，可为空")
    headers: list[str] = Field(description="表头，至少 2 列")
    rows: list[list[str]] = Field(description="数据行，每行单元格数与表头一致")


class ReportBundle(BaseModel):
    title: str = Field(description="报告标题，包含行业/对象与时间范围")
    # min_length keeps the schema from accepting a hollow report: constrained decoding
    # then cannot emit "", which used to validate and produce a 130-character "report".
    summary: str = Field(min_length=1, description="摘要，3-5 句")
    findings: list[str] = Field(min_length=2, description="主要发现，3-6 条")
    tables: list[TableBlock] = Field(
        default_factory=list,
        description="对比数据表；约束要求表格时至少填写一个",
    )
    analysis: str = Field(min_length=1, description="分析段落")
    conclusion: str = Field(min_length=1, description="结论与建议")


# Constrained decoding cannot produce zero tables when the schema forbids it, which is
# the only reliable way to honour "use a table" — the prompt alone was followed ~50%.
TableRequiredReport = create_model(
    "TableRequiredReport",
    __base__=ReportBundle,
    tables=(
        list[TableBlock],
        Field(min_length=1, description="至少一个对比数据表，不得为空"),
    ),
)


def report_schema(constraints: list[str]) -> type[ReportBundle]:
    return TableRequiredReport if requires_table(constraints) else ReportBundle


def _render_table(table: TableBlock) -> str:
    width = len(table.headers)
    if not width:
        return ""
    header = "| " + " | ".join(_cell(text) for text in table.headers) + " |"
    divider = "| " + " | ".join("---" for _ in table.headers) + " |"
    rows = []
    for row in table.rows:
        cells = [_cell(cell) for cell in row][:width]
        cells += [""] * (width - len(cells))
        rows.append("| " + " | ".join(cells) + " |")
    caption = f"**{_cell(table.title)}**\n" if table.title else ""
    return caption + "\n".join([header, divider, *rows])


def _cell(value: object) -> str:
    """A pipe or a newline inside a cell breaks the row; escape both."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_report(bundle: ReportBundle, source_count: int) -> str:
    findings = "\n".join(f"- {item}" for item in bundle.findings) or "- 无"
    tables = "\n\n".join(rendered for rendered in map(_render_table, bundle.tables) if rendered)
    findings_section = "## 主要发现\n" + findings + (("\n\n" + tables) if tables else "")
    body = "\n\n".join(
        [
            f"# {bundle.title}",
            "## 摘要\n" + bundle.summary,
            findings_section,
            "## 分析\n" + bundle.analysis,
            "## 结论\n" + bundle.conclusion,
        ]
    )
    return clamp_anchors(body, source_count)


_COMPLETENESS_REMINDER = SystemMessage(
    content="上一次输出不完整：存在空字段或在半句话处截断的字段。请基于给定的分析结果补全后重新输出："
    "摘要、分析、结论与每条主要发现都必须以完整句子收尾（句读或引用锚点结束），不要新增事实。"
)

_ANCHOR_TAIL = re.compile(r"(?:\[\d+\]\s*)+$")
_SENTENCE_END = "。！？；…」』\"'.!?;"


def _ends_cleanly(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_ANCHOR_TAIL.search(stripped)) or stripped[-1] in _SENTENCE_END


def _is_complete(bundle: ReportBundle) -> bool:
    """Reject hollow bundles and fields closed mid-sentence.

    Constrained decoding lets the model close a JSON string at any token, so a
    schema-valid bundle can still carry half a sentence ("…市场格局呈") with
    finish_reason=stop. Length alone cannot see that; every prose field must end
    with sentence punctuation or a citation anchor.
    """
    if len(bundle.findings) < 2:
        return False
    if not all(_ends_cleanly(item) for item in bundle.findings):
        return False
    return all(
        len(field.strip()) >= 20 and _ends_cleanly(field)
        for field in (bundle.summary, bundle.analysis, bundle.conclusion)
    )


async def writer(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    sources = dedupe_sources(state.get("sources"))
    analysis = state.get("analysis") or ""
    if not analysis:
        return {"messages": [AIMessage(content="分析结果为空，无法撰写报告。")]}

    sources = sources[:MAX_SOURCES_FOR_REPORT]

    if settings.USE_FAKE_MODEL:
        return {
            "report": "# 测试报告\n\n（假模型模式）",
            "messages": [AIMessage(content="报告已生成。")],
        }

    model: Any = resolve_model(config)
    constraints_list = state.get("constraints") or []
    structured = structured_output(model, report_schema(constraints_list), "writer")
    constraints = "\n".join(f"- {c}" for c in constraints_list) or "（无）"
    include_excerpts = INCLUDE_SOURCE_EXCERPTS or len(analysis) < MIN_ANALYSIS_CHARS_FOR_INDEX
    messages = [
        SystemMessage(content=WRITER_PROMPT),
        SystemMessage(
            content=f"调研意图：{state.get('intent') or '（未解析）'}\n\n"
            f"交付与范围约束：\n{constraints}\n\n"
            f"分析结果：\n{analysis}\n\n"
            f"可用来源编号表：\n{numbered_sources(sources, include_excerpts)}"
        ),
    ]
    bundle: ReportBundle = await structured.ainvoke(messages)
    if not _is_complete(bundle):
        logger.warning("Writer returned an incomplete report, retrying once")
        record_degradation("report_incomplete")
        bundle = await structured.ainvoke([*messages, _COMPLETENESS_REMINDER])
        if not _is_complete(bundle):
            logger.warning("Writer report still incomplete after retry")
    report = render_report(bundle, len(sources))

    return {"report": report, "messages": [AIMessage(content=f"报告已完成：《{bundle.title}》")]}
