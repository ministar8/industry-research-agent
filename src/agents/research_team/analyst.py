"""Analyst: clean, deduplicate, cross-validate, and structure the collected sources."""

import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.research_team.contracts import (
    EvidenceCoverage,
    EvidenceStatement,
    SubjectEvidenceCoverage,
)
from agents.research_team.normalize import (
    MAX_MERGED_CONTRADICTIONS,
    MAX_MERGED_FACTS,
    MAX_MERGED_GAPS,
    MAX_MERGED_PATTERNS,
    merge_coverage,
    merge_lines,
    merge_statements,
    merge_subject_coverage,
    normalize_coverage,
    normalize_evidence_statements,
    normalize_subject_coverage,
    to_ledger_coverage,
    to_ledger_statements,
    to_ledger_subject_coverage,
)
from agents.research_team.prompts import load_prompt, render_prompt
from agents.research_team.sources import (
    batch_window,
    dedupe_sources,
    excerpt_limit_for,
    join_gaps,
    numbered_sources,
    remap_anchors,
    url_key,
)
from agents.research_team.state import (
    ResearchState,
)
from agents.research_team.structured import structured_output
from core import resolve_model, settings
from core.metrics import Degradation, record_degradation

logger = logging.getLogger(__name__)

ANALYST_PROMPT = load_prompt("research_team.analyst.instructions")
_ANALYST_CONTEXT_PROMPT = load_prompt("research_team.analyst.context")

# One pass over the sources collected in a single retrieval round. Retrieval in this
# graph is effectively single-round (supervisor forces analysis before a second search),
# so the window has to cover a normal haul in one call: observed hauls are 7-16 pages.
# Excerpts are spread over the budget rather than capped per source, which keeps the
# wider window at the same token cost.
MAX_SOURCES_FOR_ANALYSIS = 20

# The fact base is prose-heavy but bounded; this leaves several times the normal
# output as headroom while still capping a degenerate runaway generation.
ANALYSIS_MAX_TOKENS = 8192

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
    coverage: list[EvidenceCoverage] = Field(
        min_length=1,
        description="对每个 ResearchTask 的证据覆盖判断，不得新增或遗漏任务",
    )
    subject_coverage: list[SubjectEvidenceCoverage] = Field(
        default_factory=list,
        description="对每个 ResearchSubject 的对象和证据维度覆盖判断",
    )
    evidence_statements: list[EvidenceStatement] = Field(
        default_factory=list,
        description="经归一化的离散点、变化、趋势、预测和状态证据陈述",
    )


def _remap_bundle_text(bundle: AnalysisBundle, mapping: dict[int, int]) -> AnalysisBundle:
    """Move every free-text citation the model wrote into ledger numbering."""

    def remap(lines: Sequence[str]) -> list[str]:
        return [remap_anchors(line, mapping) for line in lines]

    return bundle.model_copy(
        update={
            "facts": remap(bundle.facts),
            "patterns": remap(bundle.patterns),
            "contradictions": remap(bundle.contradictions),
            "gaps": remap(bundle.gaps),
            "coverage": [
                item.model_copy(
                    update={
                        "gaps": remap(item.gaps),
                        "reason": remap_anchors(item.reason, mapping),
                    }
                )
                for item in bundle.coverage
            ],
            "subject_coverage": [
                item.model_copy(
                    update={
                        "gaps": remap(item.gaps),
                        "reason": remap_anchors(item.reason, mapping),
                        "aspects": [
                            aspect.model_copy(
                                update={
                                    "gaps": remap(aspect.gaps),
                                    "reason": remap_anchors(aspect.reason, mapping),
                                }
                            )
                            for aspect in item.aspects
                        ],
                    }
                )
                for item in bundle.subject_coverage
            ],
            "evidence_statements": [
                item.model_copy(update={"statement": remap_anchors(item.statement, mapping)})
                for item in bundle.evidence_statements
            ],
        }
    )


def render_analysis(bundle: AnalysisBundle, coverage: list[EvidenceCoverage]) -> str:
    def bullets(items: list[str], empty: str) -> str:
        return "\n".join(f"- {item}" for item in items) if items else f"- {empty}"

    coverage_lines = []
    for item in coverage:
        sources = "".join(f"[{index}]" for index in item.source_indices) or "无"
        gaps = join_gaps(item.gaps)
        coverage_lines.append(
            f"- [{item.task_id}] {item.question}：{item.status}；"
            f"证据强度 {item.evidence_strength}；来源 {sources}；"
            f"缺口：{gaps}；理由：{item.reason}"
        )
    subject_lines = []
    for item in bundle.subject_coverage:
        aspects = (
            "；".join(f"{aspect.name}:{aspect.status}" for aspect in item.aspects) or "未生成维度"
        )
        gaps = join_gaps(item.gaps)
        subject_lines.append(
            f"- [{item.subject_id}] {item.subject}：{item.status}；"
            f"证据强度 {item.evidence_strength}；维度 {aspects}；"
            f"缺口：{gaps}；理由：{item.reason}"
        )
    statement_lines = []
    for item in bundle.evidence_statements:
        sources = "".join(f"[{index}]" for index in item.source_indices)
        periods = "、".join(item.periods) or item.period or "未标明时间"
        qualifier = (
            "来源明确表述" if item.trend_origin == "explicit_source_statement" else "结构化证据"
        )
        statement_lines.append(
            f"- {item.kind}；时期 {periods}；类型 {item.value_type}；"
            f"指标 {item.metric or '未标明'}；单位 {item.unit or '未标明'}；"
            f"{qualifier}；来源 {sources}：{item.statement}"
        )
    return "\n\n".join(
        [
            "## 事实基线\n" + bullets(bundle.facts, "无可用事实"),
            "## 关键模式\n" + bullets(bundle.patterns, "来源不足以提炼模式"),
            "## 证据类型\n" + "\n".join(statement_lines or ["- 未生成结构化证据陈述"]),
            "## 交叉验证与矛盾\n" + bullets(bundle.contradictions, "未发现明显矛盾"),
            "## 信息缺口\n" + bullets(bundle.gaps, "未识别出关键缺口"),
            "## 子问题证据覆盖\n" + "\n".join(coverage_lines),
            "## 对象证据覆盖\n" + "\n".join(subject_lines or ["- 未生成对象覆盖判断"]),
        ]
    )


async def analyst(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    ledger = dedupe_sources(state.get("sources"))
    read = set(state.get("analyzed_source_ids") or [])
    sources, mapping = batch_window(ledger, read, MAX_SOURCES_FOR_ANALYSIS)
    if not sources:
        return {"messages": [AIMessage(content="来源为空，无法进入分析阶段。")]}
    if all(url_key(source.url) in read for source in sources):
        # No unread material: the window fell back to the tail, so this call re-reads
        # pages already analysed. Visible so the wasted round can be traced.
        record_degradation(Degradation.ANALYSIS_NO_NEW_SOURCES)
    pending = len(ledger) - len(read) - len(sources)
    if pending > 0:
        # One visit reads one batch. Whatever is left stays unread for the whole turn,
        # so the appendix would list pages no judgement ever saw.
        logger.warning("%d sources left unread after this analysis batch", pending)
        record_degradation(Degradation.ANALYSIS_SOURCES_LEFT_UNREAD)

    if settings.USE_FAKE_MODEL:
        return {
            "analysis": "（假模型模式）无分析内容。",
            "messages": [AIMessage(content="分析完成。")],
        }

    model: Any = resolve_model(config)
    structured = structured_output(model, AnalysisBundle, "analyst", max_tokens=ANALYSIS_MAX_TOKENS)
    constraints = "\n".join(f"- {c}" for c in state.get("constraints") or []) or "（无）"
    tasks = state.get("tasks") or []
    subjects = state.get("subjects") or []
    task_lines = (
        "\n".join(f"- [{task.id}] {task.question}" for task in tasks) or "（无已拆解子问题）"
    )
    subject_lines = (
        "\n".join(
            f"- [{subject.id}] {subject.name}（别名：{'、'.join(subject.aliases) or '无'}）"
            for subject in subjects
        )
        or "（无已识别研究对象）"
    )
    messages = render_prompt(ANALYST_PROMPT) + render_prompt(
        _ANALYST_CONTEXT_PROMPT,
        intent=state.get("intent") or "（未解析）",
        tasks=task_lines,
        subjects=subject_lines,
        constraints=constraints,
        sources=numbered_sources(sources, excerpt_chars=excerpt_limit_for(len(sources))),
    )
    bundle: AnalysisBundle = await structured.ainvoke(messages)
    if len([item for item in bundle.facts if item.strip()]) < 2:
        # Schema already forbids an empty list; this catches a list of blank strings.
        logger.warning("Analyst returned a thin fact base, retrying once")
        record_degradation(Degradation.ANALYSIS_INCOMPLETE)
        bundle = await structured.ainvoke([*messages, _THIN_ANALYSIS_RETRY])
    bundle = _remap_bundle_text(bundle, mapping)
    # Normalise against the batch first: the model only ever sees 1..len(sources).
    coverage = to_ledger_coverage(normalize_coverage(bundle.coverage, tasks, len(sources)), mapping)
    subject_coverage = to_ledger_subject_coverage(
        normalize_subject_coverage(bundle.subject_coverage, subjects, len(sources)), mapping
    )
    evidence_statements = to_ledger_statements(
        normalize_evidence_statements(bundle.evidence_statements, len(sources)), mapping
    )

    # Carry what earlier batches established. Analyst output replaces the state, so
    # without this merge a second batch would erase evidence it was never shown.
    coverage = merge_coverage(state.get("evidence_coverage") or [], coverage)
    subject_coverage = merge_subject_coverage(state.get("subject_coverage") or [], subject_coverage)
    evidence_statements = merge_statements(
        state.get("evidence_statements") or [], evidence_statements
    )
    facts = merge_lines(state.get("facts") or [], bundle.facts, MAX_MERGED_FACTS)
    patterns = merge_lines(
        state.get("analysis_patterns") or [], bundle.patterns, MAX_MERGED_PATTERNS
    )
    contradictions = merge_lines(
        state.get("analysis_contradictions") or [], bundle.contradictions, MAX_MERGED_CONTRADICTIONS
    )
    gaps = merge_lines(state.get("analysis_gaps") or [], bundle.gaps, MAX_MERGED_GAPS)
    bundle = bundle.model_copy(
        update={
            "facts": facts,
            "patterns": patterns,
            "contradictions": contradictions,
            "gaps": gaps,
            "subject_coverage": subject_coverage,
            "evidence_statements": evidence_statements,
        }
    )
    analysis = render_analysis(bundle, coverage)

    summary = (
        f"分析完成：事实 {len(bundle.facts)} 条，模式 {len(bundle.patterns)} 条，"
        f"矛盾 {len(bundle.contradictions)} 处，缺口 {len(bundle.gaps)} 处，"
        f"任务覆盖 {len(coverage)} 条，对象覆盖 {len(subject_coverage)} 条，"
        f"结构化证据 {len(evidence_statements)} 条。"
    )
    return {
        "analysis": analysis,
        "evidence_coverage": coverage,
        "subject_coverage": subject_coverage,
        "evidence_statements": evidence_statements,
        "facts": facts,
        "analysis_patterns": patterns,
        "analysis_contradictions": contradictions,
        "analysis_gaps": gaps,
        "analyzed_source_ids": list(
            dict.fromkeys(
                [*(state.get("analyzed_source_ids") or []), *(url_key(s.url) for s in sources)]
            )
        ),
        "analysis_batches": (state.get("analysis_batches") or 0) + 1,
        "messages": [AIMessage(content=summary)],
    }
