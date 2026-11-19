"""Analyst: clean, deduplicate, cross-validate, and structure the collected sources."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.research_team.contracts import (
    AspectCoverage,
    EvidenceCoverage,
    EvidenceStatement,
    ResearchSubject,
    ResearchTask,
    SubjectEvidenceCoverage,
)
from agents.research_team.prompts import load_prompt, render_prompt
from agents.research_team.sources import dedupe_sources, join_gaps, numbered_sources
from agents.research_team.state import (
    ResearchState,
)
from agents.research_team.structured import structured_output
from core import resolve_model, settings
from core.metrics import record_degradation

logger = logging.getLogger(__name__)

ANALYST_PROMPT = load_prompt("research_team.analyst.instructions")
_ANALYST_CONTEXT_PROMPT = load_prompt("research_team.analyst.context")

MAX_SOURCES_FOR_ANALYSIS = 12

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


def _normalize_coverage(
    coverage: list[EvidenceCoverage],
    tasks: list[ResearchTask],
    source_count: int,
) -> list[EvidenceCoverage]:
    by_task: dict[str, EvidenceCoverage] = {}
    for item in coverage:
        if item.task_id not in {task.id for task in tasks}:
            logger.warning("Analyst returned coverage for unknown task %s", item.task_id)
            record_degradation("coverage_unknown_task")
            continue
        if item.task_id in by_task:
            logger.warning("Analyst returned duplicate coverage for task %s", item.task_id)
            record_degradation("coverage_duplicate_task")
            continue
        valid_indices = [index for index in item.source_indices if 1 <= index <= source_count]
        if len(valid_indices) != len(item.source_indices):
            logger.warning("Analyst returned invalid source indices for task %s", item.task_id)
            record_degradation("coverage_invalid_source_index")
        strength = item.evidence_strength
        if not valid_indices and strength != "insufficient":
            logger.warning(
                "Coverage task %s has no sources but strength %s", item.task_id, strength
            )
            record_degradation("coverage_strength_without_sources")
            strength = "insufficient"
        elif item.status == "missing" and strength != "insufficient":
            logger.warning("Coverage task %s is missing but strength is %s", item.task_id, strength)
            record_degradation("coverage_missing_strength_mismatch")
            strength = "insufficient"
        elif item.status == "conflicting" and strength == "strong":
            logger.warning("Coverage task %s is conflicting but strength is strong", item.task_id)
            record_degradation("coverage_conflicting_strength_mismatch")
            strength = "moderate"
        by_task[item.task_id] = item.model_copy(
            update={"source_indices": valid_indices, "evidence_strength": strength}
        )

    normalized: list[EvidenceCoverage] = []
    for task in tasks:
        item = by_task.get(task.id)
        if item is None:
            logger.warning("Analyst omitted coverage for task %s", task.id)
            record_degradation("coverage_incomplete")
            normalized.append(
                EvidenceCoverage(
                    task_id=task.id,
                    question=task.question,
                    source_indices=[],
                    status="missing",
                    evidence_strength="insufficient",
                    gaps=["未生成该子问题的覆盖判断"],
                    reason="Analyst 未返回该子问题的覆盖记录",
                )
            )
            continue
        if item.question != task.question:
            item = item.model_copy(update={"question": task.question})
        normalized.append(item)
    return normalized


_STRENGTH_ORDER = {"insufficient": 0, "weak": 1, "moderate": 2, "strong": 3}


def _normalize_subject_coverage(
    coverage: list[SubjectEvidenceCoverage],
    subjects: list[ResearchSubject],
    source_count: int,
) -> list[SubjectEvidenceCoverage]:
    known = {subject.id: subject for subject in subjects}
    by_subject: dict[str, SubjectEvidenceCoverage] = {}
    for item in coverage:
        subject = known.get(item.subject_id)
        if subject is None:
            logger.warning("Analyst returned coverage for unknown subject %s", item.subject_id)
            record_degradation("subject_coverage_unknown_subject")
            continue
        if item.subject_id in by_subject:
            logger.warning("Analyst returned duplicate coverage for subject %s", item.subject_id)
            record_degradation("subject_coverage_duplicate_subject")
            continue

        aspects: list[AspectCoverage] = []
        seen_aspects: set[str] = set()
        for aspect in item.aspects:
            name = aspect.name.strip()
            if not name or name in seen_aspects:
                if not name:
                    record_degradation("subject_coverage_empty_aspect")
                else:
                    record_degradation("subject_coverage_duplicate_aspect")
                continue
            seen_aspects.add(name)
            valid_indices = sorted(
                {index for index in aspect.source_indices if 1 <= index <= source_count}
            )
            if len(valid_indices) != len(set(aspect.source_indices)):
                record_degradation("subject_coverage_invalid_source_index")
            strength = aspect.evidence_strength
            if not valid_indices and strength != "insufficient":
                strength = "insufficient"
                record_degradation("subject_coverage_strength_without_sources")
            if aspect.status == "missing" and strength != "insufficient":
                strength = "insufficient"
                record_degradation("subject_coverage_missing_strength_mismatch")
            if aspect.status == "conflicting" and strength == "strong":
                strength = "moderate"
                record_degradation("subject_coverage_conflicting_strength_mismatch")
            aspects.append(
                aspect.model_copy(
                    update={
                        "name": name,
                        "source_indices": valid_indices,
                        "evidence_strength": strength,
                    }
                )
            )

        subject_indices = sorted(
            {index for index in item.source_indices if 1 <= index <= source_count}
            | {index for aspect in aspects for index in aspect.source_indices}
        )
        if len(subject_indices) != len(
            {index for index in item.source_indices if 1 <= index <= source_count}
        ):
            record_degradation("subject_coverage_aspect_source_merge")

        statuses = [aspect.status for aspect in aspects]
        if not statuses or all(status == "missing" for status in statuses):
            status = "missing"
        elif "conflicting" in statuses:
            status = "conflicting"
        elif any(status in {"missing", "partial"} for status in statuses):
            status = "partial"
        else:
            status = "sufficient"

        if status == "missing" or not aspects:
            strength = "insufficient"
        else:
            strength = min(
                (aspect.evidence_strength for aspect in aspects),
                key=lambda value: _STRENGTH_ORDER[value],
            )
            if status == "conflicting" and strength == "strong":
                strength = "moderate"

        gaps = list(dict.fromkeys(item.gaps + [gap for aspect in aspects for gap in aspect.gaps]))
        by_subject[item.subject_id] = item.model_copy(
            update={
                "subject": subject.name,
                "aspects": aspects,
                "source_indices": subject_indices,
                "status": status,
                "evidence_strength": strength,
                "gaps": gaps,
            }
        )

    normalized: list[SubjectEvidenceCoverage] = []
    for subject in subjects:
        item = by_subject.get(subject.id)
        if item is None:
            logger.warning("Analyst omitted coverage for subject %s", subject.id)
            record_degradation("subject_coverage_incomplete")
            normalized.append(
                SubjectEvidenceCoverage(
                    subject_id=subject.id,
                    subject=subject.name,
                    aspects=[],
                    status="missing",
                    evidence_strength="insufficient",
                    source_indices=[],
                    gaps=["未生成该对象的覆盖判断"],
                    reason="Analyst 未返回该对象的覆盖记录",
                )
            )
        else:
            normalized.append(item)
    return normalized


def _normalize_evidence_statements(
    statements: list[EvidenceStatement],
    source_count: int,
) -> list[EvidenceStatement]:
    normalized: list[EvidenceStatement] = []
    for item in statements:
        statement = item.statement.strip()
        if not statement:
            record_degradation("evidence_statement_empty")
            continue
        source_indices = sorted(
            {index for index in item.source_indices if 1 <= index <= source_count}
        )
        if not source_indices:
            record_degradation("evidence_statement_without_sources")
            continue
        if len(source_indices) != len(set(item.source_indices)):
            record_degradation("evidence_statement_invalid_source_index")

        periods = list(dict.fromkeys(period.strip() for period in item.periods if period.strip()))
        if item.period and item.period.strip() not in periods:
            periods.insert(0, item.period.strip())
        kind = item.kind
        trend_origin = item.trend_origin
        if kind == "trend":
            if item.value_type in {"planned", "forecast"}:
                kind = "forecast"
                trend_origin = "unknown"
                record_degradation("evidence_statement_forecast_not_trend")
                normalized.append(
                    item.model_copy(
                        update={
                            "statement": statement,
                            "kind": kind,
                            "period": periods[0] if periods else item.period,
                            "periods": periods,
                            "metric": item.metric.strip() if item.metric else item.metric,
                            "unit": item.unit.strip() if item.unit else item.unit,
                            "source_indices": source_indices,
                            "trend_origin": trend_origin,
                        }
                    )
                )
                continue
            explicit = trend_origin == "explicit_source_statement"
            comparable = (
                len(periods) >= 3
                and item.value_type in {"actual", "reported"}
                and bool(item.metric and item.metric.strip())
                and bool(item.unit and item.unit.strip())
            )
            if not explicit and not comparable:
                kind = "change" if len(periods) >= 2 else "point"
                trend_origin = "unknown"
                record_degradation("evidence_statement_trend_downgraded")
        elif kind == "change" and len(periods) < 2:
            kind = "point"
            record_degradation("evidence_statement_change_downgraded")
        elif kind == "forecast" and item.value_type not in {"planned", "forecast"}:
            kind = "change" if len(periods) >= 2 else "point"
            record_degradation("evidence_statement_forecast_downgraded")

        normalized.append(
            item.model_copy(
                update={
                    "statement": statement,
                    "kind": kind,
                    "period": periods[0] if periods else item.period,
                    "periods": periods,
                    "metric": item.metric.strip() if item.metric else item.metric,
                    "unit": item.unit.strip() if item.unit else item.unit,
                    "source_indices": source_indices,
                    "trend_origin": trend_origin,
                }
            )
        )
    return normalized


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
    sources = dedupe_sources(state.get("sources"))[:MAX_SOURCES_FOR_ANALYSIS]
    if not sources:
        return {"messages": [AIMessage(content="来源为空，无法进入分析阶段。")]}

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
        sources=numbered_sources(sources),
    )
    bundle: AnalysisBundle = await structured.ainvoke(messages)
    if len([item for item in bundle.facts if item.strip()]) < 2:
        # Schema already forbids an empty list; this catches a list of blank strings.
        logger.warning("Analyst returned a thin fact base, retrying once")
        record_degradation("analysis_incomplete")
        bundle = await structured.ainvoke([*messages, _THIN_ANALYSIS_RETRY])
    coverage = _normalize_coverage(bundle.coverage, tasks, len(sources))
    subject_coverage = _normalize_subject_coverage(bundle.subject_coverage, subjects, len(sources))
    evidence_statements = _normalize_evidence_statements(bundle.evidence_statements, len(sources))
    bundle = bundle.model_copy(
        update={
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
        "messages": [AIMessage(content=summary)],
    }
