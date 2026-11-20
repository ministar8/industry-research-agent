"""Unit tests for the research graph's pure decision logic."""

import asyncio
import json
import re
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from openai import LengthFinishReasonError

from agents.research_team.analyst import (
    AnalysisBundle,
    analyst,
)
from agents.research_team.contracts import (
    AUTHORITY_UNKNOWN,
    AspectCoverage,
    EvidenceCoverage,
    EvidenceStatement,
    ResearchSubject,
    ResearchTask,
    Source,
    SubjectEvidenceCoverage,
)
from agents.research_team.normalize import (
    merge_lines,
    merge_statements,
    merge_subject_coverage,
    normalize_coverage,
    normalize_evidence_statements,
    normalize_subject_coverage,
)
from agents.research_team.prompts import (
    PROMPTS_DIR,
    read_prompt,
    render_prompt,
)
from agents.research_team.researcher import (
    MAX_CANDIDATES,
    MAX_CONTENT_CHARS,
    SourceJudgements,
    _candidate,
    _collect_batches,
    _deep_read,
    _empty_run_reason,
    _filter_candidates,
    _judge,
    _plan_queries,
    _researcher_system,
    _task_brief,
    researcher,
)
from agents.research_team.search import SearchRejected, SearchResponseError, fetch, search
from agents.research_team.sources import (
    EXCERPT_BUDGET_CHARS,
    EXCERPT_CHARS,
    batch_window,
    clamp_anchors,
    clean_fetched_markdown,
    clean_url,
    confidence_from_authority,
    dedupe_sources,
    excerpt,
    excerpt_limit_for,
    numbered_sources,
    remap_anchors,
    url_key,
)
from agents.research_team.structured import inline_schema, structured_output
from agents.research_team.supervisor import (
    _TURN_RESET,
    MAX_ROUNDS,
    RoutingDecision,
    _constraints_for,
    _enforce,
    _recorded_target,
    _turn_state,
    finalize,
    route,
    supervisor,
)
from agents.research_team.writer import (
    ReportBundle,
    TableBlock,
    TableRequiredReport,
    _is_complete,
    _render_subject_coverage,
    render_report,
    report_schema,
    requires_table,
    writer,
)
from core.metrics import begin_turn, current, end_turn


def make_source(url: str, confidence: float = 0.8) -> Source:
    return Source(url=url, title=url, snippet="snippet", confidence=confidence)


def test_dedupe_sources_keeps_first_occurrence():
    sources = [
        make_source("https://a.example/1"),
        make_source("https://a.example/2"),
        make_source("https://a.example/1/"),
    ]
    unique = dedupe_sources(sources)
    assert [s.url for s in unique] == ["https://a.example/1", "https://a.example/2"]


def test_dedupe_sources_handles_none():
    assert dedupe_sources(None) == []


def test_dedupe_upgrades_first_copy_with_later_deep_read():
    shallow = Source(url="https://x.example/a", title="X", summary="短摘要", content="")
    deep = Source(
        url="https://x.example/a", title="X", summary="短摘要", content="Fetch 抓到的正文"
    )
    kept = dedupe_sources([shallow, deep])

    assert len(kept) == 1
    assert kept[0].content == "Fetch 抓到的正文"
    # The state-held originals must not be mutated by a read-path dedupe.
    assert shallow.content == ""


def test_normalize_coverage_matches_tasks_and_drops_invalid_source_indices():
    tasks = [
        ResearchTask(id="T1", question="问题一"),
        ResearchTask(id="T2", question="问题二"),
    ]
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="错误问题",
            source_indices=[1, 3],
            status="sufficient",
            evidence_strength="strong",
            gaps=[],
            reason="已覆盖",
        ),
        EvidenceCoverage(
            task_id="UNKNOWN",
            question="额外问题",
            source_indices=[1],
            status="sufficient",
            evidence_strength="strong",
            gaps=[],
            reason="不应保留",
        ),
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        normalized = normalize_coverage(coverage, tasks, source_count=2)

    assert [item.task_id for item in normalized] == ["T1", "T2"]
    assert normalized[0].question == "问题一"
    assert normalized[0].source_indices == [1]
    assert normalized[1].status == "missing"
    assert degraded.call_count == 3


def test_normalize_coverage_downgrades_impossible_evidence_strength():
    task = ResearchTask(id="T1", question="问题一")
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="问题一",
            source_indices=[],
            status="missing",
            evidence_strength="strong",
            gaps=["没有来源"],
            reason="错误的强度",
        )
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        normalized = normalize_coverage(coverage, [task], source_count=1)

    assert normalized[0].evidence_strength == "insufficient"
    assert degraded.call_count == 1


def test_normalize_subject_coverage_aggregates_aspects_and_fills_missing_subjects():
    subjects = [
        ResearchSubject(id="S1", name="宁德时代", aliases=["CATL"]),
        ResearchSubject(id="S2", name="比亚迪", aliases=["BYD"]),
    ]
    coverage = [
        SubjectEvidenceCoverage(
            subject_id="S1",
            subject="CATL",
            aspects=[
                AspectCoverage(
                    name="工厂",
                    status="sufficient",
                    evidence_strength="strong",
                    source_indices=[1],
                    gaps=[],
                    reason="有直接来源",
                ),
                AspectCoverage(
                    name="产能",
                    status="missing",
                    evidence_strength="insufficient",
                    source_indices=[],
                    gaps=["缺少产能"],
                    reason="没有直接数据",
                ),
            ],
            status="sufficient",
            evidence_strength="strong",
            source_indices=[1, 3],
            gaps=[],
            reason="错误的总体判断",
        )
    ]

    normalized = normalize_subject_coverage(coverage, subjects, source_count=2)

    assert [item.subject_id for item in normalized] == ["S1", "S2"]
    assert normalized[0].subject == "宁德时代"
    assert normalized[0].status == "partial"
    assert normalized[0].evidence_strength == "insufficient"
    assert normalized[0].source_indices == [1]
    assert normalized[1].status == "missing"
    assert normalized[1].gaps == ["未生成该对象的覆盖判断"]


def test_normalize_evidence_statements_downgrades_unsupported_trends():
    statements = [
        EvidenceStatement(
            statement="产能连续增长 [1]",
            kind="trend",
            periods=["2024", "2025"],
            value_type="actual",
            metric="产能",
            unit="GWh",
            source_indices=[1, 3],
        ),
        EvidenceStatement(
            statement="计划产能将持续提升 [1]",
            kind="trend",
            periods=["2025", "2026", "2027"],
            value_type="planned",
            metric="产能",
            unit="GWh",
            source_indices=[1],
        ),
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        normalized = normalize_evidence_statements(statements, source_count=1)

    assert [item.kind for item in normalized] == ["change", "forecast"]
    assert normalized[0].source_indices == [1]
    assert degraded.call_count == 3


def test_normalize_evidence_statements_allows_three_comparable_actual_periods():
    statement = EvidenceStatement(
        statement="产能呈上升趋势 [1][2]",
        kind="trend",
        periods=["2022", "2023", "2024"],
        value_type="actual",
        metric="产能",
        unit="GWh",
        source_indices=[1, 2],
    )

    normalized = normalize_evidence_statements([statement], source_count=2)

    assert normalized[0].kind == "trend"
    assert normalized[0].trend_origin == "unknown"


def test_normalize_evidence_statements_downgrades_missing_comparability_fields():
    statement = EvidenceStatement(
        statement="规模呈增长趋势 [1]",
        kind="trend",
        periods=["2022", "2023", "2024"],
        value_type="actual",
        source_indices=[1],
    )

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        normalized = normalize_evidence_statements([statement], source_count=1)

    assert normalized[0].kind == "change"
    assert degraded.call_count == 1


def test_clamp_anchors_drops_unknown_indices():
    text = "成立 [1]，存疑 [3]。"
    assert clamp_anchors(text, 2) == "成立 [1]，存疑 。"


def test_route_returns_recorded_action():
    assert route({"next_action": "researcher", "rounds": 1}) == "researcher"


def test_route_dispatches_without_rechecking_the_round_cap():
    # Supervisor applies MAX_ROUNDS when it records next_action; route deciding again
    # only creates a second place for the two to disagree.
    assert route({"next_action": "researcher", "rounds": MAX_ROUNDS + 1}) == "researcher"


def test_recorded_action_is_capped_by_the_round_limit():
    # The cap lives in one place: supervisor records finalize, so route can dispatch
    # state["next_action"] without re-deriving the policy.
    state = {
        "messages": [HumanMessage(content="调研")],
        "sources": [make_source("https://a.example/1")],
        "analysis": "已有分析",
    }

    assert _recorded_target(MAX_ROUNDS, decision("researcher"), state) == "researcher"
    assert _recorded_target(MAX_ROUNDS + 1, decision("researcher"), state) == "finalize"


def test_route_defaults_to_finalize_without_action():
    assert route({}) == "finalize"


def decision(target: str) -> RoutingDecision:
    return RoutingDecision(
        intent="测试", is_research=True, sub_questions=[], constraints=[], next=target, reason="r"
    )


def test_analyst_requires_sources():
    state = {"messages": [HumanMessage(content="调研")], "sources": []}
    assert _enforce(decision("analyst"), state) == "researcher"


@pytest.mark.parametrize("target", ["researcher", "analyst", "writer"])
def test_non_research_decision_always_finalizes(target: str):
    non_research = RoutingDecision(
        intent="无法识别调研对象",
        is_research=False,
        sub_questions=[],
        constraints=[],
        next=target,  # type: ignore[arg-type]
        reason="没有外部调研需求",
    )
    state = {
        "messages": [HumanMessage(content="谢谢")],
        "sources": [make_source("https://a.example/1")],
        "analysis": "已有分析",
    }

    assert _enforce(non_research, state) == "finalize"


def test_writer_requires_analysis():
    state = {
        "messages": [HumanMessage(content="调研")],
        "sources": [make_source("https://a.example/1")],
    }
    assert _enforce(decision("writer"), state) == "analyst"


def test_writer_allowed_with_analysis():
    state = {
        "messages": [HumanMessage(content="调研")],
        "sources": [make_source("https://a.example/1")],
        "analysis": "## 事实基线",
    }
    assert _enforce(decision("writer"), state) == "writer"


def test_researcher_blocked_when_sources_await_analysis():
    state = {
        "messages": [HumanMessage(content="调研")],
        "sources": [make_source("https://a.example/1")],
    }
    assert _enforce(decision("researcher"), state) == "analyst"


def test_researcher_allowed_without_sources():
    state = {"messages": [HumanMessage(content="调研")], "sources": []}
    assert _enforce(decision("researcher"), state) == "researcher"


def test_researcher_allowed_after_analysis():
    state = {
        "messages": [HumanMessage(content="调研")],
        "sources": [make_source("https://a.example/1")],
        "analysis": "## 事实基线",
    }
    assert _enforce(decision("researcher"), state) == "researcher"


@pytest.mark.asyncio
async def test_finalize_appends_source_appendix():
    state = {
        "messages": [HumanMessage(content="调研")],
        "report": "# 报告\n\n结论成立 [1]。",
        "sources": [make_source("https://a.example/1", 0.9)],
    }
    result = await finalize(state)
    content = result["messages"][0].content
    assert content.startswith("# 报告")
    assert "### 参考来源" in content
    assert "https://a.example/1" in content
    assert "0.90" in content


@pytest.mark.asyncio
async def test_finalize_gives_hint_when_no_research_started():
    result = await finalize(
        {"messages": [HumanMessage(content="你好")], "sources": [], "turn_attempted": False}
    )
    content = result["messages"][0].content
    assert "没能从这段输入里识别出可调研的对象" in content
    assert "未能采集到可用信息" not in content


@pytest.mark.asyncio
async def test_finalize_hint_persists_on_later_idle_turns():
    # rounds is thread-scoped and keeps growing; the hint must not decay into a failure.
    result = await finalize(
        {
            "messages": [HumanMessage(content="谢谢")],
            "sources": [],
            "rounds": 6,
            "turn_attempted": False,
        }
    )
    assert "没能从这段输入里识别出可调研的对象" in result["messages"][0].content


@pytest.mark.asyncio
async def test_finalize_reports_failure_after_a_failed_run():
    result = await finalize(
        {
            "messages": [HumanMessage(content="调研")],
            "sources": [],
            "rounds": 4,
            "turn_attempted": True,
        }
    )
    assert "未能采集到可用信息" in result["messages"][0].content


@pytest.mark.asyncio
async def test_finalize_distinguishes_collected_sources_from_empty_run():
    result = await finalize(
        {
            "messages": [HumanMessage(content="调研")],
            "sources": [make_source("https://a.example/1"), make_source("https://a.example/2")],
            "rounds": 6,
            "turn_attempted": True,
        }
    )
    content = result["messages"][0].content
    assert "已采集 2 条来源" in content
    assert "未能完成分析" in content
    assert "未能采集到可用信息" not in content


@pytest.mark.asyncio
async def test_finalize_prefers_staged_analysis_over_hint():
    result = await finalize(
        {
            "messages": [HumanMessage(content="调研")],
            "sources": [make_source("https://a.example/1")],
            "analysis": "## 事实基线\n- 已确证 [1]",
            "rounds": 9,
        }
    )
    content = result["messages"][0].content
    assert "调研分析（未生成成稿报告）" in content
    assert "https://a.example/1" in content


@pytest.mark.asyncio
async def test_supervisor_resets_turn_flag_on_a_new_human_message():
    with patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", True):
        update = await supervisor(
            {
                "messages": [HumanMessage(content="你好")],
                "rounds": 3,
                "turn_attempted": True,
            },
            None,  # type: ignore[arg-type]
        )
    assert update["turn_attempted"] is False
    assert update["next_action"] == "finalize"


@pytest.mark.asyncio
async def test_supervisor_keeps_turn_flag_during_a_run():
    with patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", True):
        update = await supervisor(
            {
                "messages": [HumanMessage(content="调研"), AIMessage(content="检索完成")],
                "rounds": 2,
                "turn_attempted": True,
            },
            None,  # type: ignore[arg-type]
        )
    assert update["turn_attempted"] is True


@pytest.mark.asyncio
async def test_supervisor_clears_abort_state_on_a_new_turn():
    with patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", True):
        update = await supervisor(
            {
                "messages": [HumanMessage(content="换个问题")],
                "abort_reason": "search_rejected",
                "rounds": 2,
                "turn_attempted": True,
            },
            None,  # type: ignore[arg-type]
        )

    assert update["abort_reason"] == ""


@pytest.mark.asyncio
async def test_supervisor_clears_staged_output_when_new_turn_finishes():
    with patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", True):
        update = await supervisor(
            {
                "messages": [HumanMessage(content="谢谢")],
                "report": "上一轮报告",
                "analysis": "上一轮分析",
                "rounds": 1,
                "turn_attempted": True,
            },
            None,  # type: ignore[arg-type]
        )

    assert update["next_action"] == "finalize"
    assert update["report"] == ""
    assert update["analysis"] == ""


@pytest.mark.asyncio
async def test_supervisor_does_not_reuse_tasks_on_a_new_turn():
    decision = RoutingDecision(
        intent="调研欧洲储能市场",
        is_research=True,
        sub_questions=[],
        constraints=[],
        next="researcher",
        reason="需要外部信息",
    )
    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor(
            {
                "messages": [HumanMessage(content="换个问题")],
                "tasks": [ResearchTask(id="T1", question="旧问题")],
                "rounds": 1,
                "turn_attempted": True,
            },
            None,  # type: ignore[arg-type]
        )

    assert [task.question for task in update["tasks"]] == ["调研欧洲储能市场"]


@pytest.mark.asyncio
async def test_supervisor_resets_research_state_for_a_new_turn_id():
    decision = RoutingDecision(
        intent="调研工业机器人",
        is_research=True,
        sub_questions=["工业机器人市场规模"],
        constraints=[],
        next="researcher",
        reason="需要新一轮检索",
    )
    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor(
            {
                "messages": [HumanMessage(content="调研工业机器人")],
                "active_turn_id": "turn-1",
                "rounds": 5,
                "tasks": [ResearchTask(id="T1", question="储能旧问题")],
                "sources": [make_source("https://old.example/1")],
                "analysis": "储能旧分析",
                "evidence_coverage": [
                    EvidenceCoverage(
                        task_id="T1",
                        question="储能旧问题",
                        source_indices=[1],
                        status="sufficient",
                        evidence_strength="strong",
                        gaps=[],
                        reason="旧证据",
                    )
                ],
                "report": "储能旧报告",
                "evidence_statements": [
                    EvidenceStatement(
                        statement="旧产能数据 [1]",
                        kind="point",
                        period="2024",
                        source_indices=[1],
                    )
                ],
                "turn_attempted": True,
            },
            {"configurable": {"turn_id": "turn-2"}},  # type: ignore[arg-type]
        )

    assert update["active_turn_id"] == "turn-2"
    assert update["rounds"] == 1
    assert update["sources"] == []
    assert update["analysis"] == ""
    assert update["evidence_coverage"] == []
    assert update["evidence_statements"] == []
    assert update["report"] == ""
    assert [task.question for task in update["tasks"]] == ["工业机器人市场规模"]


class _StubDecider:
    def __init__(self, decision: RoutingDecision) -> None:
        self._decision = decision

    async def ainvoke(self, messages: list) -> dict:
        # Mirrors `include_raw=True`: the real runnable returns a dict, not the model.
        return {"raw": None, "parsed": self._decision, "parsing_error": None}


class _StubModel:
    def __init__(self, decision: RoutingDecision) -> None:
        self._decision = decision

    def with_structured_output(
        self, schema: type, method: str | None = None, include_raw: bool = False, **kwargs
    ) -> _StubDecider:
        return _StubDecider(self._decision)


@pytest.mark.asyncio
async def test_supervisor_parses_noisy_input_into_tasks_and_constraints():
    decision = RoutingDecision(
        intent="评估国内储能电芯竞争格局",
        is_research=True,
        sub_questions=["2025 年国内储能电芯厂商出货量排名", "2025 年储能电芯均价走势"],
        constraints=["时间范围：2025 年", "地区：中国", "用表格呈现", "不含上游锂矿"],
        next="researcher",
        reason="需要外部数据",
    )
    noisy = (
        "你好，我最近在做投资，特别关注新能源赛道，之前看过一些报告但都不太系统。"
        "麻烦你帮我看看储能电芯的竞争格局，时间只要 2025 年，地区限中国。"
        "另外我比较喜欢表格，别把上游锂矿扯进来。谢谢！"
    )

    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor({"messages": [HumanMessage(content=noisy)], "rounds": 0}, None)

    assert update["next_action"] == "researcher"
    assert update["intent"] == decision.intent
    assert [task.question for task in update["tasks"]] == decision.sub_questions
    assert update["constraints"] == decision.constraints


def test_task_brief_truncates_the_raw_excerpt():
    brief = _task_brief(
        {
            "messages": [HumanMessage(content="背景铺垫" * 300)],
            "intent": "储能电芯竞争格局",
            "tasks": [],
            "constraints": ["用表格呈现"],
        }
    )
    assert "必须遵守的范围与交付约束" in brief
    assert "用表格呈现" in brief
    assert "仅供参考" in brief
    assert "…" in brief
    assert len(brief) < 1200


class _CapturingDecider:
    def __init__(self, result: object, captured: list[str]) -> None:
        self._result = result
        self._captured = captured

    async def ainvoke(self, messages: list) -> dict:
        self._captured.append("\n".join(str(message.content) for message in messages))
        return {"raw": None, "parsed": self._result, "parsing_error": None}


class _CapturingModel:
    def __init__(self, result: object, captured: list[str]) -> None:
        self._result = result
        self._captured = captured

    def with_structured_output(
        self, schema: type, method: str | None = None, include_raw: bool = False, **kwargs
    ) -> _CapturingDecider:
        return _CapturingDecider(self._result, self._captured)


@pytest.mark.asyncio
async def test_analyst_prompt_carries_constraints():
    captured: list[str] = []
    bundle = AnalysisBundle(
        facts=["事实甲 [1]", "事实乙 [1]"],
        patterns=["模式 [1]"],
        contradictions=[],
        gaps=["缺口 [1]"],
        coverage=[
            EvidenceCoverage(
                task_id="T1",
                question="2025 年储能电芯出货量排名",
                source_indices=[1],
                status="partial",
                evidence_strength="moderate",
                gaps=["缺少完整排名"],
                reason="已有来源但覆盖不完整",
            )
        ],
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "储能电芯竞争格局",
        "constraints": ["时间范围：2025 年", "不含上游锂矿"],
        "tasks": [ResearchTask(id="T1", question="2025 年储能电芯出货量排名")],
        "sources": [make_source("https://a.example/1")],
    }

    with (
        patch("agents.research_team.analyst.settings.USE_FAKE_MODEL", False),
        patch(
            "agents.research_team.analyst.resolve_model",
            return_value=_CapturingModel(bundle, captured),
        ),
    ):
        await analyst(state, None)  # type: ignore[arg-type]

    prompt = captured[0]
    assert "时间范围：2025 年" in prompt
    assert "不含上游锂矿" in prompt
    assert "[T1] 2025 年储能电芯出货量排名" in prompt
    assert "coverage 必须逐一对应" in prompt


@pytest.mark.asyncio
async def test_analyst_prompt_carries_subjects_and_returns_missing_subject_coverage():
    captured: list[str] = []
    bundle = AnalysisBundle(
        facts=["事实甲 [1]", "事实乙 [1]"],
        patterns=["模式 [1]"],
        contradictions=[],
        gaps=["比亚迪产能缺口"],
        coverage=[
            EvidenceCoverage(
                task_id="T1",
                question="双方欧洲产能",
                source_indices=[1],
                status="partial",
                evidence_strength="moderate",
                gaps=["比亚迪产能缺失"],
                reason="单方来源",
            )
        ],
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "宁德时代与比亚迪欧洲产能",
        "tasks": [ResearchTask(id="T1", question="双方欧洲产能")],
        "subjects": [
            ResearchSubject(id="S1", name="宁德时代", aliases=["CATL"]),
            ResearchSubject(id="S2", name="比亚迪", aliases=["BYD"]),
        ],
        "sources": [make_source("https://a.example/1")],
    }

    with (
        patch("agents.research_team.analyst.settings.USE_FAKE_MODEL", False),
        patch(
            "agents.research_team.analyst.resolve_model",
            return_value=_CapturingModel(bundle, captured),
        ),
    ):
        result = await analyst(state, None)  # type: ignore[arg-type]

    assert "[S1] 宁德时代" in captured[0]
    assert "[S2] 比亚迪" in captured[0]
    assert [item.subject_id for item in result["subject_coverage"]] == ["S1", "S2"]
    assert all(item.status == "missing" for item in result["subject_coverage"])


@pytest.mark.asyncio
async def test_judge_prompt_carries_task_anchor():
    captured: list[str] = []
    state = {
        "messages": [HumanMessage(content="调研")],
        "tasks": [ResearchTask(id="T1", question="2025 年储能电芯出货量排名")],
        "constraints": ["地区：中国"],
    }
    candidates = [
        {
            "url": "https://a.example/1",
            "title": "标题",
            "summary": "摘要",
            "site_name": "示例站",
            "published_at": "2026-01-01",
        }
    ]

    with patch(
        "agents.research_team.researcher.resolve_model",
        return_value=_CapturingModel(SourceJudgements(sources=[]), captured),
    ):
        result = await _judge(candidates, state, None)  # type: ignore[arg-type]

    prompt = captured[0]
    assert "2025 年储能电芯出货量排名" in prompt
    assert "地区：中国" in prompt
    assert result.sources == []


@pytest.mark.asyncio
async def test_supervisor_clears_stale_report_on_reanalysis():
    decision = RoutingDecision(
        intent="重新分析储能电芯格局",
        is_research=True,
        sub_questions=[],
        constraints=[],
        next="analyst",
        reason="来源已足够，需要重算分析",
    )
    state = {
        "messages": [HumanMessage(content="重新分析一下")],
        "active_turn_id": "turn-1",
        "sources": [make_source("https://a.example/1")],
        "analysis": "旧分析",
        "report": "# 旧报告",
        "evidence_coverage": [
            EvidenceCoverage(
                task_id="T1",
                question="旧问题",
                source_indices=[1],
                status="sufficient",
                evidence_strength="strong",
                gaps=[],
                reason="旧证据",
            )
        ],
        "evidence_statements": [
            EvidenceStatement(
                statement="旧产能数据 [1]",
                kind="point",
                period="2024",
                source_indices=[1],
            )
        ],
    }

    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor(
            state,
            {"configurable": {"turn_id": "turn-2", "context_mode": "reuse"}},
        )  # type: ignore[arg-type]

    assert update["next_action"] == "analyst"
    assert update["report"] == ""
    assert [source.url for source in update["sources"]] == ["https://a.example/1"]
    assert update["evidence_coverage"] == []
    assert update["evidence_statements"] == []


@pytest.mark.asyncio
async def test_supervisor_merges_constraints_within_a_turn():
    decision = RoutingDecision(
        intent="继续调研",
        is_research=True,
        sub_questions=[],
        constraints=["地区：中国"],
        next="analyst",
        reason="来源已足够",
    )
    state = {
        "messages": [HumanMessage(content="调研"), AIMessage(content="检索完成")],
        "constraints": ["时间范围：2025 年", "用表格呈现"],
        "sources": [make_source("https://a.example/1")],
        "rounds": 2,
    }
    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor(state, None)  # type: ignore[arg-type]

    assert update["constraints"] == ["时间范围：2025 年", "用表格呈现", "地区：中国"]


@pytest.mark.asyncio
async def test_supervisor_resets_constraints_on_a_new_turn():
    decision = RoutingDecision(
        intent="新话题",
        is_research=True,
        sub_questions=["欧洲产能"],
        constraints=["地区：欧洲"],
        next="researcher",
        reason="需要外部数据",
    )
    state = {
        "messages": [HumanMessage(content="换个话题")],
        "constraints": ["时间范围：2025 年"],
        "rounds": 0,
    }
    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor(state, None)  # type: ignore[arg-type]

    assert update["constraints"] == ["地区：欧洲"]


@pytest.mark.asyncio
async def test_supervisor_breaks_pre_analysis_research_loop():
    decision = RoutingDecision(
        intent="调研",
        is_research=True,
        sub_questions=["问题"],
        constraints=[],
        next="researcher",
        reason="r",
    )
    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        first = await supervisor(
            {"messages": [HumanMessage(content="调研")], "rounds": 0},
            None,  # type: ignore[arg-type]
        )
        assert first["next_action"] == "researcher"
        token = begin_turn("loop-test")
        try:
            second = await supervisor(
                {
                    "messages": [HumanMessage(content="调研"), AIMessage(content="检索完成")],
                    "sources": [make_source("https://a.example/1")],
                    "rounds": 1,
                    "intent": "调研",
                },
                None,  # type: ignore[arg-type]
            )
        finally:
            metrics = current()
            await end_turn(token)
        assert second["next_action"] == "analyst"
        assert metrics is not None
        assert "researcher_forced_analyst" in metrics.degradations


def test_routing_decision_schema_requires_lists():
    schema = RoutingDecision.model_json_schema()
    assert "sub_questions" in schema["required"]
    assert "constraints" in schema["required"]


def test_routing_decision_accepts_empty_lists():
    decision = RoutingDecision.model_validate(
        {
            "intent": "你好",
            "is_research": False,
            "sub_questions": [],
            "constraints": [],
            "next": "finalize",
            "reason": "闲聊",
        }
    )
    assert decision.sub_questions == []
    assert decision.constraints == []


def test_routing_decision_tolerates_missing_lists():
    decision = RoutingDecision.model_validate(
        {"intent": "测试", "is_research": True, "next": "researcher", "reason": "r"}
    )
    assert decision.sub_questions == []
    assert decision.constraints == []


@pytest.mark.asyncio
async def test_supervisor_falls_back_to_intent_task():
    decision = RoutingDecision(
        intent="调研储能电芯格局",
        is_research=True,
        sub_questions=[],
        constraints=[],
        next="researcher",
        reason="r",
    )
    token = begin_turn("fallback-test")
    try:
        with (
            patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
            patch(
                "agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)
            ),
        ):
            update = await supervisor(
                {"messages": [HumanMessage(content="调研")], "rounds": 0},
                None,  # type: ignore[arg-type]
            )
    finally:
        metrics = current()
        await end_turn(token)
    assert [task.question for task in update["tasks"]] == ["调研储能电芯格局"]
    assert metrics is not None
    assert "sub_questions_missing" in metrics.degradations


@pytest.mark.asyncio
async def test_supervisor_keeps_existing_tasks_without_sub_questions():
    decision = RoutingDecision(
        intent="继续",
        is_research=True,
        sub_questions=[],
        constraints=[],
        next="researcher",
        reason="r",
    )
    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor(
            {
                "messages": [HumanMessage(content="调研"), AIMessage(content="检索完成")],
                "tasks": [ResearchTask(id="T1", question="既有问题")],
                "rounds": 1,
            },
            None,  # type: ignore[arg-type]
        )
    assert "tasks" not in update


@pytest.mark.asyncio
async def test_supervisor_does_not_overwrite_existing_tasks_with_later_partial_questions():
    decision = RoutingDecision(
        intent="继续分析",
        is_research=True,
        sub_questions=["只覆盖一个对象的问题"],
        subjects=[ResearchSubject(id="S1", name="宁德时代")],
        constraints=[],
        next="analyst",
        reason="来源已足够",
    )
    state = {
        "messages": [HumanMessage(content="调研"), AIMessage(content="检索完成")],
        "tasks": [
            ResearchTask(id="T1", question="宁德时代问题", subject_ids=["S1"]),
            ResearchTask(id="T2", question="比亚迪问题", subject_ids=["S2"]),
        ],
        "subjects": [
            ResearchSubject(id="S1", name="宁德时代"),
            ResearchSubject(id="S2", name="比亚迪"),
        ],
        "rounds": 1,
    }

    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor(state, None)  # type: ignore[arg-type]

    assert "tasks" not in update
    assert "subjects" not in update


@pytest.mark.asyncio
async def test_supervisor_does_not_fall_back_for_analyst():
    decision = RoutingDecision(
        intent="分析",
        is_research=True,
        sub_questions=[],
        constraints=[],
        next="analyst",
        reason="r",
    )
    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor(
            {
                "messages": [HumanMessage(content="调研"), AIMessage(content="检索完成")],
                "sources": [make_source("https://a.example/1")],
                "rounds": 1,
            },
            None,  # type: ignore[arg-type]
        )
    assert update["next_action"] == "analyst"
    assert "tasks" not in update


def test_url_key_collapses_language_variants():
    simplified = "https://www.infolink-group.com/cn/energy-2025"
    traditional = "https://infolink-group.com/tw/energy-2025"
    assert url_key(simplified) == url_key(traditional)


def test_dedupe_sources_merges_language_variants():
    sources = [
        Source(url="https://x.example/cn/post", title="简体版"),
        Source(url="https://x.example/tw/post", title="繁體版"),
    ]
    assert [s.title for s in dedupe_sources(sources)] == ["简体版"]


def test_url_key_collapses_mobile_variants():
    desktop = "https://www.infolink-group.com/energy-2025"
    mobile = "https://m.infolink-group.com/energy-2025"
    assert url_key(desktop) == url_key(mobile)
    assert url_key("https://m.x.example/cn/post") == url_key("https://www.x.example/tw/post")


def test_dedupe_sources_merges_mobile_variants():
    sources = [
        Source(url="https://x.example/post", title="桌面版"),
        Source(url="https://m.x.example/post", title="手机版"),
    ]
    assert [s.title for s in dedupe_sources(sources)] == ["桌面版"]


def test_clean_url_keeps_identity_query_but_drops_tracking():
    assert clean_url("https://x.example/a?id=1&utm_source=weibo#frag") == "https://x.example/a?id=1"
    # Parameter order must not create a second key for the same article.
    assert url_key(clean_url("https://x.example/a?b=2&a=1")) == url_key(
        clean_url("https://x.example/a?a=1&b=2")
    )


def test_dedupe_keeps_distinct_query_id_articles():
    sources = [
        Source(url="https://x.example/d?id=1", title="文章一"),
        Source(url="https://x.example/d?id=2", title="文章二"),
    ]
    assert [s.title for s in dedupe_sources(sources)] == ["文章一", "文章二"]


def test_dedupe_merges_tracking_variants_of_one_query_id_article():
    sources = [
        Source(url="https://x.example/d?id=1&utm_source=weibo", title="微博转发"),
        Source(url="https://x.example/d?id=1&utm_source=wechat", title="微信转发"),
    ]
    assert [s.title for s in dedupe_sources(sources)] == ["微博转发"]


def test_confidence_from_authority():
    assert confidence_from_authority(4, 2.6) == 1.0
    assert confidence_from_authority(3, 1.6) == 0.85
    assert confidence_from_authority(3, 0.5) == 0.8
    assert confidence_from_authority(0, 0.0) == 0.4
    assert confidence_from_authority(-1, 0.0) == 0.45


@pytest.mark.asyncio
async def test_search_clamps_payload_to_api_limits():
    captured: dict = {}

    async def fake_post(path: str, payload: dict, timeout: float) -> dict:
        captured.update(path=path, payload=payload)
        return {"results": [{"url": "https://a.example/1"}]}

    with patch("agents.research_team.search._post", side_effect=fake_post):
        results = await search("q" * 200, count=99, ori_question="i" * 200)

    assert len(results) == 1
    assert captured["path"] == "search"
    assert len(captured["payload"]["query"]) == 128
    assert captured["payload"]["count"] == 50
    assert len(captured["payload"]["ori_question"]) == 128
    assert captured["payload"]["need_content"] is False


@pytest.mark.asyncio
async def test_search_rejects_a_malformed_success_payload():
    async def fake_post(path: str, payload: dict, timeout: float) -> object:
        return {"results": {"url": "https://a.example/1"}}

    with patch("agents.research_team.search._post", side_effect=fake_post):
        with pytest.raises(RuntimeError, match="invalid results payload"):
            await search("query")


def test_candidate_downgrades_invalid_scores_instead_of_raising():
    candidate = _candidate(
        {
            "url": "https://a.example/1",
            "authority": "not-a-number",
            "rerank_score": True,
        },
        "query",
    )

    assert candidate is not None
    assert candidate["authority"] == -1
    assert candidate["rerank_score"] == 0.0


def test_candidate_skips_a_non_object_result():
    assert _candidate(["not", "an", "object"], "query") is None


def test_filter_candidates_removes_only_structural_invalids_and_duplicates():
    candidates, counts = _filter_candidates(
        [
            (
                "query",
                [
                    {"url": "ftp://bad.example/1", "title": "无效协议"},
                    {"url": "https://empty.example/1", "title": ""},
                    {"url": "https://good.example/1", "title": "有效", "authority": 0},
                    {"url": "https://good.example/1", "title": "重复", "authority": 4},
                ],
            )
        ],
        [],
    )

    assert [candidate["url"] for candidate in candidates] == ["https://good.example/1"]
    assert candidates[0]["authority"] == 0
    assert counts == {
        "invalid_item": 0,
        "invalid_url": 1,
        "empty_content": 1,
        "duplicate": 1,
        "over_limit": 0,
    }


def test_filter_candidates_distinguishes_existing_duplicate():
    candidates, counts = _filter_candidates(
        [("query", [{"url": "https://known.example/1", "title": "已有来源"}])],
        [Source(url="https://known.example/1")],
    )

    assert candidates == []
    assert counts["duplicate"] == 1


def test_filter_candidates_preserves_search_order_before_judging():
    candidates, counts = _filter_candidates(
        [
            (
                "query",
                [
                    {
                        "url": "https://first.example/1",
                        "title": "首个结果",
                        "authority": 0,
                        "rerank_score": 0.1,
                    },
                    {
                        "url": "https://second.example/1",
                        "title": "第二个结果",
                        "authority": 4,
                        "rerank_score": 3.0,
                    },
                ],
            )
        ],
        [],
    )

    assert [candidate["url"] for candidate in candidates] == [
        "https://first.example/1",
        "https://second.example/1",
    ]
    assert counts["over_limit"] == 0


def test_filter_candidates_keeps_highest_reranked_results_when_over_limit():
    candidates, counts = _filter_candidates(
        [
            (
                "query",
                [
                    {
                        "url": f"https://example.com/{index}",
                        "title": f"结果 {index}",
                        "authority": 0,
                        "rerank_score": 3 * index / MAX_CANDIDATES,
                    }
                    for index in range(MAX_CANDIDATES + 1)
                ],
            )
        ],
        [],
    )

    assert len(candidates) == MAX_CANDIDATES
    assert candidates[0]["rerank_score"] == 3.0
    assert all(candidate["rerank_score"] > 0 for candidate in candidates)
    assert counts["over_limit"] == 1


@pytest.mark.asyncio
async def test_researcher_keeps_valid_results_when_another_result_is_malformed():
    state = {
        "messages": [HumanMessage(content="调研储能")],
        "tasks": [ResearchTask(id="T1", question="储能市场")],
        "sources": [make_source("https://old.example/1")],
    }

    async def fake_search(query: str, count: int, ori_question: str | None = None) -> list:
        return [
            {"url": "https://bad.example/1", "authority": {}, "rerank_score": "bad"},
            {
                "url": "https://good.example/1",
                "title": "有效结果",
                "summary": "摘要",
                "authority": 3,
                "rerank_score": 2.0,
            },
            ["不是对象"],
        ]

    async def fake_judge(candidates: list, state: dict, config: object) -> SourceJudgements:
        return SourceJudgements(
            sources=[{"url": "https://good.example/1", "keep": True, "reason": "相关"}]
        )

    with (
        patch("agents.research_team.researcher.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.researcher._plan_queries", return_value=["储能市场"]),
        patch("agents.research_team.researcher.search", side_effect=fake_search),
        patch("agents.research_team.researcher._judge", side_effect=fake_judge),
        patch("agents.research_team.researcher._deep_read", return_value=0),
    ):
        update = await researcher(state, None)  # type: ignore[arg-type]

    assert [source.url for source in update["sources"]] == [
        "https://old.example/1",
        "https://good.example/1",
    ]


@pytest.mark.asyncio
async def test_fetch_rejects_unsafe_urls():
    token = begin_turn("skip-probe")
    try:
        assert await fetch("ftp://example.com/x") is None
        assert await fetch("https://user:pass@example.com/x") is None
        assert await fetch("") is None
        metrics = current()
        assert metrics is not None
        # Skips are counted separately: no HTTP call was made, so they must not
        # inflate fetch (success) or fetch_failed.
        assert metrics.fetch_skipped == 3
        assert metrics.fetch_calls == 0
        assert metrics.fetch_failed == 0
    finally:
        await end_turn(token, "ok")


@pytest.mark.asyncio
async def test_fetch_returns_none_when_content_is_empty():
    async def fake_post(path: str, payload: dict, timeout: float) -> dict:
        return {"title": "空页", "content": ""}

    with patch("agents.research_team.search._post", side_effect=fake_post):
        assert await fetch("https://example.com/x") is None


@pytest.mark.asyncio
async def test_fetch_rejects_an_invalid_response_payload():
    async def fake_post(path: str, payload: dict, timeout: float) -> dict:
        return []  # type: ignore[return-value]

    with patch("agents.research_team.search._post", side_effect=fake_post):
        with pytest.raises(SearchResponseError, match="invalid response payload"):
            await fetch("https://example.com/x")


@pytest.mark.asyncio
async def test_deep_read_only_fetches_authoritative_sources():
    low = Source(url="https://low.example/1", authority=1, rerank_score=0.5, confidence=0.5)
    high = Source(url="https://high.example/1", authority=3, rerank_score=2.0, confidence=0.85)
    calls: list[str] = []

    async def fake_fetch(url: str) -> dict[str, str]:
        calls.append(url)
        return {"title": "正文", "content": "# 正文"}

    with patch("agents.research_team.researcher.fetch", side_effect=fake_fetch):
        fetched = await _deep_read([low, high])

    assert fetched == 1
    assert calls == ["https://high.example/1"]
    assert high.content == "# 正文"
    assert low.content == ""


@pytest.mark.asyncio
async def test_deep_read_propagates_cancellation():
    source = Source(url="https://high.example/1", authority=3, rerank_score=2.0)

    async def cancelled_fetch(url: str) -> dict[str, str]:
        raise asyncio.CancelledError

    with patch("agents.research_team.researcher.fetch", side_effect=cancelled_fetch):
        with pytest.raises(asyncio.CancelledError):
            await _deep_read([source])


@pytest.mark.asyncio
async def test_researcher_propagates_search_cancellation():
    async def cancelled_search(query: str, count: int, ori_question: str | None = None) -> list:
        raise asyncio.CancelledError

    state = {
        "messages": [HumanMessage(content="调研储能")],
        "tasks": [ResearchTask(id="T1", question="储能市场")],
    }
    with (
        patch("agents.research_team.researcher.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.researcher._plan_queries", return_value=["储能市场"]),
        patch("agents.research_team.researcher.search", side_effect=cancelled_search),
    ):
        with pytest.raises(asyncio.CancelledError):
            await researcher(state, None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_plan_queries_falls_back_to_task_text_on_llm_failure():
    state = {
        "messages": [HumanMessage(content="随便聊聊")],
        "tasks": [ResearchTask(id="T1", question="2025 年储能电芯出货排名")],
    }
    with patch("agents.research_team.researcher.resolve_model", side_effect=RuntimeError("boom")):
        queries = await _plan_queries(state, None)  # type: ignore[arg-type]

    assert queries == ["2025 年储能电芯出货排名"]


@pytest.mark.asyncio
async def test_plan_queries_falls_back_to_raw_message_without_tasks():
    state = {"messages": [HumanMessage(content="  帮我看看储能电芯  ")], "tasks": []}
    with patch("agents.research_team.researcher.resolve_model", side_effect=RuntimeError("boom")):
        queries = await _plan_queries(state, None)  # type: ignore[arg-type]

    assert queries == ["帮我看看储能电芯"]


def make_candidate(url: str, authority: int, rerank_score: float) -> dict:
    return {
        "url": url,
        "title": url,
        "summary": "摘要",
        "site_name": "示例站",
        "published_at": "",
        "authority": authority,
        "rerank_score": rerank_score,
    }


@pytest.mark.asyncio
async def test_judge_falls_back_to_score_thresholds_on_llm_failure():
    candidates = [
        make_candidate("https://high.example/1", authority=3, rerank_score=2.0),
        make_candidate("https://medium.example/1", authority=0, rerank_score=1.8),
        make_candidate("https://low.example/1", authority=1, rerank_score=0.4),
    ]
    state = {"messages": [HumanMessage(content="调研")], "tasks": []}

    with patch("agents.research_team.researcher.resolve_model", side_effect=RuntimeError("boom")):
        result = await _judge(candidates, state, None)  # type: ignore[arg-type]

    kept = [item.url for item in result.sources]
    assert kept == ["https://high.example/1", "https://medium.example/1"]


def test_excerpt_leaves_short_text_untouched():
    text = "短文本，不需要标注"
    assert excerpt(text) == text


def test_prompt_renderer_rejects_missing_and_unexpected_variables():
    template = ChatPromptTemplate.from_messages([("system", "任务：{task}")])

    with pytest.raises(ValueError, match="missing variables: task"):
        render_prompt(template)
    with pytest.raises(ValueError, match="unexpected variables: extra"):
        render_prompt(template, task="调研", extra="不应传入")


# The file name is the prompt name, so editing a template file cannot silently change
# which variables a node must supply. A stray "{foo}" in prose fails here, not at render.
_PROMPT_VARIABLES = {
    "research_team.supervisor.instructions": set(),
    "research_team.supervisor.status": {
        "intent",
        "tasks",
        "subjects",
        "constraints",
        "source_count",
        "analysis_done",
        "coverage_summary",
        "subject_coverage_count",
        "report_done",
        "rounds",
        "max_rounds",
    },
    "research_team.researcher.instructions": {"max_query_chars"},
    "research_team.researcher.task_brief": {
        "intent",
        "questions",
        "subjects",
        "constraints",
        "known_urls",
        "excerpt",
    },
    "research_team.researcher.judge_brief": {"questions", "constraints"},
    "research_team.researcher.judge_results": {"listing"},
    "research_team.analyst.instructions": set(),
    "research_team.analyst.context": {"intent", "tasks", "subjects", "constraints", "sources"},
    "research_team.writer.instructions": set(),
    "research_team.writer.context": {
        "intent",
        "constraints",
        "analysis",
        "coverage",
        "subject_coverage",
        "evidence_statements",
        "sources",
    },
}


@pytest.mark.parametrize(("name", "expected"), sorted(_PROMPT_VARIABLES.items()))
def test_prompt_file_declares_the_expected_variables(name: str, expected: set[str]) -> None:
    template = ChatPromptTemplate.from_messages([("system", read_prompt(name))])
    assert set(template.input_variables) == expected


def test_read_prompt_fails_loudly_for_a_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        read_prompt("research_team.not_a_real_prompt")


def test_every_prompt_file_is_non_empty() -> None:
    files = sorted(PROMPTS_DIR.rglob("*.md"))
    assert len(files) == len(_PROMPT_VARIABLES)
    for path in files:
        assert path.read_text(encoding="utf-8").strip(), f"{path} is empty"


def test_clean_fetched_markdown_removes_page_noise_and_exact_duplicates():
    content = """[首页](https://example.com/)
[登录](javascript:void(0))
    站点 > 新闻 > 正文
    字体：小 中 大

# 标题

正文数字：2025 年产量为 100 万吨。
| 指标 | 数值 |
| --- | --- |
| 产量 | 100 万吨 |
正文数字：2025 年产量为 100 万吨。

![站点 Logo](https://example.com/logo.png)
[相关稿件](https://example.com/related)
责任编辑：张三
【纠错】 【责任编辑：李四】
Copyright © 2000 - 2026 Example
版权所有：Example
"""

    cleaned = clean_fetched_markdown(content)

    assert "[首页]" not in cleaned
    assert "javascript:" not in cleaned
    assert "站点 > 新闻 > 正文" not in cleaned
    assert "字体：" not in cleaned
    assert "Logo" not in cleaned
    assert "[相关稿件]" not in cleaned
    assert "责任编辑" not in cleaned
    assert "Copyright" not in cleaned
    assert "版权所有" not in cleaned
    assert cleaned.count("正文数字") == 1
    assert "| 产量 | 100 万吨 |" in cleaned


def test_clean_fetched_markdown_keeps_repeated_table_rows_across_tables():
    content = """# 表一
| 指标 | 数值 |
| --- | --- |
| 产量 | 100 |

# 表二
| 指标 | 数值 |
| --- | --- |
| 销量 | 200 |
"""

    cleaned = clean_fetched_markdown(content)

    assert cleaned.count("| 指标 | 数值 |") == 2
    assert cleaned.count("| --- | --- |") == 2
    assert "| 销量 | 200 |" in cleaned


def test_deep_read_cleans_before_content_limit():
    source = Source(url="https://high.example/1", authority=3, rerank_score=2.0)
    noisy_prefix = "[首页](https://example.com/)\n" * 100
    useful = "核心数据。" * 1000

    async def fake_fetch(url: str) -> dict[str, str]:
        return {"title": "正文", "content": noisy_prefix + useful}

    with patch("agents.research_team.researcher.fetch", side_effect=fake_fetch):
        fetched = asyncio.run(_deep_read([source]))

    assert fetched == 1
    assert source.content.startswith("核心数据。")
    assert len(source.content) == MAX_CONTENT_CHARS


def test_excerpt_marks_truncation_with_total_length():
    text = "甲" * (EXCERPT_CHARS + 500)
    result = excerpt(text)
    assert f"全文共 {len(text)} 字" in result
    assert result.startswith("甲" * EXCERPT_CHARS)


def test_numbered_sources_reads_beyond_the_old_six_hundred_cut():
    # The tail marker sits at position 1000, which the previous 600-char cap dropped.
    body = "前文" * 500 + "【后段关键数据】" + "补充" * 400
    source = Source(url="https://a.example/1", title="标题", content=body)
    rendered = numbered_sources([source])
    assert "【后段关键数据】" in rendered
    assert f"全文共 {len(body)} 字" in rendered


class _ListWrappingModel:
    """Mimics DashScope returning `[{...}]` where the schema wants `{...}`."""

    def with_structured_output(
        self, schema: type, method: str | None = None, include_raw: bool = False, **kwargs
    ) -> object:
        payload = json.dumps(
            [
                [
                    {
                        "intent": "测试意图",
                        "is_research": True,
                        "sub_questions": ["子问题"],
                        "constraints": [],
                        "next": "researcher",
                        "reason": "需要检索",
                    }
                ]
            ],
            ensure_ascii=False,
        )

        class _Decider:
            async def ainvoke(self, messages: list) -> dict:
                return {
                    "raw": AIMessage(content=payload),
                    "parsed": None,
                    "parsing_error": ValueError("expected dict"),
                }

        return _Decider()


@pytest.mark.asyncio
async def test_structured_output_unwraps_single_element_array():
    decider = structured_output(_ListWrappingModel(), RoutingDecision)
    decision = await decider.ainvoke([])
    assert isinstance(decision, RoutingDecision)
    assert decision.next == "researcher"


class _BadRequest(Exception):
    status_code = 400


class _StrictRejectingModel:
    """Mimics DeepSeek's own endpoint, which refuses json_schema."""

    def __init__(self, decision: RoutingDecision) -> None:
        self._decision = decision
        self.methods: list[str | None] = []

    def with_structured_output(
        self, schema: type, method: str | None = None, include_raw: bool = False, **kwargs
    ) -> object:
        self.methods.append(method)
        decision = self._decision

        class _Decider:
            async def ainvoke(self, messages: list) -> dict:
                if method == "json_schema":
                    raise _BadRequest("Error code: 400 - This response_format type is unavailable")
                return {"raw": None, "parsed": decision, "parsing_error": None}

        return _Decider()


@pytest.mark.asyncio
async def test_structured_output_falls_back_when_json_schema_is_rejected():
    decision = RoutingDecision(
        intent="测试",
        is_research=True,
        sub_questions=[],
        constraints=[],
        next="researcher",
        reason="需要检索",
    )
    model = _StrictRejectingModel(decision)

    result = await structured_output(model, RoutingDecision).ainvoke([])

    assert result is decision
    assert model.methods == ["json_schema", "json_mode"]


@pytest.mark.asyncio
async def test_structured_output_falls_back_on_json_schema_serving_oom():
    decision = RoutingDecision(
        intent="测试",
        is_research=True,
        sub_questions=[],
        constraints=[],
        next="researcher",
        reason="需要检索",
    )

    class _OomModel:
        def __init__(self) -> None:
            self.methods: list[str | None] = []

        def with_structured_output(
            self, schema: type, method: str | None = None, include_raw: bool = False, **kwargs
        ) -> object:
            self.methods.append(method)

            class _Decider:
                async def ainvoke(self, messages: list) -> dict:
                    if method == "json_schema":
                        raise _BadRequest(
                            "Error code: 400 - An error occurred in model serving, "
                            "error message is: [std::bad_alloc]"
                        )
                    return {"raw": None, "parsed": decision, "parsing_error": None}

            return _Decider()

    model = _OomModel()

    result = await structured_output(model, RoutingDecision).ainvoke([])

    assert result is decision
    assert model.methods == ["json_schema", "json_mode"]


@pytest.mark.asyncio
async def test_structured_output_forwards_max_tokens_budget():
    captured: list[dict] = []

    class _BudgetModel:
        def with_structured_output(
            self, schema: type, method: str | None = None, include_raw: bool = False, **kwargs
        ) -> object:
            captured.append(kwargs)

            class _Decider:
                async def ainvoke(self, messages: list) -> dict:
                    decision = RoutingDecision(
                        intent="测试",
                        is_research=True,
                        sub_questions=[],
                        constraints=[],
                        next="researcher",
                        reason="需要检索",
                    )
                    return {"raw": None, "parsed": decision, "parsing_error": None}

            return _Decider()

    await structured_output(_BudgetModel(), RoutingDecision, "supervisor", max_tokens=2048).ainvoke(
        []
    )
    assert captured[-1].get("max_tokens") == 2048

    await structured_output(_BudgetModel(), RoutingDecision).ainvoke([])
    assert "max_tokens" not in captured[-1]


@pytest.mark.asyncio
async def test_structured_output_does_not_swallow_unrelated_errors():
    class _Broken:
        def with_structured_output(
            self, schema: type, method: str | None = None, include_raw: bool = False, **kwargs
        ) -> object:
            class _Decider:
                async def ainvoke(self, messages: list) -> dict:
                    raise RuntimeError("connection reset")

            return _Decider()

    with pytest.raises(RuntimeError):
        await structured_output(_Broken(), RoutingDecision).ainvoke([])


class _LengthLimit(LengthFinishReasonError):
    """The real exception demands a ChatCompletion; the class identity is what matters."""

    def __init__(self) -> None:
        pass


class _OverrunningModel:
    """First call overruns the completion ceiling, the retry returns a valid object."""

    def __init__(self, decision: RoutingDecision) -> None:
        self._decision = decision
        self.calls = 0

    def with_structured_output(
        self, schema: type, method: str | None = None, include_raw: bool = False, **kwargs
    ) -> object:
        decision = self._decision
        parent = self

        class _Decider:
            async def ainvoke(self, messages: list) -> dict:
                parent.calls += 1
                if parent.calls == 1:
                    raise _LengthLimit()
                return {"raw": None, "parsed": decision, "parsing_error": None}

        return _Decider()


@pytest.mark.asyncio
async def test_structured_output_retries_with_shorter_brief_after_length_overrun():
    decision = RoutingDecision(
        intent="测试",
        is_research=True,
        sub_questions=[],
        constraints=[],
        next="writer",
        reason="已有分析",
    )
    model = _OverrunningModel(decision)

    result = await structured_output(model, RoutingDecision).ainvoke([])

    assert result is decision
    assert model.calls == 2


def test_render_report_emits_markdown_table():
    bundle = ReportBundle(
        title="储能电芯",
        summary="这是摘要内容，包含关键数字与结论 [1]",
        findings=["发现甲 [1]", "发现乙 [1]"],
        tables=[
            TableBlock(
                title="出货排名",
                headers=["企业", "出货"],
                rows=[["宁德时代", "100"], ["海辰储能"]],  # short row gets padded
            )
        ],
        analysis="这是分析段落，解释机制、格局与风险 [1]",
        conclusion="这是结论与建议，给出可执行判断 [1]",
    )
    report = render_report(bundle, source_count=3)

    assert "| 企业 | 出货 |" in report
    assert "| --- | --- |" in report
    assert "| 宁德时代 | 100 |" in report
    assert "| 海辰储能 |  |" in report
    # The table belongs to the findings section, not a fifth heading.
    assert report.count("## ") == 4


def test_render_report_emits_deterministic_subject_completeness():
    bundle = ReportBundle(
        title="欧洲产能",
        summary="这是摘要内容，包含关键数字与结论 [1]",
        findings=["发现甲 [1]", "发现乙 [1]"],
        analysis="这是分析段落，解释机制、格局与风险 [1]",
        conclusion="这是结论与建议，给出可执行判断 [1]",
    )
    coverage = [
        SubjectEvidenceCoverage(
            subject_id="S1",
            subject="宁德时代",
            aspects=[
                AspectCoverage(
                    name="产能",
                    status="sufficient",
                    evidence_strength="moderate",
                    source_indices=[1],
                    gaps=[],
                    reason="已有来源",
                )
            ],
            status="sufficient",
            evidence_strength="moderate",
            source_indices=[1],
            gaps=[],
            reason="证据足够",
        ),
        SubjectEvidenceCoverage(
            subject_id="S2",
            subject="比亚迪",
            aspects=[],
            status="missing",
            evidence_strength="insufficient",
            source_indices=[],
            gaps=["缺少产能数据"],
            reason="没有直接来源",
        ),
    ]

    report = render_report(bundle, source_count=1, subject_coverage=coverage)

    assert "## 对象证据完整度" in report
    assert "宁德时代：状态 sufficient" in report
    assert "比亚迪：状态 missing" in report
    assert "缺少产能数据" in report


def test_requires_table_ignores_a_negated_request():
    assert requires_table(["交付格式：表格"]) is True
    assert requires_table(["交付格式：要点式，不用表格"]) is False
    assert requires_table(["时间范围：2025 年"]) is False


def test_requires_table_scopes_negation_to_the_table_clause():
    # An unrelated refusal in the same constraint must not cancel the table request.
    assert requires_table(["请用表格呈现，不要分析动力电池业务"]) is True
    assert requires_table(["用表格", "不要分析动力电池业务"]) is True
    assert requires_table(["不要用任何表格"]) is False
    # Negation after the table mention is not a refusal of tables.
    assert requires_table(["表格不用太多"]) is True
    # An explicit refusal wins over a positive request elsewhere.
    assert requires_table(["用表格", "不用表格"]) is False


def _coverage_with_a_conflicting_aspect() -> list[SubjectEvidenceCoverage]:
    return [
        SubjectEvidenceCoverage(
            subject_id="S1",
            subject="示例市场",
            aspects=[
                AspectCoverage(
                    name="产能",
                    status="conflicting",
                    evidence_strength="moderate",
                    source_indices=[1, 2],
                    gaps=["两家口径不一致"],
                    reason="来源互相矛盾",
                ),
                AspectCoverage(
                    name="价格",
                    status="sufficient",
                    evidence_strength="strong",
                    source_indices=[1],
                    gaps=[],
                    reason="单一权威来源",
                ),
            ],
            status="conflicting",
            evidence_strength="moderate",
            source_indices=[1, 2],
            gaps=["两家口径不一致"],
            reason="来源互相矛盾",
        )
    ]


def test_subject_coverage_renders_a_conflicting_aspect_only_in_the_list_form():
    # The table prints coverage and gaps in adjacent columns, so listing the conflicting
    # aspect there would put it in both columns with opposite meanings.
    coverage = _coverage_with_a_conflicting_aspect()

    assert "产能" in _render_subject_coverage(coverage, include_table=False)

    table = _render_subject_coverage(coverage, include_table=True)
    assert "| 示例市场 | 价格 |" in table


def test_table_required_schema_forbids_an_empty_list():
    schema = report_schema(["交付格式：表格"]).model_json_schema()

    assert schema["properties"]["tables"]["minItems"] == 1
    assert (
        report_schema(["交付格式：要点式"])
        .model_json_schema()["properties"]["tables"]
        .get("minItems")
        is None
    )


def test_route_stops_immediately_when_retrieval_is_rejected():
    state = {"rounds": 1, "next_action": "researcher", "abort_reason": "search_rejected"}
    assert route(state) == "finalize"


@pytest.mark.asyncio
async def test_finalize_explains_a_rejected_retrieval():
    result = await finalize(
        {"messages": [HumanMessage(content="调研")], "abort_reason": "search_rejected"}
    )
    content = result["messages"][0].content
    assert "拒绝了本次请求" in content
    assert "余额不足" in content


@pytest.mark.asyncio
async def test_researcher_aborts_when_every_query_is_rejected():
    state = {
        "messages": [HumanMessage(content="调研储能")],
        "tasks": [ResearchTask(id="T1", question="储能电芯出货排名")],
    }
    with (
        patch(
            "agents.research_team.researcher._plan_queries",
            return_value=["储能电芯 2025", "储能电芯 排名"],
        ),
        patch(
            "agents.research_team.researcher.search",
            side_effect=SearchRejected("HTTP 402: Insufficient Balance"),
        ),
    ):
        update = await researcher(state, None)  # type: ignore[arg-type]

    assert update["abort_reason"] == "search_rejected"
    assert "检索未返回结果" in update["messages"][0].content


def test_inline_schema_removes_defs_and_keeps_constraints():
    schema = inline_schema(TableRequiredReport.model_json_schema())
    dumped = json.dumps(schema)

    assert "$defs" not in dumped
    assert "$ref" not in dumped
    assert schema["properties"]["tables"]["minItems"] == 1
    assert set(schema["properties"]["tables"]["items"]["properties"]) == {
        "title",
        "headers",
        "rows",
    }


def _writer_state(analysis: str) -> dict:
    return {
        "messages": [HumanMessage(content="调研储能电芯")],
        "intent": "储能电芯竞争格局",
        "constraints": ["交付格式：表格"],
        "analysis": analysis,
        "sources": [
            Source(
                url="https://a.example/1",
                title="标题甲",
                content="原文摘录标记" * 60,
                summary="摘要标记" * 40,
            )
        ],
    }


@pytest.mark.asyncio
async def test_writer_omits_excerpts_when_analysis_is_rich():
    captured: list[str] = []
    bundle = ReportBundle(
        title="标题",
        summary="这是摘要内容，包含关键数字与结论 [1]",
        findings=["发现甲 [1]", "发现乙 [1]"],
        tables=[TableBlock(headers=["企业", "出货"], rows=[["宁德时代", "100"]])],
        analysis="这是分析段落，解释机制、格局与风险 [1]",
        conclusion="这是结论与建议，给出可执行判断 [1]",
    )
    with patch(
        "agents.research_team.writer.resolve_model",
        return_value=_CapturingModel(bundle, captured),
    ):
        await writer(_writer_state("完整分析" * 400), None)  # type: ignore[arg-type]

    prompt = captured[-1]
    assert "摘录:" not in prompt
    assert "原文摘录标记" not in prompt


@pytest.mark.asyncio
async def test_writer_keeps_excerpts_when_analysis_is_thin():
    captured: list[str] = []
    bundle = ReportBundle(
        title="标题",
        summary="这是摘要内容，包含关键数字与结论 [1]",
        findings=["发现甲 [1]", "发现乙 [1]"],
        tables=[TableBlock(headers=["企业"], rows=[["宁德时代"]])],
        analysis="这是分析段落，解释机制、格局与风险 [1]",
        conclusion="这是结论与建议，给出可执行判断 [1]",
    )
    with patch(
        "agents.research_team.writer.resolve_model",
        return_value=_CapturingModel(bundle, captured),
    ):
        await writer(_writer_state("太短"), None)  # type: ignore[arg-type]

    assert "原文摘录标记" in captured[-1]


def test_render_table_escapes_pipes_and_newlines():
    bundle = ReportBundle(
        title="标题",
        summary="这是摘要内容，包含关键数字与结论 [1]",
        findings=["发现甲 [1]", "发现乙 [1]"],
        tables=[
            TableBlock(
                headers=["企业", "价格区间"],
                rows=[["宁德\n时代", "0.26|0.38"]],
            )
        ],
        analysis="这是分析段落，解释机制、格局与风险 [1]",
        conclusion="这是结论与建议，给出可执行判断 [1]",
    )
    report = render_report(bundle, source_count=1)

    assert "| 宁德 时代 | 0.26\\|0.38 |" in report
    # Count unescaped pipes only: the escaped one must not create a phantom column.
    row = next(line for line in report.splitlines() if "宁德" in line)
    assert len(re.findall(r"(?<!\\)\|", row)) == 3


@pytest.mark.asyncio
async def test_supervisor_skips_the_model_call_when_retrieval_was_rejected():
    with patch(
        "agents.research_team.supervisor.resolve_model", side_effect=AssertionError("no call")
    ):
        update = await supervisor(
            {"messages": [HumanMessage(content="调研")], "abort_reason": "search_rejected"},
            None,  # type: ignore[arg-type]
        )

    assert update == {"next_action": "finalize"}


def test_numbered_sources_omits_the_excerpt_line_when_disabled():
    source = Source(url="https://a.example/1", title="标题", content="正文" * 100)
    rendered = numbered_sources([source], include_excerpt=False)

    assert "摘录:" not in rendered
    assert "标题" in rendered and "https://a.example/1" in rendered


class _SequenceModel:
    """Returns a different bundle per call, recording the prompts it received."""

    def __init__(self, results: list[object], captured: list[str]) -> None:
        self._results = results
        self._captured = captured

    def with_structured_output(
        self, schema: type, method: str | None = None, include_raw: bool = False, **kwargs
    ) -> object:
        results = self._results
        captured = self._captured

        class _Decider:
            index = 0

            async def ainvoke(self, messages: list) -> dict:
                captured.append("\n".join(str(message.content) for message in messages))
                result = results[min(_Decider.index, len(results) - 1)]
                _Decider.index += 1
                return {"raw": None, "parsed": result, "parsing_error": None}

        return _Decider()


@pytest.mark.asyncio
async def test_writer_retries_when_the_report_is_hollow():
    hollow = ReportBundle(
        title="标题",
        summary="半句",
        findings=["无", "无"],
        tables=[TableBlock(headers=["企业"], rows=[["宁德时代"]])],
        analysis=" ",
        conclusion=" ",
    )
    complete = ReportBundle(
        title="标题",
        summary="这是补全后的摘要，包含关键数字与结论 [1]",
        findings=["发现甲 [1]", "发现乙 [1]"],
        tables=[TableBlock(headers=["企业"], rows=[["宁德时代"]])],
        analysis="这是补全后的分析段落，解释机制与格局 [1]",
        conclusion="这是补全后的结论与建议 [1]",
    )
    captured: list[str] = []
    with patch(
        "agents.research_team.writer.resolve_model",
        return_value=_SequenceModel([hollow, complete], captured),
    ):
        out = await writer(_writer_state("完整分析" * 400), None)  # type: ignore[arg-type]

    assert len(captured) == 2
    assert "上一次输出不完整" in captured[1]
    assert "这是补全后的分析段落" in out["report"]


def _bundle(**overrides) -> ReportBundle:
    fields = {
        "title": "标题",
        "summary": "全年销量约 2.9 万辆，同比增长 95% [1]。",
        "findings": ["发现甲，销量第一 [1]。", "发现乙，份额 24.9% [2]。"],
        "tables": [TableBlock(headers=["企业", "份额"], rows=[["徐工", "24.9%"]])],
        "analysis": "换电路线份额被充电路线侵蚀，增量集中在资源运输场景 [7]。",
        "conclusion": "格局未固化，宜按季度跟踪份额变化 [1][2]。",
    }
    return ReportBundle(**(fields | overrides))


def test_is_complete_rejects_mid_sentence_fields():
    # Observed against DashScope constrained decoding: finish_reason=stop, valid
    # JSON, but the model closed the string right after "呈" / "中".
    truncated = _bundle(
        summary="全年销量约 2.9 万辆，同比增长 95%，占新能源重卡市场 35% [1][2]。市场格局呈",
        analysis="增量主要由局部资源运输需求驱动，而非全国性普适渗透。竞争格局呈",
        conclusion="全年前五强完整份额结构不完整；此外来源 [6][8] 份额表中",
    )
    assert _is_complete(truncated) is False


def test_is_complete_accepts_punctuation_and_anchor_endings():
    assert _is_complete(_bundle()) is True
    assert _is_complete(_bundle(conclusion="运营商份额数据缺失，无法确证 [9][10]")) is True
    assert _is_complete(_bundle(findings=["发现甲 [1]", "发现乙 [2]"])) is True


def test_is_complete_rejects_truncated_finding():
    assert _is_complete(_bundle(findings=["发现甲，销量第一 [1]。", "发现乙，份额"])) is False


@pytest.mark.asyncio
async def test_writer_retries_when_fields_end_mid_sentence():
    truncated = _bundle(summary="全年销量约 2.9 万辆，同比增长 95% [1]。市场格局呈")
    complete = _bundle()
    captured: list[str] = []
    with patch(
        "agents.research_team.writer.resolve_model",
        return_value=_SequenceModel([truncated, complete], captured),
    ):
        out = await writer(_writer_state("完整分析" * 400), None)  # type: ignore[arg-type]

    assert len(captured) == 2
    assert "上一次输出不完整" in captured[1]
    assert "市场格局呈" not in out["report"]


@pytest.mark.asyncio
async def test_analyst_retries_when_the_fact_base_is_blank():
    captured: list[str] = []
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="储能出货",
            source_indices=[1],
            status="sufficient",
            evidence_strength="strong",
            gaps=[],
            reason="来源支持",
        )
    ]
    thin = AnalysisBundle(
        facts=[" ", " "],
        patterns=["模式"],
        contradictions=[],
        gaps=["缺口"],
        coverage=coverage,
    )
    good = AnalysisBundle(
        facts=["事实甲 [1]", "事实乙 [1]"],
        patterns=["模式 [1]"],
        contradictions=[],
        gaps=["缺口 [1]"],
        coverage=coverage,
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "储能",
        "constraints": [],
        "tasks": [ResearchTask(id="T1", question="储能出货")],
        "sources": [make_source("https://a.example/1")],
    }
    with patch(
        "agents.research_team.analyst.resolve_model",
        return_value=_SequenceModel([thin, good], captured),
    ):
        out = await analyst(state, None)  # type: ignore[arg-type]

    assert len(captured) == 2
    assert "事实基线" in captured[1]
    assert "事实甲" in out["analysis"]
    assert out["evidence_coverage"][0].task_id == "T1"


@pytest.mark.asyncio
async def test_finalize_does_not_claim_the_round_cap_when_it_was_not_hit():
    result = await finalize(
        {
            "messages": [HumanMessage(content="调研")],
            "sources": [make_source("https://a.example/1")],
            "analysis": "## 事实基线\n- 已确证 [1]",
            "rounds": 2,
        }
    )
    content = result["messages"][0].content

    assert "未生成成稿报告" in content
    assert "达到调度上限" not in content


def test_remap_anchors_moves_batch_numbers_to_ledger_positions():
    assert remap_anchors("事实甲 [1]，事实乙 [2]", {1: 13, 2: 14}) == "事实甲 [13]，事实乙 [14]"


def test_remap_anchors_drops_unknown_numbers_and_keeps_years():
    # [2024] is a year, not a citation; dropping it would corrupt the sentence.
    assert remap_anchors("2024 [1] 年规模，[2024] 与 [9]", {1: 5}) == "2024 [5] 年规模，[2024] 与 "


def test_batch_window_starts_at_the_first_unread_source():
    sources = [Source(url=f"https://s.example/{index}") for index in range(1, 15)]
    read = {url_key(source.url) for source in sources[:12]}

    window, mapping = batch_window(sources, read, 12)

    assert [source.url for source in window] == ["https://s.example/13", "https://s.example/14"]
    assert mapping == {1: 13, 2: 14}


def test_batch_window_falls_back_to_the_tail_when_everything_was_read():
    sources = [Source(url=f"https://s.example/{index}") for index in range(1, 5)]
    read = {url_key(source.url) for source in sources}

    window, mapping = batch_window(sources, read, 2)

    assert [source.url for source in window] == ["https://s.example/3", "https://s.example/4"]
    assert mapping == {1: 3, 2: 4}


def test_merge_lines_dedupes_on_whitespace_normalised_text():
    merged = merge_lines(["事实甲 [1]"], ["事实甲  [1]", "事实乙 [2]"], 10)

    assert merged == ["事实甲 [1]", "事实乙 [2]"]


@pytest.mark.asyncio
async def test_second_batch_reads_unread_sources_and_keeps_the_earlier_judgement():
    captured: list[str] = []
    sources = [
        Source(url=f"https://s.example/{index}", title=f"来源{index}", content="正文" * 50)
        for index in range(1, 15)
    ]
    bundle = AnalysisBundle(
        facts=["新事实甲 [1]", "新事实乙 [1]"],
        patterns=["新模式 [1]"],
        contradictions=[],
        gaps=["新缺口 [1]"],
        coverage=[
            EvidenceCoverage(
                task_id="T1",
                question="问题一",
                source_indices=[1],
                status="sufficient",
                evidence_strength="moderate",
                gaps=[],
                reason="本批来源支持",
            ),
            # The model cannot see the first batch, so it reports this one uncovered.
            EvidenceCoverage(
                task_id="T2",
                question="问题二",
                source_indices=[],
                status="missing",
                evidence_strength="insufficient",
                gaps=["本批无材料"],
                reason="本批未涉及",
            ),
        ],
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "意图",
        "tasks": [
            ResearchTask(id="T1", question="问题一"),
            ResearchTask(id="T2", question="问题二"),
        ],
        "sources": sources,
        "analyzed_source_ids": [url_key(source.url) for source in sources[:12]],
        "evidence_coverage": [
            EvidenceCoverage(
                task_id="T2",
                question="问题二",
                source_indices=[3],
                status="sufficient",
                evidence_strength="strong",
                gaps=[],
                reason="首批已覆盖",
            )
        ],
    }

    with (
        patch("agents.research_team.analyst.settings.USE_FAKE_MODEL", False),
        patch(
            "agents.research_team.analyst.resolve_model",
            return_value=_CapturingModel(bundle, captured),
        ),
        patch("agents.research_team.normalize.record_degradation") as degraded,
    ):
        result = await analyst(state, None)  # type: ignore[arg-type]

    prompt = captured[0]
    assert "来源13" in prompt
    assert "来源14" in prompt
    assert "来源12" not in prompt

    coverage = {item.task_id: item for item in result["evidence_coverage"]}
    # Batch-local [1] must land on its ledger position, not stay as [1].
    assert coverage["T1"].source_indices == [13]
    # The batch saw nothing for T2; the earlier judgement survives instead of regressing.
    assert coverage["T2"].source_indices == [3]
    assert coverage["T2"].status == "sufficient"
    assert "coverage_stale_batch" in [call.args[0] for call in degraded.call_args_list]
    assert result["analyzed_source_ids"] == [url_key(source.url) for source in sources]


@pytest.mark.asyncio
async def test_writer_numbers_the_whole_ledger_and_keeps_high_anchors():
    captured: list[str] = []
    sources = [
        Source(url=f"https://s.example/{index}", title=f"来源{index}") for index in range(1, 31)
    ]
    bundle = ReportBundle(
        title="报告",
        summary="摘要包含关键判断 [30]。",
        findings=["发现甲 [30]", "发现乙 [1]"],
        analysis="分析段落解释机制与风险 [30]。",
        conclusion="结论给出可执行判断 [1]。",
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "意图",
        # Long analysis keeps the writer on the index-only path.
        "analysis": "分析" * 1000,
        "sources": sources,
    }

    with patch(
        "agents.research_team.writer.resolve_model",
        return_value=_CapturingModel(bundle, captured),
    ):
        result = await writer(state, None)  # type: ignore[arg-type]

    assert "[30] 来源30" in captured[0]
    # The ledger numbering must survive clamping, or the citation is silently lost.
    assert "[30]" in result["report"]


def test_normalize_coverage_drops_anchors_from_a_missing_judgement():
    tasks = [ResearchTask(id="T1", question="问题一")]
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="问题一",
            source_indices=[1, 4, 6],
            status="missing",
            evidence_strength="insufficient",
            gaps=["缺少排他数据"],
            reason="只有泛泛提及",
        )
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_coverage(coverage, tasks, source_count=6)

    assert result[0].source_indices == []
    assert result[0].status == "missing"
    assert result[0].evidence_strength == "insufficient"
    degraded.assert_any_call("coverage_missing_with_sources")


def test_normalize_coverage_keeps_anchors_when_the_status_is_not_missing():
    tasks = [ResearchTask(id="T1", question="问题一")]
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="问题一",
            source_indices=[1, 4],
            status="partial",
            evidence_strength="weak",
            gaps=["缺少 2025 年数据"],
            reason="已有两年数据",
        )
    ]

    with patch("agents.research_team.analyst.record_degradation") as degraded:
        result = normalize_coverage(coverage, tasks, source_count=6)

    assert result[0].source_indices == [1, 4]
    assert "coverage_missing_with_sources" not in [call.args[0] for call in degraded.call_args_list]


def test_normalize_subject_coverage_drops_anchors_from_missing_aspects():
    subjects = [ResearchSubject(id="S1", name="对象一")]
    coverage = [
        SubjectEvidenceCoverage(
            subject_id="S1",
            subject="对象一",
            aspects=[
                AspectCoverage(
                    name="产能",
                    status="missing",
                    evidence_strength="insufficient",
                    source_indices=[2],
                    gaps=["无直接数据"],
                    reason="仅提及规划",
                )
            ],
            status="missing",
            evidence_strength="insufficient",
            source_indices=[2],
            gaps=["无直接数据"],
            reason="无",
        )
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_subject_coverage(coverage, subjects, source_count=6)

    assert result[0].aspects[0].source_indices == []
    assert result[0].source_indices == []
    degraded.assert_any_call("subject_coverage_missing_with_sources")


def test_excerpt_limit_shrinks_as_the_window_widens():
    assert excerpt_limit_for(0) == EXCERPT_CHARS
    assert excerpt_limit_for(12) == EXCERPT_CHARS
    assert excerpt_limit_for(20) == EXCERPT_BUDGET_CHARS // 20
    assert excerpt_limit_for(30) == EXCERPT_BUDGET_CHARS // 30


def test_excerpt_limit_never_exceeds_the_configured_cap():
    # A short window must not spend more per source than a full-length read.
    assert all(excerpt_limit_for(count) <= EXCERPT_CHARS for count in (1, 5, 12, 20, 60))


@pytest.mark.asyncio
async def test_analyst_reads_every_source_of_a_normal_haul():
    captured: list[str] = []
    sources = [
        Source(url=f"https://s.example/{index}", title=f"来源{index}", content="正文" * 400)
        for index in range(1, 17)
    ]
    bundle = AnalysisBundle(
        facts=["事实甲 [1]", "事实乙 [1]"],
        patterns=["模式 [1]"],
        contradictions=[],
        gaps=["缺口 [1]"],
        coverage=[
            EvidenceCoverage(
                task_id="T1",
                question="问题一",
                source_indices=[16],
                status="partial",
                evidence_strength="moderate",
                gaps=["缺少 2025 年数据"],
                reason="已有多源数据",
            )
        ],
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "意图",
        "tasks": [ResearchTask(id="T1", question="问题一")],
        "sources": sources,
    }

    with (
        patch("agents.research_team.analyst.settings.USE_FAKE_MODEL", False),
        patch(
            "agents.research_team.analyst.resolve_model",
            return_value=_CapturingModel(bundle, captured),
        ),
    ):
        result = await analyst(state, None)  # type: ignore[arg-type]

    prompt = captured[0]
    # A 16-source haul is normal; the old 12-source window dropped a quarter of it.
    assert "[16] 来源16" in prompt
    assert result["evidence_coverage"][0].source_indices == [16]
    assert len(result["analyzed_source_ids"]) == 16


def test_analyst_prompt_excerpt_budget_is_flat_across_window_sizes():
    def total(count: int) -> int:
        sources = [
            Source(url=f"https://s.example/{index}", title=f"来源{index}", content="甲" * 4000)
            for index in range(1, count + 1)
        ]
        return len(numbered_sources(sources, excerpt_chars=excerpt_limit_for(count)))

    # Wider windows add per-source metadata, not more excerpt text.
    assert total(20) - total(12) < 2500


@pytest.mark.asyncio
async def test_search_reuses_one_connection_pool_per_event_loop():
    from agents.research_team import search as search_module

    first = await search_module._shared_client(60.0)
    second = await search_module._shared_client(30.0)

    # One pool per round: a client per request pays a TLS handshake each time.
    assert first is second
    assert not first.is_closed

    await search_module.close_shared_client()
    assert first.is_closed


def test_search_replaces_the_pool_when_the_event_loop_changes():
    from agents.research_team import search as search_module

    async def touch() -> object:
        return await search_module._shared_client(60.0)

    first = asyncio.run(touch())
    # A pool belongs to the loop that created it; the stale one must be dropped, not
    # reused, and closed rather than leaked.
    second = asyncio.run(touch())
    asyncio.run(search_module.close_shared_client())

    assert first is not second


@pytest.mark.asyncio
async def test_new_turn_reset_is_shared_by_both_entry_points():
    # Two hand-maintained reset lists had already drifted; both paths now read one
    # table, and this asserts they still agree field by field.
    state = {
        "messages": [HumanMessage(content="换一个问题")],
        "analysis": "旧分析",
        "report": "旧报告",
        "tasks": [ResearchTask(id="T1", question="旧问题")],
        "subjects": [ResearchSubject(id="S1", name="旧对象")],
        "constraints": ["旧约束"],
        "plan_notes": "旧理由",
        "evidence_coverage": [],
        "facts": ["旧事实"],
        "analyzed_source_ids": ["a.example/1"],
        "analysis_batches": 2,
        "analysis_patterns": ["旧模式"],
        "analysis_contradictions": ["旧矛盾"],
        "analysis_gaps": ["旧缺口"],
        "rounds": 3,
        "turn_attempted": True,
        "abort_reason": "search_rejected",
        "active_turn_id": "old",
    }

    reset = _turn_state(state, True, "fresh")
    for key, value in _TURN_RESET.items():
        assert reset[key] == value, key

    with patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", True):
        update = await supervisor(dict(state), {"configurable": {"turn_id": "new"}})

    # rounds and plan_notes are written after the reset: the first is this turn's
    # count, the second is why it routed this way.
    assert update["rounds"] == 1
    assert update["plan_notes"]
    for key, value in _TURN_RESET.items():
        if key in {"rounds", "plan_notes"}:
            continue
        assert update[key] == value, key
    assert update["active_turn_id"] == "new"


def test_clamp_anchors_keeps_a_bracketed_year():
    # [2024] is prose, not a citation; dropping it used to leave a hole in the sentence.
    assert clamp_anchors("2024 [1] 年 [2024] 与 [9] 数据", 5) == "2024 [1] 年 [2024] 与  数据"


def test_unknown_authority_sits_between_the_lowest_known_levels():
    source = Source(url="https://a.example/1")

    assert source.authority == AUTHORITY_UNKNOWN
    # Not a real level: it must not outrank known-good sources, nor sink to the floor.
    assert (
        confidence_from_authority(0, 0.0)
        < confidence_from_authority(AUTHORITY_UNKNOWN, 0.0)
        < confidence_from_authority(1, 0.0)
    )


def test_researcher_prompt_states_the_limit_the_adapter_enforces():
    text = _researcher_system()[0].content

    assert "128" in text
    assert "100 个字符" not in text


def test_strict_capability_is_reprobed_after_the_retry_window(monkeypatch):
    from agents.research_team import structured as module

    monkeypatch.setattr(module, "_strict_blocked_until", {})
    now = module.time.monotonic()
    module._mark_strict_unsupported("model@host")

    with patch("agents.research_team.structured.time.monotonic", return_value=now + 1.0):
        assert module._strict_allowed("model@host") is False

    window = module.STRICT_RETRY_AFTER_SECONDS
    with patch("agents.research_team.structured.time.monotonic", return_value=now + window + 1.0):
        # The provider may have recovered; a permanent verdict would keep the process
        # in the weaker JSON mode for its whole lifetime.
        assert module._strict_allowed("model@host") is True


def test_normalize_coverage_flags_a_judgement_that_cites_every_source():
    tasks = [ResearchTask(id="T1", question="问题一")]
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="问题一",
            source_indices=[1, 2, 3],
            status="sufficient",
            evidence_strength="strong",
            gaps=[],
            reason="全部来源都支持",
        )
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        normalize_coverage(coverage, tasks, source_count=3)

    degraded.assert_any_call("coverage_all_sources_cited")


def test_normalize_coverage_does_not_flag_a_selective_judgement():
    tasks = [ResearchTask(id="T1", question="问题一")]
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="问题一",
            source_indices=[2],
            status="partial",
            evidence_strength="moderate",
            gaps=["缺少排名"],
            reason="只有一个来源直接支持",
        )
    ]

    with patch("agents.research_team.analyst.record_degradation") as degraded:
        normalize_coverage(coverage, tasks, source_count=3)

    assert "coverage_all_sources_cited" not in [call.args[0] for call in degraded.call_args_list]


def test_analysis_window_covers_a_full_retrieval_haul():
    # Retrieval cannot collect more than MAX_CANDIDATES pages in a round, so a window
    # at least that wide leaves nothing permanently unread. This is the only place the
    # two constants are compared: raising the candidate cap must also raise the window,
    # otherwise the extra pages silently never reach a judgement.
    from agents.research_team.analyst import MAX_SOURCES_FOR_ANALYSIS
    from agents.research_team.researcher import MAX_CANDIDATES

    assert MAX_SOURCES_FOR_ANALYSIS >= MAX_CANDIDATES


@pytest.mark.asyncio
async def test_analyst_reports_sources_left_unread_beyond_the_window():
    captured: list[str] = []
    sources = [
        Source(url=f"https://s.example/{index}", title=f"来源{index}", content="正文" * 50)
        for index in range(1, 26)
    ]
    bundle = AnalysisBundle(
        facts=["事实甲 [1]", "事实乙 [1]"],
        patterns=["模式 [1]"],
        contradictions=[],
        gaps=["缺口 [1]"],
        coverage=[
            EvidenceCoverage(
                task_id="T1",
                question="问题一",
                source_indices=[1],
                status="partial",
                evidence_strength="moderate",
                gaps=[],
                reason="单源",
            )
        ],
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "意图",
        "tasks": [ResearchTask(id="T1", question="问题一")],
        "sources": sources,
    }

    with (
        patch("agents.research_team.analyst.settings.USE_FAKE_MODEL", False),
        patch(
            "agents.research_team.analyst.resolve_model",
            return_value=_CapturingModel(bundle, captured),
        ),
        patch("agents.research_team.analyst.record_degradation") as degraded,
    ):
        result = await analyst(state, None)  # type: ignore[arg-type]

    # Derived from the constant: raising the window should not fail this test for the
    # wrong reason, only make it read more of the 25.
    from agents.research_team.analyst import MAX_SOURCES_FOR_ANALYSIS

    assert len(result["analyzed_source_ids"]) == MAX_SOURCES_FOR_ANALYSIS
    assert "analysis_sources_left_unread" in [call.args[0] for call in degraded.call_args_list]


@pytest.mark.asyncio
async def test_analyst_does_not_flag_unread_when_the_window_covers_the_haul():
    captured: list[str] = []
    sources = [
        Source(url=f"https://s.example/{index}", title=f"来源{index}", content="正文" * 50)
        for index in range(1, 13)
    ]
    bundle = AnalysisBundle(
        facts=["事实甲 [1]", "事实乙 [1]"],
        patterns=["模式 [1]"],
        contradictions=[],
        gaps=["缺口 [1]"],
        coverage=[
            EvidenceCoverage(
                task_id="T1",
                question="问题一",
                source_indices=[1],
                status="partial",
                evidence_strength="moderate",
                gaps=[],
                reason="单源",
            )
        ],
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "意图",
        "tasks": [ResearchTask(id="T1", question="问题一")],
        "sources": sources,
    }

    with (
        patch("agents.research_team.analyst.settings.USE_FAKE_MODEL", False),
        patch(
            "agents.research_team.analyst.resolve_model",
            return_value=_CapturingModel(bundle, captured),
        ),
        patch("agents.research_team.analyst.record_degradation") as degraded,
    ):
        result = await analyst(state, None)  # type: ignore[arg-type]

    assert len(result["analyzed_source_ids"]) == 12
    assert "analysis_sources_left_unread" not in [call.args[0] for call in degraded.call_args_list]


def _aspect(
    name: str = "产能",
    *,
    status: str = "sufficient",
    strength: str = "strong",
    indices: tuple[int, ...] = (1,),
) -> AspectCoverage:
    return AspectCoverage(
        name=name,
        status=status,  # type: ignore[arg-type]
        evidence_strength=strength,  # type: ignore[arg-type]
        source_indices=list(indices),
        gaps=[],
        reason="r",
    )


def _subject_row(
    subject_id: str = "S1",
    aspects: tuple[AspectCoverage, ...] = (),
    *,
    status: str = "sufficient",
    strength: str = "strong",
    indices: tuple[int, ...] = (1,),
) -> SubjectEvidenceCoverage:
    return SubjectEvidenceCoverage(
        subject_id=subject_id,
        subject="宁德时代",
        aspects=list(aspects),
        status=status,  # type: ignore[arg-type]
        evidence_strength=strength,  # type: ignore[arg-type]
        source_indices=list(indices),
        gaps=[],
        reason="r",
    )


def test_subject_coverage_drops_unknown_and_duplicate_subjects():
    subjects = [ResearchSubject(id="S1", name="宁德时代")]
    coverage = [
        _subject_row("S9", (_aspect(),)),
        _subject_row("S1", (_aspect(),)),
        _subject_row("S1", (_aspect("工厂"),)),
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_subject_coverage(coverage, subjects, source_count=3)

    assert [item.subject_id for item in result] == ["S1"]
    assert [aspect.name for aspect in result[0].aspects] == ["产能"]
    keys = [call.args[0].value for call in degraded.call_args_list]
    assert "subject_coverage_unknown_subject" in keys
    assert "subject_coverage_duplicate_subject" in keys


def test_subject_coverage_skips_nameless_and_duplicate_aspects():
    subjects = [ResearchSubject(id="S1", name="宁德时代")]
    coverage = [
        _subject_row(
            aspects=(_aspect(""), _aspect("产能"), _aspect(" 产能 "), _aspect("工厂")),
        )
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_subject_coverage(coverage, subjects, source_count=3)

    assert [aspect.name for aspect in result[0].aspects] == ["产能", "工厂"]
    keys = [call.args[0].value for call in degraded.call_args_list]
    assert "subject_coverage_empty_aspect" in keys
    assert "subject_coverage_duplicate_aspect" in keys


def test_subject_coverage_status_follows_its_aspects():
    subjects = [
        ResearchSubject(id="S1", name="宁德时代"),
        ResearchSubject(id="S2", name="比亚迪"),
        ResearchSubject(id="S3", name="国轩高科"),
    ]
    coverage = [
        _subject_row(
            "S1",
            (_aspect(status="conflicting", strength="moderate"), _aspect("工厂", strength="weak")),
            status="conflicting",
        ),
        _subject_row("S2", (_aspect(status="partial", strength="weak"),), status="partial"),
        _subject_row("S3", (_aspect(), _aspect("工厂", strength="moderate")), status="sufficient"),
    ]

    result = normalize_subject_coverage(coverage, subjects, source_count=3)

    # conflicting wins over the others; strength is the weakest aspect, not the status'.
    assert (result[0].status, result[0].evidence_strength) == ("conflicting", "weak")
    assert (result[1].status, result[1].evidence_strength) == ("partial", "weak")
    assert (result[2].status, result[2].evidence_strength) == ("sufficient", "moderate")


def test_subject_coverage_without_aspects_is_missing_and_drops_its_sources():
    subjects = [ResearchSubject(id="S1", name="宁德时代")]

    result = normalize_subject_coverage([_subject_row("S1", ())], subjects, source_count=3)

    assert (result[0].status, result[0].evidence_strength) == ("missing", "insufficient")
    assert result[0].source_indices == []


def test_subject_coverage_merges_aspect_sources_into_the_subject():
    subjects = [ResearchSubject(id="S1", name="宁德时代")]
    coverage = [
        _subject_row(
            aspects=(_aspect(indices=(2, 3)), _aspect("工厂", indices=(4,))),
            indices=(1,),
        )
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_subject_coverage(coverage, subjects, source_count=5)

    assert result[0].source_indices == [1, 2, 3, 4]
    assert "subject_coverage_aspect_source_merge" in [
        c.args[0].value for c in degraded.call_args_list
    ]


def test_merge_subject_coverage_keeps_a_judgement_the_new_batch_cannot_see():
    previous = [
        _subject_row("S1", (_aspect(),), status="sufficient", strength="strong", indices=(1,))
    ]
    # A later batch reports the same object as missing because its evidence is elsewhere.
    current = [_subject_row("S1", (), status="missing", strength="insufficient", indices=())]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        merged = merge_subject_coverage(previous, current)

    assert merged[0].source_indices == [1]
    assert merged[0].status == "sufficient"
    assert "subject_coverage_stale_batch" in [c.args[0].value for c in degraded.call_args_list]


def test_missing_coverage_with_strength_reports_one_reason_not_two():
    # The "missing clears sources" rule runs first, so the strength repair is now the
    # *no sources* rule; the stale missing-strength key must not appear.
    tasks = [ResearchTask(id="T1", question="问题一")]
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="问题一",
            source_indices=[1],
            status="missing",
            evidence_strength="moderate",
            gaps=[],
            reason="r",
        )
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_coverage(coverage, tasks, source_count=3)

    keys = [call.args[0].value for call in degraded.call_args_list]
    assert result[0].evidence_strength == "insufficient"
    assert result[0].source_indices == []
    assert "coverage_missing_with_sources" in keys
    assert "coverage_strength_without_sources" in keys


def test_conflicting_coverage_with_strong_evidence_is_capped():
    tasks = [ResearchTask(id="T1", question="问题一")]
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="问题一",
            source_indices=[1],
            status="conflicting",
            evidence_strength="strong",
            gaps=[],
            reason="r",
        )
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_coverage(coverage, tasks, source_count=3)

    assert result[0].evidence_strength == "moderate"
    assert "coverage_conflicting_strength_mismatch" in [
        call.args[0].value for call in degraded.call_args_list
    ]


def test_duplicate_coverage_rows_are_dropped_not_merged():
    tasks = [ResearchTask(id="T1", question="问题一")]
    coverage = [
        EvidenceCoverage(
            task_id="T1",
            question="问题一",
            source_indices=[1],
            status="sufficient",
            evidence_strength="strong",
            gaps=[],
            reason="first",
        ),
        EvidenceCoverage(
            task_id="T1",
            question="问题一",
            source_indices=[2],
            status="partial",
            evidence_strength="weak",
            gaps=[],
            reason="second",
        ),
    ]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_coverage(coverage, tasks, source_count=3)

    assert len(result) == 1
    assert result[0].reason == "first"
    assert "coverage_duplicate_task" in [c.args[0].value for c in degraded.call_args_list]


def test_subject_coverage_drops_an_out_of_range_aspect_index():
    subjects = [ResearchSubject(id="S1", name="宁德时代")]

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_subject_coverage(
            [_subject_row(aspects=(_aspect(indices=(9,)),))], subjects, source_count=3
        )

    assert result[0].aspects[0].source_indices == []
    assert result[0].aspects[0].evidence_strength == "insufficient"
    assert "subject_coverage_invalid_source_index" in [
        call.args[0].value for call in degraded.call_args_list
    ]


def test_evidence_statements_drop_empty_and_sourceless_rows():
    def row(statement: str, indices: list[int]) -> EvidenceStatement:
        return EvidenceStatement(
            statement=statement,
            kind="point",
            period="2024",
            periods=["2024"],
            value_type="actual",
            metric="销量",
            unit="万辆",
            source_indices=indices,
            trend_origin="unknown",
        )

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_evidence_statements(
            [row("   ", [1]), row("有数据", [9]), row("保留 [1]", [1])], source_count=3
        )

    assert [item.statement for item in result] == ["保留 [1]"]
    keys = [call.args[0].value for call in degraded.call_args_list]
    assert "evidence_statement_empty" in keys
    assert "evidence_statement_without_sources" in keys


def test_evidence_statement_keeps_the_single_period_at_the_front():
    item = EvidenceStatement(
        statement="2024 年销量 [1]",
        kind="point",
        period="2024",
        periods=["2025"],
        value_type="actual",
        metric="销量",
        unit="万辆",
        source_indices=[1],
        trend_origin="unknown",
    )

    result = normalize_evidence_statements([item], source_count=3)

    # The declared single period wins the front slot; the list keeps both.
    assert result[0].period == "2024"
    assert result[0].periods == ["2024", "2025"]


def test_evidence_statements_downgrade_shapes_their_periods_cannot_support():
    def row(kind: str, value_type: str, periods: list[str]) -> EvidenceStatement:
        return EvidenceStatement(
            statement="句子 [1]",
            kind=kind,  # type: ignore[arg-type]
            period=periods[0],
            periods=periods,
            value_type=value_type,  # type: ignore[arg-type]
            metric="销量",
            unit="万辆",
            source_indices=[1],
            trend_origin="unknown",
        )

    with patch("agents.research_team.normalize.record_degradation") as degraded:
        result = normalize_evidence_statements(
            [
                row("change", "actual", ["2024"]),  # one period cannot be a change
                row("forecast", "actual", ["2024"]),  # actual data is not a forecast
            ],
            source_count=3,
        )

    assert [item.kind for item in result] == ["point", "point"]
    keys = [call.args[0].value for call in degraded.call_args_list]
    assert "evidence_statement_change_downgraded" in keys
    assert "evidence_statement_forecast_downgraded" in keys


def test_merge_statements_appends_only_statements_it_has_not_seen():
    def row(statement: str, periods: list[str]) -> EvidenceStatement:
        return EvidenceStatement(
            statement=statement,
            kind="point",
            period=periods[0],
            periods=periods,
            value_type="actual",
            metric="销量",
            unit="万辆",
            source_indices=[1],
            trend_origin="unknown",
        )

    previous = [row("销量 100 [1]", ["2024"])]
    current = [row("销量  100 [1]", ["2024"]), row("销量 120 [1]", ["2025"])]

    merged = merge_statements(previous, current)

    # Same metric/period/text collapses even with different whitespace.
    assert [item.statement for item in merged] == ["销量 100 [1]", "销量 120 [1]"]


@pytest.mark.parametrize(
    ("raw_count", "failures", "filters", "expected"),
    [
        (
            0,
            2,
            {"invalid_item": 0, "invalid_url": 0, "empty_content": 0, "duplicate": 0},
            "未返回结果",
        ),
        (0, 0, {"invalid_item": 0, "invalid_url": 0, "empty_content": 0, "duplicate": 0}, "空结果"),
        (3, 0, {"invalid_item": 1, "invalid_url": 1, "empty_content": 1, "duplicate": 0}, "全无效"),
        (3, 0, {"invalid_item": 0, "invalid_url": 0, "empty_content": 0, "duplicate": 3}, "全重复"),
        (
            3,
            0,
            {"invalid_item": 1, "invalid_url": 0, "empty_content": 0, "duplicate": 0},
            "过滤后无可用结果",
        ),
    ],
)
def test_empty_run_reason_explains_why_nothing_survived(
    raw_count: int, failures: int, filters: dict[str, int], expected: str
):
    assert _empty_run_reason(raw_count, failures, filters) == expected


def test_collect_batches_counts_failures_and_reraises_cancellation():
    usable, failures, rejected = _collect_batches(
        ["a", "b", "c"],
        [["hit"], SearchRejected("denied"), RuntimeError("boom")],
    )

    assert usable == [("a", ["hit"])]
    assert (failures, rejected) == (2, 1)

    with pytest.raises(asyncio.CancelledError):
        _collect_batches(["a"], [asyncio.CancelledError()])


def test_constraints_merge_within_a_turn_and_reset_across_turns():
    decision = RoutingDecision(
        intent="x",
        is_research=True,
        sub_questions=[],
        constraints=["地区：中国", "时间：2024"],
        next="researcher",
        reason="r",
    )

    # Inside a turn a later round may under-report, so the parsed set is merged in.
    assert _constraints_for(["排除：乘用车"], decision, False) == [
        "排除：乘用车",
        "地区：中国",
        "时间：2024",
    ]
    # A new request must not inherit the previous one's scope.
    assert _constraints_for(["排除：乘用车"], decision, True) == ["地区：中国", "时间：2024"]
