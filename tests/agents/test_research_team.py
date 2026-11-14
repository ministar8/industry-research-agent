"""Unit tests for the research graph's pure decision logic."""

import json
import re
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from openai import LengthFinishReasonError

from agents.research_team.analyst import AnalysisBundle, analyst
from agents.research_team.helpers import (
    EXCERPT_CHARS,
    clamp_anchors,
    clean_url,
    confidence_from_authority,
    dedupe_sources,
    excerpt,
    inline_schema,
    numbered_sources,
    structured_output,
    url_key,
)
from agents.research_team.researcher import (
    FALLBACK_REASON,
    SourceJudgements,
    _deep_read,
    _judge,
    _plan_queries,
    _task_brief,
    researcher,
)
from agents.research_team.search import SearchRejected, fetch, search
from agents.research_team.state import ResearchTask, Source
from agents.research_team.supervisor import (
    RoutingDecision,
    _enforce,
    finalize,
    route,
    supervisor,
)
from agents.research_team.writer import (
    ReportBundle,
    TableBlock,
    TableRequiredReport,
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


def test_clamp_anchors_drops_unknown_indices():
    text = "成立 [1]，存疑 [3]。"
    assert clamp_anchors(text, 2) == "成立 [1]，存疑 。"


def test_route_returns_recorded_action():
    assert route({"next_action": "researcher", "rounds": 1}) == "researcher"


def test_route_forces_finalize_past_round_limit():
    assert route({"next_action": "researcher", "rounds": 9, "max_rounds": 8}) == "finalize"


def test_route_defaults_to_finalize_without_action():
    assert route({}) == "finalize"


def decision(target: str) -> RoutingDecision:
    return RoutingDecision(
        intent="测试", is_research=True, sub_questions=[], next=target, reason="r"
    )


def test_analyst_requires_sources():
    state = {"messages": [HumanMessage(content="调研")], "sources": []}
    assert _enforce(decision("analyst"), state) == "researcher"


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
        self, schema: type, method: str | None = None, include_raw: bool = False
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
        self, schema: type, method: str | None = None, include_raw: bool = False
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
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "储能电芯竞争格局",
        "constraints": ["时间范围：2025 年", "不含上游锂矿"],
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
        "sources": [make_source("https://a.example/1")],
        "analysis": "旧分析",
        "report": "# 旧报告",
    }

    with (
        patch("agents.research_team.supervisor.settings.USE_FAKE_MODEL", False),
        patch("agents.research_team.supervisor.resolve_model", return_value=_StubModel(decision)),
    ):
        update = await supervisor(state, None)  # type: ignore[arg-type]

    assert update["next_action"] == "analyst"
    assert update["report"] == ""


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
        end_turn(token, "ok")


@pytest.mark.asyncio
async def test_fetch_returns_none_when_content_is_empty():
    async def fake_post(path: str, payload: dict, timeout: float) -> dict:
        return {"title": "空页", "content": ""}

    with patch("agents.research_team.search._post", side_effect=fake_post):
        assert await fetch("https://example.com/x") is None


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
    assert all(item.reason == FALLBACK_REASON for item in result.sources)


def test_excerpt_leaves_short_text_untouched():
    text = "短文本，不需要标注"
    assert excerpt(text) == text


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
        self, schema: type, method: str | None = None, include_raw: bool = False
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
        self, schema: type, method: str | None = None, include_raw: bool = False
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
        intent="测试", is_research=True, next="researcher", reason="需要检索"
    )
    model = _StrictRejectingModel(decision)

    result = await structured_output(model, RoutingDecision).ainvoke([])

    assert result is decision
    assert model.methods == ["json_schema", "json_mode"]


@pytest.mark.asyncio
async def test_structured_output_does_not_swallow_unrelated_errors():
    class _Broken:
        def with_structured_output(
            self, schema: type, method: str | None = None, include_raw: bool = False
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
        self, schema: type, method: str | None = None, include_raw: bool = False
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
    decision = RoutingDecision(intent="测试", is_research=True, next="writer", reason="已有分析")
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
        self, schema: type, method: str | None = None, include_raw: bool = False
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


@pytest.mark.asyncio
async def test_analyst_retries_when_the_fact_base_is_blank():
    captured: list[str] = []
    thin = AnalysisBundle(facts=[" ", " "], patterns=["模式"], contradictions=[], gaps=["缺口"])
    good = AnalysisBundle(
        facts=["事实甲 [1]", "事实乙 [1]"],
        patterns=["模式 [1]"],
        contradictions=[],
        gaps=["缺口 [1]"],
    )
    state = {
        "messages": [HumanMessage(content="调研")],
        "intent": "储能",
        "constraints": [],
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


@pytest.mark.asyncio
async def test_finalize_does_not_claim_the_round_cap_when_it_was_not_hit():
    result = await finalize(
        {
            "messages": [HumanMessage(content="调研")],
            "sources": [make_source("https://a.example/1")],
            "analysis": "## 事实基线\n- 已确证 [1]",
            "rounds": 2,
            "max_rounds": 5,
        }
    )
    content = result["messages"][0].content

    assert "未生成成稿报告" in content
    assert "达到调度上限" not in content
