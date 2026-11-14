"""Researcher: multi-query discovery with provenance taken from the Search API."""

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.research_team.helpers import (
    clean_title,
    clean_url,
    confidence_from_authority,
    dedupe_sources,
    resolve_model,
    structured_output,
    url_key,
)
from agents.research_team.search import SearchRejected, fetch, search
from agents.research_team.state import ResearchState, Source
from core import settings
from core.metrics import record_degradation, record_fetch, record_search

logger = logging.getLogger(__name__)

MAX_QUERIES = 4
RESULTS_PER_QUERY = 8
MAX_CANDIDATES = 16
FETCH_TOP_K = 3
MIN_AUTHORITY_FOR_FETCH = 3
MIN_RERANK_FOR_FETCH = 1.5
MAX_CONTENT_CHARS = 4000
MAX_RAW_EXCERPT = 400
MAX_QUERY_CHARS = 128
FALLBACK_MIN_AUTHORITY = 2
FALLBACK_MIN_RERANK = 1.5
FALLBACK_REASON = "评分阈值降级（相关性判定不可用）"

# Query plans and relevance judgements are small outputs; cap them so a degenerate
# runaway generation dies cheap instead of burning the full shared LLM_MAX_TOKENS.
QUERY_PLAN_MAX_TOKENS = 2048
JUDGE_MAX_TOKENS = 4096

RESEARCHER_PROMPT = """你是行业与竞品调研团队的检索员。你只做一件事：针对分配到的子问题采集原始素材，
并如实记录来源。你不下结论，也不写报告。

采集要求：
- 优先采集可验证的事实：数据、公告、财报、政策文件、产品参数、市场份额、价格。
- 每条 query 必须具体，带上年份、地区、公司名或产品名，且不超过 100 个字符。
- 严格围绕给定子问题检索，不要因为用户描述里的背景信息扩大范围。

来源可信度由检索接口自带：authority（-1 未知，3 以上为高权威）与 rerank_score
（0-3，1.5 以上为高质量）。你不需要自己给来源打分，只需要判断相关性。"""


class QueryPlan(BaseModel):
    queries: list[str] = Field(
        min_length=1, max_length=MAX_QUERIES, description="具体可检索的搜索语句，每条不超过 100 字"
    )


class SourceJudgement(BaseModel):
    url: str
    keep: bool = Field(description="与子问题直接相关且内容有效则保留")
    reason: str = Field(default="", description="一句话判定理由")


class SourceJudgements(BaseModel):
    sources: list[SourceJudgement]


def _task_brief(state: ResearchState) -> str:
    tasks = state.get("tasks") or []
    questions = "\n".join(f"- {t.question}" for t in tasks) or "- （未拆解，按用户原问题检索）"
    constraints = "\n".join(f"- {c}" for c in state.get("constraints") or []) or "（无）"
    known = dedupe_sources(state.get("sources"))
    known_urls = "\n".join(f"- {s.url}" for s in known[-15:]) or "（无）"
    last_user = ""
    for message in reversed(state["messages"]):
        if message.type == "human":
            last_user = str(message.content)
            break
    excerpt = last_user[:MAX_RAW_EXCERPT] + ("…" if len(last_user) > MAX_RAW_EXCERPT else "")
    return f"""本轮意图：{state.get("intent") or "（未解析）"}
待调研子问题：
{questions}

必须遵守的范围与交付约束：
{constraints}

已采集过的来源（不要重复）：
{known_urls}

用户原始描述（节选，仅供参考，不得据此增减子问题，也不得把无关内容当作检索目标）：
{excerpt}"""


def _judge_brief(state: ResearchState) -> str:
    """Anchor for relevance judgement: what was asked, under which limits."""
    tasks = state.get("tasks") or []
    questions = "\n".join(f"- {t.question}" for t in tasks) or "- （未拆解）"
    constraints = "\n".join(f"- {c}" for c in state.get("constraints") or []) or "（无）"
    return f"""本轮调研任务：
{questions}

必须遵守的范围与交付约束：
{constraints}"""


def _fallback_queries(state: ResearchState) -> list[str]:
    """Use the decomposed task text as-is when query planning is unavailable."""
    questions = [
        task.question.strip() for task in state.get("tasks") or [] if task.question.strip()
    ]
    if not questions:
        for message in reversed(state["messages"]):
            if message.type == "human":
                questions = [str(message.content).strip()]
                break
    return [question[:MAX_QUERY_CHARS] for question in questions if question][:MAX_QUERIES]


async def _plan_queries(state: ResearchState, config: RunnableConfig) -> list[str]:
    try:
        model: Any = resolve_model(config)
        planner = structured_output(model, QueryPlan, "planner", max_tokens=QUERY_PLAN_MAX_TOKENS)
        messages = [
            SystemMessage(content=RESEARCHER_PROMPT),
            SystemMessage(content=_task_brief(state)),
        ]
        plan: QueryPlan = await planner.ainvoke(messages)
    except Exception as exc:
        logger.warning("Query planning failed, using task text instead: %s", exc)
        record_degradation("query_planning_failed")
        return _fallback_queries(state)

    queries = [q.strip() for q in plan.queries if q.strip()][:MAX_QUERIES]
    if not queries:
        logger.warning("Query planning returned no usable query, using task text instead")
        record_degradation("query_planning_empty")
        return _fallback_queries(state)
    return queries


def _score_passes_threshold(candidate: dict[str, Any]) -> bool:
    return (
        candidate["authority"] >= FALLBACK_MIN_AUTHORITY
        or candidate["rerank_score"] >= FALLBACK_MIN_RERANK
    )


async def _judge(
    candidates: list[dict[str, Any]], state: ResearchState, config: RunnableConfig
) -> SourceJudgements:
    listing = "\n\n".join(
        f"候选 {index}\nURL: {item['url']}\n标题: {item['title']}\n"
        f"站点: {item['site_name'] or '未知'} | 发布: {item['published_at'] or '未知'}\n"
        f"摘要: {item['summary'][:500]}"
        for index, item in enumerate(candidates, start=1)
    )
    try:
        model: Any = resolve_model(config)
        judge = structured_output(model, SourceJudgements, "judge", max_tokens=JUDGE_MAX_TOKENS)
        messages = [
            SystemMessage(content=RESEARCHER_PROMPT),
            SystemMessage(content=_judge_brief(state)),
            SystemMessage(
                content="下面是候选检索结果，请逐条判断是否与任务直接相关。"
                "不需要评估可信度，接口已提供 authority 与 rerank_score。\n\n" + listing
            ),
        ]
        return await judge.ainvoke(messages)
    except Exception as exc:
        logger.warning("Relevance judgement failed, falling back to score thresholds: %s", exc)
        record_degradation("judge_fallback")
        return SourceJudgements(
            sources=[
                SourceJudgement(url=candidate["url"], keep=True, reason=FALLBACK_REASON)
                for candidate in candidates
                if _score_passes_threshold(candidate)
            ]
        )


def _candidate(item: dict[str, Any], query: str) -> dict[str, Any]:
    authority = item.get("authority")
    return {
        "url": clean_url(str(item.get("url") or "")),
        "title": clean_title(str(item.get("title") or "")),
        "snippet": str(item.get("snippet") or ""),
        "summary": str(item.get("summary") or ""),
        "site_name": str(item.get("site_name") or ""),
        "published_at": str(item.get("date") or ""),
        "authority": int(authority) if authority is not None else -1,
        "rerank_score": float(item.get("rerank_score") or 0.0),
        "query": query,
    }


def _to_source(candidate: dict[str, Any]) -> Source:
    return Source(
        url=candidate["url"],
        title=candidate["title"],
        snippet=candidate["snippet"],
        summary=candidate["summary"],
        site_name=candidate["site_name"],
        published_at=candidate["published_at"],
        authority=candidate["authority"],
        rerank_score=candidate["rerank_score"],
        query=candidate["query"],
        confidence=confidence_from_authority(candidate["authority"], candidate["rerank_score"]),
    )


async def _deep_read(sources: list[Source]) -> int:
    targets = [
        source
        for source in sources
        if source.authority >= MIN_AUTHORITY_FOR_FETCH
        or source.rerank_score >= MIN_RERANK_FOR_FETCH
    ][:FETCH_TOP_K]
    if not targets:
        return 0

    pages = await asyncio.gather(*(fetch(source.url) for source in targets), return_exceptions=True)
    fetched = 0
    for source, page in zip(targets, pages, strict=True):
        if isinstance(page, BaseException):
            logger.warning("Fetch failed for %s: %s", source.url, page)
            record_fetch(ok=False)
            continue
        if not page:
            continue
        source.content = page["content"][:MAX_CONTENT_CHARS]
        fetched += 1
    return fetched


async def researcher(state: ResearchState, config: RunnableConfig) -> dict[str, Any]:
    if settings.USE_FAKE_MODEL:
        return {"turn_attempted": True, "messages": [AIMessage(content="（假模型模式）跳过检索。")]}

    queries = await _plan_queries(state, config)
    intent = state.get("intent") or None
    batches = await asyncio.gather(
        *(search(query, RESULTS_PER_QUERY, ori_question=intent) for query in queries),
        return_exceptions=True,
    )

    failures = 0
    rejected = 0
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for query, batch in zip(queries, batches, strict=True):
        if isinstance(batch, BaseException):
            failures += 1
            if isinstance(batch, SearchRejected):
                rejected += 1
            logger.warning("Search failed for %r: %s", query, batch)
            record_search(ok=False)
            continue
        for item in batch:
            candidate = _candidate(item, query)
            key = url_key(candidate["url"])
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    candidates = candidates[:MAX_CANDIDATES]

    if not candidates:
        detail = f"（{failures} 条查询调用失败）" if failures else ""
        update: dict[str, Any] = {
            "turn_attempted": True,
            "messages": [
                AIMessage(content=f"本轮检索未返回结果{detail}，将尝试重新拆解查询条件。")
            ],
        }
        if rejected and rejected == len(queries):
            # Every query was refused, so the next round would fail identically.
            record_degradation("search_rejected")
            update["abort_reason"] = "search_rejected"
        return update

    judgements = await _judge(candidates, state, config)
    keep = {url_key(item.url) for item in judgements.sources if item.keep}
    collected = [_to_source(c) for c in candidates if url_key(c["url"]) in keep]

    if not collected:
        return {
            "turn_attempted": True,
            "messages": [AIMessage(content="本轮候选结果均被判定为不相关，未纳入来源库。")],
        }

    fetched = await _deep_read(collected)
    kept = len(collected)
    average = sum(s.confidence for s in collected) / kept
    failure_note = f"，{failures} 条查询失败" if failures else ""
    summary = (
        f"检索完成：执行 {len(queries)} 条查询{failure_note}，保留 {kept} 条来源"
        f"（平均可信度 {average:.2f}），深读 {fetched} 篇正文。\n"
        + "\n".join(f"- [{s.title or s.url}]({s.url})" for s in collected[:5])
    )
    return {
        "sources": collected,
        "turn_attempted": True,
        "messages": [AIMessage(content=summary)],
    }
