"""Researcher: multi-query discovery with provenance taken from the Search API."""

import asyncio
import logging
import math
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agents.research_team.contracts import Source
from agents.research_team.prompts import load_prompt, render_prompt
from agents.research_team.search import SearchRejected, fetch, is_fetchable, search
from agents.research_team.sources import (
    clean_fetched_markdown,
    clean_title,
    clean_url,
    confidence_from_authority,
    dedupe_sources,
    url_key,
)
from agents.research_team.state import ResearchState
from agents.research_team.structured import structured_output
from core import resolve_model, settings
from core.metrics import record_degradation, record_fetch, record_filter, record_search

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

# Query plans and relevance judgements are small outputs; cap them so a degenerate
# runaway generation dies cheap instead of burning the full shared LLM_MAX_TOKENS.
QUERY_PLAN_MAX_TOKENS = 2048
JUDGE_MAX_TOKENS = 4096

RESEARCHER_PROMPT = load_prompt("research_team.researcher.instructions")
_TASK_BRIEF_PROMPT = load_prompt("research_team.researcher.task_brief")
_JUDGE_BRIEF_PROMPT = load_prompt("research_team.researcher.judge_brief")
_JUDGE_RESULTS_PROMPT = load_prompt("research_team.researcher.judge_results")


class QueryPlan(BaseModel):
    queries: list[str] = Field(
        min_length=1, max_length=MAX_QUERIES, description="具体可检索的搜索语句，每条不超过 100 字"
    )


class SourceJudgement(BaseModel):
    # Only the verdict is read; a "reason" field here cost ~400 output tokens per
    # judgement round and was discarded by every consumer.
    url: str
    keep: bool = Field(description="与子问题直接相关且内容有效则保留")


class SourceJudgements(BaseModel):
    sources: list[SourceJudgement]


def _task_brief(state: ResearchState) -> str:
    tasks = state.get("tasks") or []
    questions = "\n".join(f"- {t.question}" for t in tasks) or "- （未拆解，按用户原问题检索）"
    subjects = state.get("subjects") or []
    subject_lines = (
        "\n".join(
            f"- {subject.name}（别名：{'、'.join(subject.aliases) or '无'}）"
            for subject in subjects
        )
        or "- （未单独识别）"
    )
    constraints = "\n".join(f"- {c}" for c in state.get("constraints") or []) or "（无）"
    known = dedupe_sources(state.get("sources"))
    known_urls = "\n".join(f"- {s.url}" for s in known[-15:]) or "（无）"
    last_user = ""
    for message in reversed(state["messages"]):
        if message.type == "human":
            last_user = str(message.content)
            break
    excerpt = last_user[:MAX_RAW_EXCERPT] + ("…" if len(last_user) > MAX_RAW_EXCERPT else "")
    return str(
        render_prompt(
            _TASK_BRIEF_PROMPT,
            intent=state.get("intent") or "（未解析）",
            questions=questions,
            subjects=subject_lines,
            constraints=constraints,
            known_urls=known_urls,
            excerpt=excerpt,
        )[0].content
    )


def _judge_brief(state: ResearchState) -> str:
    """Anchor for relevance judgement: what was asked, under which limits."""
    tasks = state.get("tasks") or []
    questions = "\n".join(f"- {t.question}" for t in tasks) or "- （未拆解）"
    constraints = "\n".join(f"- {c}" for c in state.get("constraints") or []) or "（无）"
    return str(
        render_prompt(
            _JUDGE_BRIEF_PROMPT,
            questions=questions,
            constraints=constraints,
        )[0].content
    )


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
        messages = render_prompt(RESEARCHER_PROMPT) + [SystemMessage(content=_task_brief(state))]
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
        messages = (
            render_prompt(RESEARCHER_PROMPT)
            + [SystemMessage(content=_judge_brief(state))]
            + render_prompt(_JUDGE_RESULTS_PROMPT, listing=listing)
        )
        return await judge.ainvoke(messages)
    except Exception as exc:
        logger.warning("Relevance judgement failed, falling back to score thresholds: %s", exc)
        record_degradation("judge_fallback")
        return SourceJudgements(
            sources=[
                SourceJudgement(url=candidate["url"], keep=True)
                for candidate in candidates
                if _score_passes_threshold(candidate)
            ]
        )


def _safe_score(item: Mapping[str, Any], field: str, default: int | float) -> int | float:
    value = item.get(field)
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
        if not math.isfinite(number):
            raise ValueError
        if field == "authority":
            if not number.is_integer() or not -1 <= number <= 4:
                raise ValueError
            return int(number)
        if not 0 <= number <= 3:
            raise ValueError
        return number
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid Search %s field of type %s", field, type(value).__name__)
        record_degradation("search_result_invalid_field")
        return default


def _candidate(item: Any, query: str) -> dict[str, Any] | None:
    if not isinstance(item, Mapping):
        logger.warning("Skipping Search result with type %s", type(item).__name__)
        record_degradation("search_result_invalid_item")
        return None
    return {
        "url": clean_url(str(item.get("url") or "")),
        "title": clean_title(str(item.get("title") or "")),
        "snippet": str(item.get("snippet") or ""),
        "summary": str(item.get("summary") or ""),
        "site_name": str(item.get("site_name") or ""),
        "published_at": str(item.get("date") or ""),
        "authority": _safe_score(item, "authority", -1),
        "rerank_score": _safe_score(item, "rerank_score", 0.0),
        "query": query,
    }


def _filter_candidates(
    batches: list[tuple[str, list[Any]]], known_sources: list[Source]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove only structurally unusable or duplicate Search results.

    Quality scores are deliberately not consulted here; relevance remains the
    Judge's responsibility and score-derived confidence is assigned unchanged.
    """
    counts = {
        "invalid_item": 0,
        "invalid_url": 0,
        "empty_content": 0,
        "duplicate": 0,
        "over_limit": 0,
    }
    candidates: list[dict[str, Any]] = []
    seen = {url_key(source.url) for source in known_sources if url_key(source.url)}
    for query, batch in batches:
        for item in batch:
            candidate = _candidate(item, query)
            if candidate is None:
                counts["invalid_item"] += 1
                continue
            if not is_fetchable(candidate["url"]):
                counts["invalid_url"] += 1
                continue
            if not any(candidate[field].strip() for field in ("title", "summary", "snippet")):
                counts["empty_content"] += 1
                continue
            key = url_key(candidate["url"])
            if not key or key in seen:
                counts["duplicate"] += 1
                continue
            seen.add(key)
            candidates.append(candidate)

    if len(candidates) > MAX_CANDIDATES:
        counts["over_limit"] = len(candidates) - MAX_CANDIDATES
        candidates.sort(
            key=lambda candidate: (candidate["rerank_score"], candidate["authority"]),
            reverse=True,
        )
        candidates = candidates[:MAX_CANDIDATES]
    return candidates, counts


def _filter_note(counts: dict[str, int], raw_count: int, candidate_count: int) -> str:
    labels = {
        "invalid_item": "结构无效",
        "invalid_url": "URL 无效",
        "empty_content": "内容为空",
        "duplicate": "重复",
        "over_limit": "超出限额",
    }
    details = "，".join(f"{labels[key]} {value} 条" for key, value in counts.items() if value)
    return f"候选过滤：原始 {raw_count} 条，去重并限额后交给 Judge {candidate_count} 条" + (
        f"（{details}）" if details else ""
    )


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
            if isinstance(page, asyncio.CancelledError):
                raise page
            if isinstance(page, Exception):
                logger.warning("Fetch failed for %s: %s", source.url, page)
                record_fetch(ok=False)
                continue
            raise page
        if not page:
            continue
        source.content = clean_fetched_markdown(page["content"])[:MAX_CONTENT_CHARS]
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
    batches_for_filter: list[tuple[str, list[Any]]] = []
    for query, batch in zip(queries, batches, strict=True):
        if isinstance(batch, BaseException):
            if isinstance(batch, asyncio.CancelledError):
                raise batch
            if isinstance(batch, Exception):
                failures += 1
                if isinstance(batch, SearchRejected):
                    rejected += 1
                logger.warning("Search failed for %r: %s", query, batch)
                record_search(ok=False)
                continue
            raise batch
        batches_for_filter.append((query, batch))

    raw_count = sum(len(batch) for _, batch in batches_for_filter)
    candidates, filter_counts = _filter_candidates(batches_for_filter, state.get("sources") or [])
    for reason, count in filter_counts.items():
        record_filter(reason, count)
    filter_note = _filter_note(filter_counts, raw_count, len(candidates))

    if not candidates:
        if raw_count == 0 and failures:
            outcome = "未返回结果"
        elif raw_count == 0:
            outcome = "空结果"
        elif (
            filter_counts["invalid_item"]
            + filter_counts["invalid_url"]
            + filter_counts["empty_content"]
            == raw_count
        ):
            outcome = "全无效"
        elif filter_counts["duplicate"] == raw_count:
            outcome = "全重复"
        else:
            outcome = "过滤后无可用结果"
        detail = f"（{failures} 条查询调用失败）" if failures else ""
        update: dict[str, Any] = {
            "turn_attempted": True,
            "messages": [AIMessage(content=f"本轮检索{outcome}{detail}。{filter_note}")],
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
        f"（平均来源评分 {average:.2f}），深读 {fetched} 篇正文。\n"
        f"{filter_note}\n" + "\n".join(f"- [{s.title or s.url}]({s.url})" for s in collected[:5])
    )
    return {
        "sources": dedupe_sources([*(state.get("sources") or []), *collected]),
        "turn_attempted": True,
        "messages": [AIMessage(content=summary)],
    }
