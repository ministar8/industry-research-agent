"""Per-turn metrics for the research graph.

Nodes record into a context-local collector; the service opens one per HTTP request
and emits a single structured line when the turn ends. Counters live outside
`ResearchState` on purpose: they describe one request, not the conversation, and
writing them into the checkpoint would grow the row on every turn for data nobody
reads back.

The collector object is shared by reference across the tasks LangGraph spawns, so
writes made inside a node are visible to the caller that opened the turn.
"""

import logging
from collections import deque
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    thread_id: str = "-"
    user_id: str = "-"
    model: str = "-"
    query_summary: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_perf: float = field(default_factory=perf_counter)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    search_calls: int = 0
    fetch_calls: int = 0
    search_failed: int = 0
    fetch_failed: int = 0
    fetch_skipped: int = 0
    degradations: list[str] = field(default_factory=list)
    filter_counts: dict[str, int] = field(default_factory=dict)
    sources_count: int = 0
    report_title: str = ""
    error_detail: str = ""
    # Per-node attribution: {"judge": {"calls": 1, "in": 9000, "out": 600}, ...}
    calls: dict[str, dict[str, int]] = field(default_factory=dict)

    def snapshot(self, status: str) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "status": status,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "model": self.model,
            "query_summary": self.query_summary,
            "report_title": self.report_title,
            "sources_count": self.sources_count,
            "error_detail": self.error_detail,
            "elapsed": round(perf_counter() - self.started_perf, 1),
            "llm_calls": self.llm_calls,
            "tokens_in": self.input_tokens,
            "tokens_out": self.output_tokens,
            "search_calls": self.search_calls,
            "fetch_calls": self.fetch_calls,
            "search_failed": self.search_failed,
            "fetch_failed": self.fetch_failed,
        }

    def record_usage(self, usage: Mapping[str, Any] | None, label: str = "-") -> None:
        self.llm_calls += 1
        input_tokens = int((usage or {}).get("input_tokens") or 0)
        output_tokens = int((usage or {}).get("output_tokens") or 0)
        details = (usage or {}).get("output_token_details") or {}
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.reasoning_tokens += int(details.get("reasoning") or 0)
        bucket = self.calls.setdefault(label, {"calls": 0, "in": 0, "out": 0})
        bucket["calls"] += 1
        bucket["in"] += input_tokens
        bucket["out"] += output_tokens

    def emit(self, status: str) -> None:
        # `thread_id` comes from the request. A newline in it would split this line in
        # two and let the caller forge an entry that looks legitimate to log parsers.
        thread = str(self.thread_id).replace("\r", "\\r").replace("\n", "\\n")
        logger.info(
            "turn thread=%s status=%s elapsed=%.1fs llm_calls=%d tokens_in=%d "
            "tokens_out=%d reasoning=%d search=%d fetch=%d search_failed=%d "
            "fetch_failed=%d fetch_skipped=%d degraded=%s",
            thread,
            status,
            perf_counter() - self.started_perf,
            self.llm_calls,
            self.input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.search_calls,
            self.fetch_calls,
            self.search_failed,
            self.fetch_failed,
            self.fetch_skipped,
            ",".join(self.degradations) or "-",
        )
        if self.filter_counts:
            logger.info(
                "turn source_filters=%s",
                ",".join(f"{key}={value}" for key, value in sorted(self.filter_counts.items())),
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "turn breakdown %s",
                " ".join(
                    f"{label}={bucket['calls']}x/{bucket['in']}in/{bucket['out']}out"
                    for label, bucket in sorted(self.calls.items())
                ),
            )


_current: ContextVar[TurnMetrics | None] = ContextVar("turn_metrics", default=None)
RECENT_LIMIT = 200
_recent: deque[dict[str, Any]] = deque(maxlen=RECENT_LIMIT)
_sink: Any | None = None


def set_turn_sink(sink: Any | None) -> None:
    """Register the durable turn store (memory.turns); None keeps runs in-process only."""
    global _sink
    _sink = sink


def begin_turn(
    thread_id: str | None = None,
    user_id: str | None = None,
    model: str | None = None,
    query: str | None = None,
) -> Token[TurnMetrics | None]:
    """Open a collector for one request. Pair with `end_turn` in a `finally`."""
    query_summary = " ".join((query or "").split())
    if len(query_summary) > 200:
        query_summary = query_summary[:197] + "..."
    return _current.set(
        TurnMetrics(
            thread_id=thread_id or "-",
            user_id=user_id or "-",
            model=model or "-",
            query_summary=query_summary,
        )
    )


def record_result(
    *,
    sources_count: int | None = None,
    report_title: str | None = None,
    error: BaseException | None = None,
) -> None:
    metrics = _current.get()
    if metrics is None:
        return
    if sources_count is not None:
        metrics.sources_count = max(0, sources_count)
    if report_title:
        normalized = " ".join(report_title.split())
        metrics.report_title = normalized.removeprefix("# ").strip()[:300]
    if error is not None:
        detail = f"{type(error).__name__}: {' '.join(str(error).split())}"
        metrics.error_detail = detail[:500]


async def end_turn(token: Token[TurnMetrics | None], status: str = "ok") -> None:
    metrics = _current.get()
    if metrics is not None:
        snapshot = {"user_id": metrics.user_id, **metrics.snapshot(status)}
        _recent.append(snapshot)
        if _sink is not None:
            try:
                await _sink.append(snapshot)
            except Exception:
                logger.exception("Failed to persist turn metrics")
        metrics.emit(status)
    _current.reset(token)


def recent_turns(user_id: str | None = None) -> list[dict[str, Any]]:
    """Return recent real service runs, optionally restricted to one user."""
    return [
        {key: value for key, value in item.items() if key != "user_id"}
        for item in reversed(_recent)
        if user_id is None or item["user_id"] == user_id
    ]


def record_usage(usage: Mapping[str, Any] | None, label: str = "-") -> None:
    """Count one LLM call. `label` names the node so cost can be attributed."""
    metrics = _current.get()
    if metrics is not None:
        metrics.record_usage(usage, label)


def current() -> TurnMetrics | None:
    """The collector for the turn in flight, if any. Used by the eval harness."""
    return _current.get()


def record_search(ok: bool) -> None:
    metrics = _current.get()
    if metrics is None:
        return
    if ok:
        metrics.search_calls += 1
    else:
        metrics.search_failed += 1


def record_fetch(ok: bool) -> None:
    metrics = _current.get()
    if metrics is None:
        return
    if ok:
        metrics.fetch_calls += 1
    else:
        metrics.fetch_failed += 1


def record_fetch_skipped() -> None:
    """Count a deep-read target dropped before any HTTP call (unfetchable URL).

    Distinct from fetch_failed: no request was made, so it must not inflate the
    failure rate or imply a billed call.
    """
    metrics = _current.get()
    if metrics is not None:
        metrics.fetch_skipped += 1


def record_degradation(reason: str) -> None:
    """Mark a fallback path. Repeated reasons collapse so the line stays short."""
    metrics = _current.get()
    if metrics is not None and reason not in metrics.degradations:
        metrics.degradations.append(reason)


def record_filter(reason: str, count: int = 1) -> None:
    """Record how many Search candidates were removed before relevance judging."""
    metrics = _current.get()
    if metrics is not None and count > 0:
        metrics.filter_counts[reason] = metrics.filter_counts.get(reason, 0) + count
