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
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    thread_id: str = "-"
    started_at: float = field(default_factory=perf_counter)
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
    # Per-node attribution: {"judge": {"calls": 1, "in": 9000, "out": 600}, ...}
    calls: dict[str, dict[str, int]] = field(default_factory=dict)

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
            perf_counter() - self.started_at,
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "turn breakdown %s",
                " ".join(
                    f"{label}={bucket['calls']}x/{bucket['in']}in/{bucket['out']}out"
                    for label, bucket in sorted(self.calls.items())
                ),
            )


_current: ContextVar[TurnMetrics | None] = ContextVar("turn_metrics", default=None)


def begin_turn(thread_id: str | None = None) -> Token[TurnMetrics | None]:
    """Open a collector for one request. Pair with `end_turn` in a `finally`."""
    return _current.set(TurnMetrics(thread_id=thread_id or "-"))


def end_turn(token: Token[TurnMetrics | None], status: str = "ok") -> None:
    metrics = _current.get()
    if metrics is not None:
        metrics.emit(status)
    _current.reset(token)


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
