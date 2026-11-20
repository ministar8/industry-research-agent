"""Tests for the per-turn metrics collector."""

import logging
import pathlib

import pytest

from core.metrics import (
    Degradation,
    begin_turn,
    end_turn,
    record_degradation,
    record_fetch,
    record_fetch_skipped,
    record_result,
    record_search,
    record_usage,
    set_turn_sink,
)


@pytest.mark.asyncio
async def test_usage_is_summed_across_calls(caplog):
    caplog.set_level(logging.INFO, logger="core.metrics")
    token = begin_turn("thread-1")
    record_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "output_token_details": {"reasoning": 5},
        }
    )
    record_usage({"input_tokens": 50, "output_tokens": 10})
    # A call without usage still counts, otherwise retries vanish from the total.
    record_usage(None)
    await end_turn(token, "ok")

    line = caplog.records[-1].getMessage()
    assert "llm_calls=3" in line
    assert "tokens_in=150" in line
    assert "tokens_out=30" in line
    assert "reasoning=5" in line


@pytest.mark.asyncio
async def test_turn_line_reports_every_dimension(caplog):
    caplog.set_level(logging.INFO, logger="core.metrics")
    token = begin_turn("thread-42")
    record_usage({"input_tokens": 1200, "output_tokens": 300})
    record_search(ok=True)
    record_search(ok=False)
    record_fetch(ok=True)
    record_fetch(ok=False)
    record_degradation(Degradation.JUDGE_FALLBACK)
    record_degradation(Degradation.JUDGE_FALLBACK)  # repeated reason collapses
    await end_turn(token, "ok")

    line = caplog.records[-1].getMessage()
    assert "thread=thread-42" in line
    assert "status=ok" in line
    assert "search=1" in line
    assert "search_failed=1" in line
    assert "fetch=1" in line
    assert "fetch_failed=1" in line
    assert "fetch_skipped=0" in line
    assert "degraded=judge_fallback" in line


@pytest.mark.asyncio
async def test_fetch_skipped_is_distinct_from_failed(caplog):
    caplog.set_level(logging.INFO, logger="core.metrics")
    token = begin_turn("thread-skip")
    record_fetch_skipped()
    record_fetch(ok=False)
    await end_turn(token, "ok")

    line = caplog.records[-1].getMessage()
    assert "fetch_skipped=1" in line
    assert "fetch_failed=1" in line


def test_recording_without_an_open_turn_is_a_noop():
    # Nodes are also called directly in tests; recording must not raise there.
    record_usage({"input_tokens": 1})
    record_search(ok=True)
    record_degradation(Degradation.LENGTH_RETRY)


@pytest.mark.asyncio
async def test_thread_id_cannot_forge_a_second_log_line(caplog):
    # thread_id is caller-supplied; an unescaped newline would let it write a line that
    # log parsers treat as a separate, legitimate-looking event.
    caplog.set_level(logging.INFO, logger="core.metrics")
    token = begin_turn("a\nforged line=1\r")
    await end_turn(token, "ok")

    line = caplog.records[-1].getMessage()
    assert "\n" not in line
    assert "\r" not in line
    assert "thread=a\\nforged line=1\\r" in line


@pytest.mark.asyncio
async def test_end_turn_persists_to_the_sink_and_survives_sink_failure():
    class Sink:
        def __init__(self):
            self.rows = []

        async def append(self, snapshot):
            self.rows.append(snapshot)

    sink = Sink()
    set_turn_sink(sink)
    try:
        token = begin_turn("thread-persist", user_id="alice")
        record_usage({"input_tokens": 10, "output_tokens": 2})
        await end_turn(token, "ok")
    finally:
        set_turn_sink(None)
    assert len(sink.rows) == 1
    assert sink.rows[0]["user_id"] == "alice"
    assert sink.rows[0]["thread_id"] == "thread-persist"
    assert sink.rows[0]["tokens_in"] == 10

    class BrokenSink:
        async def append(self, snapshot):
            raise RuntimeError("db down")

    set_turn_sink(BrokenSink())
    try:
        token = begin_turn("thread-broken")
        await end_turn(token, "ok")
    finally:
        set_turn_sink(None)


@pytest.mark.asyncio
async def test_snapshot_contains_runtime_context_and_error():
    class Sink:
        def __init__(self):
            self.rows = []

        async def append(self, snapshot):
            self.rows.append(snapshot)

    sink = Sink()
    set_turn_sink(sink)
    try:
        token = begin_turn(
            "thread-context",
            user_id="alice",
            model="fake-model",
            query="  研究   储能市场  ",
        )
        record_result(sources_count=4, report_title="# 储能市场报告")
        await end_turn(token, "ok")
    finally:
        set_turn_sink(None)

    row = sink.rows[0]
    assert row["model"] == "fake-model"
    assert row["query_summary"] == "研究 储能市场"
    assert row["report_title"] == "储能市场报告"
    assert row["sources_count"] == 4
    assert row["started_at"].endswith("+00:00")
    assert row["finished_at"].endswith("+00:00")

    sink = Sink()
    set_turn_sink(sink)
    try:
        token = begin_turn("thread-error", query="failed")
        record_result(error=RuntimeError("provider unavailable\nsecret"))
        await end_turn(token, "error")
    finally:
        set_turn_sink(None)
    assert sink.rows[0]["error_detail"] == "RuntimeError: provider unavailable secret"


def test_every_degradation_key_comes_from_the_enum():
    # A string literal still logs and still looks like a real signal, so nothing but a
    # source scan catches a typo or a key that never made it into the registry.
    root = pathlib.Path(__file__).resolve().parents[2] / "src"
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in root.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if 'record_degradation("' in line or 'record_degradation(f"' in line
    ]

    assert offenders == []


def test_degradation_values_are_unique_snake_case():
    values = [member.value for member in Degradation]

    assert len(values) == len(set(values))
    for member in Degradation:
        assert member == member.value  # StrEnum: logs and JSON keep the plain string
        assert member.value.islower()
        assert " " not in member.value
