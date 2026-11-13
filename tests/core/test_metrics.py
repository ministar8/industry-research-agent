"""Tests for the per-turn metrics collector."""

import logging

from core.metrics import (
    begin_turn,
    end_turn,
    record_degradation,
    record_fetch,
    record_search,
    record_usage,
)


def test_usage_is_summed_across_calls(caplog):
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
    end_turn(token, "ok")

    line = caplog.records[-1].getMessage()
    assert "llm_calls=3" in line
    assert "tokens_in=150" in line
    assert "tokens_out=30" in line
    assert "reasoning=5" in line


def test_turn_line_reports_every_dimension(caplog):
    caplog.set_level(logging.INFO, logger="core.metrics")
    token = begin_turn("thread-42")
    record_usage({"input_tokens": 1200, "output_tokens": 300})
    record_search(ok=True)
    record_search(ok=False)
    record_fetch(ok=True)
    record_fetch(ok=False)
    record_degradation("judge_fallback")
    record_degradation("judge_fallback")  # repeated reason collapses
    end_turn(token, "ok")

    line = caplog.records[-1].getMessage()
    assert "thread=thread-42" in line
    assert "status=ok" in line
    assert "search=1" in line
    assert "search_failed=1" in line
    assert "fetch=1" in line
    assert "fetch_failed=1" in line
    assert "degraded=judge_fallback" in line


def test_recording_without_an_open_turn_is_a_noop():
    # Nodes are also called directly in tests; recording must not raise there.
    record_usage({"input_tokens": 1})
    record_search(ok=True)
    record_degradation("whatever")
