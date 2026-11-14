"""Tests for the eval runner's cost estimation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from run_eval import estimate_cost  # noqa: E402


def test_estimate_cost_prices_snapshot_suffix_like_base_model():
    assert estimate_cost("deepseek-v4.1-flash-1017", 1_000_000, 1_000_000) == estimate_cost(
        "deepseek-v4.1-flash", 1_000_000, 1_000_000
    )


def test_estimate_cost_known_model():
    assert estimate_cost("deepseek-v4.1-flash", 1_000_000, 1_000_000) == "¥15.26"


def test_estimate_cost_unknown_model_reports_no_guess():
    assert estimate_cost("mystery-model", 1_000_000, 1_000_000) == "—"
    assert estimate_cost("mystery-9999", 1_000_000, 1_000_000) == "—"
