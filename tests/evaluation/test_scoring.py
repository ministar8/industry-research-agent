"""Tests for the deterministic eval scoring helpers."""

from evaluation.scoring import (
    anchor_count,
    anchor_failures,
    asserts,
    bullet_items,
    run_checks,
    table_rows,
)


def test_asserts_ignores_negated_mentions():
    assert asserts("按排除项要求，涉及上游锂矿的相关论述未纳入本报告", "锂矿") is False
    assert asserts("上游锂矿价格在 2025 年继续下行", "锂矿") is True


def test_anchor_failures_reports_indices_past_the_source_count():
    report = "结论甲 [1]，结论乙 [3]，结论丙 [9]"
    assert anchor_failures(report, source_count=3) == [9]
    assert anchor_failures(report, source_count=9) == []
    assert anchor_count(report) == 3


def test_table_and_bullet_detection():
    table = "| 企业 | 出货 |\n| --- | --- |\n| 宁德时代 | 100 |"
    assert table_rows(table) == 3
    assert bullet_items("- 甲\n- 乙\n- 丙\n- 丁\n- 戊") == 5


def test_run_checks_scores_each_named_predicate():
    report = "## 摘要\n测试 [1]\n\n## 主要发现\n- 甲\n- 乙\n- 丙\n- 丁\n- 戊"
    checks = run_checks(
        ["constraint:2025", "table", "anchors", "bullets", "sources_min:3", "mention:储能"],
        answer=report + "\n储能",
        report=report,
        constraints=["时间范围：2025 年"],
        source_count=2,
    )
    by_name = {item.name: item for item in checks}

    assert by_name["constraint:2025"].passed
    assert by_name["table"].passed is False  # no table in this report
    assert by_name["anchors"].passed  # [1] resolves to 2 sources
    assert by_name["bullets"].passed
    assert by_name["sources_min:3"].passed is False
    assert by_name["mention:储能"].passed
