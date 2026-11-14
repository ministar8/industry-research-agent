"""Tests for the deterministic eval scoring helpers."""

from evaluation.scoring import (
    anchor_count,
    anchor_failures,
    asserts,
    bullet_items,
    run_checks,
    table_rows,
    truncated_sections,
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


def test_truncated_sections_flags_mid_sentence_endings():
    report = (
        "## 摘要\n销量约 2.9 万辆 [1]。市场格局呈\n\n"
        "## 主要发现\n- 发现甲 [1]。\n- 发现乙，份额\n\n"
        "## 分析\n增量集中在资源运输场景。竞争格局呈\n\n"
        "## 结论\n此外来源 [6][8] 份额表中"
    )
    assert truncated_sections(report) == ["摘要", "主要发现", "分析", "结论"]


def test_truncated_sections_accepts_punctuation_anchors_and_tables():
    report = (
        "## 摘要\n销量约 2.9 万辆，同比增长 95% [1][2]。\n\n"
        "## 主要发现\n- 发现甲 [1]\n- 发现乙 [2]。\n\n"
        "**对比表**\n| 企业 | 份额 |\n| --- | --- |\n| 徐工 | 24.9% |\n\n"
        "## 分析\n换电份额被充电侵蚀 [7]。\n\n"
        "## 结论\n运营商份额缺失，无法确证 [9][10]"
    )
    assert truncated_sections(report) == []


def test_complete_check_in_run_checks():
    report = "## 摘要\n销量约 2.9 万辆 [1]。市场格局呈"
    results = run_checks(["complete"], answer=report, report=report, constraints=[], source_count=1)
    assert results[0].passed is False
    assert "摘要" in results[0].detail


def test_meta_check_flags_qa_self_check_block():
    report = (
        "## 结论\n可执行判断如下 [1]。\n\n---\n\n"
        "**交付说明（供内部核对，非报告正文）**\n- 表格：本次未要求表格，tables 字段留空。"
    )
    results = run_checks(["meta"], answer="", report=report, constraints=[], source_count=1)
    assert results[0].passed is False
    assert "元文本标记" in results[0].detail


def test_meta_check_flags_internal_input_terms():
    report = "## 摘要\n比亚迪信息在来源编号表中完全缺失，全部给定来源均未涉及 [1]。"
    results = run_checks(["meta"], answer="", report=report, constraints=[], source_count=1)
    assert results[0].passed is False


def test_meta_check_passes_legitimate_near_misses():
    report = (
        "## 结论\n按交付要求以表格呈现，交付排期延至 2026 年 [1]。"
        "银行表外业务规模扩张，现有来源未覆盖该数据 [2]。"
    )
    results = run_checks(["meta"], answer="", report=report, constraints=[], source_count=2)
    assert results[0].passed is True


def test_meta_check_passes_empty_report():
    results = run_checks(["meta"], answer="", report="", constraints=[], source_count=0)
    assert results[0].passed is True
