"""Deterministic scoring for the evaluation set.

Only structural properties are checked here: constraint fidelity, delivery format,
citation anchors that resolve to a recorded source, and exclusions that leaked into
the body. Whether the cross-validation is *good* is not machine-checkable — the
`caveat` check is a keyword proxy, not a judgement, and the summary labels it as such.
"""

import re
from dataclasses import dataclass

# A mention inside a sentence that also says "not / excluded" is compliance, not a leak.
NEGATION_MARKERS = ("不", "未", "排除", "剔除", "不含", "无关", "除外")

REQUIRED_SECTIONS = ("## 摘要", "## 主要发现", "## 分析", "## 结论")

CAVEAT_MARKERS = ("单源", "单一来源", "未验证", "无法验证", "矛盾", "不一致", "口径")

HINT_MARKER = "识别出可调研的对象"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"[。！？；\n]", text) if part.strip()]


def asserts(text: str, term: str) -> bool:
    """True when `term` appears in a sentence that does not negate or exclude it."""
    for sentence in _sentences(text):
        if term in sentence and not any(marker in sentence for marker in NEGATION_MARKERS):
            return True
    return False


def anchor_failures(report: str, source_count: int) -> list[int]:
    """Anchors that point past the recorded sources. Appendix links are not anchors."""
    used = {int(match) for match in re.findall(r"\[(\d+)\]", report)}
    return sorted(index for index in used if index < 1 or index > source_count)


def anchor_count(report: str) -> int:
    return len(re.findall(r"\[(\d+)\]", report))


def table_rows(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith("|"))


def bullet_items(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*(?:[-*]|\d+[.)])\s+\S", line))


def run_checks(
    checks: list[str],
    *,
    answer: str,
    report: str,
    constraints: list[str],
    source_count: int,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    joined_constraints = " / ".join(constraints)

    for check in checks:
        name, _, argument = check.partition(":")

        if name == "constraint":
            hit = argument in joined_constraints
            results.append(
                CheckResult(
                    check, hit, "" if hit else f"constraints={joined_constraints or '（空）'}"
                )
            )
        elif name == "exclude":
            leaked = asserts(report, argument)
            results.append(
                CheckResult(
                    check, not leaked, "" if not leaked else f"正文仍出现「{argument}」的断言"
                )
            )
        elif name == "mention":
            hit = argument in answer
            results.append(CheckResult(check, hit, "" if hit else f"未提及「{argument}」"))
        elif name == "table":
            rows = table_rows(report)
            results.append(CheckResult(check, rows >= 3, f"表格行数 {rows}"))
        elif name == "bullets":
            items = bullet_items(report)
            results.append(CheckResult(check, items >= 5, f"条目数 {items}"))
        elif name == "anchors":
            failures = anchor_failures(report, source_count)
            results.append(
                CheckResult(
                    check,
                    not failures,
                    f"{anchor_count(report)} 处锚点" if not failures else f"越界锚点 {failures}",
                )
            )
        elif name == "sections":
            missing = [item for item in REQUIRED_SECTIONS if item not in report]
            results.append(CheckResult(check, not missing, f"缺失 {missing}" if missing else ""))
        elif name == "caveat":
            hit = any(marker in answer for marker in CAVEAT_MARKERS)
            results.append(CheckResult(check, hit, "" if hit else "未见单源/矛盾/口径类提示"))
        elif name == "sources_min":
            ok = source_count >= int(argument)
            results.append(CheckResult(check, ok, f"来源 {source_count} 条"))
        elif name == "sources_max":
            ok = source_count <= int(argument)
            results.append(CheckResult(check, ok, f"来源 {source_count} 条"))
        elif name == "hint":
            hit = HINT_MARKER in answer
            results.append(CheckResult(check, hit, "" if hit else "未返回识别失败提示"))
        else:
            results.append(CheckResult(check, False, "未知判据"))

    return results
