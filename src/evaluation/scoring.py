"""Deterministic scoring for the evaluation set.

Only structural properties are checked here: constraint fidelity, delivery format,
citation anchors that resolve to a recorded source, exclusions that leaked into
the body, and the minimum evidence required for strong trend wording. Whether the
cross-validation is *good* is not machine-checkable — the `caveat` check is a
keyword proxy, not a judgement.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# A mention inside a sentence that also says "not / excluded" is compliance, not a leak.
NEGATION_MARKERS = ("不", "未", "排除", "剔除", "不含", "无关", "除外")

REQUIRED_SECTIONS = ("## 摘要", "## 主要发现", "## 分析", "## 结论")

CAVEAT_MARKERS = ("单源", "单一来源", "未验证", "无法验证", "矛盾", "不一致", "口径")

# Internal scaffolding terms that must never surface in a user-facing report.
# "交付说明"/"表外" stay out: legitimate business/finance words, not leak markers.
META_LEAK = re.compile(r"tables\s*字段|字段留空|供内部核对|非报告正文|来源编号表|给定来源")

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


_ANCHOR_TAIL = re.compile(r"(?:\[\d+\]\s*)+$")
_SENTENCE_END = "。！？；…」』\"'.!?;"
_PROSE_SECTIONS = ("摘要", "分析", "结论")
_STRONG_TREND_LANGUAGE = re.compile(
    r"(?:持续(?:增长|上涨|下降|扩张|提升|改善|回落)|"
    r"连续(?:增长|上涨|下降|提升|回落)|"
    r"(?:呈|表现出|显示出)(?:明显|持续|长期|总体)?"
    r"(?:上升|下降|增长|下滑|上涨|回落)趋势|"
    r"长期(?:上升|增长|上涨|下降))"
)


def _ends_cleanly(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_ANCHOR_TAIL.search(stripped)) or stripped[-1] in _SENTENCE_END


def truncated_sections(report: str) -> list[str]:
    """Sections whose prose stops mid-sentence instead of ending on punctuation.

    Constrained decoding can close a JSON string at any token, so a rendered
    report may carry "…市场格局呈" with no error anywhere upstream. Table rows and
    captions inside 主要发现 are not prose and are skipped.
    """
    bad: list[str] = []
    for section in re.split(r"^## ", report, flags=re.MULTILINE)[1:]:
        title, _, body = section.partition("\n")
        title = title.strip()
        if title == "主要发现":
            bullets = [
                line for line in body.splitlines() if re.match(r"^\s*(?:[-*]|\d+[.)])\s+\S", line)
            ]
            if any(not _ends_cleanly(line) for line in bullets):
                bad.append(title)
        elif title in _PROSE_SECTIONS:
            if body.strip() and not _ends_cleanly(body):
                bad.append(title)
    return bad


def _trend_language_in_report(report: str) -> bool:
    prose: list[str] = []
    for section in re.split(r"^## ", report, flags=re.MULTILINE)[1:]:
        title, _, body = section.partition("\n")
        if title.strip() in _PROSE_SECTIONS or title.strip() == "主要发现":
            prose.append(body)
    return bool(_STRONG_TREND_LANGUAGE.search("\n".join(prose)))


def trend_failures(
    statements: Sequence[Mapping[str, Any]],
    report: str,
    source_count: int,
) -> list[str]:
    """Return violations of the lightweight trend evidence contract."""
    failures: list[str] = []
    eligible_trends = 0
    for index, item in enumerate(statements, start=1):
        kind = item.get("kind")
        source_indices = item.get("source_indices") or []
        valid_sources = [
            source
            for source in source_indices
            if isinstance(source, int) and 1 <= source <= source_count
        ]
        if not valid_sources:
            failures.append(f"证据{index}没有有效来源")

        periods = [period for period in (item.get("periods") or []) if str(period).strip()]
        value_type = item.get("value_type", "unknown")
        if kind == "trend":
            origin = item.get("trend_origin", "unknown")
            if value_type in {"planned", "forecast"}:
                failures.append(f"证据{index}将规划/预测标为实际趋势")
            elif origin == "explicit_source_statement":
                eligible_trends += 1
            elif (
                len(periods) >= 3
                and value_type in {"actual", "reported"}
                and str(item.get("metric") or "").strip()
                and str(item.get("unit") or "").strip()
            ):
                eligible_trends += 1
            else:
                failures.append(f"证据{index}的趋势缺少三个同口径时间点或完整口径")
        elif kind == "change" and len(periods) < 2:
            failures.append(f"证据{index}的期间变化少于两个时间点")
        elif kind == "forecast" and value_type not in {"planned", "forecast"}:
            failures.append(f"证据{index}的预测类型不是规划或预测")

    if _trend_language_in_report(report) and not eligible_trends:
        failures.append("正文使用强趋势措辞但没有合格的趋势证据")
    return failures


def run_checks(
    checks: list[str],
    *,
    answer: str,
    report: str,
    constraints: list[str],
    source_count: int,
    evidence_statements: Sequence[Mapping[str, Any]] | None = None,
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
        elif name == "complete":
            bad = truncated_sections(report)
            results.append(CheckResult(check, not bad, "" if not bad else f"半句话收尾 {bad}"))
        elif name == "meta":
            hit = sorted(set(META_LEAK.findall(report)))
            results.append(CheckResult(check, not hit, "" if not hit else f"元文本标记 {hit}"))
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

    if evidence_statements is not None:
        failures = trend_failures(evidence_statements, report, source_count)
        results.append(
            CheckResult(
                "trend",
                not failures,
                "" if not failures else "；".join(failures),
            )
        )

    return results
