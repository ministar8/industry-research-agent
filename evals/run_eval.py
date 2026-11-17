"""Run the fixed evaluation set against the research graph and score it.

Usage:
    PYTHONPATH=src uv run python evals/run_eval.py           # all cases
    PYTHONPATH=src uv run python evals/run_eval.py swap      # filter by case id
    PYTHONPATH=src uv run python evals/run_eval.py --list    # show case ids
    PYTHONPATH=src uv run python evals/run_eval.py --budget 150000   # stop at 150k tokens

The score is deterministic: it checks constraint fidelity, delivery format, citation
anchors, exclusion leaks, and whether strong trend claims have eligible structured
evidence. It does not judge whether the analysis is otherwise good.

Every run writes `evals/runs/<timestamp>/` with one JSON per case plus `summary.md`,
so two runs can be compared line by line after a prompt or model change.
"""

import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agents.research_team.graph import research_supervisor
from agents.research_team.helpers import dedupe_sources
from core import settings
from core.metrics import begin_turn, current, end_turn
from evaluation.scoring import CheckResult, run_checks

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases.json"

# Per-million-token prices in USD. Unknown models report no cost rather than a guess.
PRICE_PER_MTOK = {
    "deepseek-v4.1-flash": (0.424, 1.696),
    "deepseek-flash": (0.15, 0.60),
}
USD_TO_CNY = 7.2


def load_cases(selection: str | None) -> list[dict]:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if selection:
        cases = [case for case in cases if selection in case["id"]]
    if not cases:
        raise SystemExit(f"没有匹配的用例：{selection}")
    return cases


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> str:
    # DashScope snapshot suffixes (-1017 etc.) are priced like the base model.
    prices = PRICE_PER_MTOK.get(model) or PRICE_PER_MTOK.get(re.sub(r"-\d{4}$", "", model))
    if not prices:
        return "—"
    usd = tokens_in / 1e6 * prices[0] + tokens_out / 1e6 * prices[1]
    return f"¥{usd * USD_TO_CNY:.2f}"


async def run_case(case: dict, model: str) -> dict:
    thread_id = f"eval-{case['id']}-{datetime.now():%H%M%S}"
    token = begin_turn(thread_id)
    started = perf_counter()
    error = None
    state: dict = {}

    try:
        state = await research_supervisor.ainvoke(
            {"messages": [HumanMessage(content=case["message"])]},
            {"configurable": {"model": model, "thread_id": thread_id}},
        )
    except Exception as exc:  # noqa: BLE001 - a failed case is data, not a crash
        error = f"{type(exc).__name__}: {exc}"

    metrics = current()
    await end_turn(token, "error" if error else "ok")
    elapsed = perf_counter() - started

    sources = dedupe_sources(state.get("sources"))
    messages = state.get("messages") or []
    answer = str(messages[-1].content) if messages else ""
    report = state.get("report") or ""

    if error:
        checks = [CheckResult(name, False, "用例执行失败") for name in case["checks"]]
    else:
        checks = run_checks(
            case["checks"],
            answer=answer,
            report=report,
            constraints=state.get("constraints") or [],
            source_count=len(sources),
            evidence_statements=[
                item.model_dump() for item in state.get("evidence_statements") or []
            ],
        )

    return {
        "id": case["id"],
        "title": case["title"],
        "model": model,
        "elapsed": round(elapsed, 1),
        "error": error,
        "intent": state.get("intent"),
        "constraints": state.get("constraints") or [],
        "subjects": [subject.model_dump() for subject in state.get("subjects") or []],
        "tasks": [task.question for task in state.get("tasks") or []],
        "evidence_coverage": [item.model_dump() for item in state.get("evidence_coverage") or []],
        "subject_coverage": [item.model_dump() for item in state.get("subject_coverage") or []],
        "evidence_statements": [
            item.model_dump() for item in state.get("evidence_statements") or []
        ],
        "source_count": len(sources),
        "fetched_count": sum(1 for source in sources if source.content),
        "metrics": {
            "llm_calls": metrics.llm_calls if metrics else 0,
            "tokens_in": metrics.input_tokens if metrics else 0,
            "tokens_out": metrics.output_tokens if metrics else 0,
            "reasoning": metrics.reasoning_tokens if metrics else 0,
            "search": metrics.search_calls if metrics else 0,
            "fetch": metrics.fetch_calls if metrics else 0,
            "search_failed": metrics.search_failed if metrics else 0,
            "fetch_failed": metrics.fetch_failed if metrics else 0,
            "fetch_skipped": metrics.fetch_skipped if metrics else 0,
            "source_filters": dict(metrics.filter_counts) if metrics else {},
            "degraded": list(metrics.degradations) if metrics else [],
        },
        "cost": estimate_cost(
            model,
            metrics.input_tokens if metrics else 0,
            metrics.output_tokens if metrics else 0,
        ),
        "checks": [
            {"name": item.name, "passed": item.passed, "detail": item.detail} for item in checks
        ],
        "report": report,
        "answer": answer,
    }


def write_summary(out_dir: Path, records: list[dict], model: str) -> str:
    lines = [
        f"# 评估结果 {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- 模型：`{model}`",
        f"- 用例：{len(records)} 个",
        "",
        "| 用例 | 结果 | 失败判据 | 耗时 | tokens(in/out) | 检索 | 成本 |",
        "|---|---|---|---|---|---|---|",
    ]
    total_passed = total_checks = 0
    for record in records:
        passed = sum(1 for check in record["checks"] if check["passed"])
        total = len(record["checks"])
        total_passed += passed
        total_checks += total
        failed = [check["name"] for check in record["checks"] if not check["passed"]]
        metrics = record["metrics"]
        lines.append(
            f"| {record['id']} | {passed}/{total} | {', '.join(failed) or '—'} | "
            f"{record['elapsed']}s | {metrics['tokens_in']}/{metrics['tokens_out']} | "
            f"{metrics['search']}+{metrics['fetch']} | {record['cost']} |"
        )

    lines += [
        "",
        f"**合计：{total_passed}/{total_checks} 项通过 （{total_passed / total_checks:.0%}）**",
        "",
        "判据为确定性检查：约束是否落到 `constraints`、交付格式是否出现、引用锚点是否越界、",
        "排除项是否渗入正文，以及强趋势措辞是否有合格的结构化证据。"
        "`caveat` 是关键词代理（单源/矛盾/口径），不代表分析质量。",
        "",
        "各用例的完整报告见同目录下的 `<case>.json`。",
    ]
    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


async def main() -> None:
    load_dotenv(".env", override=True)
    logging.basicConfig(level=settings.LOG_LEVEL.to_logging_level())
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    cases = load_cases(args[0] if args else None)

    if "--list" in sys.argv:
        for case in cases:
            print(f"{case['id']:24} {case['title']}")
        return

    budget = 0
    for flag in sys.argv[1:]:
        if flag.startswith("--budget="):
            budget = int(flag.split("=", 1)[1])
    if "--budget" in sys.argv:
        budget = int(sys.argv[sys.argv.index("--budget") + 1])

    model = str(settings.DEFAULT_MODEL.value if settings.DEFAULT_MODEL else "")
    if not model:
        raise SystemExit("未配置 DEFAULT_MODEL")

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = ROOT / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    limit = f" | 预算 {budget:,} token" if budget else ""
    print(f"模型 {model} | 用例 {len(cases)} 个{limit} | 输出 {out_dir}")
    records = []
    spent = 0
    for index, case in enumerate(cases, start=1):
        if budget and spent >= budget:
            print(f"预算已用尽（{spent:,} token），跳过剩余 {len(cases) - index + 1} 个用例")
            break
        print(f"[{index}/{len(cases)}] {case['id']} …", end=" ", flush=True)
        record = await run_case(case, model)
        (out_dir / f"{case['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(record)
        spent += record["metrics"]["tokens_in"] + record["metrics"]["tokens_out"]
        passed = sum(1 for check in record["checks"] if check["passed"])
        failed = [check["name"] for check in record["checks"] if not check["passed"]]
        status = record["error"] or f"{passed}/{len(record['checks'])}"
        print(f"{status}  {record['elapsed']}s  {'失败: ' + ','.join(failed) if failed else ''}")

    if budget:
        print(f"累计 {spent:,} token（预算 {budget:,}）")
    summary = write_summary(out_dir, records, model)
    print()
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
