"""Writer context A/B: full source excerpts vs index only.

The same sources and analysis are held constant; only the writer's view of the source
list changes. Run it before trading recall for tokens.

    PYTHONPATH=src uv run python evals/ab_writer.py
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

import agents.research_team.writer as writer_mod
from agents.research_team.graph import research_supervisor
from core import settings
from core.metrics import begin_turn, current, end_turn

ROOT = Path(__file__).resolve().parent
CASE_ID = "ess-cell-cn-2025"


def numbers_in(text: str) -> set[str]:
    return {match.replace(",", "") for match in re.findall(r"\d+(?:\.\d+)?", text)}


def stats(report: str, allowed: set[str]) -> dict:
    used = numbers_in(report)
    return {
        "chars": len(report),
        "numbers": len(re.findall(r"\d+(?:\.\d+)?", report)),
        "unsourced": len(used - allowed),
        "anchors": len(re.findall(r"\[\d+\]", report)),
        "table_rows": sum(1 for line in report.splitlines() if line.strip().startswith("|")),
        "sections": report.count("## "),
    }


async def main() -> None:
    load_dotenv(".env", override=True)
    logging.basicConfig(level=settings.LOG_LEVEL.to_logging_level())

    case = next(
        item
        for item in json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
        if item["id"] == CASE_ID
    )
    config = {"configurable": {"model": str(settings.DEFAULT_MODEL.value)}}

    print("采集中间状态（一次完整调研）…")
    state = await research_supervisor.ainvoke(
        {"messages": [HumanMessage(content=case["message"])]},
        {**config, "configurable": {**config["configurable"], "thread_id": "ab-writer-source"}},
    )
    sources = state.get("sources") or []
    analysis = state.get("analysis") or ""
    grounded = numbers_in(
        analysis + "\n".join(source.content or source.summary for source in sources)
    )
    print(f"来源 {len(sources)} 条，分析 {len(analysis)} 字，可溯源数字 {len(grounded)} 个\n")

    results = []
    for label, flag in (("with_excerpts", True), ("index_only", False)):
        writer_mod.INCLUDE_SOURCE_EXCERPTS = flag
        token = begin_turn(f"ab-writer-{label}")
        out = await writer_mod.writer(state, config)  # type: ignore[arg-type]
        metrics = current()
        end_turn(token, "ok")
        report = out.get("report") or ""
        results.append(
            {
                "variant": label,
                "tokens_in": metrics.input_tokens if metrics else 0,
                "tokens_out": metrics.output_tokens if metrics else 0,
                "report": report,
                **stats(report, grounded),
            }
        )

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / f"ab_writer_{stamp}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    header = (
        f"{'变体':16} {'输入':>7} {'输出':>7} {'字数':>6} "
        f"{'数字':>5} {'未溯源':>6} {'锚点':>5} {'表格行':>6} {'章节':>4}"
    )
    print(header)
    for item in results:
        print(
            f"{item['variant']:16} {item['tokens_in']:>7} {item['tokens_out']:>7} "
            f"{item['chars']:>6} {item['numbers']:>5} {item['unsourced']:>6} "
            f"{item['anchors']:>5} {item['table_rows']:>6} {item['sections']:>4}"
        )
    delta = results[0]["tokens_in"] - results[1]["tokens_in"]
    print(f"\n输入节省 {delta} token（{delta / max(results[0]['tokens_in'], 1):.0%}）")
    print(f"完整结果：evals/ab_writer_{stamp}.json")


if __name__ == "__main__":
    asyncio.run(main())
