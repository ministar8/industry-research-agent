"""A/B benchmark: run the same research question on two models and compare cost."""

import asyncio
import json
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv(".env", override=True)

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import agents.research_team.researcher as researcher_mod
from agents.research_team import graph as graph_mod
from schema.models import DeepseekModelName, QwenModelName

QUESTION = (
    "你好啊，我最近在做投资，特别关注新能源赛道，之前看过几份报告但都不太系统。"
    "麻烦你帮我看看 2025 年国内储能电芯的出货量排名和价格走势，时间只要 2025 年，"
    "地区限中国，能用表格呈现最好，另外别把上游锂矿那些扯进来。谢谢！"
)


class UsageCollector(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.calls += 1
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                self.input_tokens += usage.get("input_tokens", 0) or 0
                self.output_tokens += usage.get("output_tokens", 0) or 0
                details = usage.get("output_token_details") or {}
                self.reasoning_tokens += details.get("reasoning", 0) or 0


async def run(model: Any, label: str) -> dict[str, Any]:
    stats = {"search": 0, "fetch": 0}
    usage = UsageCollector()

    original_search = researcher_mod.search
    original_fetch = researcher_mod.fetch

    async def counted_search(*args: Any, **kwargs: Any) -> Any:
        stats["search"] += 1
        return await original_search(*args, **kwargs)

    async def counted_fetch(*args: Any, **kwargs: Any) -> Any:
        stats["fetch"] += 1
        return await original_fetch(*args, **kwargs)

    researcher_mod.search = counted_search
    researcher_mod.fetch = counted_fetch

    graph = graph_mod.workflow.compile(checkpointer=MemorySaver())
    started = time.time()
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=QUESTION)]},
            config={
                "configurable": {"model": model, "thread_id": label},
                "callbacks": [usage],
            },
        )
    except Exception as exc:
        return {
            "label": label,
            "model": model.value,
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "elapsed": round(time.time() - started, 1),
            **stats,
        }
    finally:
        researcher_mod.search = original_search
        researcher_mod.fetch = original_fetch

    elapsed = round(time.time() - started, 1)
    content = str(result["messages"][-1].content)
    return {
        "label": label,
        "model": model.value,
        "elapsed": elapsed,
        "llm_calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "search_calls": stats["search"],
        "fetch_calls": stats["fetch"],
        "report_len": len(content),
        "rounds": result.get("rounds"),
        "intent": result.get("intent"),
        "tasks": [t.question for t in result.get("tasks") or []],
        "constraints": result.get("constraints"),
        "report": content,
    }


async def main() -> None:
    runs = []
    for model, label in [
        (DeepseekModelName.DEEPSEEK_FLASH, "deepseek"),
        (QwenModelName.QWEN_38_27B, "qwen-no-thinking"),
    ]:
        print(f"=== running {label} ({model.value}) ===", flush=True)
        outcome = await run(model, label)
        runs.append(outcome)
        print(
            f"  -> {outcome.get('elapsed')}s, llm={outcome.get('llm_calls')}, "
            f"in={outcome.get('input_tokens')}, out={outcome.get('output_tokens')}, "
            f"reasoning={outcome.get('reasoning_tokens')}, "
            f"search={outcome.get('search_calls')}, fetch={outcome.get('fetch_calls')}",
            flush=True,
        )

    with open("ab_result.json", "w", encoding="utf-8") as handle:
        json.dump(runs, handle, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    for run_result in runs:
        if run_result.get("error"):
            print(f"{run_result['label']}: ERROR {run_result['error']}")
            continue
        print(
            f"{run_result['label']:18} {run_result['elapsed']:>6}s  "
            f"llm={run_result['llm_calls']}  in={run_result['input_tokens']:>6}  "
            f"out={run_result['output_tokens']:>5}  reasoning={run_result['reasoning_tokens']:>6}  "
            f"search={run_result['search_calls']}  fetch={run_result['fetch_calls']}  "
            f"len={run_result['report_len']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
