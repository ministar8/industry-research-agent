# CLAUDE.md

<!-- Maintainer note (HTML comments are stripped before this file enters context):
     The human-facing overview, architecture diagram and config table live in README.md.
     Keep this file under 200 lines. Add a line only for something Claude cannot derive
     from the repo or would otherwise re-learn next session. Move directory-specific
     rules to .claude/rules/ with a `paths:` frontmatter, and long procedures to skills. -->

## Commands

- Install: `uv sync`.
- Lint: `uv run ruff check src/ tests/ evals/`; fix with `uv run ruff check --fix ...`.
- Format: `uv run ruff format --check src/ tests/ evals/`; fix with `uv run ruff format ...`.
- Typecheck: `uv run pyrefly check` (expect 0 errors).
- Full suite: `uv run pytest -q` (expect no failures; the pass count grows as cases are added).
- One file / one case: `uv run pytest tests/agents/test_research_team.py -q`, or append
  `::test_name`.
- Offline gate for any code, prompt or schema change: ruff + pyrefly + pytest. These cost
  **zero API tokens** — prove what can be proven by assertion before spending a live run.
- Regression baseline (real API, ~110k tokens per full run):
  `PYTHONPATH=src uv run python evals/run_eval.py`
  Single case: append `ess-cell`. List ids: `--list`. Hard cap: `--budget 150000`.
  The recorded baseline is **28/28** for `deepseek-v4.1-flash-1017`. Two runs are
  comparable only when the model id in `summary.md` matches. The checks are deterministic
  (constraints recorded, format, anchors, exclusions, source count, sections); they do
  **not** measure analysis quality — read one saved report before claiming improvement.
- Run: `uv run python src/run_service.py` (:8080). PostgreSQL must be running first:
  `docker compose up -d postgres`.

## Conventions

- Comments: default to none. Write one only when the *why* is non-obvious — a hidden
  constraint, an invariant, or a provider workaround. Never explain *what* the code does;
  never write multi-paragraph blocks. New files may carry one short module docstring.
- `constraints` must reach every node whose output feeds the report: Researcher (query
  scope), Analyst (exclusions applied to the fact base), Writer (delivery format). A node
  that consumes `sources` without `constraints` is a bug; that exact omission shipped
  silently once.
- Noise stripping is lossy for prose, never for constraints. Time range, region,
  statistical scope, explicit exclusions, format, length and audience must land in
  `constraints` verbatim. There is no chat node and no small-talk channel: every turn is
  one long noisy paragraph that *contains* a research request.
- The raw user message is never a task source for Researcher. `_task_brief` carries a short
  explicitly-marked excerpt only as fallback; `tasks` are the search basis.
- `sources` accumulates across rounds, deduplicated by `url_key` (which collapses `/cn/`
  and `/tw/` copies of the same article), never overwritten.
- Any prompt truncation must state the omission. `excerpt()` appends
  `（节选，全文共 N 字）`, because a model that cannot see the cut reports "absent" where
  the truth is "truncated" — which corrupts the information-gap section it is asked to write.
- Every `[n]` in `report` must resolve to an entry in `sources`; `clamp_anchors` drops
  out-of-range anchors. Writer never introduces an unrecorded URL.
- `finalize` assembles, never generates. It must not call a model; that is what keeps a
  turn with no researchable object free of charge.
- Routing decisions stay structured (`RoutingDecision`) so routing is testable without an LLM.
- Cross-node contracts are tested by capturing the **actual prompt** sent to a stub model,
  not by asserting a field was written to state. A field produced in A, consumed in C and
  skipped in B is invisible to state-only assertions.
- Every structured call passes a node `label` (`supervisor`/`planner`/`judge`/`analyst`/
  `writer`). Any new external call or fallback path needs its `record_*` call in
  `core/metrics.py`, or cost and failure rates silently disappear.

## Gotchas

Provider and measurement facts that are not derivable from the code. Check them before
"simplifying" any of these paths.

- `structured_output()` in `research_team/structured.py` tries `response_format: json_schema`
  first, falling back to `json_mode` only on a 400/422 naming `response_format`. DashScope
  accepts strict decoding; DeepSeek's own endpoint returns `This response_format type is
  unavailable now`, and forced `tool_choice` is rejected under thinking mode. Capability is
  cached per `model@base_url`, so only the first call pays the probe.
- `inline_schema()` must stay. Pydantic emits `$defs`/`$ref` for nested models
  (`TableBlock`) and DashScope answers `503 Cannot find field $defs`.
- `include_raw=True` is mandatory. Once output is reduced to a pydantic object,
  `usage_metadata` is gone and every token count reads zero.
- `streaming=False` on every provider. Nothing in this graph streams tokens — `finalize`
  emits one complete message. JSON mode once returned token ids as content; constrained
  decoding removed that failure class.
- A completion hitting `LLM_MAX_TOKENS` raises `LengthFinishReasonError`. The wrapper
  retries once with a tighter brief and records the wasted tokens. The retry returns a
  shorter report — that is the trade, not a bug.
- Schema-valid is not content-non-empty. `ReportBundle` needs `min_length`/`minItems` plus
  `_is_complete()` retry; `AnalysisBundle` needs the same plus a blank-fact-base retry. A
  hollow bundle once rendered a 130-character report with empty sections and passed parsing.
- When a table is requested, `report_schema()` returns `TableRequiredReport`
  (`tables.minItems=1`) and code renders the Markdown. `requires_table()` must keep
  handling negations such as `不用表格`. Prompt instructions alone produced a table only
  about half the time across repeated runs.
- Writer receives the **numbered source index, not raw excerpts**
  (`INCLUDE_SOURCE_EXCERPTS=False`). Measured on identical sources, 12 excerpts cost ~8k
  extra input tokens and produced shorter reports, one truncated mid-analysis. Excerpts
  return automatically when analysis is below `MIN_ANALYSIS_CHARS_FOR_INDEX`.
- Non-retry 4xx from Search/Fetch must raise `SearchRejected`, never return `{}`. Treating
  a 402 (balance exhausted) as "no results" burned five rounds of LLM calls and reported a
  generic failure. `abort_reason` short-circuits `route()` to `finalize`.
- `DEEPSEEK_API_KEY` authenticates Search/Fetch even when the LLM runs on DashScope. Do
  not delete it as a leftover provider key.
- Thinking is off on DashScope. `thinking_budget` is not honoured on hosted DeepSeek (a
  4096 budget still ran to the completion ceiling), so keep
  `DASHSCOPE_ENABLE_THINKING=false`. `LLM_MAX_TOKENS=16384`.
- Windows + PostgreSQL: uvicorn >= 0.54 hard-codes `ProactorEventLoop` and ignores
  `set_event_loop_policy`; psycopg's async mode refuses it. `run_service.py` builds
  `uvicorn.Server` manually with `asyncio.SelectorEventLoop`. That path has **no hot
  reload**. Do not revert it to `uvicorn.run(...)`.
- `PRICE_PER_MTOK` has no entry for `deepseek-v4.1-flash-1017`, so the eval cost column
  shows `—`. Do not invent a price.
