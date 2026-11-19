"""Structured output binding with provider fallbacks.

This layer talks to providers, not to the graph.
Caps, retries and payload repair all depend on how an endpoint handles
`response_format`, and none of it touches the research domain models.
"""

import json
import logging
import re
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AnyMessage, SystemMessage
from openai import LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from core.metrics import record_degradation, record_usage

logger = logging.getLogger(__name__)

# Whether a provider accepts `response_format: json_schema`, keyed by model id.
# Unknown providers are probed once; a 400 flips the entry to False for the process.
_STRICT_SUPPORTED: dict[str, bool] = {}

_LENGTH_RETRY = SystemMessage(
    content="上一次输出超出长度上限被截断。请压缩篇幅后重新输出："
    "摘要不超过 4 句，主要发现不超过 5 条，分析不超过 300 字，结论不超过 150 字。"
)


def structured_output(
    model: Any, schema: type, label: str = "", max_tokens: int | None = None
) -> Any:
    """Bind a pydantic schema, preferring constrained decoding.

    `json_schema` is the reliable path: the provider constrains generation, so the
    model cannot emit prose, a bare array, or token ids. DashScope supports it;
    DeepSeek's own endpoint does not ("This response_format type is unavailable
    now"), and forcing `tool_choice` is rejected under thinking mode.

    So: try strict first, and on a 400 that names `response_format`, fall back to
    JSON mode with the schema serialised into the prompt. Without that hint the
    model invents key names (`next_agent` instead of `next`).

    `include_raw=True` keeps the AIMessage, which is the only place token usage is
    readable once the output has been reduced to a pydantic model.

    `max_tokens` overrides the model's completion ceiling for this call only, so a
    small decision schema can cap a degenerate runaway generation cheaply instead
    of burning the full shared budget.
    """
    return _StructuredOutput(model, schema, label, max_tokens)


class _StructuredOutput:
    def __init__(
        self, model: Any, schema: type, label: str = "", max_tokens: int | None = None
    ) -> None:
        self._model = model
        self._schema = schema
        self._label = label or "-"
        self._max_tokens = max_tokens
        # Capability belongs to the endpoint, not the model name: the same id can be
        # served by a provider that rejects json_schema.
        self._key = (
            f"{getattr(model, 'model_name', '') or id(model)}"
            f"@{getattr(model, 'openai_api_base', '') or ''}"
        )
        self._api_schema = inline_schema(schema.model_json_schema())
        self._hint = _schema_hint(self._api_schema)
        self._strict: Any = None
        self._loose: Any = None

    def _budget_kwargs(self) -> dict[str, Any]:
        return {"max_tokens": self._max_tokens} if self._max_tokens is not None else {}

    async def ainvoke(self, messages: list[AnyMessage]) -> Any:
        try:
            return await self._attempt(messages)
        except LengthFinishReasonError as exc:
            # The writer can overrun the ceiling on a source-heavy turn. Failing the
            # whole request wastes the retrieval and analysis already paid for, so
            # retry once with an explicit budget instead of surfacing the error.
            _record_failed_usage(exc, self._label)
            logger.warning(
                "%s hit the completion ceiling, retrying with a tighter brief", self._key
            )
            record_degradation("length_retry")
            return await self._attempt([*messages, _LENGTH_RETRY])

    async def _attempt(self, messages: list[AnyMessage]) -> Any:
        if _STRICT_SUPPORTED.get(self._key, True):
            try:
                result = await self._strict_call(messages)
            except Exception as exc:
                if not _rejects_json_schema(exc):
                    raise
                _STRICT_SUPPORTED[self._key] = False
                logger.warning(
                    "%s rejected json_schema, falling back to JSON mode: %s", self._key, exc
                )
            else:
                parsed = _coerce(result.get("parsed"), self._schema)
                if parsed is None:
                    parsed = self._repair(result.get("raw"))
                if parsed is not None:
                    return parsed
        return await self._loose_call(messages)

    async def _strict_call(self, messages: list[AnyMessage]) -> dict[str, Any]:
        if self._strict is None:
            self._strict = self._model.with_structured_output(
                self._api_schema, method="json_schema", include_raw=True, **self._budget_kwargs()
            )
        result = await self._strict.ainvoke(messages)
        record_usage(getattr(result.get("raw"), "usage_metadata", None), self._label)
        return result

    async def _loose_call(self, messages: list[AnyMessage]) -> Any:
        if self._loose is None:
            self._loose = self._model.with_structured_output(
                self._schema, method="json_mode", include_raw=True, **self._budget_kwargs()
            )
        payload = [self._hint, *messages]
        result = await self._call(payload)
        parsed = _coerce(result.get("parsed"), self._schema) or self._repair(result.get("raw"))
        if parsed is None:
            # One retry: JSON mode occasionally returns prose or a bare array.
            result = await self._call(payload)
            parsed = _coerce(result.get("parsed"), self._schema) or self._repair(result.get("raw"))
        if parsed is None:
            raise OutputParserException(
                f"Structured output could not be parsed: {result.get('parsing_error')}"
            )
        return parsed

    async def _call(self, payload: list[AnyMessage]) -> dict[str, Any]:
        assert self._loose is not None
        result = await self._loose.ainvoke(payload)
        record_usage(getattr(result.get("raw"), "usage_metadata", None), self._label)
        return result

    def _repair(self, raw: Any) -> Any | None:
        """Recover the shapes json_mode still lets through.

        Observed against DashScope: the model wraps the object in one or two
        single-element arrays (`[[{...}]]`), so the schema rejects a perfectly
        usable answer. Only single-element wrappers are unwrapped — anything else
        is a real mismatch and should surface as an error.
        """
        content = getattr(raw, "content", None)
        if isinstance(content, list):
            # Some providers return content blocks instead of a plain string.
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        if not isinstance(content, str):
            return None
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        try:
            data = json.loads(text)
        except ValueError:
            return None
        while isinstance(data, list) and len(data) == 1:
            data = data[0]
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                return None
        if not isinstance(data, dict):
            return None
        try:
            return self._schema.model_validate(data)
        except ValidationError:
            return None


def _record_failed_usage(exc: Exception, label: str = "-") -> None:
    """Count the tokens a failed call still cost.

    A length-capped completion is billed in full but never returns, so without this
    the turn metrics understate both spend and the real cost of retrying.
    """
    usage = getattr(getattr(exc, "completion", None), "usage", None)
    if usage is None:
        return
    record_usage(
        {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
        },
        label,
    )


def inline_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve `$ref`/`$defs` into a self-contained JSON schema.

    Pydantic emits `$ref: "#/$defs/TableBlock"` for every nested model. DashScope's
    grammar compiler chokes on it ("Cannot find field $defs in #/$defs/TableBlock",
    surfaced as a 503), and any provider that resolves refs lazily has the same
    failure mode. Inlining removes the indirection; the graph's models are shallow,
    so there is no recursion to guard against.
    """
    definitions = schema.get("$defs") or {}

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            reference = node.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                target = definitions.get(reference.rsplit("/", 1)[-1])
                if target is not None:
                    return resolve(target)
            return {key: resolve(value) for key, value in node.items() if key != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def _rejects_json_schema(exc: Exception) -> bool:
    """True when the provider refuses `response_format: json_schema`.

    Only a 4xx that names the field counts. A timeout, a rate limit or a schema
    mismatch must propagate — silently degrading those to JSON mode would hide the
    real fault behind a lower-quality path.
    """
    status = getattr(exc, "status_code", None)
    text = str(exc)
    return status in {400, 422} and ("response_format" in text or "json_schema" in text)


def _coerce(parsed: Any, schema: type) -> Any | None:
    """Normalise a parsed payload to the pydantic model.

    Binding a JSON-schema dict returns plain dicts, while binding the model returns
    model instances; both have to end up as the latter.
    """
    if parsed is None:
        return None
    if isinstance(parsed, schema):
        return parsed
    if isinstance(parsed, BaseModel):
        # A subclass schema (e.g. TableRequiredReport) rejects a base instance by
        # identity, so round-trip through its fields instead.
        try:
            return schema.model_validate(parsed.model_dump())
        except ValidationError:
            return None
    if isinstance(parsed, dict):
        try:
            return schema.model_validate(parsed)
        except ValidationError:
            return None
    return None


def _schema_hint(parameters: dict[str, Any]) -> SystemMessage:
    return SystemMessage(
        content=(
            "输出要求：只输出一个 JSON 对象，必须严格符合下面的 JSON Schema。"
            "不要输出解释文字，不要用 Markdown 代码块包裹，不要添加 schema 之外的字段。\n"
            + json.dumps(parameters, ensure_ascii=False)
        )
    )
