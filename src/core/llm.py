from functools import cache
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from core.settings import settings
from schema.models import (
    AllModelEnum,
    DashScopeModelName,
    DeepseekModelName,
    FakeModelName,
    OpenAICompatibleName,
    OpenAIModelName,
)

_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in DashScopeModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in FakeModelName}
)


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


type ModelT = ChatOpenAI | FakeToolModel


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, streaming=True)

    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL,
            temperature=0.5,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
        )

    if model_name in DashScopeModelName:
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY must be configured")
        extra_body: dict[str, Any] = {"enable_thinking": settings.DASHSCOPE_ENABLE_THINKING}
        if settings.DASHSCOPE_ENABLE_THINKING and settings.DASHSCOPE_THINKING_BUDGET:
            extra_body["thinking_budget"] = settings.DASHSCOPE_THINKING_BUDGET
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.3,
            # DashScope's compatible endpoint intermittently returns token ids in
            # place of text when streaming is combined with `json_object`, which the
            # structured-output layer always requests. Nothing in this graph streams
            # tokens to the client — every answer is emitted as a whole message — so
            # disabling streaming costs nothing and removes the failure mode.
            streaming=False,
            max_tokens=settings.LLM_MAX_TOKENS,
            openai_api_base=settings.DASHSCOPE_BASE_URL,
            openai_api_key=settings.DASHSCOPE_API_KEY,
            extra_body=extra_body,
        )

    if model_name in DeepseekModelName:
        # Thinking mode shares the completion budget with the visible answer: without a
        # cap the model can burn the whole allowance on reasoning and return no text.
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.3,
            streaming=False,
            max_tokens=settings.LLM_MAX_TOKENS,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )

    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")


def resolve_model(config: RunnableConfig | None = None) -> ModelT:
    """Pick the model for one call: the request override wins over the default."""
    configured = config.get("configurable", {}).get("model") if config else None
    model_name: AllModelEnum | None = configured or settings.DEFAULT_MODEL
    if model_name is None:
        raise ValueError("No default model configured")
    return get_model(model_name)
