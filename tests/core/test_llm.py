import os
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from core.llm import get_model
from core.settings import settings
from schema.models import DashScopeModelName, DeepseekModelName, FakeModelName, OpenAIModelName


def test_get_model_openai():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        model = get_model(OpenAIModelName.GPT_6_LUNA)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gpt-6-luna"
        assert model.streaming is True


# get_model is @cache'd, so these call the uncached function directly: clearing the
# shared cache would evict entries later tests (and app startup) rely on.
_get_model_uncached = get_model.__wrapped__


@pytest.mark.parametrize("name", list(DeepseekModelName))
def test_get_model_deepseek(name):
    with patch("core.settings.settings.DEEPSEEK_API_KEY", SecretStr("test_key")):
        model = _get_model_uncached(name)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == name.value
        assert model.openai_api_base == "https://api.deepseek.com"
        assert model.temperature == 0.3
        # Token streaming is off: json_mode + streaming returns malformed content on
        # DashScope, and nothing in this graph streams tokens to the client.
        assert model.streaming is False


def test_get_model_qwen_disables_thinking():
    with patch("core.settings.settings.DASHSCOPE_API_KEY", SecretStr("test_key")):
        model = _get_model_uncached(DashScopeModelName.QWEN_38_27B)
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "qwen3.8-27b"
        assert model.extra_body == {"enable_thinking": False}
        assert model.max_tokens == settings.LLM_MAX_TOKENS


def test_get_model_fake():
    model = get_model(FakeModelName.FAKE)
    assert isinstance(model, FakeListChatModel)
    assert model.responses == ["This is a test response from the fake model."]


def test_get_model_invalid():
    with pytest.raises(ValueError, match="Unsupported model:"):
        # Using type: ignore since we're intentionally testing invalid input
        get_model("invalid_model")  # type: ignore
