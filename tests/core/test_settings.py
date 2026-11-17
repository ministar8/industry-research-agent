import logging
import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from core.settings import DatabaseType, LogLevel, Settings
from schema.models import DashScopeModelName, DeepseekModelName, FakeModelName, OpenAIModelName


def test_settings_default_values():
    settings = Settings(_env_file=None)
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8080
    assert settings.DATABASE_TYPE == DatabaseType.POSTGRES
    assert settings.USE_FAKE_MODEL is False


def test_settings_no_api_keys():
    # Test that settings raises error when no API keys are provided
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
            _ = Settings(_env_file=None)


def test_settings_with_openai_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.OPENAI_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_6_LUNA
        assert settings.AVAILABLE_MODELS == set(OpenAIModelName)


def test_settings_with_dashscope_key():
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.DEFAULT_MODEL == DashScopeModelName.DEEPSEEK_V41_FLASH_1017
        assert settings.AVAILABLE_MODELS == set(DashScopeModelName)


def test_settings_dashscope_wins_when_both_keys_present():
    with patch.dict(
        os.environ,
        {"DASHSCOPE_API_KEY": "test_dashscope_key", "DEEPSEEK_API_KEY": "test_search_key"},
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.DEFAULT_MODEL == DashScopeModelName.DEEPSEEK_V41_FLASH_1017
        expected_models = set(DashScopeModelName)
        expected_models.update(set(DeepseekModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_with_deepseek_key():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.DEEPSEEK_API_KEY == SecretStr("test_key")
        assert settings.DEFAULT_MODEL == DeepseekModelName.DEEPSEEK_FLASH
        assert settings.AVAILABLE_MODELS == set(DeepseekModelName)
        # SecretStr keeps the key out of logs and tracebacks that render Settings.
        assert "test_key" not in repr(settings)


def test_settings_deepseek_wins_when_both_keys_present():
    with patch.dict(
        os.environ,
        {"DEEPSEEK_API_KEY": "test_deepseek_key", "OPENAI_API_KEY": "test_openai_key"},
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.DEFAULT_MODEL == DeepseekModelName.DEEPSEEK_FLASH
        expected_models = set(DeepseekModelName)
        expected_models.update(set(OpenAIModelName))
        assert settings.AVAILABLE_MODELS == expected_models


def test_settings_use_fake_model():
    with patch.dict(os.environ, {"USE_FAKE_MODEL": "true"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.DEFAULT_MODEL == FakeModelName.FAKE
        assert settings.AVAILABLE_MODELS == set(FakeModelName)


def test_settings_use_fake_model_wins_over_ambient_real_keys():
    # USE_FAKE_MODEL must win the default even when real provider keys are present.
    with patch.dict(
        os.environ,
        {
            "USE_FAKE_MODEL": "true",
            "OPENAI_API_KEY": "test_openai_key",
            "DEEPSEEK_API_KEY": "test_deepseek_key",
        },
        clear=True,
    ):
        settings = Settings(_env_file=None)
        assert settings.DEFAULT_MODEL == FakeModelName.FAKE
        # AVAILABLE_MODELS still unions every active provider.
        assert set(FakeModelName).issubset(settings.AVAILABLE_MODELS)
        assert set(OpenAIModelName).issubset(settings.AVAILABLE_MODELS)


def test_settings_explicit_default_model_overrides_fake():
    # An explicit DEFAULT_MODEL takes precedence over USE_FAKE_MODEL.
    with patch.dict(
        os.environ,
        {"USE_FAKE_MODEL": "true", "OPENAI_API_KEY": "test_openai_key"},
        clear=True,
    ):
        settings = Settings(_env_file=None, DEFAULT_MODEL=OpenAIModelName.GPT_6_LUNA)
        assert settings.DEFAULT_MODEL == OpenAIModelName.GPT_6_LUNA


def test_settings_default_model_of_unconfigured_provider_fails_at_startup():
    # Without this the service starts, advertises an unusable default model, and only
    # fails on the first request (or when the UI indexes models by default_model).
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        with pytest.raises(ValueError, match="requires DASHSCOPE_API_KEY"):
            Settings(_env_file=None, DEFAULT_MODEL=DashScopeModelName.QWEN_38_27B)


def test_settings_fake_mode_tolerates_a_leftover_default_model():
    # Fake mode never calls a provider, so a DEFAULT_MODEL left over from a real run
    # must not block startup.
    with patch.dict(os.environ, {"USE_FAKE_MODEL": "true"}, clear=True):
        settings = Settings(_env_file=None, DEFAULT_MODEL=DeepseekModelName.DEEPSEEK_FLASH)
        assert settings.DEFAULT_MODEL == DeepseekModelName.DEEPSEEK_FLASH


def test_settings_base_url():
    # A wildcard bind address is not routable as a destination (10049 on Windows).
    settings = Settings(HOST="0.0.0.0", PORT=8000, _env_file=None)
    assert settings.BASE_URL == "http://127.0.0.1:8000"
    # A concrete bind address must be preserved: it is the address clients should use.
    settings = Settings(HOST="192.168.1.10", PORT=8000, _env_file=None)
    assert settings.BASE_URL == "http://192.168.1.10:8000"
    # Trailing whitespace from .env would otherwise end up inside the URL.
    settings = Settings(HOST="0.0.0.0 ", PORT=8000, _env_file=None)
    assert settings.BASE_URL == "http://127.0.0.1:8000"
    # IPv6 literals need brackets, otherwise the port is read as part of the address.
    settings = Settings(HOST="::1", PORT=8000, _env_file=None)
    assert settings.BASE_URL == "http://[::1]:8000"


def test_settings_is_dev():
    settings = Settings(MODE="dev", _env_file=None)
    assert settings.is_dev() is True

    settings = Settings(MODE="prod", _env_file=None)
    assert settings.is_dev() is False


def test_log_level_enum():
    """Test LogLevel enum and its conversion to logging levels."""
    assert LogLevel.DEBUG.to_logging_level() == logging.DEBUG
    assert LogLevel.INFO.to_logging_level() == logging.INFO
    assert LogLevel.WARNING.to_logging_level() == logging.WARNING
    assert LogLevel.ERROR.to_logging_level() == logging.ERROR
    assert LogLevel.CRITICAL.to_logging_level() == logging.CRITICAL


def test_settings_log_level_default():
    """Test that LOG_LEVEL defaults to WARNING."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.LOG_LEVEL == LogLevel.WARNING
        assert settings.LOG_LEVEL.to_logging_level() == logging.WARNING


def test_settings_log_level_from_env():
    """Test that LOG_LEVEL can be set from environment variable."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "LOG_LEVEL": "DEBUG"}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.LOG_LEVEL == LogLevel.DEBUG
        assert settings.LOG_LEVEL.to_logging_level() == logging.DEBUG


def test_settings_log_level_invalid():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "LOG_LEVEL": "INVALID"}, clear=True):
        with pytest.raises(ValueError, match="validation error for Settings\nLOG_LEVEL\n"):
            Settings(_env_file=None)
