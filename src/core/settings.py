from enum import StrEnum
from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from schema.models import (
    AllModelEnum,
    DashScopeModelName,
    DeepseekModelName,
    FakeModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    Provider,
)


class DatabaseType(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGO = "mongo"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """Convert to Python logging level constant."""
        import logging

        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return mapping[self]


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "0.0.0.0"
    PORT: int = 8080
    GRACEFUL_SHUTDOWN_TIMEOUT: int = 30
    LOG_LEVEL: LogLevel = LogLevel.WARNING

    AUTH_SECRET: SecretStr | None = None

    DEEPSEEK_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None
    USE_FAKE_MODEL: bool = False

    # Qwen via Model Studio (DashScope) OpenAI-compatible endpoint
    DASHSCOPE_API_KEY: SecretStr | None = None
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # Thinking is on by default for Qwen3.8 and shares the completion budget with the
    # answer. Set DASHSCOPE_THINKING_BUDGET to keep a bounded amount of deliberation instead
    # of switching it off entirely.
    DASHSCOPE_ENABLE_THINKING: bool = False
    DASHSCOPE_THINKING_BUDGET: int | None = None
    # Completion ceiling. It must cover thinking plus the visible answer when thinking is on.
    LLM_MAX_TOKENS: int = 16384

    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    # Set openai compatible api, mainly used for self-hosted gateways
    COMPATIBLE_MODEL: str | None = None
    COMPATIBLE_API_KEY: SecretStr | None = None
    COMPATIBLE_BASE_URL: str | None = None

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    # Database Configuration
    DATABASE_TYPE: DatabaseType = DatabaseType.SQLITE
    SQLITE_DB_PATH: str = "checkpoints.db"

    # PostgreSQL Configuration
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_APPLICATION_NAME: str = "industry-research-agent"
    POSTGRES_MIN_CONNECTIONS_PER_POOL: int = 1
    POSTGRES_MAX_CONNECTIONS_PER_POOL: int = 1

    # MongoDB Configuration
    MONGO_HOST: str | None = None
    MONGO_PORT: int | None = None
    MONGO_DB: str | None = None
    MONGO_USER: str | None = None
    MONGO_PASSWORD: SecretStr | None = None
    MONGO_AUTH_SOURCE: str | None = None
    MONGO_TLS: bool = False  # opt-in TLS for MongoDB; set to True for production/Atlas

    def model_post_init(self, __context: Any) -> None:
        # DashScope-hosted deepseek-v4.1-flash is the default: same model family,
        # ~4x faster with thinking off, and thinking stays controllable there.
        api_keys = {
            Provider.DASHSCOPE: self.DASHSCOPE_API_KEY,
            Provider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.OPENAI_COMPATIBLE: self.COMPATIBLE_BASE_URL and self.COMPATIBLE_MODEL,
            Provider.FAKE: self.USE_FAKE_MODEL,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        # USE_FAKE_MODEL must win the default even when real provider keys are present.
        if self.USE_FAKE_MODEL and self.DEFAULT_MODEL is None:
            self.DEFAULT_MODEL = FakeModelName.FAKE

        for provider in active_keys:
            match provider:
                case Provider.DASHSCOPE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DashScopeModelName.DEEPSEEK_V41_FLASH_1017
                    self.AVAILABLE_MODELS.update(set(DashScopeModelName))
                case Provider.DEEPSEEK:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DeepseekModelName.DEEPSEEK_FLASH
                    self.AVAILABLE_MODELS.update(set(DeepseekModelName))
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_6_LUNA
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.OPENAI_COMPATIBLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
                    self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()
