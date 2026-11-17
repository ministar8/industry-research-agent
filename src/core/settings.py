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
    # Trusted proxy IPs for X-Forwarded-* headers. Behind a reverse proxy (Render,
    # Docker, nginx) set "*" or the proxy CIDR, otherwise the request scheme/host stay
    # internal and the cookie-auth Origin check rejects every cross-HTTPS POST.
    FORWARDED_ALLOW_IPS: str = "127.0.0.1"
    LOG_LEVEL: LogLevel = LogLevel.WARNING

    AUTH_SECRET: SecretStr | None = None
    JWT_SECRET: SecretStr | None = None
    JWT_ACCESS_TTL_MINUTES: int = 15
    JWT_REFRESH_TTL_DAYS: int = 30
    AUTH_COOKIE_SECURE: bool = False

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

        # A model name is only usable when its provider is configured. Without this
        # check a mismatched DEFAULT_MODEL starts cleanly and then fails at the first
        # request (500), or crashes the UI on `models.index(default_model)`.
        # Fake mode short-circuits every node, so a leftover DEFAULT_MODEL from a real
        # run must not block startup: the model is never called.
        if (
            not self.USE_FAKE_MODEL
            and self.DEFAULT_MODEL is not None
            and self.DEFAULT_MODEL not in self.AVAILABLE_MODELS
        ):
            required = {
                DashScopeModelName: "DASHSCOPE_API_KEY",
                DeepseekModelName: "DEEPSEEK_API_KEY",
                OpenAIModelName: "OPENAI_API_KEY",
                OpenAICompatibleName: "COMPATIBLE_BASE_URL and COMPATIBLE_MODEL",
                FakeModelName: "USE_FAKE_MODEL=true",
            }
            missing = next(
                (key for provider, key in required.items() if self.DEFAULT_MODEL in provider),
                "a configured provider",
            )
            available = ", ".join(sorted(model.value for model in self.AVAILABLE_MODELS)) or "none"
            raise ValueError(
                f"DEFAULT_MODEL={self.DEFAULT_MODEL.value} requires {missing}, which is not "
                f"configured. Configure it, or unset DEFAULT_MODEL to auto-select. "
                f"Available models: {available}"
            )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_URL(self) -> str:
        # HOST is a bind address. 0.0.0.0 means "listen on every interface" and is not a
        # valid destination: connecting to it fails on Windows with WSAEADDRNOTAVAIL
        # (10049), while POSIX silently routes it to loopback. Clients need a concrete
        # address, so wildcards collapse to loopback here.
        host = self.HOST.strip() or "0.0.0.0"
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        if ":" in host:
            # IPv6 literals are bracketed in URLs; without this the port is parsed as
            # part of the address.
            host = f"[{host}]"
        return f"http://{host}:{self.PORT}"

    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()
