from enum import StrEnum, auto


class Provider(StrEnum):
    DASHSCOPE = auto()
    DEEPSEEK = auto()
    OPENAI = auto()
    OPENAI_COMPATIBLE = auto()
    FAKE = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models"""

    GPT_6_LUNA = "gpt-6-luna"
    GPT_61_TERRA = "gpt-6.1-terra"


class DashScopeModelName(StrEnum):
    """Models hosted on Alibaba Model Studio (DashScope).

    Unlike DeepSeek's own endpoint, DashScope honours `enable_thinking` for the
    DeepSeek models it hosts, which makes thinking cost controllable.
    """

    DEEPSEEK_V41_FLASH = "deepseek-v4.1-flash"
    DEEPSEEK_V41_FLASH_1017 = "deepseek-v4.1-flash-1017"
    QWEN_38_27B = "qwen3.8-27b"
    QWEN_4_MAX = "qwen4-max"
    GLM_53 = "glm-5.3"


class DeepseekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing"""

    DEEPSEEK_FLASH = "deepseek-flash"
    DEEPSEEK_V4_PRO = "deepseek-v4-pro"


class OpenAICompatibleName(StrEnum):
    """Any OpenAI-compatible endpoint (vLLM, SGLang, internal gateway)."""

    OPENAI_COMPATIBLE = "openai-compatible"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


type AllModelEnum = (
    DashScopeModelName | OpenAIModelName | OpenAICompatibleName | DeepseekModelName | FakeModelName
)
