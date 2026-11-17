import json
import re
from datetime import datetime
from typing import Annotated, Any, Literal, NotRequired

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    StringConstraints,
    field_validator,
)
from typing_extensions import TypedDict

from schema.models import AllModelEnum, DeepseekModelName, OpenAIModelName

AGENT_CONFIG_MAX_BYTES = 32 * 1024
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+$")
Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=255, pattern=_IDENTIFIER_PATTERN.pattern),
]


def validate_agent_config_size(value: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("agent_config must contain JSON-compatible values") from exc
    if len(encoded) > AGENT_CONFIG_MAX_BYTES:
        raise ValueError(f"agent_config must be at most {AGENT_CONFIG_MAX_BYTES} UTF-8 bytes")
    return value


def validate_identifier(value: str, field_name: str = "identifier") -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} may contain only ASCII letters, digits, '.', '_', ':' and '-'"
        )
    return value


class AgentInfo(BaseModel):
    """Info about an available agent."""

    key: str = Field(
        description="Agent key.",
        examples=["research-supervisor"],
    )
    description: str = Field(
        description="Description of the agent.",
        examples=["A research assistant for generating research papers."],
    )


class ServiceMetadata(BaseModel):
    """Metadata about the service including available agents and models."""

    agents: list[AgentInfo] = Field(
        description="List of available agents.",
    )
    models: list[AllModelEnum] = Field(
        description="List of available LLMs.",
    )
    default_agent: str = Field(
        description="Default agent used when none is specified.",
        examples=["research-supervisor"],
    )
    default_model: AllModelEnum = Field(
        description="Default model used when none is specified.",
    )


class RegisterInput(BaseModel):
    """Credentials for creating a user account."""

    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)


class LoginInput(RegisterInput):
    """Credentials for signing in."""


class UserResponse(BaseModel):
    """Public user information."""

    id: str
    username: str
    created_at: datetime | None = None


class AuthResponse(BaseModel):
    """Authenticated user returned after a session change."""

    user: UserResponse


class UserInput(BaseModel):
    """Basic user input for the agent."""

    model_config = ConfigDict(validate_assignment=True)

    message: str = Field(
        description="User input to the agent.",
        max_length=20_000,
        examples=["What is the weather in Tokyo?"],
    )
    model: SerializeAsAny[AllModelEnum] | None = Field(
        title="Model",
        description="LLM Model to use for the agent. Defaults to the default model set in the settings of the service.",
        default=None,
        examples=[DeepseekModelName.DEEPSEEK_FLASH, OpenAIModelName.GPT_6_LUNA],
    )
    thread_id: Identifier | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    context_mode: Literal["fresh", "reuse"] = Field(
        description="Whether the research turn starts fresh or reuses the previous turn's sources.",
        default="fresh",
    )
    user_id: Identifier | None = Field(
        description="User ID to persist and continue a conversation across multiple threads.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_config: dict[str, Any] = Field(
        description="Additional configuration to pass through to the agent",
        default={},
        examples=[{"spicy_level": 0.8}],
    )

    @field_validator("agent_config")
    @classmethod
    def validate_agent_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        # Validate before StreamingResponse sends headers or any checkpoint is read.
        validate_agent_config_size(value)
        reserved_keys = {
            "thread_id",
            "turn_id",
            "context_mode",
            "user_id",
            "model",
            "checkpoint_id",
            "checkpoint_ns",
        }
        if overlap := reserved_keys & value.keys():
            raise ValueError(f"agent_config contains reserved keys: {sorted(overlap)}")
        return value

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""
    args: dict[str, Any]
    """The arguments to the tool call."""
    id: str | None
    """An identifier associated with the tool call."""
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        # `str.title()` would render "ai" as "Ai"; the label is a display name, not a
        # capitalized identifier.
        label = "AI" if self.type == "ai" else self.type.title()
        base_title = label + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: Identifier = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    user_id: Identifier | None = Field(
        description="User ID that owns the thread.",
        default=None,
    )


class ChatHistory(BaseModel):
    messages: list[ChatMessage]


class UserThreadsInput(BaseModel):
    """Input for listing a user's conversation threads."""

    user_id: Identifier | None = Field(
        description="User ID to list threads for.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    limit: int = Field(
        description="Maximum number of threads to return.",
        default=20,
        ge=1,
        le=100,
    )


class ThreadSummary(BaseModel):
    """Summary of a single conversation thread."""

    thread_id: str = Field(
        description="Thread ID of the conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    agent_id: str = Field(
        description="Agent this thread was run with.",
        examples=["research-supervisor"],
    )
    updated_at: datetime | None = Field(
        description="Timestamp of the most recent checkpoint in this thread.",
        default=None,
        examples=["2024-07-31T20:14:19.804150+00:00"],
    )
    title: str | None = Field(
        description="Title for the thread, derived from the first human message.",
        default=None,
        examples=["What is the weather in Tokyo?"],
    )


class UserThreads(BaseModel):
    threads: list[ThreadSummary]
