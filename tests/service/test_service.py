import json
import logging
from unittest.mock import AsyncMock, patch

import langsmith
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.types import Interrupt, StateSnapshot

from agents.agents import Agent
from core.settings import settings
from schema import ChatHistory, ChatMessage, ServiceMetadata
from schema.models import FakeModelName, OpenAICompatibleName, OpenAIModelName


def test_invoke(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent(test_client, mock_agent) -> None:
    """Test that /invoke works with a custom agent_id path parameter."""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    CUSTOM_ANSWER = "The weather in Tokyo is sunny."
    DEFAULT_ANSWER = "This is from the default agent."

    # Create a separate mock for the default agent
    default_mock = AsyncMock()
    default_mock.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=DEFAULT_ANSWER)]})
    ]

    # Configure our custom mock agent
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=CUSTOM_ANSWER)]})]

    # Patch get_agent to return the correct agent based on the provided agent_id
    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return mock_agent
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(f"/{CUSTOM_AGENT}/invoke", json={"message": QUESTION})
        assert response.status_code == 200

        # Verify custom agent was called and default wasn't
        mock_agent.ainvoke.assert_awaited_once()
        default_mock.ainvoke.assert_not_awaited()

        input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
        assert input_message.content == QUESTION

        output = ChatMessage.model_validate(response.json())
        assert output.type == "ai"
        assert output.content == CUSTOM_ANSWER  # Verify we got the custom agent's response


def test_invoke_model_param(test_client, mock_agent) -> None:
    """Test that the model parameter is correctly passed to the agent if specified."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_MODEL = OpenAIModelName.GPT_61_TERRA
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post("/invoke", json={"message": QUESTION, "model": CUSTOM_MODEL})
    assert response.status_code == 200

    # Verify the model was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["model"] == CUSTOM_MODEL

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify a valid enum outside the configured allowlist returns a 400.
    # The allowlist depends on which keys the environment provides, so pick a model
    # that is genuinely unavailable here instead of hardcoding one.
    unavailable = [
        model
        for enum in (FakeModelName, OpenAICompatibleName)
        for model in enum
        if model not in settings.AVAILABLE_MODELS
    ]
    if not unavailable:
        pytest.skip("every model enum is available in this environment")

    response = test_client.post("/invoke", json={"message": QUESTION, "model": unavailable[0]})
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]

    # Verify a malformed model string still fails request validation.
    INVALID_MODEL = "gpt-7-notreal"
    response = test_client.post("/invoke", json={"message": QUESTION, "model": INVALID_MODEL})
    assert response.status_code == 422


def test_invoke_no_model_param_uses_none_default(test_client, mock_agent) -> None:
    """Test that when no model is specified, UserInput defaults to None and isn't passed to the runnable config (not hardcoded gpt-6-luna)."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    # Don't specify model in the request
    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert "model" not in config["configurable"]  # Should not be present when None

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


def test_invoke_custom_agent_config(test_client, mock_agent) -> None:
    """Test that the agent_config parameter is correctly passed to the agent."""
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is sunny."
    CUSTOM_CONFIG = {"spicy_level": 0.1, "additional_param": "value_foo"}

    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content=ANSWER)]})]

    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": CUSTOM_CONFIG}
    )
    assert response.status_code == 200

    # Verify the agent_config was passed correctly in the config
    mock_agent.ainvoke.assert_awaited_once()
    config = mock_agent.ainvoke.await_args.kwargs["config"]
    assert config["configurable"]["spicy_level"] == 0.1
    assert config["configurable"]["additional_param"] == "value_foo"

    # Verify the response is still correct
    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER

    # Verify a reserved key in agent_config throws a validation error
    INVALID_CONFIG = {"model": "gpt-6-luna"}
    response = test_client.post(
        "/invoke", json={"message": QUESTION, "agent_config": INVALID_CONFIG}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "endpoint", ["/invoke", "/stream", "/custom_agent/invoke", "/custom_agent/stream"]
)
@pytest.mark.parametrize("key", ["thread_id", "user_id", "model", "checkpoint_id", "checkpoint_ns"])
@pytest.mark.parametrize("value", ["override", "", None])
def test_agent_config_rejects_reserved_keys(test_client, mock_agent, endpoint, key, value) -> None:
    response = test_client.post(
        endpoint,
        json={"message": "Continue research", "agent_config": {key: value}},
    )

    assert response.status_code == 422
    assert "reserved keys" in response.text
    assert key in response.text
    mock_agent.aget_state.assert_not_awaited()
    mock_agent.ainvoke.assert_not_awaited()
    mock_agent.astream.assert_not_called()


@pytest.mark.parametrize(
    "endpoint", ["/invoke", "/stream", "/custom_agent/invoke", "/custom_agent/stream"]
)
def test_message_rejects_longer_value(test_client, mock_agent, endpoint) -> None:
    response = test_client.post(endpoint, json={"message": "x" * 20_001})

    assert response.status_code == 422
    assert "at most 20000 characters" in response.text
    mock_agent.aget_state.assert_not_awaited()
    mock_agent.ainvoke.assert_not_awaited()
    mock_agent.astream.assert_not_called()


def test_invoke_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    INTERRUPT = "Confirm weather check"
    mock_agent.ainvoke.return_value = [
        ("values", {"messages": [AIMessage(content=ANSWER)]}),
        ("updates", {"__interrupt__": [Interrupt(value=INTERRUPT)]}),
    ]

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == INTERRUPT


@patch("service.service.LangsmithClient")
def test_feedback(mock_client: langsmith.Client, test_client) -> None:
    ls_instance = mock_client.return_value
    ls_instance.create_feedback.return_value = None
    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    ls_instance.create_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="human-feedback-stars",
        score=0.8,
    )


@patch("service.service.LangsmithClient")
def test_feedback_create_error_returns_structured_500(mock_client, test_client) -> None:
    """A LangSmith call failure is a structured 500, not a raw traceback."""
    mock_client.return_value.create_feedback.side_effect = RuntimeError("langsmith down")
    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    assert response.status_code == 500
    assert response.json() == {"detail": "Unexpected error"}


@patch("service.service.LangsmithClient", side_effect=RuntimeError("no api key"))
def test_feedback_client_init_error_returns_structured_500(mock_client, test_client) -> None:
    """A missing LangSmith key fails at client construction; still a structured 500."""
    body = {
        "run_id": "847c6285-8fc9-4560-a83f-4e6285809254",
        "key": "human-feedback-stars",
        "score": 0.8,
    }
    response = test_client.post("/feedback", json=body)
    assert response.status_code == 500
    assert response.json() == {"detail": "Unexpected error"}


def test_history(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    user_question = HumanMessage(content=QUESTION)
    agent_response = AIMessage(content=ANSWER)
    mock_agent.aget_state.return_value = StateSnapshot(
        values={"messages": [user_question, agent_response]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    response = test_client.post(
        "/history", json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"}
    )
    assert response.status_code == 200

    output = ChatHistory.model_validate(response.json())
    assert output.messages[0].type == "human"
    assert output.messages[0].content == QUESTION
    assert output.messages[1].type == "ai"
    assert output.messages[1].content == ANSWER


def test_history_custom_agent(test_client) -> None:
    """Test that /{agent_id}/history reads the thread through the requested agent's graph."""
    CUSTOM_AGENT = "custom_agent"
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."

    custom_snapshot = StateSnapshot(
        values={"messages": [HumanMessage(content=QUESTION), AIMessage(content=ANSWER)]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )
    # The default agent's graph doesn't know about this thread, so it returns no messages.
    default_snapshot = StateSnapshot(
        values={"messages": []},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    custom_mock = AsyncMock()
    custom_mock.aget_state.return_value = custom_snapshot
    custom_mock.checkpointer = None
    default_mock = AsyncMock()
    default_mock.aget_state.return_value = default_snapshot
    default_mock.checkpointer = None

    def agent_lookup(agent_id):
        if agent_id == CUSTOM_AGENT:
            return custom_mock
        return default_mock

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.post(
            f"/{CUSTOM_AGENT}/history",
            json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"},
        )
        assert response.status_code == 200

        # The custom agent's graph was used, not the default one.
        custom_mock.aget_state.assert_awaited_once()
        default_mock.aget_state.assert_not_awaited()

        output = ChatHistory.model_validate(response.json())
        assert output.messages[0].type == "human"
        assert output.messages[0].content == QUESTION
        assert output.messages[1].type == "ai"
        assert output.messages[1].content == ANSWER


@pytest.mark.asyncio
async def test_stream(test_client, mock_agent) -> None:
    """Test streaming tokens and messages."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming
    with test_client.stream(
        "POST",
        "/stream",
        json={
            "message": QUESTION,
            "stream_tokens": True,
            "agent_config": {"spicy_level": 0.8},
        },
    ) as response:
        assert response.status_code == 200
        config = mock_agent.aget_state.await_args.kwargs["config"]
        assert config["configurable"]["spicy_level"] == 0.8

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify streamed tokens
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == len(TOKENS)
        for i, msg in enumerate(token_messages):
            assert msg["content"] == TOKENS[i]

        # Verify final message
        final_messages = [msg for msg in messages if msg["type"] == "message"]
        assert len(final_messages) == 1
        assert final_messages[0]["content"]["content"] == FINAL_ANSWER
        assert final_messages[0]["content"]["type"] == "ai"


@pytest.mark.asyncio
async def test_stream_no_tokens(test_client, mock_agent) -> None:
    """Test streaming without tokens."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    # Configure mock to use our async iterator function
    events = [
        (
            "messages",
            (
                AIMessageChunk(content=token),
                {"tags": []},
            ),
        )
        for token in TOKENS
    ] + [
        (
            "updates",
            {"chat_model": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify no token messages
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == 0

        # Verify final message
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["content"]["content"] == FINAL_ANSWER
        assert messages[0]["content"]["type"] == "ai"


def test_stream_interrupt(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    INTERRUPT = "Confirm weather check"
    # Configure mock to use our async iterator function
    events = [
        (
            "updates",
            {"__interrupt__": [Interrupt(value=INTERRUPT)]},
        )
    ]

    async def mock_astream(**kwargs):
        for event in events:
            yield event

    mock_agent.astream = mock_astream

    # Make request with streaming disabled
    with test_client.stream(
        "POST", "/stream", json={"message": QUESTION, "stream_tokens": False}
    ) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify interrupt message
        assert len(messages) == 1
        assert messages[0]["content"]["content"] == INTERRUPT
        assert messages[0]["content"]["type"] == "ai"


def test_info(test_client, mock_settings) -> None:
    """Test that /info returns the correct service metadata."""

    base_agent = Agent(description="A base agent.", graph_like=None)
    mock_settings.AUTH_SECRET = None
    mock_settings.DEFAULT_MODEL = OpenAIModelName.GPT_6_LUNA
    mock_settings.AVAILABLE_MODELS = {OpenAIModelName.GPT_6_LUNA, OpenAIModelName.GPT_61_TERRA}
    with patch.dict("agents.agents.agents", {"base-agent": base_agent}, clear=True):
        response = test_client.get("/info")
        assert response.status_code == 200
        output = ServiceMetadata.model_validate(response.json())

    assert output.default_agent == "research-supervisor"
    assert len(output.agents) == 1
    assert output.agents[0].key == "base-agent"
    assert output.agents[0].description == "A base agent."

    assert output.default_model == OpenAIModelName.GPT_6_LUNA
    assert output.models == [OpenAIModelName.GPT_6_LUNA, OpenAIModelName.GPT_61_TERRA]


@pytest.mark.parametrize("endpoint", ["/invoke", "/stream", "/history", "/threads"])
def test_unknown_agent_returns_404(test_client, endpoint) -> None:
    """Vanilla endpoints must match AG-UI: unknown agent_id is 404, not 500."""
    if endpoint == "/threads":
        response = test_client.get("/no-such-agent/threads", params={"user_id": "u"})
    elif endpoint == "/history":
        response = test_client.post("/no-such-agent/history", json={"thread_id": "t"})
    else:
        response = test_client.post(f"/no-such-agent{endpoint}", json={"message": "x"})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_invoke_metrics_use_generated_thread_id(test_client, mock_agent, caplog) -> None:
    """When the client omits thread_id, the metrics line records the generated one, not '-'."""
    caplog.set_level(logging.INFO, logger="core.metrics")
    mock_agent.ainvoke.return_value = [("values", {"messages": [AIMessage(content="ok")]})]

    response = test_client.post("/invoke", json={"message": "hi"})
    assert response.status_code == 200

    resolved = mock_agent.ainvoke.await_args.kwargs["config"]["configurable"]["thread_id"]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("turn thread=")]
    assert len(lines) == 1
    assert f"thread={resolved}" in lines[0]
    assert "thread=-" not in lines[0]


@pytest.mark.asyncio
async def test_stream_metrics_use_generated_thread_id(test_client, mock_agent, caplog) -> None:
    """The /stream path correlates metrics to the generated thread_id too."""
    caplog.set_level(logging.INFO, logger="core.metrics")

    async def mock_astream(**kwargs):
        yield ("updates", {"chat_model": {"messages": [AIMessage(content="ok")]}})

    mock_agent.astream = mock_astream

    with test_client.stream("POST", "/stream", json={"message": "hi"}) as response:
        assert response.status_code == 200
        for _ in response.iter_lines():
            pass

    resolved = mock_agent.aget_state.await_args.kwargs["config"]["configurable"]["thread_id"]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("turn thread=")]
    assert len(lines) == 1
    assert f"thread={resolved}" in lines[0]
    assert "thread=-" not in lines[0]
