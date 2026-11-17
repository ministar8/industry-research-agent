"""Bounds on caller-supplied identifiers.

thread_id doubles as the checkpoint key, so an unbounded value is both a log-injection
vector and unbounded storage. Limits apply to request models only; response models stay
unconstrained so threads persisted before the limit still render.
"""

import pytest
from pydantic import ValidationError

from schema import ChatHistoryInput, ChatMessage, StreamInput, UserInput, UserThreadsInput


def test_identifiers_accept_the_limit():
    assert UserInput(message="x", thread_id="a" * 255).thread_id == "a" * 255
    assert UserInput(message="x", user_id="a" * 255).user_id == "a" * 255


def test_identifiers_reject_longer_values():
    with pytest.raises(ValidationError):
        UserInput(message="x", thread_id="a" * 256)
    with pytest.raises(ValidationError):
        UserInput(message="x", user_id="a" * 256)
    with pytest.raises(ValidationError):
        ChatHistoryInput(thread_id="a" * 256)
    with pytest.raises(ValidationError):
        UserThreadsInput(user_id="a" * 256)


@pytest.mark.parametrize("request_model", [UserInput, StreamInput])
@pytest.mark.parametrize("key", ["thread_id", "user_id", "model", "checkpoint_id", "checkpoint_ns"])
def test_agent_config_rejects_reserved_keys(request_model, key):
    with pytest.raises(ValidationError, match="reserved keys"):
        request_model(message="x", agent_config={key: "override"})


@pytest.mark.parametrize("request_model", [UserInput, StreamInput])
@pytest.mark.parametrize("config", [{}, {"spicy_level": 0.8, "checkpoint_note": "research"}])
def test_agent_config_preserves_business_parameters(request_model, config):
    assert request_model(message="x", agent_config=config).agent_config == config


@pytest.mark.parametrize("request_model", [UserInput, StreamInput])
def test_message_accepts_limit(request_model):
    assert len(request_model(message="x" * 20_000).message) == 20_000


@pytest.mark.parametrize("request_model", [UserInput, StreamInput])
def test_message_rejects_longer_value(request_model):
    with pytest.raises(ValidationError, match="at most 20000 characters"):
        request_model(message="x" * 20_001)


@pytest.mark.parametrize("request_model", [UserInput, StreamInput])
@pytest.mark.parametrize("message", ["", " \t\r\n"])
def test_message_rejects_blank_value(request_model, message):
    with pytest.raises(ValidationError, match="must not be blank"):
        request_model(message=message)


def test_agent_config_rejects_oversized_json():
    with pytest.raises(ValidationError, match="32768 UTF-8 bytes"):
        UserInput(message="x", agent_config={"value": "a" * 32_768})


@pytest.mark.parametrize("identifier", ["", "has space", "line\nbreak", "中文"])
def test_identifiers_reject_unsafe_values(identifier):
    with pytest.raises(ValidationError):
        UserInput(message="x", thread_id=identifier)


def test_identifier_allows_existing_thread_formats():
    for identifier in ("test-thread", "anonymous:847c6285-8fc9-4560-a83f-4e6285809254"):
        assert UserInput(message="x", thread_id=identifier).thread_id == identifier


def test_assignment_validates_user_input_fields():
    request = UserInput(message="x")
    with pytest.raises(ValidationError):
        request.thread_id = "has space"
    with pytest.raises(ValidationError):
        request.agent_config = {"thread_id": "override"}


def test_pretty_repr_uses_a_display_label():
    # "ai".title() renders "Ai", which reads as a typo in a user-facing header.
    assert "AI Message" in ChatMessage(type="ai", content="x").pretty_repr()
    assert "Human Message" in ChatMessage(type="human", content="x").pretty_repr()
