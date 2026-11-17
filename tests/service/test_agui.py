import json
import logging
from typing import Any
from unittest.mock import patch

import pytest
from ag_ui.core import RunAgentInput
from ag_ui.core.events import Event
from ag_ui.encoder import EventEncoder
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import TypeAdapter

from core import metrics, settings
from schema.models import FakeModelName
from service.agui import _event_stream

FAKE_RESPONSE = "The answer is 42"

event_adapter: TypeAdapter[Event] = TypeAdapter(Event)

# Captures the configurable seen by the agent, for the forwardedProps test
captured_configurable: dict[str, Any] = {}


async def call_model(state: MessagesState, config: RunnableConfig) -> MessagesState:
    captured_configurable.update(config["configurable"])
    model = FakeListChatModel(responses=[FAKE_RESPONSE])
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


model_graph = StateGraph(MessagesState)
model_graph.add_node("model", call_model)
model_graph.set_entry_point("model")
model_graph.add_edge("model", END)
model_agent = model_graph.compile(checkpointer=MemorySaver())


async def ask_color(state: MessagesState) -> MessagesState:
    answer = interrupt("What is your favorite color?")
    return {"messages": [AIMessage(content=f"Your favorite color is {answer}")]}


interrupt_graph = StateGraph(MessagesState)
interrupt_graph.add_node("ask", ask_color)
interrupt_graph.set_entry_point("ask")
interrupt_graph.add_edge("ask", END)
interrupt_agent = interrupt_graph.compile(checkpointer=MemorySaver())


async def record_then_reply(state: MessagesState, config: RunnableConfig) -> MessagesState:
    # Writes into the per-turn collector the way real nodes do. If the contextvar
    # opened in _event_stream did not propagate through ag-ui-langgraph, these counts
    # would stay zero and the metrics test would fail.
    metrics.record_usage({"input_tokens": 100, "output_tokens": 20}, label="stub")
    metrics.record_search(ok=True)
    return {"messages": [AIMessage(content="recorded")]}


metrics_graph = StateGraph(MessagesState)
metrics_graph.add_node("record", record_then_reply)
metrics_graph.set_entry_point("record")
metrics_graph.add_edge("record", END)
metrics_agent = metrics_graph.compile(checkpointer=MemorySaver())


async def record_then_fail(state: MessagesState, config: RunnableConfig) -> MessagesState:
    metrics.record_usage({"input_tokens": 5, "output_tokens": 0}, label="stub")
    raise RuntimeError("boom")


failing_graph = StateGraph(MessagesState)
failing_graph.add_node("fail", record_then_fail)
failing_graph.set_entry_point("fail")
failing_graph.add_edge("fail", END)
failing_agent = failing_graph.compile(checkpointer=MemorySaver())


@pytest.fixture(autouse=True)
def _reset_captured_configurable():
    """Every test using model_agent writes to this shared dict; reset before each test."""
    captured_configurable.clear()
    yield


@pytest.fixture
def allow_fake_model(monkeypatch):
    """Make FakeModelName.FAKE pass the AVAILABLE_MODELS allowlist check."""
    monkeypatch.setattr(settings, "AVAILABLE_MODELS", {FakeModelName.FAKE})


@pytest.fixture
def mock_agui_agent():
    def agent_lookup(agent_id: str):
        agents = {
            "model-agent": model_agent,
            "interrupt-agent": interrupt_agent,
            "metrics-agent": metrics_agent,
            "failing-agent": failing_agent,
        }
        try:
            return agents[agent_id]
        except KeyError:
            raise KeyError(agent_id)

    with patch("service.agui.get_agent", side_effect=agent_lookup):
        yield


def run_input(thread_id: str = "test-thread", **overrides: Any) -> dict[str, Any]:
    body = {
        "threadId": thread_id,
        "runId": "test-run",
        "messages": [{"id": "msg-1", "role": "user", "content": "Hello"}],
        "tools": [],
        "context": [],
        "state": {},
        "forwardedProps": {},
    }
    body.update(overrides)
    return body


def collect_events(test_client, path: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    """POST to an AG-UI endpoint and parse the SSE response into event dicts."""
    with test_client.stream("POST", path, json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
        return events


def test_agui_stream_lifecycle(mock_agui_agent, test_client) -> None:
    """A basic run emits a protocol-conformant AG-UI event stream."""
    events = collect_events(test_client, "/agui/model-agent/run", run_input())

    # Every event must parse as a valid AG-UI event (protocol conformance).
    for event in events:
        event_adapter.validate_python(event)

    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"

    # Tokens assemble into the model response
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert text == FAKE_RESPONSE

    # The final messages snapshot includes the exchange
    snapshots = [e for e in events if e["type"] == "MESSAGES_SNAPSHOT"]
    assert snapshots
    roles = [(m["role"], m.get("content")) for m in snapshots[-1]["messages"]]
    assert ("user", "Hello") in roles
    assert ("assistant", FAKE_RESPONSE) in roles


def test_agui_no_raw_events(mock_agui_agent, test_client) -> None:
    """RAW passthrough events are filtered out - they expose server-side internals."""
    events = collect_events(test_client, "/agui/model-agent/run", run_input())
    assert all(e["type"] != "RAW" for e in events)


def test_agui_default_agent_route(mock_agui_agent, test_client) -> None:
    """POST /agui/run falls back to the default agent."""
    with patch("service.agui.get_agent", return_value=model_agent) as mock_get_agent:
        events = collect_events(test_client, "/agui/run", run_input())
    from agents import DEFAULT_AGENT

    mock_get_agent.assert_called_once_with(DEFAULT_AGENT)
    assert events[-1]["type"] == "RUN_FINISHED"


def test_agui_unknown_agent(mock_agui_agent, test_client) -> None:
    response = test_client.post("/agui/no-such-agent/run", json=run_input())
    assert response.status_code == 404


def test_agui_configurable_passthrough(mock_agui_agent, allow_fake_model, test_client) -> None:
    """forwardedProps.configurable values reach the agent's configurable."""
    body = run_input(
        thread_id="passthrough-thread",
        forwardedProps={"configurable": {"model": "fake", "user_id": "user-123"}},
    )
    collect_events(test_client, "/agui/model-agent/run", body)
    assert captured_configurable.get("model") == "fake"
    assert captured_configurable.get("user_id") == "user-123"


def test_agui_rejects_invalid_context_mode(mock_agui_agent, test_client) -> None:
    body = run_input(forwardedProps={"configurable": {"context_mode": "unknown"}})

    response = test_client.post("/agui/model-agent/run", json=body)

    assert response.status_code == 422
    assert "context_mode" in response.json()["detail"]


def test_agui_jwt_identity_overrides_client_user_id(
    monkeypatch, mock_agui_agent, test_client
) -> None:
    """With JWT auth the session identity wins over a client-asserted user_id."""
    from uuid import uuid4

    from pydantic import SecretStr

    from auth.security import issue_access_token
    from auth.store import AuthUser
    from service import app

    user_id = uuid4()
    monkeypatch.setattr(settings, "JWT_SECRET", SecretStr("test-secret"))

    class Store:
        async def get_user(self, uid):
            return AuthUser(user_id, "alice", 0) if uid == user_id else None

    monkeypatch.setattr(app.state, "auth_store", Store(), raising=False)
    test_client.cookies.set("access_token", issue_access_token(user_id, "alice", 0))

    body = run_input(
        thread_id="jwt-identity-thread",
        forwardedProps={"configurable": {"user_id": "victim"}},
    )
    collect_events(test_client, "/agui/model-agent/run", body)

    assert captured_configurable.get("user_id") == str(user_id)
    tup = model_agent.checkpointer.get_tuple(
        RunnableConfig(configurable={"thread_id": "jwt-identity-thread"})
    )
    assert tup is not None
    assert tup.metadata["user_id"] == str(user_id)


def test_agui_rejects_thread_owned_by_another_user(mock_agui_agent, test_client) -> None:
    """Ownership is checked before SSE headers: the caller gets a real 404, not a cut stream."""
    body = run_input(
        thread_id="owned-thread",
        forwardedProps={"configurable": {"user_id": "user-123"}},
    )
    collect_events(test_client, "/agui/model-agent/run", body)

    intruder = run_input(
        thread_id="owned-thread",
        forwardedProps={"configurable": {"user_id": "user-456"}},
    )
    response = test_client.post("/agui/model-agent/run", json=intruder)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_agui_jwt_user_cannot_open_another_users_thread(
    monkeypatch, mock_agui_agent, test_client
) -> None:
    from uuid import uuid4

    from pydantic import SecretStr

    from auth.security import issue_access_token
    from auth.store import AuthUser
    from service import app

    collect_events(
        test_client,
        "/agui/model-agent/run",
        run_input(
            thread_id="victim-thread", forwardedProps={"configurable": {"user_id": "victim"}}
        ),
    )

    user_id = uuid4()
    monkeypatch.setattr(settings, "JWT_SECRET", SecretStr("test-secret"))

    class Store:
        async def get_user(self, uid):
            return AuthUser(user_id, "alice", 0) if uid == user_id else None

    monkeypatch.setattr(app.state, "auth_store", Store(), raising=False)
    test_client.cookies.set("access_token", issue_access_token(user_id, "alice", 0))

    response = test_client.post("/agui/model-agent/run", json=run_input(thread_id="victim-thread"))

    assert response.status_code == 404


def test_agui_records_thread_metadata(mock_agui_agent, test_client) -> None:
    """AG-UI runs record user_id/agent_id so their threads are listed by /threads."""
    body = run_input(
        thread_id="metadata-thread",
        forwardedProps={"configurable": {"user_id": "user-123"}},
    )
    collect_events(test_client, "/agui/model-agent/run", body)

    tup = model_agent.checkpointer.get_tuple(
        RunnableConfig(configurable={"thread_id": "metadata-thread"})
    )
    assert tup is not None
    assert tup.metadata["user_id"] == "user-123"
    assert tup.metadata["agent_id"] == "model-agent"


def test_agui_generates_user_id_when_missing(mock_agui_agent, test_client) -> None:
    """Without a client-supplied user_id the run still records one, as /invoke does."""
    collect_events(test_client, "/agui/model-agent/run", run_input(thread_id="anon-thread"))

    tup = model_agent.checkpointer.get_tuple(
        RunnableConfig(configurable={"thread_id": "anon-thread"})
    )
    assert tup is not None
    assert tup.metadata["user_id"]
    assert captured_configurable.get("user_id") == tup.metadata["user_id"]


def test_agui_configurable_reserved_keys(mock_agui_agent, test_client) -> None:
    body = run_input(forwardedProps={"configurable": {"thread_id": "hijack"}})
    response = test_client.post("/agui/model-agent/run", json=body)
    assert response.status_code == 422
    assert "reserved" in response.json()["detail"]


def test_agui_configurable_wrong_type(mock_agui_agent, test_client) -> None:
    body = run_input(forwardedProps={"configurable": "not-a-dict"})
    response = test_client.post("/agui/model-agent/run", json=body)
    assert response.status_code == 422


def test_agui_configurable_model_not_available(mock_agui_agent, test_client) -> None:
    """A model outside the operator's AVAILABLE_MODELS allowlist is rejected before the run starts."""
    body = run_input(
        thread_id="model-not-available-thread",
        forwardedProps={"configurable": {"model": "not-a-real-model"}},
    )
    response = test_client.post("/agui/model-agent/run", json=body)
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]


def test_agui_interrupt_and_resume(mock_agui_agent, test_client) -> None:
    """An interrupt surfaces as an on_interrupt CUSTOM event and can be resumed."""
    thread_id = "interrupt-thread"
    events = collect_events(test_client, "/agui/interrupt-agent/run", run_input(thread_id))
    assert events[-1]["type"] == "RUN_FINISHED"
    interrupts = [e for e in events if e["type"] == "CUSTOM" and e["name"] == "on_interrupt"]
    assert len(interrupts) == 1
    assert interrupts[0]["value"] == "What is your favorite color?"

    # Resume the run with an answer
    resume_body = run_input(thread_id, messages=[], forwardedProps={"command": {"resume": "blue"}})
    events = collect_events(test_client, "/agui/interrupt-agent/run", resume_body)
    assert events[-1]["type"] == "RUN_FINISHED"
    snapshots = [e for e in events if e["type"] == "MESSAGES_SNAPSHOT"]
    contents = [m.get("content") for m in snapshots[-1]["messages"]]
    assert "Your favorite color is blue" in contents


def test_agui_auth(mock_settings, mock_agui_agent, test_client) -> None:
    """The AG-UI endpoints enforce the same bearer auth as the rest of the service."""
    from pydantic import SecretStr

    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post("/agui/model-agent/run", json=run_input())
    assert response.status_code == 401

    response = test_client.post(
        "/agui/model-agent/run",
        json=run_input(),
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert response.status_code == 401

    response = test_client.post(
        "/agui/model-agent/run",
        json=run_input(),
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 200


def _turn_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("turn thread=")]


def test_agui_emits_turn_metrics(mock_agui_agent, test_client, caplog) -> None:
    """AG-UI runs open the per-turn collector and node counts reach it.

    A zero count would mean the contextvar did not propagate through ag-ui-langgraph
    and the metrics line is silently empty - the exact failure this guards against.
    """
    caplog.set_level(logging.INFO, logger="core.metrics")
    events = collect_events(
        test_client, "/agui/metrics-agent/run", run_input(thread_id="metrics-thread")
    )
    assert events[-1]["type"] == "RUN_FINISHED"

    lines = _turn_lines(caplog)
    assert len(lines) == 1
    assert "thread=metrics-thread" in lines[0]
    assert "status=ok" in lines[0]
    assert "llm_calls=1" in lines[0]
    assert "tokens_in=100" in lines[0]
    assert "search=1" in lines[0]


def test_agui_metrics_status_error_on_run_error(mock_agui_agent, test_client, caplog) -> None:
    """A graph failure surfaces as RUN_ERROR and the turn line reports status=error."""
    caplog.set_level(logging.INFO, logger="core.metrics")
    events = collect_events(
        test_client, "/agui/failing-agent/run", run_input(thread_id="error-thread")
    )
    assert any(e["type"] == "RUN_ERROR" for e in events)

    lines = _turn_lines(caplog)
    assert len(lines) == 1
    assert "status=error" in lines[0]
    assert "llm_calls=1" in lines[0]


@pytest.mark.asyncio
async def test_agui_metrics_closed_on_early_disconnect(caplog) -> None:
    """Closing the stream before the run finishes records status=closed, not ok."""
    caplog.set_level(logging.INFO, logger="core.metrics")
    input_data = RunAgentInput.model_validate(run_input(thread_id="closed-thread"))
    config = RunnableConfig(configurable={"thread_id": "closed-thread", "user_id": "u"})
    encoder = EventEncoder(accept="text/event-stream")
    gen = _event_stream("metrics-agent", metrics_agent, input_data, config, encoder)
    await gen.__anext__()
    await gen.aclose()

    lines = _turn_lines(caplog)
    assert len(lines) == 1
    assert "thread=closed-thread" in lines[0]
    assert "status=closed" in lines[0]
