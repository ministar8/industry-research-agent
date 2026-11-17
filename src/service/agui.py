"""AG-UI protocol endpoint for the agent service.

Exposes any agent in the service over the AG-UI protocol (https://docs.ag-ui.com)
so it can be used with AG-UI compatible frontends like CopilotKit. The
LangGraph -> AG-UI event translation is handled by the official `ag-ui-langgraph`
package; this module only wires it into the service's agent registry, auth, and
tracing.

See docs/AGUI.md for usage, including how to connect a client.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from ag_ui.core import EventType, RunAgentInput
from ag_ui.encoder import EventEncoder
from ag_ui_langgraph import LangGraphAgent
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig

from agents import DEFAULT_AGENT, AgentGraph, get_agent
from auth.security import jwt_enabled
from core.metrics import begin_turn, end_turn, record_result
from schema.schema import validate_agent_config_size, validate_identifier
from service.utils import ensure_model_available, thread_metadata_matches

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agui")

# Managed by the protocol/checkpointer, so clients may not override these values via
# forwardedProps.configurable.
RESERVED_CONFIGURABLE_KEYS = {"thread_id", "turn_id", "checkpoint_id", "checkpoint_ns"}


def _base_config(
    input_data: RunAgentInput, agent_id: str, authenticated_user_id: str | None = None
) -> RunnableConfig:
    """Build the base RunnableConfig for an AG-UI run.

    Clients can pass configurable values (e.g. `model` or custom agent config) in
    `forwardedProps.configurable` - the AG-UI equivalent of the vanilla API's
    `model` / `agent_config` fields. `thread_id` is taken from the AG-UI input by
    the `ag-ui-langgraph` package itself. When JWT auth is enabled the user ID
    comes from the authenticated session, never from the client payload.
    """
    forwarded: dict[str, Any] = input_data.forwarded_props or {}
    configurable = forwarded.get("configurable") or {}
    if not isinstance(configurable, dict):
        raise HTTPException(status_code=422, detail="forwardedProps.configurable must be an object")
    try:
        validate_agent_config_size(configurable)
        validate_identifier(input_data.thread_id, "thread_id")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if overlap := RESERVED_CONFIGURABLE_KEYS & configurable.keys():
        raise HTTPException(
            status_code=422,
            detail=f"forwardedProps.configurable contains reserved keys: {overlap}",
        )

    if (model := configurable.get("model")) is not None:
        ensure_model_available(model)

    configurable = dict(configurable)
    if authenticated_user_id is not None:
        configurable["user_id"] = authenticated_user_id
    user_id = configurable.setdefault("user_id", f"anonymous:{input_data.thread_id}")
    if not isinstance(user_id, str):
        raise HTTPException(status_code=422, detail="user_id must be a string")
    try:
        validate_identifier(user_id, "user_id")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    context_mode = configurable.setdefault("context_mode", "fresh")
    if context_mode not in {"fresh", "reuse"}:
        raise HTTPException(
            status_code=422,
            detail="forwardedProps.configurable.context_mode must be 'fresh' or 'reuse'",
        )

    turn_id = str(uuid4())
    configurable["thread_id"] = input_data.thread_id
    configurable["turn_id"] = turn_id
    return RunnableConfig(
        configurable=configurable,
        # Recorded in checkpoint metadata so AG-UI threads show up in /threads too.
        metadata={"user_id": user_id, "agent_id": agent_id},
    )


async def _event_stream(
    agent_id: str,
    graph: AgentGraph,
    input_data: RunAgentInput,
    config: RunnableConfig,
    encoder: EventEncoder,
) -> AsyncGenerator[str, None]:
    # Open the same per-turn collector the vanilla /stream path uses, so AG-UI runs
    # report cost/degradation/failure instead of vanishing from the metrics line.
    messages = input_data.messages or []
    query = next(
        (
            str(message.content)
            for message in reversed(messages)
            if getattr(message, "role", None) == "user"
        ),
        "",
    )
    turn = begin_turn(
        input_data.thread_id,
        config["configurable"].get("user_id"),
        str(config["configurable"].get("model") or "-"),
        query,
    )
    # "closed" means the consumer went away before the run finished.
    outcome = "closed"
    try:
        # A new LangGraphAgent per request: it holds per-run state and is cheap to build.
        agent = LangGraphAgent(name=agent_id, graph=graph, config=config)  # type: ignore[arg-type]
        async for event in agent.run(input_data):
            # Don't forward RAW passthrough events. Standard AG-UI clients ignore them,
            # and they expose server-side internals - including fully rendered prompts
            # from on_chat_model_start - to the caller. Remove this filter only if the
            # endpoint is consumed by a trusted middle layer and you want the full
            # event firehose (e.g. for the AG-UI Event Inspector).
            if event.type == EventType.RAW:
                continue
            # The package turns graph exceptions into RUN_ERROR events rather than
            # raising, so error status must be read off the stream, not a try/except.
            if event.type == EventType.RUN_ERROR:
                outcome = "error"
            yield encoder.encode(event)
        if outcome == "closed":
            outcome = "ok"
        state = await graph.aget_state(config=config)
        report = state.values.get("report") or ""
        title = report.splitlines()[0][2:].strip() if report.startswith("# ") else ""
        record_result(sources_count=len(state.values.get("sources") or []), report_title=title)
    except Exception as exc:
        outcome = "error"
        record_result(error=exc)
        logger.exception("AG-UI run failed")
        raise
    finally:
        await end_turn(turn, outcome)


@router.post("/run", operation_id="agui_run_default")
@router.post("/{agent_id}/run", operation_id="agui_run")
async def agui_run(
    input_data: RunAgentInput, request: Request, agent_id: str = DEFAULT_AGENT
) -> StreamingResponse:
    """
    Run an agent over the AG-UI protocol, streaming AG-UI events via SSE.

    Point an AG-UI client (e.g. CopilotKit's runtime or HttpAgent) at this endpoint.
    Use the same threadId across runs to continue a conversation - threads are
    persisted in the service's checkpointer and shared with the vanilla API.
    """
    try:
        graph: AgentGraph = get_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    config = _base_config(input_data, agent_id, getattr(request.state, "auth_user_id", None))
    # Check thread ownership before the SSE headers go out: an HTTPException raised
    # inside the streaming generator would reach the client as a truncated 200 stream.
    state = await graph.aget_state(config=config)
    metadata = state.metadata or {}
    if not thread_metadata_matches(
        metadata,
        expected_user_id=config["configurable"]["user_id"],
        expected_agent_id=agent_id,
        reject_missing_user=jwt_enabled()
        and bool(state.metadata or state.values or state.tasks or state.next),
    ):
        raise HTTPException(status_code=404, detail="Thread not found")
    encoder = EventEncoder(accept=request.headers.get("accept", ""))
    return StreamingResponse(
        _event_stream(agent_id, graph, input_data, config, encoder),
        media_type=encoder.get_content_type(),
    )
