import inspect
import json
import logging
import re
import secrets
import warnings
from collections.abc import AsyncGenerator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt
from starlette.types import Scope

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info
from auth.routes import router as auth_router
from auth.security import decode_token, jwt_enabled
from auth.store import get_auth_store
from core import settings
from core.metrics import begin_turn, end_turn, recent_turns, record_result, set_turn_sink
from memory import initialize_database, initialize_store, initialize_turn_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    ServiceMetadata,
    StreamInput,
    UserInput,
    UserThreads,
    UserThreadsInput,
)
from service.agui import router as agui_router
from service.threads import list_user_threads
from service.utils import (
    convert_message_content_to_string,
    ensure_model_available,
    langchain_to_chat_message,
    messages_from_checkpoint,
    remove_tool_calls,
    thread_metadata_matches,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


async def verify_bearer(
    request: Request,
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    token = request.cookies.get("access_token") or (http_auth.credentials if http_auth else None)
    if jwt_enabled():
        if request.cookies.get("access_token") and request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        try:
            claims = decode_token(token, "access")
            user_id = UUID(claims["sub"])
            store = getattr(request.app.state, "auth_store", None)
            user = await store.get_user(user_id) if store else None
            if user is None or user.token_version != claims.get("version"):
                raise ValueError("revoked")
        except (ValueError, KeyError):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        request.state.auth_user_id = str(user_id)
        return
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or not secrets.compare_digest(http_auth.credentials, auth_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer, store,
    and static agents.
    """
    try:
        app.state.auth_store = None
        app.state.turn_store = None
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with AsyncExitStack() as stack:
            saver = await stack.enter_async_context(initialize_database())
            store = await stack.enter_async_context(initialize_store())
            if jwt_enabled():
                if settings.DATABASE_TYPE.value != "postgres":
                    raise ValueError("JWT 用户认证需要 DATABASE_TYPE=postgres。")
                app.state.auth_store = await stack.enter_async_context(get_auth_store())
            # Set up both components
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            app.state.turn_store = await stack.enter_async_context(initialize_turn_store())
            set_turn_sink(app.state.turn_store)

            try:
                if not settings.AUTH_SECRET and not jwt_enabled():
                    logger.warning(
                        "No API authentication is configured — all protected endpoints are open. "
                        "Set JWT_SECRET to enable username login or AUTH_SECRET for legacy bearer auth."
                    )
                if not jwt_enabled():
                    logger.warning(
                        "JWT_SECRET is not configured — username login is disabled. "
                        "Set JWT_SECRET to enable user authentication."
                    )

                # Configure every registered agent before serving requests.
                agents = get_all_agent_info()
                for a in agents:
                    agent = get_agent(a.key)
                    # Set checkpointer for thread-scoped memory (conversation history)
                    agent.checkpointer = saver
                    # Set store for long-term memory (cross-conversation knowledge)
                    agent.store = store
                    logger.info(f"Agent loaded: {a.key}")
                yield
            finally:
                set_turn_sink(None)
    except Exception:
        logger.exception("Error during database/store/agents initialization")
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)
app.include_router(auth_router)
router = APIRouter(dependencies=[Depends(verify_bearer)])
# AG-UI protocol endpoints inherit the same bearer auth - see service/agui.py
router.include_router(agui_router)


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


@router.get("/runtime-metrics")
async def runtime_metrics(request: Request) -> dict[str, Any]:
    """Return real service runs, persisted when a turn store is configured."""
    user_id = getattr(request.state, "auth_user_id", None)
    store = getattr(request.app.state, "turn_store", None)
    if store is not None:
        return {"turns": await store.recent(user_id)}
    return {"turns": recent_turns(user_id)}


def _get_agent_or_404(agent_id: str) -> AgentGraph:
    try:
        return get_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")


def _record_graph_result(values: dict[str, Any]) -> None:
    report = values.get("report") or ""
    title = ""
    if report:
        match = re.search(r"^#\s+(.+)$", report, re.MULTILINE)
        title = match.group(1).strip() if match else ""
    record_result(sources_count=len(values.get("sources") or []), report_title=title)


async def _handle_input(
    user_input: UserInput,
    agent: AgentGraph,
    agent_id: str,
    authenticated_user_id: str | None = None,
) -> tuple[dict[str, Any], UUID, str]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation, the run_id, and the resolved thread_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    user_id = authenticated_user_id or user_input.user_id or f"anonymous:{thread_id}"

    configurable = {
        "thread_id": thread_id,
        "user_id": user_id,
        "context_mode": user_input.context_mode,
    }
    if user_input.model is not None:
        ensure_model_available(user_input.model)
        configurable["model"] = user_input.model

    configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        metadata={"user_id": user_id, "agent_id": agent_id},
        run_id=run_id,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    stored_metadata = state.metadata or {}
    if not thread_metadata_matches(
        stored_metadata,
        expected_user_id=user_id,
        expected_agent_id=agent_id,
        reject_missing_user=jwt_enabled()
        and bool(state.metadata or state.values or state.tasks or state.next),
    ):
        raise HTTPException(status_code=404, detail="Thread not found")

    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]
    active_turn_id = state.values.get("active_turn_id")
    turn_id = str(active_turn_id) if interrupted_tasks and active_turn_id else str(run_id)
    active_context_mode = state.values.get("active_context_mode")
    context_mode = (
        str(active_context_mode)
        if interrupted_tasks and active_context_mode
        else user_input.context_mode
    )
    config["configurable"]["turn_id"] = turn_id
    config["configurable"]["context_mode"] = context_mode

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id, thread_id


@router.post("/{agent_id}/invoke", operation_id="invoke_with_agent_id")
@router.post("/invoke")
async def invoke(
    request: Request, user_input: UserInput, agent_id: str = DEFAULT_AGENT
) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. A run ID is
    attached to messages for correlating a single request.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: AgentGraph = _get_agent_or_404(agent_id)
    kwargs, run_id, thread_id = await _handle_input(
        user_input, agent, agent_id, getattr(request.state, "auth_user_id", None)
    )

    configurable = kwargs["config"]["configurable"]
    turn = begin_turn(
        thread_id,
        configurable.get("user_id"),
        str(configurable.get("model") or settings.DEFAULT_MODEL),
        user_input.message,
    )
    outcome = "error"
    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        # A run that stops on an interrupt reports it on the final event of either stream
        # mode, so check for the interrupt before falling back to the last message.
        if "__interrupt__" in response:
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        elif response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
            _record_graph_result(response)
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        outcome = "ok"
        return output
    except Exception as exc:
        record_result(error=exc)
        logger.exception("An exception occurred during invoke")
        raise HTTPException(status_code=500, detail="Unexpected error")
    finally:
        await end_turn(turn, outcome)


async def message_generator(
    user_input: StreamInput,
    agent: AgentGraph,
    agent_id: str = DEFAULT_AGENT,
    authenticated_user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint. The agent is resolved by
    the caller so an unknown agent_id fails with 404 before SSE headers are sent.
    """
    kwargs, run_id, thread_id = await _handle_input(
        user_input, agent, agent_id, authenticated_user_id
    )

    configurable = kwargs["config"]["configurable"]
    turn = begin_turn(
        thread_id,
        configurable.get("user_id"),
        str(configurable.get("model") or settings.DEFAULT_MODEL),
        user_input.message,
    )
    # "closed" means the consumer went away before the graph finished.
    outcome = "closed"
    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(  # type: ignore[no-matching-overload]
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # Handle different stream event structures based on subgraphs
            if len(stream_event) == 3:
                # With subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # Without subgraphs: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages: list[Any] = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if "supervisor" in node or "sub-agent" in node:
                        # the only tools that come from the actual agent are the handoff and handback tools
                        if update_messages and isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # If this is a sub-agent, we want to keep the last 2 messages - the handback tool, and it's result
                                update_messages = update_messages[-2:]
                            else:
                                # If this is a supervisor, we want to keep the last message only - the handoff result. The tool comes from the 'agent' node.
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception:
                    logger.exception("Error parsing message")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
        outcome = "ok"
        state = await agent.aget_state(config=kwargs["config"])
        _record_graph_result(state.values)
    except Exception as exc:
        outcome = "error"
        record_result(error=exc)
        logger.exception("Error in message generator")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        await end_turn(turn, outcome)
        # Yielding after GeneratorExit raises "async generator ignored GeneratorExit";
        # a client that walked away does not need the sentinel.
        if outcome != "closed":
            yield "data: [DONE]\n\n"


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(
    request: Request, user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    agent = _get_agent_or_404(agent_id)
    return StreamingResponse(
        message_generator(
            user_input, agent, agent_id, getattr(request.state, "auth_user_id", None)
        ),
        media_type="text/event-stream",
    )


@router.post("/{agent_id}/history", operation_id="history_with_agent_id")
@router.post("/history")
async def history(
    request: Request, input: ChatHistoryInput, agent_id: str = DEFAULT_AGENT
) -> ChatHistory:
    """
    Get chat history for a thread and agent.

    If agent_id is not provided, the default agent will be used.
    """
    agent: AgentGraph = _get_agent_or_404(agent_id)
    config = RunnableConfig(configurable={"thread_id": input.thread_id})
    try:
        messages: list[BaseMessage] = []
        stored_metadata: Mapping[str, Any] = {}
        thread_exists = False
        # Functional-API agents keep the conversation in `__previous__`, which aget_state
        # doesn't return, so read the raw checkpoint first and only fall back for graphs.
        checkpointer = getattr(agent, "checkpointer", None)
        if checkpointer:
            tup = await checkpointer.aget_tuple(config)
            thread_exists = tup is not None
            stored_metadata = tup.metadata or {} if tup else {}
            if tup and "__previous__" in (tup.checkpoint.get("channel_values") or {}):
                messages = messages_from_checkpoint(tup.checkpoint)
        if not messages:
            state_snapshot = await agent.aget_state(config=config)
            thread_exists = thread_exists or bool(
                state_snapshot.metadata
                or state_snapshot.values
                or state_snapshot.tasks
                or state_snapshot.next
            )
            if state_snapshot.metadata:
                stored_metadata = state_snapshot.metadata
            messages = state_snapshot.values.get("messages", [])
        expected_user_id = getattr(request.state, "auth_user_id", None) or input.user_id
        if not thread_metadata_matches(
            stored_metadata,
            expected_user_id=expected_user_id,
            expected_agent_id=agent_id,
            reject_missing_user=jwt_enabled() and thread_exists,
        ):
            raise HTTPException(status_code=404, detail="Thread not found")
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except HTTPException:
        raise
    except Exception:
        logger.exception("An exception occurred while reading chat history")
        raise HTTPException(status_code=500, detail="Unexpected error")


@router.get("/{agent_id}/threads", operation_id="threads_with_agent_id")
@router.get("/threads")
async def threads(
    request: Request, input: UserThreadsInput = Depends(), agent_id: str = DEFAULT_AGENT
) -> UserThreads:
    """
    List a user's conversation threads for an agent, most recently updated first.

    When JWT authentication is enabled, the user ID is taken from the authenticated
    session rather than the query string.
    """
    agent: AgentGraph = _get_agent_or_404(agent_id)
    checkpointer = getattr(agent, "checkpointer", None)
    if not checkpointer:
        return UserThreads(threads=[])

    try:
        user_id = getattr(request.state, "auth_user_id", None) or input.user_id
        if user_id is None:
            raise HTTPException(status_code=422, detail="user_id is required.")
        summaries = await list_user_threads(checkpointer, user_id, agent_id, input.limit)
    except HTTPException:
        raise
    except Exception:
        logger.exception("An exception occurred while listing user threads")
        raise HTTPException(status_code=500, detail="Unexpected error")

    return UserThreads(threads=summaries)


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    return health_status


app.include_router(router)
_static_dir = Path(__file__).resolve().parents[2] / "static"


class RevalidatingStaticFiles(StaticFiles):
    # Without Cache-Control browsers cache heuristically by Last-Modified age, which
    # served stale JS for hours after a deploy; no-cache forces a 304 revalidation.
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# 网页挂在根路径：访问 / 即展示页，无需跳转子路径。
# API 路由先注册、优先匹配，不受静态挂载影响；/ui 保留以兼容旧链接。
app.mount(
    "/ui",
    RevalidatingStaticFiles(directory=_static_dir, html=True),
    name="ui",
)
app.mount(
    "/",
    RevalidatingStaticFiles(directory=_static_dir, html=True),
    name="root",
)
