import logging
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI

from schema import AgentInfo


@pytest.mark.asyncio
async def test_lifespan(monkeypatch, caplog) -> None:
    """Test that the lifespan sets up the database and wires the agents."""
    from service import service

    fake_saver_setup = False
    fake_store_setup = False

    class FakeSaver:
        async def setup(self) -> None:
            nonlocal fake_saver_setup
            fake_saver_setup = True

    class FakeStore:
        async def setup(self) -> None:
            nonlocal fake_store_setup
            fake_store_setup = True

    fake_saver = FakeSaver()
    fake_store = FakeStore()

    @asynccontextmanager
    async def fake_initialize_database():
        yield fake_saver

    @asynccontextmanager
    async def fake_initialize_store():
        yield fake_store

    fake_turn_store = object()

    @asynccontextmanager
    async def fake_initialize_turn_store():
        yield fake_turn_store

    agents = {
        "good": type("Agent", (), {"checkpointer": None, "store": None})(),
        "bad": type("Agent", (), {"checkpointer": None, "store": None})(),
    }

    def fake_get_agent(agent_key: str):
        if agent_key == "bad":
            raise RuntimeError("boom")
        return agents[agent_key]

    monkeypatch.setattr(service, "initialize_database", fake_initialize_database)
    monkeypatch.setattr(service, "initialize_store", fake_initialize_store)
    monkeypatch.setattr(service, "initialize_turn_store", fake_initialize_turn_store)
    turn_sinks = []
    monkeypatch.setattr(service, "set_turn_sink", turn_sinks.append)
    monkeypatch.setattr(service, "get_agent", fake_get_agent)
    monkeypatch.setattr(
        service,
        "get_all_agent_info",
        lambda: [
            AgentInfo(key="good", description=""),
            AgentInfo(key="bad", description=""),
        ],
    )

    caplog.set_level(logging.INFO, logger=service.logger.name)

    app = FastAPI()
    with pytest.raises(RuntimeError, match="boom"):
        async with service.lifespan(app):
            pytest.fail("lifespan should fail before serving requests")

    assert fake_saver_setup
    assert fake_store_setup
    assert agents["good"].checkpointer is fake_saver
    assert agents["good"].store is fake_store
    assert agents["bad"].checkpointer is None
    assert agents["bad"].store is None
    assert turn_sinks == [fake_turn_store, None]

    assert "Agent loaded: good" in caplog.text
    assert "Error during database/store/agents initialization" in caplog.text
