"""Tests for the LazyLoadingAgent base class."""

import asyncio
from unittest.mock import Mock

import pytest

from agents.lazy_agent import LazyLoadingAgent


class LazyLoadingAgentFixture(LazyLoadingAgent):
    """Test implementation of LazyLoadingAgent."""

    def __init__(self):
        super().__init__()

    async def _load(self):
        """Test load implementation."""
        mock_graph = Mock()
        mock_graph.name = "test-graph"
        return mock_graph


class TestLazyLoadingAgentBase:
    """Test the LazyLoadingAgent base class functionality."""

    def test_initialization(self):
        """Test that agent initializes correctly."""
        agent = LazyLoadingAgentFixture()
        assert not agent.is_loaded
        assert agent._graph is None

    @pytest.mark.asyncio
    async def test_load(self):
        """Test that load works correctly."""
        agent = LazyLoadingAgentFixture()
        await agent.load()
        assert agent.is_loaded
        assert agent.get_graph().name == "test-graph"

    @pytest.mark.asyncio
    async def test_load_is_idempotent(self):
        agent = LazyLoadingAgentFixture()
        await agent.load()
        graph = agent.get_graph()
        await agent.load()
        assert agent.get_graph() is graph

    @pytest.mark.asyncio
    async def test_concurrent_load_creates_one_graph(self):
        agent = LazyLoadingAgentFixture()
        await asyncio.gather(agent.load(), agent.load())
        assert agent.is_loaded
        assert agent.get_graph().name == "test-graph"

    def test_get_graph_before_load(self):
        """Test that get_graph raises error before load."""
        agent = LazyLoadingAgentFixture()
        with pytest.raises(RuntimeError, match="Agent not loaded"):
            agent.get_graph()

    def test_get_graph_after_load(self):
        """Test that get_graph works after load."""
        agent = LazyLoadingAgentFixture()
        agent._loaded = True
        agent._graph = Mock()

        graph = agent.get_graph()
        assert graph is not None

    def test_get_graph_no_graph_created(self):
        """Test that get_graph raises error if no graph was created."""
        agent = LazyLoadingAgentFixture()
        agent._loaded = True
        agent._graph = None

        with pytest.raises(RuntimeError, match="Agent graph not created"):
            agent.get_graph()
