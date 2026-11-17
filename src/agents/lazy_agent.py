"""Agent types with async initialization and dynamic graph creation."""

import asyncio
from abc import ABC, abstractmethod

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel


class LazyLoadingAgent(ABC):
    """Base class for agents that require async loading."""

    def __init__(self) -> None:
        """Initialize the agent."""
        self._loaded = False
        self._graph: CompiledStateGraph | Pregel | None = None
        self._load_lock = asyncio.Lock()

    async def load(self) -> None:
        """Load the agent once and validate the graph created by the subclass."""
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            graph = await self._load()
            if graph is None:
                raise RuntimeError("Agent load() did not create a graph.")
            self._graph = graph
            self._loaded = True

    @abstractmethod
    async def _load(self) -> CompiledStateGraph | Pregel:
        """
        Perform async loading for this agent.

        This method is called during service startup and should handle:
        - Setting up external connections (MCP clients, databases, etc.)
        - Loading tools or resources
        - Any other async setup required
        - Creating the agent's graph
        """
        raise NotImplementedError  # pragma: no cover

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has completed loading successfully."""
        return self._loaded

    def get_graph(self) -> CompiledStateGraph | Pregel:
        """
        Get the agent's graph.

        Returns the graph instance that was created during load().

        Returns:
            The agent's graph (CompiledStateGraph or Pregel)
        """
        if not self._loaded:
            raise RuntimeError("Agent not loaded. Call load() first.")
        if self._graph is None:
            raise RuntimeError("Agent graph not created during load().")
        return self._graph
