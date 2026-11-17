"""Tests for the static agent registry."""

import pytest

from agents.agents import get_agent


class TestAgentRegistry:
    def test_get_agent_static_agent(self):
        """Test getting a static agent."""
        agent = get_agent("research-supervisor")
        assert agent is not None

    def test_get_agent_nonexistent(self):
        """Test getting a non-existent agent."""
        with pytest.raises(KeyError):
            get_agent("nonexistent-agent")
