from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.research_team import research_supervisor
from schema import AgentInfo

DEFAULT_AGENT = "research-supervisor"

# Type alias to handle LangGraph's different agent patterns
# - @entrypoint functions return Pregel
# - StateGraph().compile() returns CompiledStateGraph
AgentGraph = CompiledStateGraph | Pregel  # What get_agent() returns (always loaded)


@dataclass
class Agent:
    description: str
    graph: AgentGraph


agents: dict[str, Agent] = {
    "research-supervisor": Agent(
        description="行业与竞品信息调研团队：意图解析、任务拆解、检索、交叉分析，产出带引用锚点的报告。",
        graph=research_supervisor,
    ),
}


def get_agent(agent_id: str) -> AgentGraph:
    """Get a statically compiled agent graph."""
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
