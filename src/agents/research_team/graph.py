"""Graph assembly for the industry research team."""

from langgraph.graph import END, StateGraph

from agents.research_team.analyst import analyst
from agents.research_team.researcher import researcher
from agents.research_team.state import ResearchState
from agents.research_team.supervisor import finalize, route, supervisor
from agents.research_team.writer import writer

workflow = StateGraph(ResearchState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher)
workflow.add_node("analyst", analyst)
workflow.add_node("writer", writer)
workflow.add_node("finalize", finalize)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "finalize": "finalize",
    },
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("analyst", "supervisor")
workflow.add_edge("writer", "supervisor")
workflow.add_edge("finalize", END)

research_supervisor = workflow.compile()
