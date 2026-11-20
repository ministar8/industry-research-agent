"""Structure of the compiled research graph.

Node unit tests never look at the wiring, so a wrong edge or a dispatch key that drifted
from `NextAction` would only surface in a paid evaluation run. These assertions are free.
"""

from typing import get_args

from agents.research_team.graph import research_supervisor
from agents.research_team.state import NextAction


def _edges(conditional: bool | None = None) -> set[tuple[str, str]]:
    return {
        (edge.source, edge.target)
        for edge in research_supervisor.get_graph().edges
        if conditional is None or edge.conditional is conditional
    }


def test_every_next_action_is_a_node() -> None:
    # A literal without a node is dispatched into nothing; a node without a literal is
    # unreachable. The two lists are declared in different files, so they are checked here.
    assert set(get_args(NextAction)) <= set(research_supervisor.get_graph().nodes)


def test_supervisor_dispatches_to_every_next_action() -> None:
    dispatched = {
        (source, target) for source, target in _edges(conditional=True) if source == "supervisor"
    }
    assert {(("supervisor", target)) for target in get_args(NextAction)} <= dispatched


def test_workers_report_back_to_the_supervisor() -> None:
    edges = _edges()
    for worker in ("researcher", "analyst", "writer"):
        assert (worker, "supervisor") in edges


def test_entry_point_and_termination() -> None:
    edges = _edges()
    assert ("__start__", "supervisor") in edges
    assert ("finalize", "__end__") in edges
