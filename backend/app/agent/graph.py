"""LangGraph workflow for adaptive scholarly retrieval."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.app.agent.nodes import (
    analyze_coverage,
    decide_iteration,
    evolve_query,
    plan_search,
    retrieve_papers,
)
from backend.app.agent.state import PaperSearchState


def _next_after_decision(state: PaperSearchState) -> str:
    return "evolve_query" if state.get("should_continue") else "end"


def build_paper_search_graph():
    """Build the executable retrieval controller used by the production API."""
    graph = StateGraph(PaperSearchState)

    graph.add_node("plan_search", plan_search)
    graph.add_node("retrieve", retrieve_papers)
    graph.add_node("analyze_coverage", analyze_coverage)
    graph.add_node("decide_iteration", decide_iteration)
    graph.add_node("evolve_query", evolve_query)

    graph.add_edge(START, "plan_search")
    graph.add_edge("plan_search", "retrieve")
    graph.add_edge("retrieve", "analyze_coverage")
    graph.add_edge("analyze_coverage", "decide_iteration")
    graph.add_conditional_edges(
        "decide_iteration",
        _next_after_decision,
        {
            "evolve_query": "evolve_query",
            "end": END,
        },
    )
    graph.add_edge("evolve_query", "retrieve")
    return graph.compile()


paper_search_graph = build_paper_search_graph()
