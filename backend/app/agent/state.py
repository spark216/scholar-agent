"""LangGraph state definitions for the academic paper search agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def merge_unique_lists(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Merge two lists while preserving order and uniqueness."""
    merged: list[str] = []
    for item in (left or []) + (right or []):
        if item not in merged:
            merged.append(item)
    return merged


def append_dicts(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return (left or []) + (right or [])


class PaperCandidate(TypedDict, total=False):
    id: str
    title: str
    abstract: str
    authors: list[str]
    year: int | None
    venue: str | None
    citation_count: int
    relevance: str
    relevance_score: float
    url: str | None
    tags: list[str]
    source_query: str
    source_iteration: int


class PaperSearchState(TypedDict, total=False):
    """End-to-end state for the scholarly search agent pipeline.

    Pipeline sketch:
      understand -> decompose -> plan_search -> retrieve
      -> filter -> iterate_or_rank -> rank -> synthesize
    """

    # --- input ---
    raw_query: str
    max_papers: int
    use_mock: bool
    evaluation_fast_mode: bool
    provider_cache_mode: str

    # --- conversation / planning memory ---
    messages: Annotated[list, add_messages]

    # --- query understanding ---
    understanding: dict[str, Any]
    intent: str
    entities: Annotated[list[str], merge_unique_lists]
    methods: Annotated[list[str], merge_unique_lists]
    datasets: Annotated[list[str], merge_unique_lists]
    venues: Annotated[list[str], merge_unique_lists]
    time_range: str | None
    sub_queries: Annotated[list[str], merge_unique_lists]
    rewritten_queries: Annotated[list[str], merge_unique_lists]

    # --- iterative search control ---
    search_plan: list[str]
    current_query: str
    executed_queries: list[str]
    current_iteration: int
    max_iterations: int
    should_continue: bool
    stop_reason: str
    target_candidates: int
    round_candidate_target: int
    min_new_candidates: int
    min_coverage: float
    max_api_calls: int
    llm_evolution_enabled: bool
    llm_evolution_calls: int

    # --- retrieval & ranking ---
    candidates: list[dict[str, Any]]
    round_candidates: list[dict[str, Any]]
    round_new_count: int
    round_history: Annotated[list[dict[str, Any]], append_dicts]
    coverage: dict[str, Any]
    coverage_gaps: list[str]
    filtered_candidates: list[dict[str, Any]]
    ranked_papers: list[dict[str, Any]]

    # --- synthesis outputs ---
    relation_graph: dict[str, Any]
    summary: dict[str, Any]
    agent_trace: Annotated[list[dict[str, Any]], append_dicts]

    # --- cost / telemetry ---
    api_calls: int
    token_usage: int
    error: str | None
