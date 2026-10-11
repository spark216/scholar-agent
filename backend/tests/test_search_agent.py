"""Tests for the production adaptive retrieval graph."""

import unittest
import random
from unittest.mock import patch

from backend.app.agent.graph import paper_search_graph
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.search_service import (
    _benchmark_fallback_understanding,
    _benchmark_understanding,
    _embedding_candidate_limit,
    _limit_embedding_candidates,
    iter_paper_search_events,
)
from backend.app.config import get_settings
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.pipeline import EvidRankPipeline
from backend.tests.conftest import FakeEncoder, FakeReranker


def _candidate(prefix: str, index: int, query: str):
    return {
        "id": f"{prefix}-{index}",
        "title": f"Watermelon yield method {prefix} {index}",
        "abstract": f"Watermelon yield improvement evidence retrieved by {query}.",
        "authors": ["Researcher"],
        "year": 2024,
        "published_at": "2024",
        "venue": "Agronomy",
        "citation_count": 0,
        "url": f"https://example.org/{prefix}/{index}",
        "tags": ["watermelon", "yield"],
        "source": prefix,
        "sources": [prefix],
        "external_ids": {"doi": f"10.1234/{prefix}.{index}"},
        "score": 1.0,
    }


def _initial_state(max_iterations=2):
    understanding = QueryUnderstanding(
        intent="检索西瓜增产方法",
        topics=["watermelon", "yield"],
        sub_queries=["watermelon yield methods", "watermelon cultivation yield"],
        rewritten_queries=["watermelon yield research", "watermelon cultivation methods"],
    )
    return {
        "raw_query": "西瓜增产的方法",
        "understanding": understanding.model_dump(),
        "max_papers": 5,
        "candidates": [],
        "round_history": [],
        "agent_trace": [],
        "api_calls": 0,
        "token_usage": 0,
        "target_candidates": 80,
        "max_iterations": max_iterations,
        "round_candidate_target": 40,
        "min_new_candidates": 5,
        "min_coverage": 0.6,
        "max_api_calls": 12,
        "llm_evolution_enabled": False,
        "llm_evolution_calls": 0,
    }


def test_benchmark_understanding_keeps_official_short_query():
    rejected = QueryUnderstanding(
        is_paper_search=False,
        reject_reason="单个词汇没有明确检索意图",
    )

    understanding = _benchmark_understanding("deafness", rejected)

    assert understanding.is_paper_search is True
    assert understanding.reject_reason is None
    assert understanding.topics == ["deafness"]
    assert understanding.sub_queries == ["deafness"]
    assert rejected.is_paper_search is False


def test_benchmark_fallback_understanding_is_explicit_and_lexical():
    understanding, meta = _benchmark_fallback_understanding(
        "ergothioneine",
        RuntimeError("intent endpoint unavailable"),
    )

    assert understanding.is_paper_search is True
    assert understanding.topics == ["ergothioneine"]
    assert understanding.raw_json["parser_version"] == "benchmark-lexical-fallback-v1"
    assert meta["benchmark_intent_fallback"] is True
    assert meta["intent_provider"] == "benchmark-lexical-fallback"


class AdaptiveSearchAgentTests(unittest.TestCase):
    def test_retrieval_preserves_independent_intent_subqueries(self):
        from backend.app.agent.nodes import retrieve_papers

        captured: dict[str, QueryUnderstanding] = {}

        def academic(query, understanding, **kwargs):
            del query, kwargs
            captured["understanding"] = understanding
            return [
                _candidate("academic", index, "watermelon yield")
                for index in range(80)
            ], {"api_calls": 1, "providers": {}, "errors": {}}

        state = _initial_state()
        state["current_query"] = "watermelon yield research"
        state["current_iteration"] = 0
        with (
            patch("backend.app.agent.nodes.search_academic_papers", side_effect=academic),
            patch("backend.app.agent.nodes.search_papers_with_tavily") as tavily_mock,
        ):
            retrieve_papers(state)

        round_understanding = captured["understanding"]
        self.assertIn("watermelon yield methods", round_understanding.sub_queries)
        self.assertIn("watermelon cultivation yield", round_understanding.sub_queries)
        self.assertIn("西瓜增产的方法", round_understanding.sub_queries)
        tavily_mock.assert_not_called()

    def test_real_gap_can_use_one_budgeted_llm_query_evolution(self):
        from backend.app.agent.nodes import evolve_query

        state = _initial_state()
        state.update(
            {
                "llm_evolution_enabled": True,
                "stop_reason": "coverage_or_recall_gap",
                "current_iteration": 0,
                "executed_queries": ["watermelon yield research"],
                "search_plan": ["watermelon yield research", "watermelon methods"],
                "coverage_gaps": ["field experiment"],
                "api_calls": 5,
                "token_usage": 100,
            }
        )
        with patch(
            "backend.app.agent.nodes.evolve_query_with_llm",
            return_value=(
                "watermelon field experiment yield",
                "补充田间实验表达",
                {"api_calls": 1, "token_usage": 30},
            ),
        ):
            result = evolve_query(state)

        self.assertEqual(result["current_query"], "watermelon field experiment yield")
        self.assertEqual(result["api_calls"], 6)
        self.assertEqual(result["token_usage"], 130)
        self.assertEqual(result["llm_evolution_calls"], 1)
        self.assertEqual(result["agent_trace"][0]["tool"], "llm-query-evolution")

    def test_embedding_budget_scales_only_for_larger_result_windows(self):
        settings = get_settings()
        self.assertEqual(
            _embedding_candidate_limit(5),
            min(settings.bge_m3_max_candidate_limit, settings.bge_m3_candidate_limit),
        )
        self.assertEqual(
            _embedding_candidate_limit(50),
            min(
                settings.bge_m3_max_candidate_limit,
                max(settings.bge_m3_candidate_limit, 150),
            ),
        )
        self.assertEqual(
            _embedding_candidate_limit(5, evaluation_fast_mode=True),
            min(
                settings.paper_test_bge_m3_max_candidate_limit,
                settings.paper_test_bge_m3_candidate_limit,
            ),
        )

    def test_runs_second_round_when_first_round_has_recall_gap(self):
        round_counter = {"value": 0}

        def academic(query, understanding, **kwargs):
            round_index = round_counter["value"]
            round_counter["value"] += 1
            papers = [
                _candidate(f"academic-{round_index}", index, query)
                for index in range(30)
            ]
            return papers, {
                "api_calls": 5,
                "source_counts": {"openalex": 30},
                "errors": {},
            }

        def tavily(query, understanding, max_results=None):
            round_index = round_counter["value"] - 1
            papers = [
                _candidate(f"tavily-{round_index}", index, query)
                for index in range(10)
            ]
            return papers, {"api_calls": 1, "tavily_raw_count": 10}

        with (
            patch("backend.app.agent.nodes.search_academic_papers", side_effect=academic),
            patch("backend.app.agent.nodes.search_papers_with_tavily", side_effect=tavily),
        ):
            final = paper_search_graph.invoke(_initial_state())

        self.assertEqual(len(final["candidates"]), 80)
        self.assertEqual(len(final["round_history"]), 2)
        self.assertEqual(len(final["executed_queries"]), 2)
        self.assertEqual(final["current_iteration"], 1)
        self.assertEqual(final["stop_reason"], "target_and_coverage_met")
        self.assertEqual(final["api_calls"], 12)
        self.assertEqual(final["token_usage"], 0)
        self.assertEqual(
            [item["api_calls"] for item in final["round_history"]],
            [6, 6],
        )
        self.assertEqual(
            [item["token_usage"] for item in final["round_history"]],
            [0, 0],
        )
        retrieve_steps = [
            step
            for step in final["agent_trace"]
            if step["step"].startswith("agent:retrieve:")
        ]
        self.assertEqual(
            [step["cumulative_api_calls"] for step in retrieve_steps],
            [6, 12],
        )
        self.assertEqual(
            [step["cumulative_token_usage"] for step in retrieve_steps],
            [0, 0],
        )
        self.assertTrue(any("agent:evolve:2" == step["step"] for step in final["agent_trace"]))

    def test_stops_after_first_round_when_target_and_coverage_are_met(self):
        def academic(query, understanding, **kwargs):
            papers = [_candidate("academic", index, query) for index in range(80)]
            return papers, {
                "api_calls": 5,
                "source_counts": {"openalex": 80},
                "errors": {},
            }

        with (
            patch("backend.app.agent.nodes.search_academic_papers", side_effect=academic),
            patch(
                "backend.app.agent.nodes.search_papers_with_tavily",
                return_value=([], {"api_calls": 1, "tavily_raw_count": 0}),
            ) as tavily_mock,
        ):
            final = paper_search_graph.invoke(_initial_state())

        self.assertEqual(len(final["round_history"]), 1)
        self.assertEqual(final["current_iteration"], 0)
        self.assertEqual(final["stop_reason"], "target_and_coverage_met")
        self.assertFalse(final["should_continue"])
        self.assertEqual(final["round_history"][0]["api_calls"], 5)
        self.assertEqual(final["round_history"][0]["token_usage"], 0)
        tavily_mock.assert_not_called()

    def test_evaluation_runs_two_distinct_variants_before_early_stop(self):
        queries = []

        def academic(query, understanding, **kwargs):
            queries.append(query)
            suffix = len(queries)
            return [
                _candidate(f"academic-{suffix}", index, query)
                for index in range(80)
            ], {
                "api_calls": 5,
                "source_counts": {"openalex": 80},
                "errors": {},
            }

        state = _initial_state()
        state["evaluation_fast_mode"] = True
        with (
            patch("backend.app.agent.nodes.search_academic_papers", side_effect=academic),
            patch(
                "backend.app.agent.nodes.search_papers_with_tavily",
                return_value=([], {"api_calls": 0, "tavily_raw_count": 0}),
            ),
        ):
            final = paper_search_graph.invoke(state)

        self.assertEqual(len(final["round_history"]), 2)
        self.assertEqual(len(queries), 2)
        self.assertNotEqual(queries[0].casefold(), queries[1].casefold())


class _Store:
    def __init__(self, candidates):
        self.candidates = candidates

    def upsert_papers(self, session_id, papers):
        return {
            "upserted": len(papers),
            "embedding_provider": "bge-m3",
            "dense_dim": 1024,
        }

    def search_top_k(self, query, session_id, top_k):
        rows = []
        for index, paper in enumerate(self.candidates[:top_k]):
            item = dict(paper)
            item["relevance_score"] = 1.0 - (index / max(top_k, 1))
            rows.append(item)
        return rows, {
            "embedding_provider": "bge-m3",
            "retriever": "dense+sparse+rrf",
            "rrf_k": 60,
        }


class _Reranker:
    def rerank(self, query, papers, top_k, *, candidate_limit=None):
        del candidate_limit
        return papers[:top_k], {
            "reranker": "bge-reranker",
            "applied": True,
            "candidates": len(papers),
        }


class SearchServiceAgentIntegrationTests(unittest.TestCase):
    def test_authority_and_recency_are_opt_in_ranking_preferences(self):
        self.assertIsNone(EvidRankPipeline._explicit_preference_mode("papers about retrieval"))
        self.assertEqual(
            EvidRankPipeline._explicit_preference_mode("seminal highly cited retrieval papers"),
            "authority",
        )
        self.assertEqual(
            EvidRankPipeline._explicit_preference_mode("latest frontier retrieval papers"),
            "recency",
        )

    def test_embedding_budget_preserves_source_and_round_diversity(self):
        candidates = []
        for source in ("openalex", "arxiv", "pubmed", "serper", "tavily"):
            for iteration in (0, 1):
                for index in range(15):
                    paper = _candidate(source, iteration * 100 + index, "query")
                    paper["source_iteration"] = iteration
                    candidates.append(paper)

        selected = _limit_embedding_candidates(candidates, 100)

        self.assertEqual(len(selected), 100)
        self.assertEqual(
            {paper["source"] for paper in selected},
            {"openalex", "arxiv", "pubmed", "serper", "tavily"},
        )
        self.assertEqual({paper["source_iteration"] for paper in selected}, {0, 1})

    def test_embedding_budget_retains_exact_dataset_match_from_the_tail(self):
        candidates = [
            _candidate("openalex", index, "generic language model")
            for index in range(80)
        ]
        target = _candidate("pasa-local-corpus", 999, "reasoning benchmark")
        target["title"] = "Evaluating Reasoning Systems on HotPotQA"
        target["abstract"] = "An experimental study using the HotPotQA dataset."
        candidates.append(target)

        selected = _limit_embedding_candidates(
            candidates,
            10,
            query="LLM experiments on HotPotQA",
            understanding=QueryUnderstanding(datasets=["HotPotQA"]),
        )

        self.assertIn(target, selected)

    def test_embedding_budget_prefers_concise_exact_anchor_over_long_title(self):
        candidates = []
        for index in range(15):
            paper = _candidate("openalex", index, "generic nutrition study")
            paper["provider_ranks"] = {"openalex": index + 1}
            candidates.append(paper)
        concise = _candidate("openalex", 100, "ergothioneine")
        concise["title"] = "Ergothioneine"
        concise["provider_ranks"] = {"openalex": 100}
        verbose = _candidate("openalex", 101, "ergothioneine")
        verbose["title"] = (
            "A very long generic review of dietary compounds including "
            "ergothioneine and many unrelated nutrients"
        )
        verbose["provider_ranks"] = {"openalex": 101}
        candidates.extend([verbose, concise])

        selected = _limit_embedding_candidates(
            candidates,
            10,
            query="ergothioneine",
            understanding=QueryUnderstanding(entities=["ergothioneine"]),
        )

        self.assertIn(concise, selected)
        self.assertLess(selected.index(concise), selected.index(verbose))

    def test_embedding_budget_is_independent_of_provider_completion_order(self):
        candidates = []
        for source in ("openalex", "arxiv", "semantic-scholar"):
            for index in range(20):
                paper = _candidate(source, index, "query")
                paper["provider_ranks"] = {source: index + 1}
                candidates.append(paper)

        shuffled = list(candidates)
        random.Random(42).shuffle(shuffled)
        first = _limit_embedding_candidates(candidates, 25)
        second = _limit_embedding_candidates(shuffled, 25)

        self.assertEqual(
            [paper["id"] for paper in first],
            [paper["id"] for paper in second],
        )

    def test_embedding_budget_protects_each_provider_head(self):
        candidates = []
        for index in range(30):
            paper = _candidate("openalex", index, "HotPotQA reasoning")
            paper["provider_ranks"] = {"openalex": index + 1}
            paper["title"] = f"HotPotQA reasoning language model {index}"
            candidates.append(paper)
        semantic_head = _candidate("semantic-scholar", 999, "unrelated wording")
        semantic_head["provider_ranks"] = {"semantic-scholar": 1}
        semantic_head["title"] = "An opaque benchmark study"
        semantic_head["abstract"] = "No query terms are present here."
        candidates.append(semantic_head)

        selected = _limit_embedding_candidates(
            candidates,
            10,
            query="HotPotQA reasoning language model",
        )

        self.assertIn(semantic_head, selected)

    def test_embedding_budget_applies_relevance_quota_after_provider_heads(self):
        candidates = []
        for source in ("openalex", "arxiv"):
            for index in range(20):
                paper = _candidate(source, index, "generic topic")
                paper["provider_ranks"] = {source: index + 1}
                candidates.append(paper)
        relevant_tail = []
        for index in range(10):
            paper = _candidate("semantic-scholar", 100 + index, "generic topic")
            paper["provider_ranks"] = {"semantic-scholar": 20 + index}
            paper["title"] = f"Exact HotPotQA reasoning benchmark {index}"
            paper["abstract"] = "Evaluation of language models on HotPotQA."
            relevant_tail.append(paper)
            candidates.append(paper)

        selected = _limit_embedding_candidates(
            candidates,
            20,
            query="language model reasoning on HotPotQA",
            understanding=QueryUnderstanding(datasets=["HotPotQA"]),
        )

        # Provider heads remain protected and the relevance pass receives its
        # configured quota in addition to them, instead of sharing that quota.
        self.assertIn(candidates[0], selected)
        self.assertGreaterEqual(sum(item in selected for item in relevant_tail), 8)

    def test_production_search_consumes_agent_candidates_before_bge(self):
        candidates = [_candidate("agent", index, "watermelon yield") for index in range(80)]
        understanding = QueryUnderstanding(
            intent="检索西瓜增产方法",
            topics=["watermelon", "yield"],
            rewritten_queries=["watermelon yield"],
        )
        agent_state = {
            "candidates": candidates,
            "current_iteration": 1,
            "executed_queries": ["watermelon yield", "watermelon cultivation"],
            "stop_reason": "target_and_coverage_met",
            "coverage": {"rate": 1.0, "covered": ["watermelon", "yield"], "total": 2},
            "coverage_gaps": [],
            "round_history": [{"iteration": 1}, {"iteration": 2}],
            "api_calls": 12,
            "token_usage": 0,
            "agent_trace": [
                {
                    "step": "agent:decide:2",
                    "detail": "停止检索：target_and_coverage_met",
                    "status": "done",
                    "phase": "plan",
                    "tool": "adaptive-stop-controller",
                    "iteration": 2,
                    "cumulative_api_calls": 12,
                    "cumulative_token_usage": 0,
                }
            ],
        }
        pipeline = EvidRankPipeline(
            encoder=FakeEncoder(),
            reranker=FakeReranker(),
            evidence_judge=DeterministicEvidenceJudge(),
        )
        with (
            patch(
                "backend.app.services.search_service.parse_query_intent",
                return_value=(
                    understanding,
                    {
                        "api_calls": 1,
                        "token_usage": 10,
                        "intent_provider": "test",
                        "local_year": 2026,
                    },
                ),
            ),
            patch.object(
                paper_search_graph,
                "stream",
                return_value=iter([agent_state]),
            ),
            patch(
                "backend.app.services.search_service.get_evidrank_pipeline",
                return_value=pipeline,
            ),
            patch("backend.app.services.search_service.save_search_result"),
            patch("backend.app.services.search_service.save_output_artifact"),
        ):
            events = list(
                iter_paper_search_events(
                    "西瓜增产的方法",
                    max_papers=5,
                    use_mock=False,
                )
            )

        result = next(event["result"] for event in events if event.get("type") == "result")
        self.assertEqual(len(result["papers"]), 5)
        self.assertEqual(result["meta"]["candidate_count_before_bge"], 80)
        self.assertEqual(result["meta"]["search_agent"]["iterations"], 2)
        self.assertEqual(
            result["meta"]["search_agent"]["stop_reason"],
            "target_and_coverage_met",
        )
        self.assertEqual(result["meta"]["api_calls"], 13)
        self.assertEqual(result["meta"]["token_usage"], 10)
        self.assertEqual(result["meta"]["search_agent"]["api_calls"], 12)
        self.assertEqual(result["meta"]["search_agent"]["token_usage"], 0)
        self.assertEqual(
            result["meta"]["search_agent"]["budget"]["max_api_calls"],
            12,
        )
        self.assertEqual(result["meta"]["cost_breakdown"]["intent_api_calls"], 1)
        self.assertEqual(result["meta"]["cost_breakdown"]["intent_token_usage"], 10)
        self.assertEqual(result["meta"]["cost_breakdown"]["agent_api_calls"], 12)
        self.assertEqual(result["meta"]["cost_breakdown"]["agent_token_usage"], 0)
        self.assertEqual(len(result["meta"]["candidate_trace"]["merged"]), 80)
        self.assertEqual(len(result["meta"]["candidate_trace"]["budget"]), 80)
        self.assertEqual(len(result["meta"]["candidate_trace"]["ranking"]), 80)
        self.assertEqual(len(result["meta"]["candidate_trace"]["final"]), 5)
        self.assertTrue(result["ranking_bundle"]["is_frozen"])
        self.assertEqual(
            result["ranking_bundle"]["bundle_hash"],
            result["output_artifact"]["bundle_hash"],
        )

    def test_partial_candidate_pool_still_enters_bge_pipeline(self):
        candidates = [_candidate("agent", index, "watermelon yield") for index in range(35)]
        understanding = QueryUnderstanding(
            intent="检索西瓜增产方法",
            topics=["watermelon", "yield"],
            rewritten_queries=["watermelon yield"],
        )
        agent_state = {
            "candidates": candidates,
            "current_iteration": 1,
            "executed_queries": ["watermelon yield", "watermelon cultivation"],
            "stop_reason": "max_iterations_reached",
            "coverage": {"rate": 1.0, "covered": ["watermelon", "yield"], "total": 2},
            "coverage_gaps": [],
            "round_history": [{"iteration": 1}, {"iteration": 2}],
            "api_calls": 12,
            "token_usage": 0,
            "agent_trace": [],
        }
        pipeline = EvidRankPipeline(
            encoder=FakeEncoder(),
            reranker=FakeReranker(),
            evidence_judge=DeterministicEvidenceJudge(),
        )
        with (
            patch(
                "backend.app.services.search_service.parse_query_intent",
                return_value=(
                    understanding,
                    {
                        "api_calls": 1,
                        "token_usage": 10,
                        "intent_provider": "test",
                        "local_year": 2026,
                    },
                ),
            ),
            patch.object(
                paper_search_graph,
                "stream",
                return_value=iter([agent_state]),
            ),
            patch(
                "backend.app.services.search_service.get_evidrank_pipeline",
                return_value=pipeline,
            ),
            patch("backend.app.services.search_service.save_search_result"),
            patch("backend.app.services.search_service.save_output_artifact"),
        ):
            events = list(
                iter_paper_search_events(
                    "西瓜增产的方法",
                    max_papers=5,
                    use_mock=False,
                )
            )

        result = next(event["result"] for event in events if event.get("type") == "result")
        self.assertEqual(len(result["papers"]), 5)
        self.assertEqual(result["meta"]["candidate_count_before_bge"], 35)
        self.assertEqual(result["meta"]["embedding_candidate_count"], 35)
        self.assertFalse(result["meta"]["academic_target_met_after_fallback"])


if __name__ == "__main__":
    unittest.main()
