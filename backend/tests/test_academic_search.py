"""Tests for multi-source academic retrieval and canonical deduplication."""

import unittest
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.ranking.adapters import candidates_from_rows
from backend.app.services.academic_search import (
    _arxiv_query,
    _openalex_abstract,
    _stable_id,
    _use_pubmed,
    build_query_variants,
    merge_academic_candidates,
    search_academic_papers,
    search_semantic_scholar,
    search_semantic_scholar_neighborhood,
)
from backend.app.services.local_pasa_corpus import search_local_pasa_corpus
from backend.app.services.academic_http import (
    ProviderRequestError,
    academic_http_request,
    reset_academic_http_state,
    request_attempts,
)


def _paper(source: str, index: int, *, doi: str | None = None, abstract: str = "abstract"):
    external_ids = {"doi": doi or f"10.1234/{source}.{index}"}
    return {
        "id": f"{source}-{index}",
        "title": f"{source} paper {index}",
        "abstract": abstract,
        "authors": [source],
        "year": 2024,
        "published_at": "2024",
        "venue": source,
        "citation_count": index,
        "url": f"https://example.org/{source}/{index}",
        "tags": [],
        "source": source,
        "sources": [source],
        "external_ids": external_ids,
        "score": 1.0 / (index + 1),
    }


class AcademicCandidateMergeTests(unittest.TestCase):
    def test_merges_by_doi_and_preserves_provenance(self):
        first = _paper("openalex", 1, doi="10.1234/shared", abstract="short")
        second = _paper(
            "pubmed",
            2,
            doi="10.1234/shared",
            abstract="a substantially longer authoritative abstract",
        )
        second["title"] = "Different punctuation in the same paper"
        second["citation_count"] = 99

        merged = merge_academic_candidates([first, second])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["sources"], ["openalex", "pubmed"])
        self.assertEqual(merged[0]["citation_count"], 99)
        self.assertEqual(merged[0]["abstract"], second["abstract"])

    def test_merges_rows_with_the_same_existing_canonical_id(self):
        first = _paper("openalex", 1)
        second = _paper("tavily", 2)
        first["external_ids"] = {}
        second["external_ids"] = {}
        first["id"] = "shared-canonical-id"
        second["id"] = "shared-canonical-id"

        merged = merge_academic_candidates([first, second])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["sources"], ["openalex", "tavily"])

    def test_same_title_with_different_years_is_not_force_merged(self):
        first = _paper("openalex", 1)
        second = _paper("tavily", 2)
        for item in (first, second):
            item["title"] = "A Reused Academic Title"
            item["external_ids"] = {}
        first["id"] = "first-provider-id"
        first["year"] = 2020
        second["id"] = "second-provider-id"
        second["year"] = 2024

        merged = merge_academic_candidates([first, second])

        self.assertEqual(len(merged), 2)
        self.assertEqual(len({item["id"] for item in merged}), 2)

    def test_arxiv_url_and_version_identifiers_merge(self):
        first = _paper("arxiv", 1)
        second = _paper("openalex", 2)
        first["external_ids"] = {"arxiv": "https://arxiv.org/abs/2401.01234v2"}
        second["external_ids"] = {"arxiv": "2401.01234"}
        first["title"] = "Provider title variant"
        second["title"] = "Canonical title"

        merged = merge_academic_candidates([first, second])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["external_ids"]["arxiv"], "2401.01234")

    def test_transitive_identity_bridge_cannot_leave_duplicate_stable_ids(self):
        arxiv_id = "2401.01234"
        doi = "10.1234/identity-bridge"

        arxiv_row = _paper("arxiv", 1)
        arxiv_row["title"] = "Shared Paper Title"
        arxiv_row["external_ids"] = {"arxiv": arxiv_id}
        arxiv_row["id"] = _stable_id(arxiv_row["external_ids"], arxiv_row["title"])

        doi_row = _paper("openalex", 2)
        doi_row["title"] = "Publisher Paper Title"
        doi_row["external_ids"] = {"doi": doi}
        doi_row["id"] = _stable_id(doi_row["external_ids"], doi_row["title"])

        # This row connects the first row by title/arXiv and the second by DOI.
        # The old first-match-only merge left two rows with the DOI-derived ID.
        bridge = _paper("serper-scholar", 3)
        bridge["title"] = arxiv_row["title"]
        bridge["external_ids"] = {"arxiv": arxiv_id, "doi": doi}
        bridge["id"] = "provider-bridge-row"

        merged = merge_academic_candidates([arxiv_row, doi_row, bridge])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["external_ids"], {"arxiv": arxiv_id, "doi": doi})
        self.assertEqual(merged[0]["id"], _stable_id(merged[0]["external_ids"], merged[0]["title"]))
        self.assertEqual(merged[0]["sources"], ["arxiv", "openalex", "serper-scholar"])
        self.assertEqual(len(candidates_from_rows(merged)), 1)

    def test_reconstructs_openalex_inverted_abstract(self):
        abstract = _openalex_abstract({"retrieval": [1], "Academic": [0], "works": [2]})
        self.assertEqual(abstract, "Academic retrieval works")


class ProviderQueryCompilerTests(unittest.TestCase):
    def test_semantic_scholar_search_normalizes_results_and_filters_year(self):
        request = httpx.Request(
            "GET",
            "https://api.semanticscholar.org/graph/v1/paper/search",
        )
        response = httpx.Response(
            200,
            request=request,
            json={
                "total": 1,
                "data": [
                    {
                        "paperId": "s2-paper-id",
                        "corpusId": 12345,
                        "url": "https://www.semanticscholar.org/paper/s2-paper-id",
                        "title": "Semantic Retrieval for Scientific Papers",
                        "abstract": "A dense and sparse retrieval study.",
                        "venue": "ACL",
                        "year": 2024,
                        "publicationDate": "2024-05-01",
                        "citationCount": 17,
                        "authors": [{"authorId": "1", "name": "Ada Researcher"}],
                        "externalIds": {
                            "DOI": "https://doi.org/10.1000/S2.TEST",
                            "ArXiv": "2401.01234",
                        },
                        "fieldsOfStudy": ["Computer Science"],
                    }
                ],
            },
        )
        settings = SimpleNamespace(
            semantic_scholar_api_key="secret-key",
            semantic_scholar_min_interval_seconds=0.0,
        )
        with (
            patch("backend.app.services.academic_search.get_settings", return_value=settings),
            patch(
                "backend.app.services.academic_search.academic_http_request",
                return_value=response,
            ) as request_mock,
        ):
            papers, meta = search_semantic_scholar(
                "scientific paper retrieval",
                QueryUnderstanding(
                    rewritten_queries=["semantic scientific paper retrieval"],
                    time_range="2024",
                ),
                limit=20,
            )

        self.assertEqual(meta["provider"], "semantic-scholar")
        self.assertEqual(meta["api_calls"], 1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["source"], "semantic-scholar")
        self.assertEqual(papers[0]["external_ids"]["doi"], "10.1000/s2.test")
        self.assertEqual(papers[0]["external_ids"]["arxiv"], "2401.01234")
        self.assertEqual(papers[0]["external_ids"]["semantic_scholar"], "s2-paper-id")
        self.assertEqual(papers[0]["external_ids"]["corpus_id"], "12345")
        call = request_mock.call_args
        self.assertEqual(call.kwargs["params"]["year"], "2024")
        self.assertEqual(call.kwargs["headers"]["x-api-key"], "secret-key")

    def test_semantic_scholar_neighborhood_uses_one_bounded_reference_hop(self):
        response = httpx.Response(
            200,
            request=httpx.Request(
                "GET",
                "https://api.semanticscholar.org/graph/v1/paper/seed/references",
            ),
            json={
                "data": [
                    {
                        "citedPaper": {
                            "paperId": "reference-id",
                            "corpusId": 99,
                            "title": "Foundational Retrieval Work",
                            "abstract": "A foundational paper.",
                            "year": 2020,
                            "citationCount": 12,
                            "authors": [{"name": "Ada"}],
                            "externalIds": {"ArXiv": "2001.00001"},
                            "fieldsOfStudy": ["Computer Science"],
                        }
                    }
                ]
            },
        )
        settings = SimpleNamespace(
            semantic_scholar_api_key="secret-key",
            semantic_scholar_min_interval_seconds=0.0,
            semantic_scholar_citation_expansion_enabled=True,
            semantic_scholar_citation_expansion_require_api_key=True,
        )
        with (
            patch("backend.app.services.academic_search.get_settings", return_value=settings),
            patch(
                "backend.app.services.academic_search.academic_http_request",
                return_value=response,
            ) as request_mock,
        ):
            papers, meta = search_semantic_scholar_neighborhood(
                "seed",
                "find foundational retrieval research",
                QueryUnderstanding(keywords=["retrieval"]),
                limit=20,
            )

        self.assertEqual(meta["direction"], "references")
        self.assertEqual(meta["api_calls"], 1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["citation_seed_id"], "seed")
        self.assertEqual(papers[0]["external_ids"]["semantic_scholar"], "reference-id")
        self.assertEqual(request_mock.call_args.kwargs["params"]["limit"], 20)

    def test_arxiv_query_removes_question_words_and_keeps_recall_branches(self):
        understanding = QueryUnderstanding(
            intent="paper search",
            topics=["in-context learning"],
            methods=["pre-training"],
        )
        dsl = _arxiv_query(
            "Give me papers about how large language models gain in-context learning during pre-training",
            understanding,
        )

        self.assertIn('all:"in-context learning"', dsl)
        self.assertIn(" OR ", dsl)
        self.assertNotIn('all:"Give"', dsl)
        self.assertNotIn('all:"papers"', dsl)

    def test_arxiv_query_drops_pure_chinese_facets_from_english_query(self):
        dsl = _arxiv_query(
            "large language model agents with memory modules",
            QueryUnderstanding(
                topics=["人工智能"],
                methods=["大语言模型 LLM"],
            ),
        )

        self.assertNotIn('all:"人工智能"', dsl)
        self.assertIn('all:"大语言模型 LLM"', dsl)

    def test_pubmed_is_routed_only_for_biomedical_queries(self):
        self.assertFalse(
            _use_pubmed(
                "large language model retrieval",
                QueryUnderstanding(topics=["information retrieval"], domain="computer science"),
            )
        )
        self.assertTrue(
            _use_pubmed(
                "protein treatment response",
                QueryUnderstanding(topics=["cancer therapy"], domain="biomedicine"),
            )
        )

    def test_query_variants_preserve_dataset_and_acronym_terms(self):
        variants = build_query_variants(
            "Find LLM methods evaluated through experiments on HotPotQA",
            QueryUnderstanding(
                topics=["large language models"],
                datasets=["HotPotQA"],
                methods=["chain-of-thought"],
            ),
        )

        self.assertTrue(any("HotPotQA" in value for value in variants))
        self.assertLessEqual(len(variants), 4)

    def test_local_pasa_title_index_retrieves_arxiv_metadata_without_qrels(self):
        understanding = QueryUnderstanding(datasets=["HotPotQA"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "id2paper.json"
            source.write_text(
                json.dumps(
                    {
                        "2401.00001": "A General Reasoning Agent",
                        "2401.00002": "Unrelated Computer Vision Paper",
                    }
                ),
                encoding="utf-8",
            )
            with zipfile.ZipFile(root / "cs_paper_2nd.zip", "w") as archive:
                archive.writestr(
                    "ageneralreasoningagent",
                    json.dumps(
                        {
                            "title": "A General Reasoning Agent",
                            "abstract": "We evaluate a large language model on the HotPotQA dataset.",
                        }
                    ),
                )
            with (
                patch(
                    "backend.app.services.local_pasa_corpus.resolve_pasa_title_corpus",
                    return_value=source,
                ),
                patch(
                    "backend.app.services.local_pasa_corpus._index_path",
                    return_value=root / "index.sqlite3",
                ),
            ):
                papers, meta = search_local_pasa_corpus(
                    "LLM experiments on HotPotQA",
                    understanding,
                    limit=5,
                )

        self.assertEqual(papers[0]["external_ids"]["arxiv"], "2401.00001")
        self.assertIn("HotPotQA", papers[0]["abstract"])
        self.assertEqual(meta["api_calls"], 0)
        self.assertTrue(meta["index_built"])
        self.assertEqual(meta["document_count"], 1)

    def test_local_pasa_search_keeps_raw_query_when_rewrite_drops_key_terms(self):
        understanding = QueryUnderstanding(
            sub_queries=[
                "generic language model pretraining research",
                "negative impact of reinforcement learning on supervised fine-tuned LLMs",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "id2paper.json"
            source.write_text(
                json.dumps(
                    {
                        "2401.00003": "Alignment Can Hurt Supervised Fine-Tuning",
                        "2401.00004": "A Survey of Language Models",
                    }
                ),
                encoding="utf-8",
            )
            with zipfile.ZipFile(root / "cs_paper_2nd.zip", "w") as archive:
                archive.writestr(
                    "alignmentcanhurtsupervisedfinetuning",
                    json.dumps(
                        {
                            "title": "Alignment Can Hurt Supervised Fine-Tuning",
                            "abstract": (
                                "Reinforcement learning can negatively impact the "
                                "performance of supervised fine-tuned language models."
                            ),
                        }
                    ),
                )
            with (
                patch(
                    "backend.app.services.local_pasa_corpus.resolve_pasa_title_corpus",
                    return_value=source,
                ),
                patch(
                    "backend.app.services.local_pasa_corpus._index_path",
                    return_value=root / "index.sqlite3",
                ),
            ):
                papers, _meta = search_local_pasa_corpus(
                    "generic language model pretraining research",
                    understanding,
                    limit=5,
                )

        self.assertEqual(papers[0]["external_ids"]["arxiv"], "2401.00003")

    def test_local_pasa_porter_index_matches_inflected_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "id2paper.json"
            source.write_text(
                json.dumps(
                    {
                        "2401.00005": "Evaluating Autonomous Agents",
                        "2401.00006": "A Vision Transformer Survey",
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "backend.app.services.local_pasa_corpus.resolve_pasa_title_corpus",
                    return_value=source,
                ),
                patch(
                    "backend.app.services.local_pasa_corpus._index_path",
                    return_value=root / "index.sqlite3",
                ),
            ):
                papers, _meta = search_local_pasa_corpus(
                    "evaluate autonomous agent",
                    QueryUnderstanding(),
                    limit=5,
                )

        self.assertEqual(papers[0]["external_ids"]["arxiv"], "2401.00005")


class AcademicHttpTests(unittest.TestCase):
    def tearDown(self):
        reset_academic_http_state()

    def test_transient_connection_error_retries_on_shared_client(self):
        request = httpx.Request("GET", "https://example.org")
        response = httpx.Response(200, request=request, json={"ok": True})

        class Client:
            def __init__(self):
                self.calls = 0

            def request(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise httpx.ConnectError("temporary", request=request)
                return response

        client = Client()
        with (
            patch(
                "backend.app.services.academic_http.get_academic_http_client",
                return_value=client,
            ),
            patch("backend.app.services.academic_http.time.sleep"),
        ):
            result = academic_http_request("arxiv", "GET", "https://example.org")

        self.assertEqual(client.calls, 2)
        self.assertEqual(request_attempts(result), 2)

    def test_repeated_failures_open_provider_circuit(self):
        request = httpx.Request("GET", "https://example.org")

        class Client:
            def request(self, *_args, **_kwargs):
                raise httpx.ConnectError("down", request=request)

        with (
            patch(
                "backend.app.services.academic_http.get_academic_http_client",
                return_value=Client(),
            ),
            patch("backend.app.services.academic_http.time.sleep"),
        ):
            with self.assertRaises(ProviderRequestError):
                academic_http_request("openalex", "GET", "https://example.org")
            with self.assertRaises(ProviderRequestError):
                academic_http_request("openalex", "GET", "https://example.org")
            with self.assertRaises(ProviderRequestError) as raised:
                academic_http_request("openalex", "GET", "https://example.org")

        self.assertTrue(raised.exception.circuit_open)
        self.assertEqual(raised.exception.attempts, 0)

    def test_429_honors_bounded_retry_after(self):
        request = httpx.Request("GET", "https://example.org")
        limited = httpx.Response(
            429,
            request=request,
            headers={"Retry-After": "2"},
        )
        success = httpx.Response(200, request=request, json={"ok": True})

        class Client:
            def __init__(self):
                self.responses = iter((limited, success))

            def request(self, *_args, **_kwargs):
                return next(self.responses)

        with (
            patch(
                "backend.app.services.academic_http.get_academic_http_client",
                return_value=Client(),
            ),
            patch("backend.app.services.academic_http.time.sleep") as sleep,
            patch("backend.app.services.academic_http.random.uniform", return_value=0.0),
        ):
            result = academic_http_request(
                "semantic-scholar",
                "GET",
                "https://example.org",
            )

        self.assertIs(result, success)
        sleep.assert_called_once_with(2.0)

    def test_429_is_reported_with_its_status_code(self):
        request = httpx.Request("GET", "https://example.org")

        class Client:
            def request(self, *_args, **_kwargs):
                return httpx.Response(429, request=request)

        with (
            patch(
                "backend.app.services.academic_http.get_academic_http_client",
                return_value=Client(),
            ),
            patch("backend.app.services.academic_http.time.sleep"),
        ):
            with self.assertRaises(ProviderRequestError) as raised:
                academic_http_request(
                    "semantic-scholar",
                    "GET",
                    "https://example.org",
                )

        self.assertEqual(raised.exception.error_type, "HTTP429")


class MultiProviderSearchTests(unittest.TestCase):
    def test_evaluation_cache_reuses_network_provider_responses(self):
        calls = {
            name: 0
            for name in ("serper-scholar", "semantic-scholar", "arxiv", "openalex")
        }

        def provider(source):
            def search(query, understanding, *, limit, offset=0):
                del query, understanding, limit
                calls[source] += 1
                papers = [_paper(source, offset)]
                return papers, {
                    "provider": source,
                    "raw_count": 1,
                    "returned_count": 1,
                    "api_calls": 1,
                }

            return search

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "backend.app.services.persistent_cache._cache_path",
                    return_value=Path(directory) / "cache.sqlite3",
                ),
                patch(
                    "backend.app.services.academic_search.resolve_pasa_title_corpus",
                    return_value=None,
                ),
                patch(
                    "backend.app.services.academic_search.search_serper_scholar",
                    provider("serper-scholar"),
                ),
                patch(
                    "backend.app.services.academic_search.search_arxiv",
                    provider("arxiv"),
                ),
                patch(
                    "backend.app.services.academic_search.search_openalex",
                    provider("openalex"),
                ),
                patch(
                    "backend.app.services.academic_search.search_semantic_scholar",
                    provider("semantic-scholar"),
                ),
            ):
                first = search_academic_papers(
                    "cached query",
                    QueryUnderstanding(topics=["retrieval"]),
                    target=1,
                    provider_limit=1,
                    allow_second_page=False,
                    use_cache=True,
                )[1]
                second = search_academic_papers(
                    "cached query",
                    QueryUnderstanding(topics=["retrieval"]),
                    target=1,
                    provider_limit=1,
                    allow_second_page=False,
                    use_cache=True,
                )[1]

        self.assertEqual(first["api_calls"], 4)
        self.assertEqual(second["api_calls"], 0)
        self.assertEqual(set(calls.values()), {1})

    def test_local_corpus_keeps_wider_zero_api_recall_window(self):
        seen_limits: dict[str, int] = {}

        def provider(source):
            def search(query, understanding, *, limit, offset=0):
                del query, understanding
                seen_limits[source] = limit
                papers = [_paper(source, offset + index) for index in range(limit)]
                return papers, {
                    "provider": source,
                    "raw_count": len(papers),
                    "returned_count": len(papers),
                    "api_calls": 0 if source == "pasa-local-corpus" else 1,
                }

            return search

        with (
            patch(
                "backend.app.services.academic_search.resolve_pasa_title_corpus",
                return_value=Path("/tmp/id2paper.json"),
            ),
            patch(
                "backend.app.services.academic_search.search_local_pasa_corpus",
                provider("pasa-local-corpus"),
            ),
            patch(
                "backend.app.services.academic_search.search_serper_scholar",
                provider("serper-scholar"),
            ),
            patch("backend.app.services.academic_search.search_arxiv", provider("arxiv")),
            patch(
                "backend.app.services.academic_search.search_openalex",
                provider("openalex"),
            ),
            patch(
                "backend.app.services.academic_search.search_semantic_scholar",
                provider("semantic-scholar"),
            ),
        ):
            search_academic_papers(
                "academic retrieval",
                QueryUnderstanding(topics=["retrieval"]),
                target=40,
                provider_limit=20,
                allow_second_page=False,
            )

        self.assertEqual(seen_limits["serper-scholar"], 20)
        self.assertEqual(seen_limits["pasa-local-corpus"], 100)

    def test_routes_general_query_to_four_relevant_providers(self):
        def provider(source):
            def search(query, understanding, *, limit, offset=0):
                papers = [_paper(source, offset + index) for index in range(25)]
                return papers, {
                    "provider": source,
                    "query": query,
                    "raw_count": len(papers),
                    "returned_count": len(papers),
                    "api_calls": 1,
                }

            return search

        understanding = QueryUnderstanding(
            intent="paper search",
            topics=["retrieval"],
            rewritten_queries=["academic retrieval"],
        )
        with (
            patch(
                "backend.app.services.academic_search.resolve_pasa_title_corpus",
                return_value=None,
            ),
            patch(
                "backend.app.services.academic_search.search_serper_scholar",
                provider("serper-scholar"),
            ),
            patch("backend.app.services.academic_search.search_arxiv", provider("arxiv")),
            patch("backend.app.services.academic_search.search_openalex", provider("openalex")),
            patch(
                "backend.app.services.academic_search.search_semantic_scholar",
                provider("semantic-scholar"),
            ),
            patch("backend.app.services.academic_search.search_pubmed", provider("pubmed")),
        ):
            papers, meta = search_academic_papers("academic retrieval", understanding)

        self.assertEqual(len(papers), 100)
        self.assertTrue(meta["target_met"])
        self.assertEqual(meta["target"], 50)
        self.assertEqual(
            set(meta["source_counts"]),
            {"serper-scholar", "semantic-scholar", "arxiv", "openalex"},
        )
        self.assertEqual(
            meta["routed_providers"],
            ["serper-scholar", "arxiv", "openalex", "semantic-scholar"],
        )
        self.assertEqual(meta["api_calls"], 4)

    def test_second_page_respects_api_call_budget(self):
        def provider(source):
            def search(query, understanding, *, limit, offset=0):
                papers = [_paper(source, offset)]
                return papers, {
                    "provider": source,
                    "query": query,
                    "raw_count": 1,
                    "returned_count": 1,
                    "api_calls": 1,
                }

            return search

        with (
            patch(
                "backend.app.services.academic_search.resolve_pasa_title_corpus",
                return_value=None,
            ),
            patch(
                "backend.app.services.academic_search.search_serper_scholar",
                provider("serper-scholar"),
            ),
            patch("backend.app.services.academic_search.search_arxiv", provider("arxiv")),
            patch("backend.app.services.academic_search.search_openalex", provider("openalex")),
            patch(
                "backend.app.services.academic_search.search_semantic_scholar",
                provider("semantic-scholar"),
            ),
        ):
            papers, meta = search_academic_papers(
                "rare retrieval task",
                QueryUnderstanding(topics=["retrieval"]),
                target=20,
                provider_limit=10,
                allow_second_page=True,
                api_call_budget=5,
            )

        self.assertEqual(meta["api_calls"], 5)
        self.assertTrue(meta["second_page_used"])
        self.assertEqual(len(papers), 5)


if __name__ == "__main__":
    unittest.main()
