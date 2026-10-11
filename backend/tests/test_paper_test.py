import json

import pytest

from backend.app.models.paper_test import PaperTestDatasetInfo, PaperTestRequest
from backend.app.models.schemas import (
    PaperItem,
    QueryUnderstanding,
    RelationGraph,
    SearchResponse,
    SearchSummary,
)
from backend.app.services.paper_test import (
    _build_report,
    _failed_query,
    _metric_at_k,
    _save_report,
    iter_paper_test_events,
    list_paper_test_datasets,
    list_paper_test_history,
    load_latest_paper_test_report,
    load_paper_test_report,
    normalize_arxiv_id,
    normalize_corpus_id,
    normalize_doi,
    normalize_pmid,
    normalize_title,
    score_search_response,
    validate_paper_test_sample_limit,
)
from backend.app.services.paper_test_asta import score_asta_response
from backend.app.services.persistent_cache import FrozenProviderCacheMiss


def _response(*papers: PaperItem) -> SearchResponse:
    return SearchResponse(
        query="query",
        understanding=QueryUnderstanding(),
        papers=list(papers),
        graph=RelationGraph(),
        summary=SearchSummary(overview="test"),
    )


def test_paper_test_request_normalizes_k_values():
    request = PaperTestRequest(ks=[10, 5, 10])
    assert request.ks == [5, 10]


def test_sample_limit_accepts_full_dataset_and_rejects_above_it(monkeypatch):
    dataset = PaperTestDatasetInfo(
        id="pasa-auto",
        name="PaSa AutoScholarQuery",
        available=True,
        sample_count=1000,
        source_path="fixture.jsonl",
        description="fixture",
    )
    monkeypatch.setattr(
        "backend.app.services.paper_test._dataset_info",
        lambda _dataset_id: dataset,
    )

    request = PaperTestRequest(dataset="pasa-auto", sample_limit=1000, ks=[7])
    assert validate_paper_test_sample_limit(request).sample_count == 1000
    with pytest.raises(ValueError, match="总量 1000"):
        validate_paper_test_sample_limit(
            PaperTestRequest(dataset="pasa-auto", sample_limit=1001, ks=[7])
        )


def test_identifier_and_title_normalization():
    assert normalize_arxiv_id("https://arxiv.org/pdf/2301.01234v2.pdf") == "2301.01234"
    assert normalize_arxiv_id("arXiv:2301.01234V3") == "2301.01234"
    assert normalize_title("  Attention—Is All You Need! ") == "attention is all you need"
    assert normalize_pmid("https://pubmed.ncbi.nlm.nih.gov/26287849/") == "26287849"
    assert normalize_pmid("PMID: 26287849") == "26287849"
    assert normalize_doi("https://doi.org/10.1000/ABC.Test") == "10.1000/abc.test"
    assert normalize_corpus_id("CorpusId:12345") == "12345"


def test_score_search_response_matches_doi_and_corpus_id_before_title():
    sample = {
        "qid": "q-identifiers",
        "question": "find target",
        "answer": ["First title can differ", "Second title can differ"],
        "answer_arxiv_id": ["", ""],
        "answer_doi": ["10.1000/target", ""],
        "answer_corpus_id": ["", "12345"],
    }
    response = _response(
        PaperItem(
            id="doi-match",
            title="Provider title differs",
            abstract="",
            doi="https://doi.org/10.1000/TARGET",
        ),
        PaperItem(
            id="corpus-match",
            title="Another provider title",
            abstract="",
            semantic_scholar_corpus_id="CorpusId:12345",
        ),
    )

    result = score_search_response(sample, response, [2], latency_ms=1)

    assert result.relevant_count == 2
    assert result.metrics["2"].hits == 2


def test_score_search_response_matches_arxiv_then_normalized_title():
    sample = {
        "qid": "q1",
        "question": "transformer papers",
        "answer": ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers"],
        "answer_arxiv_id": ["1706.03762", "1810.04805"],
    }
    response = _response(
        PaperItem(
            id="one",
            title="Different provider title",
            abstract="",
            arxiv_id="1706.03762v7",
        ),
        PaperItem(
            id="two",
            title="BERT — Pre-training of Deep Bidirectional Transformers",
            abstract="",
        ),
        PaperItem(id="three", title="Unrelated", abstract=""),
    )

    result = score_search_response(sample, response, [2, 3], latency_ms=12.5)

    assert result.relevant_count == 2
    assert result.metrics["2"].hits == 2
    assert result.metrics["2"].precision == 1.0
    assert result.metrics["2"].recall == 1.0
    assert result.metrics["2"].ndcg == 1.0
    assert result.metrics["2"].mrr == 1.0
    assert result.metrics["3"].precision == 2 / 3


def test_score_search_response_matches_trec_pm_by_pmid():
    sample = {
        "qid": "pm2017_1",
        "question": "disease: Liposarcoma; gene: CDK4 Amplification",
        "answer": [],
        "answer_arxiv_id": [],
        "answer_pmid": ["26287849", "26948001"],
    }
    response = _response(
        PaperItem(
            id="pubmed-26287849",
            title="Provider title may differ",
            abstract="",
            pmid="PMID:26287849",
        ),
        PaperItem(id="wrong", title="Unrelated", abstract="", pmid="1"),
    )

    result = score_search_response(sample, response, [2], latency_ms=1)

    assert result.relevant_count == 2
    assert result.metrics["2"].hits == 1
    assert result.metrics["2"].precision == 0.5
    assert result.metrics["2"].recall == 0.5
    assert result.matched_titles == ["26287849"]


def test_external_benchmark_dataset_is_discovered_from_package_root(monkeypatch, tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "pm2017.jsonl").write_text(
        '{"qid":"pm2017_1","question":"case","answer_pmid":["1"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("backend.app.services.paper_test._dataset_roots", lambda: [tmp_path])

    dataset = next(item for item in list_paper_test_datasets() if item.id == "trec-pm-2017")

    assert dataset.available is True
    assert dataset.sample_count == 1
    assert dataset.evaluation_family == "trec-pm"
    assert dataset.source_path == str(eval_dir / "pm2017.jsonl")


def test_report_records_latency_percentiles_and_cost_totals():
    dataset = PaperTestDatasetInfo(
        id="pasa-real",
        name="fixture",
        available=True,
        sample_count=2,
        source_path=__file__,
        description="fixture",
    )
    queries = []
    for index, latency in enumerate((10.0, 100.0), start=1):
        response = _response(PaperItem(id=str(index), title="Target", abstract=""))
        response.meta = {"api_calls": index, "token_usage": index * 10}
        queries.append(
            score_search_response(
                {"qid": f"q{index}", "question": "query", "answer": ["Target"]},
                response,
                [1],
                latency_ms=latency,
            )
        )

    report = _build_report(
        PaperTestRequest(sample_limit=2, ks=[1]),
        dataset,
        queries,
        "latency-cost",
    )

    assert report.p50_latency_ms == 10.0
    assert report.p95_latency_ms == 100.0
    assert report.total_api_calls == 3
    assert report.total_token_usage == 30


def test_report_separates_failures_from_successful_query_metrics():
    dataset = PaperTestDatasetInfo(
        id="pasa-real",
        name="fixture",
        available=True,
        sample_count=2,
        source_path=__file__,
        description="fixture",
    )
    success = score_search_response(
        {"qid": "ok", "question": "query", "answer": ["Target"]},
        _response(PaperItem(id="target", title="Target", abstract="")),
        [1],
        latency_ms=1,
    )
    failure = _failed_query(
        {"qid": "bad", "question": "query", "answer": ["Target"]},
        [1],
        1,
        "IntentClassificationError: intent service unavailable",
    )

    report = _build_report(
        PaperTestRequest(sample_limit=2, ks=[1]),
        dataset,
        [success, failure],
        "failure-breakdown",
    )

    assert report.aggregate["1"].f1 == 0.5
    assert report.aggregate_successful["1"].f1 == 1.0
    assert report.failure_rate == 0.5
    assert report.failure_breakdown == {"intent_or_query_validation": 1}


def test_one_gold_cannot_be_counted_twice_by_id_and_title_aliases():
    sample = {
        "qid": "q-one-to-one",
        "question": "find the target",
        "answer": ["Canonical Target Paper"],
        "answer_arxiv_id": ["2401.01234"],
    }
    response = _response(
        PaperItem(
            id="id-match",
            title="Provider title variant",
            abstract="",
            arxiv_id="2401.01234v2",
        ),
        PaperItem(
            id="title-match",
            title="Canonical Target Paper",
            abstract="",
        ),
    )

    result = score_search_response(sample, response, [2], latency_ms=1)

    assert result.relevant_count == 1
    assert result.metrics["2"].hits == 1
    assert result.metrics["2"].recall == 1.0


def test_conflicting_strong_identifier_blocks_title_fallback():
    sample = {
        "qid": "q-conflict",
        "question": "find target",
        "answer": ["Canonical Target Paper"],
        "answer_arxiv_id": ["2401.01234"],
    }
    response = _response(
        PaperItem(
            id="wrong-version",
            title="Canonical Target Paper",
            abstract="",
            arxiv_id="2401.99999",
        )
    )

    result = score_search_response(sample, response, [1], latency_ms=1)

    assert result.metrics["1"].hits == 0


def test_report_identifies_the_first_candidate_stage_containing_gold():
    sample = {
        "qid": "q-stage",
        "question": "find target",
        "answer": ["Target Paper"],
        "answer_arxiv_id": ["2401.01234"],
    }
    response = _response(PaperItem(id="wrong", title="Wrong Paper", abstract=""))
    response.meta["candidate_trace"] = {
        "provider": [{"id": "wrong", "title": "Wrong Paper", "external_ids": {}}],
        "merged": [
            {
                "id": "target",
                "title": "Provider title variant",
                "external_ids": {"arxiv": "2401.01234v2"},
            }
        ],
        "budget": [],
        "ranking": [],
    }
    response.meta["search_agent"] = {
        "round_history": [
            {
                "iteration": 1,
                "query": "find target",
                "academic": {
                    "providers": {
                        "arxiv": {
                            "attempts": [
                                {
                                    "query": 'all:"target"',
                                    "raw_count": 3,
                                    "returned_count": 3,
                                    "api_calls": 1,
                                    "latency_ms": 12.0,
                                }
                            ]
                        }
                    }
                },
                "tavily": {"api_calls": 0, "returned_count": 0},
                "errors": {},
            }
        ]
    }

    result = score_search_response(sample, response, [5], latency_ms=1)

    assert result.first_seen_stage == "merged"
    assert result.stage_metrics["provider"].candidate_recall == 0.0
    assert result.stage_metrics["merged"].candidate_recall == 1.0
    assert result.stage_metrics["merged"].matched_ranks == [1]
    assert result.stage_metrics["budget"].candidate_recall == 0.0
    assert result.loss_stage == "budget"
    assert result.provider_diagnostics[0].provider == "arxiv"
    assert result.provider_diagnostics[0].query == 'all:"target"'


def test_metric_at_k_uses_k_denominator_and_binary_ndcg():
    metric = _metric_at_k([0, 1], relevant_count=4, k=5)
    assert metric.hits == 1
    assert metric.precision == 0.2
    assert metric.recall == 0.25
    assert metric.mrr == 0.5
    assert 0 < metric.ndcg < 1


def test_iterator_calls_scholaragent_once_per_sample_and_aggregates(monkeypatch, tmp_path):
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "qid": "q1",
                "question": "find transformer paper",
                "answer": ["Attention Is All You Need"],
                "answer_arxiv_id": ["1706.03762"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = PaperTestDatasetInfo(
        id="pasa-real",
        name="fixture",
        available=True,
        sample_count=1,
        source_path=str(dataset_path),
        description="fixture",
    )
    monkeypatch.setattr("backend.app.services.paper_test._dataset_info", lambda _dataset_id: dataset)
    monkeypatch.setattr("backend.app.services.paper_test._save_report", lambda _report: None)
    calls = []

    def search_runner(query, *, max_papers, use_mock):
        calls.append((query, max_papers, use_mock))
        return _response(
            PaperItem(
                id="transformer",
                title="Attention Is All You Need",
                abstract="",
                arxiv_id="1706.03762v7",
            )
        )

    events = list(
        iter_paper_test_events(
            PaperTestRequest(sample_limit=1, ks=[5, 10]),
            search_runner=search_runner,
        )
    )

    assert calls == [("find transformer paper", 10, False)]
    assert [event["type"] for event in events] == [
        "start",
        "progress",
        "query_result",
        "result",
    ]
    report = events[-1]["report"]
    assert report["aggregate"]["5"]["hits"] == 1
    assert report["aggregate"]["5"]["precision"] == 0.2
    assert report["aggregate"]["5"]["recall"] == 1.0


def test_asta_semantic_metrics_use_corpus_ids_judge_and_normalizer(monkeypatch):
    response = _response(
        PaperItem(id="one", title="Perfect paper", abstract="evidence", doi="10.1/one"),
        PaperItem(id="two", title="Partial paper", abstract="evidence", doi="10.1/two"),
        PaperItem(id="three", title="Unresolved paper", abstract=""),
    )
    monkeypatch.setattr(
        "backend.app.services.paper_test_asta.resolve_corpus_ids",
        lambda _papers: ["101", "202", None],
    )
    monkeypatch.setattr(
        "backend.app.services.paper_test_asta._judge_semantic_relevance",
        lambda _papers, _ids, _criteria, _known: ({"101": 3, "202": 1}, "ok"),
    )
    sample = {
        "qid": "semantic_1",
        "question": "query",
        "_asta_total_relevant": 4,
        "_asta_criteria": {
            "known_to_be_good": [],
            "relevance_criteria": [
                {"name": "criterion", "description": "required", "weight": 1.0}
            ],
        },
    }

    scored = score_asta_response(sample, response)

    assert scored.perfect_flags == [1, 0, 0]
    assert scored.relevant_count == 4
    assert scored.official.estimated_recall == 0.25
    assert scored.official.corpus_id_validity == 2 / 3
    assert scored.official.perfectly_relevant == 1
    assert scored.official.adjusted_f1 is not None
    assert scored.official.judge_status == "ok"


def test_asta_specific_metrics_use_exact_official_corpus_ids(monkeypatch):
    response = _response(
        PaperItem(id="one", title="Target", abstract=""),
        PaperItem(id="two", title="Wrong", abstract=""),
    )
    monkeypatch.setattr(
        "backend.app.services.paper_test_asta.resolve_corpus_ids",
        lambda _papers: ["101", "999"],
    )
    sample = {
        "qid": "specific_1",
        "question": "find target",
        "_asta_total_relevant": 1,
        "_asta_criteria": {"corpus_ids": ["101"]},
    }

    scored = score_asta_response(sample, response)

    assert scored.official.standard_f1 == 2 / 3
    assert scored.official.estimated_recall == 1.0
    assert scored.official.corpus_id_validity == 1.0
    assert scored.official.ndcg == 1.0


def test_every_completed_run_is_saved_and_queryable(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    runs_dir = results_dir / "runs"
    monkeypatch.setattr("backend.app.services.paper_test.RESULTS_DIR", results_dir)
    monkeypatch.setattr("backend.app.services.paper_test.RUNS_DIR", runs_dir)
    dataset_path = tmp_path / "fixture.jsonl"
    dataset_path.write_text('{"fixture":true}\n', encoding="utf-8")
    dataset = PaperTestDatasetInfo(
        id="pasa-real",
        name="fixture",
        available=True,
        sample_count=1,
        source_path=str(dataset_path),
        description="fixture",
    )
    query_result = score_search_response(
        {
            "qid": "q1",
            "question": "query",
            "answer": ["Target"],
            "answer_arxiv_id": ["1706.03762"],
        },
        _response(PaperItem(id="one", title="Target", abstract="", arxiv_id="1706.03762")),
        [5],
        latency_ms=10,
    )
    request = PaperTestRequest(sample_limit=1, ks=[5])
    first = _build_report(request, dataset, [query_result], "run_one")
    second = _build_report(request, dataset, [query_result], "run_two")

    _save_report(first)
    _save_report(second)

    assert len(list(runs_dir.glob("*.json"))) == 2
    assert load_latest_paper_test_report().run_id == "run_two"
    history = list_paper_test_history(limit=10)
    assert history.total == 2
    assert {item.run_id for item in history.items} == {"run_one", "run_two"}
    assert all(item.max_k == 5 for item in history.items)
    assert all(item.seed == 42 for item in history.items)
    assert all(item.stage_recall["final"] == 1.0 for item in history.items)
    assert all(item.dataset_sha256 for item in history.items)
    assert all(item.config_fingerprint for item in history.items)
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.total_relevant == 1
    assert first.run_metadata.configuration["retrieval"]["local_index_version"].endswith(
        "porter-v3"
    )
    assert first.run_metadata.configuration["models"]["bge_m3"]["revision"]
    assert load_paper_test_report("run_one").aggregate["5"].f1 == pytest.approx(1 / 3)
    assert load_paper_test_report("../latest") is None


def test_frozen_provider_cache_miss_aborts_without_saving(monkeypatch, tmp_path):
    dataset_path = tmp_path / "fixture.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "qid": "q1",
                "question": "query",
                "answer": ["Target"],
                "answer_arxiv_id": ["1706.03762"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = PaperTestDatasetInfo(
        id="pasa-real",
        name="fixture",
        available=True,
        sample_count=1,
        source_path=str(dataset_path),
        description="fixture",
    )
    monkeypatch.setattr(
        "backend.app.services.paper_test.validate_paper_test_sample_limit",
        lambda _request: dataset,
    )
    monkeypatch.setattr("backend.app.services.paper_test.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("backend.app.services.paper_test.RUNS_DIR", tmp_path / "results" / "runs")

    def missing_runner(_query, **_kwargs):
        raise FrozenProviderCacheMiss("missing frozen entry")

    with pytest.raises(FrozenProviderCacheMiss, match="missing frozen entry"):
        list(
            iter_paper_test_events(
                PaperTestRequest(sample_limit=1, ks=[5], provider_cache_mode="frozen"),
                search_runner=missing_runner,
            )
        )
    assert not (tmp_path / "results" / "latest.json").exists()
