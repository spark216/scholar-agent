"""Regression tests for bounded, deduplicated reranker inference."""

from collections import OrderedDict

from backend.app.services.reranker import BGEReranker


def _reranker():
    reranker = BGEReranker.__new__(BGEReranker)
    reranker.enabled = True
    reranker.required = True
    reranker.model_path = "test-reranker"
    reranker.model_revision = "revision"
    reranker.max_length = 512
    reranker.top_n = 10
    reranker.cache_size = 8
    reranker.batch_size = 4
    reranker.device = "cpu"
    reranker.resolved_model_path = "test-reranker"
    reranker.provider = "bge-reranker"
    reranker._load_error = None
    reranker._model = object()
    reranker._score_cache = OrderedDict()
    reranker.calls = []

    def score_pairs(pairs):
        reranker.calls.append(pairs)
        return [1.0 for _ in pairs], [0.0 for _ in pairs]

    reranker._score_pairs = score_pairs
    return reranker


def test_reranker_deduplicates_identical_query_document_pairs_and_reuses_cache():
    reranker = _reranker()
    papers = [
        {"id": "p1", "title": "Same", "abstract": "Document", "rrf_score": 0.2},
        {"id": "p2", "title": "Same", "abstract": "Document", "rrf_score": 0.1},
    ]

    first, first_meta = reranker.rerank("query", papers, top_k=2)
    reranker.rerank("query", papers[:1], top_k=1)

    assert [paper["id"] for paper in first] == ["p1", "p2"]
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0]) == 1
    assert first_meta["cache_hits"] == 1
