import pytest

from backend.app.services.ranking.adapters import candidates_from_rows
from backend.app.services.ranking.hybrid_ranker import (
    HybridCandidateRanker,
    reciprocal_rank_fusion,
)
from backend.tests.conftest import FakeEncoder


def test_rrf_uses_ranks_not_score_scales():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert scores["a"] == pytest.approx(scores["b"])
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)


def test_hybrid_scores_are_kept_separately(candidate_fixture):
    papers = candidates_from_rows(candidate_fixture)
    ranker = HybridCandidateRanker(FakeEncoder(), rrf_k=60)
    ranked, scores, meta = ranker.rank("Alpha RobotSet real robot", papers)
    assert ranked
    assert meta["provider"] == "fake-bge-m3"
    first = scores[ranked[0].canonical_paper_id]
    assert first.dense_score is not None
    assert first.sparse_score is not None
    assert first.rrf_score is not None
    assert first.dense_rank is not None
    assert first.sparse_rank is not None
    assert first.rrf_rank is not None


def test_hybrid_rrf_includes_allowlisted_local_fts_rank():
    class DensePrefersSecond:
        provider = "fake-bge-m3"
        model_revision = "fixture"
        resolved_model_path = "fixture://encoder"

        def encode(self, texts):
            del texts
            return [[0.0, 1.0], [1.0, 0.0]], [{1: 1.0}, {0: 1.0}], self.provider

        def encode_query(self, text):
            del text
            return [1.0, 0.0], {0: 1.0}, self.provider

    papers = candidates_from_rows(
        [
            {
                "id": "fts-first",
                "title": "Exact lexical result",
                "source": "pasa-local-corpus",
                "provider_ranks": {"pasa-local-corpus": 1},
            },
            {
                "id": "dense-first",
                "title": "Semantic result",
                "source": "openalex",
                "provider_ranks": {"openalex": 1},
            },
        ]
    )

    ranked, _scores, meta = HybridCandidateRanker(DensePrefersSecond()).rank(
        "exact lexical query",
        papers,
    )

    assert ranked[0].canonical_paper_id == "fts-first"
    assert meta["rrf_paths"] == [
        "bge_dense",
        "bge_sparse",
        "pasa-local-corpus",
    ]
    assert meta["lexical_provider_counts"] == {"pasa-local-corpus": 1}
