"""Tests for the optional persistent PaSa full-corpus vector Provider."""

from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.local_vector_corpus import PaSaVectorCorpusStore, _time_filter


class _Encoder:
    def encode_query(self, _query):
        return [1.0, 0.0], {0: 1.0}, "test-bge-m3"


class _Client:
    def has_collection(self, _collection):
        return True

    def hybrid_search(self, **_kwargs):
        return [
            [
                {
                    "distance": 0.8,
                    "entity": {
                        "arxiv_id": "2401.01234",
                        "title": "A full corpus result",
                        "abstract": "Relevant abstract.",
                        "year": 2024,
                    },
                }
            ]
        ]


def _store():
    store = PaSaVectorCorpusStore.__new__(PaSaVectorCorpusStore)
    store.collection = "pasa-full"
    store.rrf_k = 60
    store.recall_limit = 20
    store.encoder = _Encoder()
    store.client = _Client()
    return store


def test_time_filter_uses_normalized_year_range():
    assert _time_filter(QueryUnderstanding(time_range="2020-2024")) == (
        "year >= 2020 and year <= 2024"
    )
    assert _time_filter(QueryUnderstanding(time_range="1995-2001")) == (
        "year >= 1995 and year <= 2001"
    )
    assert _time_filter(QueryUnderstanding(time_range="before 2000")) == "year < 2000"
    assert _time_filter(QueryUnderstanding(time_range="2024以后")) == "year >= 2024"
    assert _time_filter(QueryUnderstanding(time_range="2023")) == "year == 2023"
    assert _time_filter(QueryUnderstanding()) == ""


def test_full_corpus_search_returns_mergeable_arxiv_candidate():
    papers, meta = _store().search(
        "language model",
        QueryUnderstanding(time_range="2020-2026"),
        limit=5,
    )

    assert len(papers) == 1
    assert papers[0]["external_ids"] == {"arxiv": "2401.01234"}
    assert papers[0]["source"] == "pasa-local-vector"
    assert meta["embedding_provider"] == "test-bge-m3"
