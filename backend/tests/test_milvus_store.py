"""Tests for session-scoped Milvus retrieval and fallback behavior."""

import unittest

from backend.app.services.milvus_store import MilvusPaperStore, _truncate_utf8


def _hit(session_id: str, paper_id: str = "paper-1", distance: float = 0.8):
    return {
        "id": f"{session_id}:{paper_id}",
        "distance": distance,
        "entity": {
            "session_id": session_id,
            "title": "Crop yield paper",
            "abstract": "Agricultural crop yield improvement.",
            "url": "https://example.org/paper/1",
            "published_at": "2024",
            "year": 2024,
        },
    }


class _Encoder:
    def encode_query(self, query):
        return [1.0, 0.0], {1: 1.0}, "test-encoder"


class _UpsertEncoder:
    def encode(self, texts):
        return (
            [[1.0, 0.0] for _ in texts],
            [{1: 1.0} for _ in texts],
            "test-encoder",
        )


class _Client:
    def __init__(self, hybrid_results):
        self.hybrid_results = hybrid_results
        self.hybrid_reqs = []
        self.hybrid_kwargs = {}
        self.search_calls = 0

    def hybrid_search(self, **kwargs):
        self.hybrid_reqs = kwargs["reqs"]
        self.hybrid_kwargs = kwargs
        return self.hybrid_results

    def search(self, **kwargs):
        self.search_calls += 1
        session_id = kwargs["filter"].split('"')[1]
        return [[_hit(session_id)]]


class _UpsertClient:
    def __init__(self):
        self.rows = []

    def upsert(self, **kwargs):
        self.rows = kwargs["data"]

    def flush(self, collection):
        return None

    def load_collection(self, collection):
        return None


def _store(client):
    store = MilvusPaperStore.__new__(MilvusPaperStore)
    store.collection = "papers"
    store.rrf_k = 60
    store.recall_limit = 20
    store.encoder = _Encoder()
    store.client = client
    return store


def _upsert_store(client):
    store = MilvusPaperStore.__new__(MilvusPaperStore)
    store.collection = "papers"
    store.dim = 2
    store.encoder = _UpsertEncoder()
    store.client = client
    return store


class MilvusSessionRetrievalTests(unittest.TestCase):
    def test_truncates_multibyte_text_by_utf8_bytes(self):
        text = "西瓜增产" * 1000

        truncated = _truncate_utf8(text, 4096)

        self.assertLessEqual(len(truncated.encode("utf-8")), 4096)
        self.assertTrue(text.startswith(truncated))

    def test_upsert_keeps_all_varchars_within_schema_limits(self):
        client = _UpsertClient()
        paper = {
            "id": "paper-1",
            "title": "西" * 500,
            "abstract": "瓜" * 3000,
            "url": "https://example.org/" + ("路" * 500),
            "published_at": "年" * 100,
            "year": 2024,
        }

        meta = _upsert_store(client).upsert_papers("session-1", [paper])

        row = client.rows[0]
        self.assertLessEqual(len(row["title"].encode("utf-8")), 1024)
        self.assertLessEqual(len(row["abstract"].encode("utf-8")), 4096)
        self.assertLessEqual(len(row["url"].encode("utf-8")), 1024)
        self.assertLessEqual(len(row["published_at"].encode("utf-8")), 64)
        self.assertEqual(meta["truncated_varchar_fields"], 4)

    def test_applies_session_filter_to_each_ann_request(self):
        session_id = "current-session"
        client = _Client([[_hit(session_id)]])

        papers, meta = _store(client).search_top_k("crop yield", session_id, 5)

        expected = f'session_id == "{session_id}"'
        self.assertEqual([request.expr for request in client.hybrid_reqs], [expected, expected])
        self.assertNotIn("filter", client.hybrid_kwargs)
        self.assertEqual(len(papers), 1)
        self.assertEqual(meta["retriever"], "dense+sparse+rrf")
        self.assertEqual(client.search_calls, 0)

    def test_falls_back_to_dense_when_hybrid_returns_no_hits(self):
        client = _Client([[]])

        papers, meta = _store(client).search_top_k("crop yield", "current-session", 5)

        self.assertEqual(len(papers), 1)
        self.assertEqual(meta["retriever"], "dense-only-fallback")
        self.assertEqual(meta["fallback_reason"], "hybrid_search_empty")
        self.assertEqual(client.search_calls, 1)


if __name__ == "__main__":
    unittest.main()
