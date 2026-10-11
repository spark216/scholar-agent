"""Tests for bounded and cached BGE-M3 encoding."""

import unittest
from collections import OrderedDict
from threading import Lock

from backend.app.services.embeddings import BGEM3Encoder


class _Model:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        return {
            "dense_vecs": [[float(index), 1.0] for index, _ in enumerate(texts)],
            "lexical_weights": [{"1": 1.0} for _ in texts],
        }


def _encoder(model):
    encoder = BGEM3Encoder.__new__(BGEM3Encoder)
    encoder.model_path = "test-model"
    encoder.dim = 2
    encoder.batch_size = 8
    encoder.max_length = 1024
    encoder.cache_size = 8
    encoder._model = model
    encoder._cache = OrderedDict()
    encoder._inference_lock = Lock()
    return encoder


class BGEEncodingPerformanceTests(unittest.TestCase):
    def test_deduplicates_batch_and_reuses_cached_vectors(self):
        model = _Model()
        encoder = _encoder(model)

        dense, sparse, provider = encoder.encode(["same", "same", "other"])
        encoder.encode(["same", "third"])

        self.assertEqual(provider, "bge-m3")
        self.assertEqual(len(dense), 3)
        self.assertEqual(len(sparse), 3)
        self.assertEqual(model.calls[0][0], ["same", "other"])
        self.assertEqual(model.calls[1][0], ["third"])
        self.assertEqual(model.calls[0][1]["batch_size"], 8)
        self.assertEqual(model.calls[0][1]["max_length"], 1024)

    def test_cache_evicts_least_recently_used_entry(self):
        model = _Model()
        encoder = _encoder(model)
        encoder.cache_size = 2

        encoder.encode(["one", "two"])
        encoder.encode(["one"])
        encoder.encode(["three"])

        self.assertEqual(len(encoder._cache), 2)
        self.assertIn(encoder._cache_key("one"), encoder._cache)
        self.assertIn(encoder._cache_key("three"), encoder._cache)
        self.assertNotIn(encoder._cache_key("two"), encoder._cache)

    def test_full_cache_does_not_evict_a_value_needed_by_current_batch(self):
        model = _Model()
        encoder = _encoder(model)
        encoder.cache_size = 2

        encoder.encode(["one", "two"])
        dense, sparse, provider = encoder.encode(["one", "three", "four"])

        self.assertEqual(provider, "bge-m3")
        self.assertEqual(len(dense), 3)
        self.assertEqual(len(sparse), 3)
        self.assertEqual(model.calls[1][0], ["three", "four"])

    def test_dense_dimension_mismatch_fails_closed(self):
        encoder = _encoder(_Model())
        encoder.dim = 3

        with self.assertRaisesRegex(RuntimeError, "dense dimension"):
            encoder.encode(["paper"])


if __name__ == "__main__":
    unittest.main()
