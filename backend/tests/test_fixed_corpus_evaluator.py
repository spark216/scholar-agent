from __future__ import annotations

from pathlib import Path

import pytest

from backend.scripts.evaluate_fixed_corpus import _load_qrels, _metric, _rrf


FIXTURES = Path(__file__).parent / "fixtures"


def test_metric_uses_exact_ids_and_does_not_count_duplicates_twice():
    metric = _metric(["doc-a", "doc-a", "doc-b", "same title"], {"doc-a": 2, "doc-c": 1}, 3)

    assert metric["hits"] == 1
    assert metric["precision"] == pytest.approx(1 / 3)
    assert metric["recall"] == pytest.approx(1 / 2)


def test_rrf_ignores_duplicate_id_inside_one_path():
    duplicated = _rrf([["a", "a", "b"], ["b", "a"]])
    deduplicated = _rrf([["a", "b"], ["b", "a"]])

    assert duplicated == deduplicated == ["a", "b"]


def test_load_qrels_keeps_max_grade_for_exact_query_document_pair():
    assert _load_qrels(FIXTURES / "fixed_qrels.tsv") == {"q1": {"d1": 2, "d2": 0}}


def test_load_qrels_rejects_empty_identity():
    with pytest.raises(ValueError, match="empty qid or docid"):
        _load_qrels(FIXTURES / "fixed_qrels_invalid.tsv")
