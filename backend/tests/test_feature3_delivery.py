from __future__ import annotations

from collections import OrderedDict

import pytest

from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.llm_evidence_judge import ZhipuEvidenceJudge
from backend.app.services.ranking.pipeline import EvidRankPipeline
from backend.app.services.reranker import BGEReranker
from backend.tests.conftest import FakeEncoder, FakeReranker


def test_cross_encoder_scores_only_configured_top_n_and_preserves_rrf_tail():
    reranker = BGEReranker.__new__(BGEReranker)
    reranker.enabled = True
    reranker.required = True
    reranker.model_revision = "test-revision"
    reranker.model_path = "test-model"
    reranker.resolved_model_path = "fixture://reranker"
    reranker.top_n = 2
    reranker.batch_size = 2
    reranker.max_length = 128
    reranker.cache_size = 16
    reranker.device = "cpu"
    reranker.provider = "bge-reranker"
    reranker._model = object()
    reranker._score_cache = OrderedDict()
    captured: list[list[list[str]]] = []

    def score_pairs(pairs):
        captured.append(pairs)
        logits = [1.0 if "p1" in pair[1] else 2.0 for pair in pairs]
        return logits, [0.0 for _ in logits]

    reranker._score_pairs = score_pairs
    papers = [
        {"id": f"p{index}", "title": f"p{index}", "abstract": "", "rrf_score": 1 / index}
        for index in range(1, 5)
    ]

    ranked, meta = reranker.rerank("query", papers, top_k=4)

    assert [value["id"] for value in ranked] == ["p2", "p1", "p3", "p4"]
    assert sum(len(batch) for batch in captured) == 2
    assert meta["cross_input_count"] == 2
    assert meta["returned_count"] == 4


def test_evidence_batch_payload_requires_exact_paper_and_evidence_ids():
    expected = {"p1": {"c1", "c2"}, "p2": {"c1"}}
    valid = {
        "results": [
            {
                "paper_id": "p1",
                "records": [
                    {"evidence_key": "c1", "label": "support", "quoted_text": "x"},
                    {"evidence_key": "c2", "label": "unknown", "quoted_text": None},
                ],
            },
            {
                "paper_id": "p2",
                "records": [
                    {"evidence_key": "c1", "label": "violate", "quoted_text": "y"}
                ],
            },
        ]
    }
    parsed = ZhipuEvidenceJudge._validate_batch_payload(valid, expected)
    assert set(parsed) == {"p1", "p2"}

    missing_paper = {"results": valid["results"][:1]}
    with pytest.raises(ValueError, match="missing evidence paper_ids"):
        ZhipuEvidenceJudge._validate_batch_payload(missing_paper, expected)

    duplicate_key = {
        "results": [
            {
                "paper_id": "p1",
                "records": [
                    {"evidence_key": "c1", "label": "support"},
                    {"evidence_key": "c1", "label": "support"},
                    {"evidence_key": "c2", "label": "unknown"},
                ],
            },
            valid["results"][1],
        ]
    }
    with pytest.raises(ValueError, match="duplicate evidence_key"):
        ZhipuEvidenceJudge._validate_batch_payload(duplicate_key, expected)


def test_evidence_batch_accepts_partial_aliases_without_discarding_valid_records():
    expected = {"p1": {"c1", "c2"}, "p2": {"c1"}}
    partial = {
        "results": {
            "p1": [
                {
                    "facet_id": "c1",
                    "label": "支持",
                    "quoted_text": "exact quote",
                }
            ]
        }
    }
    parsed = ZhipuEvidenceJudge._validate_batch_payload(
        partial,
        expected,
        require_complete=False,
    )
    assert parsed == {
        "p1": {
            "c1": {
                "facet_id": "c1",
                "label": "support",
                "quoted_text": "exact quote",
            }
        }
    }


def test_pipeline_uses_batch_interface_in_bounded_chunks(intent_fixture, candidate_fixture):
    class CountingBatchJudge:
        provider = "counting-batch-judge"
        batch_size = 2

        def __init__(self):
            self.base = DeterministicEvidenceJudge()
            self.batch_sizes: list[int] = []

        def reset_metrics(self):
            return None

        def get_metrics(self):
            return {"api_calls": len(self.batch_sizes), "token_usage": 0, "errors": 0}

        def judge_batch(self, *, session_id, papers, intent):
            self.batch_sizes.append(len(papers))
            return {
                paper.canonical_paper_id: self.base.judge(
                    session_id=session_id,
                    paper=paper,
                    intent=intent,
                )
                for paper in papers
            }

        def judge(self, **kwargs):
            raise AssertionError("single-paper judge must not be used when judge_batch exists")

    judge = CountingBatchJudge()
    pipeline = EvidRankPipeline(
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        evidence_judge=judge,
    )
    pipeline.evidence_top_k = 3
    understanding = QueryUnderstanding.model_validate(intent_fixture["understanding"])
    bundle = pipeline.rank(
        session_id="batch-test",
        query=intent_fixture["query"],
        understanding=understanding,
        candidate_rows=candidate_fixture,
        top_k=4,
        include_rejected_in_scoring=True,
    )

    assert judge.batch_sizes == [2, 1]
    assert bundle.costs.evidence_api_calls == 2
