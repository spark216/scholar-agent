from __future__ import annotations

import json
from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def intent_fixture() -> dict:
    return json.loads((FIXTURE_DIR / "intent.json").read_text(encoding="utf-8"))


@pytest.fixture
def candidate_fixture() -> list[dict]:
    return json.loads((FIXTURE_DIR / "candidate_papers.json").read_text(encoding="utf-8"))


class FakeEncoder:
    provider = "fake-bge-m3"
    model_revision = "test-revision"
    resolved_model_path = "fixture://fake-bge-m3"

    @staticmethod
    def _vector(text: str) -> tuple[list[float], dict[int, float]]:
        value = text.casefold()
        dense = [
            float("alpha" in value),
            float("real robot" in value),
            float("robotset" in value),
            float("simulation only" in value),
        ]
        sparse = {
            index: component
            for index, component in enumerate(dense)
            if component
        }
        return dense, sparse

    def encode(self, texts: list[str]):
        values = [self._vector(text) for text in texts]
        return [v[0] for v in values], [v[1] for v in values], self.provider

    def encode_query(self, text: str):
        dense, sparse = self._vector(text)
        return dense, sparse, self.provider


class FakeReranker:
    model_revision = "test-reranker-revision"

    def rerank(
        self,
        query: str,
        papers: list[dict],
        top_k: int,
        *,
        candidate_limit: int | None = None,
    ):
        del candidate_limit
        logits = {
            "p-alpha-real": 8.0,
            "p-alpha-sim": 7.0,
            "p-alpha-unknown": 5.0,
        }
        scored = []
        for paper in papers:
            item = dict(paper)
            value = logits.get(str(item["id"]), 0.0)
            item["cross_logit"] = value
            item["cross_sigmoid"] = 1.0 / (1.0 + __import__("math").exp(-value))
            scored.append(item)
        scored.sort(key=lambda item: (-item["cross_logit"], item["id"]))
        return scored[:top_k], {
            "reranker": "fake-reranker",
            "model_revision": self.model_revision,
            "applied": True,
            "device": "cpu",
        }


class FakeSelector:
    """Deterministic stand-in for the PaSa 7B selector service."""

    provider = "fake-pasa-selector"
    enabled = True
    required = False
    _load_error = None
    last_inference_ms = None

    def infer_scores(self, query: str, papers: list[dict]):
        del query
        # Deterministic relevance: "alpha" papers score high, others low.
        scores = [0.95 if "alpha" in str(p.get("title") or "") else 0.10 for p in papers]
        self.last_inference_ms = 1.0
        return scores, {
            "provider": self.provider,
            "applied": True,
            "scored_count": len(scores),
            "inference_ms": 1.0,
        }


class DisabledFakeSelector(FakeSelector):
    enabled = False
    provider = "disabled"

    def infer_scores(self, query: str, papers: list[dict]):
        del query, papers
        return None, {"provider": self.provider, "applied": False, "scored_count": 0}
