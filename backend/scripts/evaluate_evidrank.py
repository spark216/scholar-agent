"""Reproducible offline evaluation for feature-three EvidRank.

The default fixture mode is deterministic and does not download models.  Pass
``--real-models`` to evaluate the pinned local BGE-M3 and reranker snapshots on
the same labelled candidate pool.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from backend.app.models.evidrank import RankingBundle, RankingLevel
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.pipeline import EvidRankPipeline

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


class FixtureEncoder:
    """Small deterministic encoder used only to verify the evaluation path."""

    provider = "fixture-bge-m3"
    model_revision = "fixture-v1"
    resolved_model_path = "fixture://bge-m3"

    @staticmethod
    def _vector(text: str) -> tuple[list[float], dict[int, float]]:
        value = text.casefold()
        dense = [
            float("alpha" in value),
            float("real robot" in value),
            float("robotset" in value),
            float("simulation only" in value),
        ]
        sparse = {index: score for index, score in enumerate(dense) if score}
        return dense, sparse

    def encode(self, texts: list[str]):
        values = [self._vector(text) for text in texts]
        return [value[0] for value in values], [value[1] for value in values], self.provider

    def encode_query(self, text: str):
        dense, sparse = self._vector(text)
        return dense, sparse, self.provider


class FixtureReranker:
    model_revision = "fixture-v1"

    def rerank(
        self,
        query: str,
        papers: list[dict[str, Any]],
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
        scored: list[dict[str, Any]] = []
        for paper in papers:
            item = dict(paper)
            logit = logits.get(str(item["id"]), 0.0)
            item["cross_logit"] = logit
            item["cross_sigmoid"] = 1.0 / (1.0 + math.exp(-logit))
            scored.append(item)
        scored.sort(key=lambda item: (-item["cross_logit"], str(item["id"])))
        return scored[:top_k], {
            "reranker": "fixture-reranker",
            "model_revision": self.model_revision,
            "applied": True,
            "device": "cpu",
        }


def _dcg(gains: list[int]) -> float:
    return sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def evaluate_bundle(
    bundle: RankingBundle,
    qrels: dict[str, int],
    *,
    k: int,
) -> dict[str, Any]:
    ranked = [
        item
        for item in bundle.items
        if item.level != RankingLevel.EXCLUDED
    ][:k]
    gains = [int(qrels.get(item.paper_id, 0)) for item in ranked]
    ideal = sorted((int(value) for value in qrels.values()), reverse=True)[:k]
    relevant = {paper_id for paper_id, value in qrels.items() if int(value) > 0}
    retrieved_relevant = {item.paper_id for item in ranked if int(qrels.get(item.paper_id, 0)) > 0}
    first_relevant_rank = next(
        (
            rank
            for rank, item in enumerate(ranked, start=1)
            if int(qrels.get(item.paper_id, 0)) > 0
        ),
        None,
    )
    hard_violation_count = sum(
        1
        for item in ranked
        if item.gate.decision.value == "reject"
        or any(label.value == "violate" for label in item.facet_labels.values())
    )
    unknown_facets = sum(len(item.critical_unknown_facets) for item in ranked)
    precision = len(retrieved_relevant) / len(ranked) if ranked else 0.0
    recall = (len(retrieved_relevant) / len(relevant)) if relevant else 0.0
    return {
        "k": k,
        "ndcg_at_k": (_dcg(gains) / _dcg(ideal)) if _dcg(ideal) else 0.0,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "mrr_at_k": (1.0 / first_relevant_rank) if first_relevant_rank else 0.0,
        "hard_violation_rate_at_k": hard_violation_count / len(ranked) if ranked else 0.0,
        "critical_unknown_facets_at_k": unknown_facets,
        "returned_ids": [item.paper_id for item in ranked],
        "level_counts": {
            level.value: sum(item.level == level for item in bundle.items)
            for level in RankingLevel
        },
        "latency_ms": bundle.timings.model_dump(),
        "stage_counts": bundle.stage_counts,
        "costs": bundle.costs.model_dump(),
        "model_versions": bundle.model_versions,
        "degraded": bundle.degraded,
        "degradation_reasons": bundle.degradation_reasons,
        "bundle_hash": bundle.bundle_hash,
    }


def _evaluate_ranked_ids(
    paper_ids: list[str],
    qrels: dict[str, int],
    *,
    k: int,
) -> dict[str, Any]:
    selected = paper_ids[:k]
    relevant = {paper_id for paper_id, value in qrels.items() if int(value) > 0}
    retrieved = {paper_id for paper_id in selected if int(qrels.get(paper_id, 0)) > 0}
    precision = len(retrieved) / len(selected) if selected else 0.0
    recall = len(retrieved) / len(relevant) if relevant else 0.0
    gains = [int(qrels.get(paper_id, 0)) for paper_id in selected]
    ideal = sorted((int(value) for value in qrels.values()), reverse=True)[:k]
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "ndcg_at_k": (_dcg(gains) / _dcg(ideal)) if _dcg(ideal) else 0.0,
        "returned_ids": selected,
    }


def evaluate_ablation_suite(
    bundle: RankingBundle,
    qrels: dict[str, int],
    *,
    k: int,
) -> dict[str, Any]:
    items = bundle.items

    def score_order(field: str, *, exclude_gate: bool = False, exclude_level: bool = False) -> list[str]:
        values = [
            item
            for item in items
            if (not exclude_gate or item.gate.decision.value != "reject")
            and (not exclude_level or item.level != RankingLevel.EXCLUDED)
            and getattr(item.scores, field) is not None
        ]
        values.sort(key=lambda item: (-float(getattr(item.scores, field)), item.paper_id))
        return [item.paper_id for item in values]

    variants = {
        "A_dense_only": score_order("dense_score"),
        "B_sparse_only": score_order("sparse_score"),
        "C_dense_sparse_rrf": score_order("rrf_score"),
        "D_plus_cross_encoder": score_order("cross_logit"),
        "E_plus_hard_gate": score_order("cross_logit", exclude_gate=True),
        "F_plus_abstract_evidence": score_order("cross_logit", exclude_level=True),
        # G equals F when no controller-provided full-text replay is present.
        "G_plus_selective_fulltext": score_order("cross_logit", exclude_level=True),
        "H_plus_level_and_coverage": [
            item.paper_id for item in items if item.level != RankingLevel.EXCLUDED
        ],
    }
    return {
        name: _evaluate_ranked_ids(paper_ids, qrels, k=k)
        for name, paper_ids in variants.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-models", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be positive")

    intent_data = json.loads((FIXTURE_DIR / "intent.json").read_text(encoding="utf-8"))
    candidates = json.loads((FIXTURE_DIR / "candidate_papers.json").read_text(encoding="utf-8"))
    qrels = json.loads((FIXTURE_DIR / "qrels.json").read_text(encoding="utf-8"))
    understanding = QueryUnderstanding.model_validate(intent_data["understanding"])

    if args.real_models:
        pipeline = EvidRankPipeline(evidence_judge=DeterministicEvidenceJudge())
        mode = "real-models"
    else:
        pipeline = EvidRankPipeline(
            encoder=FixtureEncoder(),
            reranker=FixtureReranker(),
            evidence_judge=DeterministicEvidenceJudge(),
        )
        mode = "fixture"
    bundle = pipeline.rank(
        session_id=f"evaluation-{mode}",
        query=intent_data["query"],
        understanding=understanding,
        candidate_rows=candidates,
        top_k=max(args.k, len(candidates)),
        include_rejected_in_scoring=True,
    )
    report = {
        "evaluation_mode": mode,
        "fixture_version": "evidrank-fixture-v1",
        **evaluate_bundle(bundle, qrels, k=args.k),
        "ablations": evaluate_ablation_suite(bundle, qrels, k=args.k),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
