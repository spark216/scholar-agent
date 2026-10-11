"""In-candidate-pool BGE-M3 dense/sparse ranking with auditable RRF."""

from __future__ import annotations

import math
from typing import Any

from backend.app.models.evidrank import CandidatePaper, RankingScores


def _dense_cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"dense vector dimension mismatch: {len(left)} != {len(right)}")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def _sparse_dot(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return float(sum(value * right.get(index, 0.0) for index, value in left.items()))


def reciprocal_rank_fusion(
    ranked_paths: list[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    if k <= 0:
        raise ValueError("rrf k must be positive")
    scores: dict[str, float] = {}
    for path in ranked_paths:
        for rank, paper_id in enumerate(path, start=1):
            scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridCandidateRanker:
    def __init__(
        self,
        encoder: Any,
        *,
        rrf_k: int = 60,
        lexical_providers: tuple[str, ...] = (
            "pasa-local-corpus",
            "fixed-corpus-fts",
        ),
    ) -> None:
        self.encoder = encoder
        self.rrf_k = rrf_k
        self.lexical_providers = lexical_providers

    def rank(
        self,
        query: str,
        papers: list[CandidatePaper],
    ) -> tuple[list[CandidatePaper], dict[str, RankingScores], dict[str, Any]]:
        if not papers:
            return [], {}, {"provider": getattr(self.encoder, "provider", "unknown")}

        texts = [f"{paper.title}\n{paper.abstract or ''}" for paper in papers]
        dense_docs, sparse_docs, provider = self.encoder.encode(texts)
        dense_query, sparse_query, query_provider = self.encoder.encode_query(query)
        if provider != query_provider:
            raise RuntimeError(f"encoder provider changed within run: {provider} != {query_provider}")

        dense_values = [
            _dense_cosine(dense_query, vector)
            for vector in dense_docs
        ]
        sparse_values = [
            _sparse_dot(sparse_query, vector)
            for vector in sparse_docs
        ]
        dense_order = sorted(
            range(len(papers)),
            key=lambda i: (-dense_values[i], papers[i].canonical_paper_id),
        )
        sparse_order = sorted(
            range(len(papers)),
            key=lambda i: (-sparse_values[i], papers[i].canonical_paper_id),
        )
        dense_ids = [papers[i].canonical_paper_id for i in dense_order]
        sparse_ids = [papers[i].canonical_paper_id for i in sparse_order]
        # Only deterministic local lexical ranks enter RRF. Unstable web
        # provider ranks remain provenance/candidate-budget signals.
        lexical_paths: dict[str, list[str]] = {}
        for lexical_provider in self.lexical_providers:
            ranked = sorted(
                (
                    paper
                    for paper in papers
                    if int(paper.provider_ranks.get(lexical_provider, 0)) > 0
                ),
                key=lambda paper: (
                    int(paper.provider_ranks[lexical_provider]),
                    paper.canonical_paper_id,
                ),
            )
            if ranked:
                lexical_paths[lexical_provider] = [
                    paper.canonical_paper_id for paper in ranked
                ]
        rrf_values = reciprocal_rank_fusion(
            [dense_ids, sparse_ids, *lexical_paths.values()],
            k=self.rrf_k,
        )
        rrf_order = sorted(
            range(len(papers)),
            key=lambda i: (-rrf_values[papers[i].canonical_paper_id], papers[i].canonical_paper_id),
        )

        dense_ranks = {paper_id: rank for rank, paper_id in enumerate(dense_ids, start=1)}
        sparse_ranks = {paper_id: rank for rank, paper_id in enumerate(sparse_ids, start=1)}
        rrf_ids = [papers[i].canonical_paper_id for i in rrf_order]
        rrf_ranks = {paper_id: rank for rank, paper_id in enumerate(rrf_ids, start=1)}
        scores: dict[str, RankingScores] = {}
        by_index = {paper.canonical_paper_id: i for i, paper in enumerate(papers)}
        for paper in papers:
            paper_id = paper.canonical_paper_id
            index = by_index[paper_id]
            scores[paper_id] = RankingScores(
                provider_scores=paper.provider_scores,
                provider_ranks=paper.provider_ranks,
                dense_score=dense_values[index],
                dense_rank=dense_ranks[paper_id],
                sparse_score=sparse_values[index],
                sparse_rank=sparse_ranks[paper_id],
                rrf_score=rrf_values[paper_id],
                rrf_rank=rrf_ranks[paper_id],
            )
        return [papers[index] for index in rrf_order], scores, {
            "provider": provider,
            "model_revision": getattr(self.encoder, "model_revision", None) or "unresolved",
            "resolved_model_path": getattr(self.encoder, "resolved_model_path", None),
            "rrf_k": self.rrf_k,
            "rrf_paths": ["bge_dense", "bge_sparse", *lexical_paths],
            "lexical_provider_counts": {
                provider: len(path) for provider, path in lexical_paths.items()
            },
            "candidate_count": len(papers),
            "dense_dimension": len(dense_query),
        }
