"""End-to-end feature-three EvidRank pipeline."""

from __future__ import annotations

import re
import time
import unicodedata
import uuid
import math
from functools import lru_cache
from typing import Any

from backend.app.config import get_settings
from backend.app.models.evidrank import (
    CandidatePaper,
    ConstraintPolarity,
    EvidenceLabel,
    FulltextDocument,
    GateDecision,
    RankingBundle,
    RankingItem,
    RankingLevel,
    RunCosts,
    StageTiming,
)
from backend.app.services.embeddings import get_bge_m3_encoder
from backend.app.services.ranking.adapters import (
    candidates_from_rows,
    research_intent_from_understanding,
)
from backend.app.services.ranking.bundle_builder import finalize_draft_hash
from backend.app.services.ranking.feedback import (
    build_coverage_gaps,
    build_uncertainty_queue,
)
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge
from backend.app.services.ranking.hard_gate import apply_hard_gate
from backend.app.services.ranking.hybrid_ranker import HybridCandidateRanker
from backend.app.services.ranking.identity_review import review_identity
from backend.app.services.ranking.level_decider import decide_level
from backend.app.services.ranking.llm_evidence_judge import get_evidence_judge
from backend.app.services.ranking.metadata_validator import validate_metadata
from backend.app.services.ranking.rank_calibrator import load_threshold_profile
from backend.app.services.ranking.fulltext_verifier import verify_fulltext
from backend.app.services.pasa_selector import get_pasa_selector
from backend.app.services.reranker import get_bge_reranker


class EvidRankPipeline:
    def __init__(
        self,
        *,
        encoder: Any | None = None,
        reranker: Any | None = None,
        selector: Any | None = None,
        evidence_judge: Any | None = None,
        decision_policy: str | None = None,
    ) -> None:
        settings = get_settings()
        # Load the 7B PaSa selector FIRST: its 4-bit load materialises the
        # 15GB bf16 checkpoint (RAM spike ~13GB) which must happen while
        # memory is still mostly free — loading it after the BGE models
        # segfaults on 16GB machines. Resident footprint afterwards is small
        # (~6GB VRAM), so the three models coexist comfortably.
        self.selector = selector or get_pasa_selector()
        self.encoder = encoder or get_bge_m3_encoder()
        self.reranker = reranker or get_bge_reranker()
        self.evidence_judge = evidence_judge or get_evidence_judge()
        self.decision_policy = decision_policy or settings.evidrank_decision_policy
        if self.decision_policy not in {"strict", "recall_first"}:
            raise ValueError(f"unsupported EvidRank decision policy: {self.decision_policy}")
        self.metadata_evidence_judge = DeterministicEvidenceJudge()
        self.rrf_k = settings.rrf_k
        self.evidence_top_k = settings.evidrank_evidence_top_k
        self.evidence_pool_multiplier = max(
            1,
            settings.evidrank_evidence_pool_multiplier,
        )
        self.high_rank_cutoff = settings.evidrank_high_rank_cutoff
        self.default_facet_target = settings.evidrank_default_facet_target
        self.cross_rank_weight = max(
            0.0,
            min(1.0, settings.bge_reranker_cross_rank_weight),
        )
        self.facet_rank_weight = max(
            0.0,
            min(1.0 - self.cross_rank_weight, settings.bge_reranker_facet_rank_weight),
        )
        self.head_protect = max(0, settings.bge_reranker_head_protect)
        self.evaluation_reranker_top_n = max(
            1,
            settings.paper_test_bge_reranker_top_n,
        )
        self.selector_rank_weight = max(
            0.0,
            min(1.0, settings.pasa_selector_rank_weight),
        )
        self.selector_protect_head = bool(settings.pasa_selector_protect_head)
        self.selector_top_n = max(1, settings.pasa_selector_top_n)
        self.evaluation_selector_top_n = max(
            1,
            settings.paper_test_pasa_selector_top_n,
        )
        self.threshold_version = settings.evidrank_threshold_version
        self.threshold_profile = load_threshold_profile(settings.evidrank_threshold_profile)
        if self.threshold_profile is not None:
            self.threshold_version = self.threshold_profile.version

    @staticmethod
    def _rerank_query(intent: Any) -> str:
        # Keep the natural request dominant. Repeating it as schema labels
        # ("Tasks", "Background policy", ...) consumes the pair token budget
        # and can make a reranker match facet names instead of the requested
        # scientific relation. Only append exact concepts absent from the raw
        # query, which is especially useful for normalized dataset acronyms.
        raw = " ".join(str(intent.raw_query or "").split()).strip()
        folded = raw.casefold()
        concepts: list[str] = []
        for value in [*intent.datasets, *intent.methods, *intent.tasks]:
            clean = " ".join(str(value or "").split()).strip()
            if clean and clean.casefold() not in folded and clean.casefold() not in {
                item.casefold() for item in concepts
            }:
                concepts.append(clean)
        return "\n".join(
            part
            for part in [raw, f"Key concepts: {', '.join(concepts[:8])}" if concepts else ""]
            if part
        )

    @staticmethod
    def _match_text(value: Any) -> str:
        normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
        return " ".join(re.sub(r"[^\w+._-]+", " ", normalized).split())

    @classmethod
    def _structured_concepts(cls, intent: Any) -> list[tuple[str, float]]:
        """Return high-value concepts that are safe to use as ranking hints.

        Broad topics are intentionally excluded. Dataset names, methods and
        explicit positive relations are much less likely to reward a merely
        topical paper, which makes this signal suitable for head reranking.
        """
        weighted: list[tuple[Any, float]] = [
            *((value, 3.0) for value in intent.datasets),
            *((value, 2.0) for value in intent.methods),
        ]
        for relation in intent.relations:
            if relation.required:
                weighted.extend(((relation.source, 1.5), (relation.target, 1.5)))
        metadata_facets = {
            "year", "language", "venue", "publication_type", "paper_type",
            "paper_id", "canonical_paper_id", "retracted",
        }
        for constraint in intent.constraints:
            if (
                constraint.polarity == ConstraintPolarity.POSITIVE
                and constraint.facet.casefold() not in metadata_facets
            ):
                values = constraint.value if isinstance(constraint.value, list) else [constraint.value]
                weighted.extend((value, 1.5) for value in values)

        result: list[tuple[str, float]] = []
        seen: set[str] = set()
        for value, weight in weighted:
            clean = cls._match_text(value)
            if len(clean) < 2 or clean in seen:
                continue
            result.append((clean, weight))
            seen.add(clean)
        return result[:12]

    @classmethod
    def _facet_match_score(
        cls,
        paper: CandidatePaper,
        concepts: list[tuple[str, float]],
    ) -> float:
        if not concepts:
            return 0.0
        title = cls._match_text(paper.title)
        abstract = cls._match_text(paper.abstract)
        score = 0.0
        matched = 0
        for concept, weight in concepts:
            if concept in title:
                score += weight * 2.0
                matched += 1
            elif concept in abstract:
                score += weight
                matched += 1
            else:
                tokens = set(concept.split())
                document_tokens = set(f"{title} {abstract}".split())
                if len(tokens) >= 2 and len(tokens & document_tokens) / len(tokens) >= 0.8:
                    score += weight * 0.5
                    matched += 1
        # Reward satisfying several independent conditions instead of one
        # repeated keyword. This is still only a ranking hint, never a gate.
        return score + (matched / len(concepts))

    @classmethod
    def _lexical_anchor_score(
        cls,
        paper: CandidatePaper,
        intent: Any,
        concepts: list[tuple[str, float]],
    ) -> float:
        """Score exact entities for bounded recall protection.

        This signal never replaces the learned ranker. It only prevents a
        named dataset/method or a short entity query from disappearing after
        cross-encoder fusion.
        """
        anchors = [value for value, _weight in concepts]
        raw = cls._match_text(intent.raw_query)
        raw_tokens = [token for token in raw.split() if len(token) >= 3]
        if not anchors and len(raw_tokens) <= 3:
            anchors = [raw]
        if not anchors:
            return 0.0
        title = cls._match_text(paper.title)
        abstract = cls._match_text(paper.abstract)
        score = 0.0
        for anchor in dict.fromkeys(anchors):
            if len(anchor) < 3:
                continue
            if anchor in title:
                score += 3.0
            elif anchor in abstract:
                score += 1.0
        return score

    @staticmethod
    def _explicit_preference_mode(query: str) -> str | None:
        """Recognize authority/recency only when the user explicitly asks.

        Citation count and publication year are poor universal relevance
        proxies, so they never affect ordinary benchmark queries. They become
        a small tie-break signal only for intents that request this trade-off.
        """
        searchable = unicodedata.normalize("NFKC", query or "").casefold()
        authority_cues = (
            "seminal", "influential", "authoritative", "highly cited",
            "landmark", "经典", "权威", "高被引", "奠基",
        )
        recent_cues = (
            "recent", "latest", "newest", "state of the art", "frontier",
            "最新", "近年", "近五年", "前沿",
        )
        if any(cue in searchable for cue in authority_cues):
            return "authority"
        if any(cue in searchable for cue in recent_cues):
            return "recency"
        return None

    @staticmethod
    def _effective_labels(
        records: list[Any],
    ) -> tuple[dict[str, EvidenceLabel], list[str]]:
        source_priority = {"metadata": 0, "abstract": 1, "fulltext": 2}
        by_condition: dict[str, list[Any]] = {}
        for record in records:
            condition_id = record.constraint_id or record.facet_id
            by_condition.setdefault(condition_id, []).append(record)

        label_priority = {
            EvidenceLabel.SUPPORT: 0,
            EvidenceLabel.UNKNOWN: 1,
            EvidenceLabel.VIOLATE: 2,
        }
        labels: dict[str, EvidenceLabel] = {}
        conflicts: set[str] = set()
        for condition_records in by_condition.values():
            established = {
                record.label
                for record in condition_records
                if record.label != EvidenceLabel.UNKNOWN
            }
            facet_id = condition_records[0].facet_id
            if established == {EvidenceLabel.SUPPORT, EvidenceLabel.VIOLATE}:
                effective_label = EvidenceLabel.UNKNOWN
                conflicts.add(facet_id)
            else:
                selected = max(
                    condition_records,
                    key=lambda record: (
                        record.label != EvidenceLabel.UNKNOWN,
                        source_priority[record.source_type],
                    ),
                )
                effective_label = selected.label
            previous = labels.get(facet_id)
            if previous is None or label_priority[effective_label] > label_priority[previous]:
                labels[facet_id] = effective_label
        return labels, sorted(conflicts)

    def _final_sort_key(self, item: RankingItem) -> tuple[Any, ...]:
        level_priority = {
            RankingLevel.HIGH: 0,
            RankingLevel.VERIFY: 1,
            RankingLevel.PARTIAL: 2,
            RankingLevel.EXCLUDED: 3,
        }
        relevance_rank = (
            item.scores.fusion_rank
            or item.scores.cross_rank
            or item.scores.rrf_rank
            or 10**9
        )
        if self.decision_policy == "recall_first":
            # In recall-first mode EvidRank remains a hard safety gate, but an
            # unresolved soft facet must not outrank a much more relevant
            # paper. Explicit exclusions still remain at the end.
            return (
                item.level == RankingLevel.EXCLUDED,
                relevance_rank,
                level_priority[item.level],
                item.paper_id,
            )
        return (
            level_priority[item.level],
            relevance_rank,
            item.paper_id,
        )

    def rank(
        self,
        *,
        session_id: str,
        query: str,
        understanding: Any,
        candidate_rows: list[dict[str, Any]],
        top_k: int,
        fulltext_documents: dict[str, FulltextDocument | dict[str, Any]] | None = None,
        fulltext_budget: int = 0,
        include_rejected_in_scoring: bool = False,
        fast_evaluation: bool = False,
        ablation: dict[str, bool] | None = None,
    ) -> RankingBundle:
        """Ablation switches (all default to enabled, so ``None`` == full
        pipeline): reranker, facet, preference, evidence, level."""
        use_reranker = bool(ablation.get("reranker", True)) if ablation else True
        use_facet = bool(ablation.get("facet", True)) if ablation else True
        use_preference = bool(ablation.get("preference", True)) if ablation else True
        use_selector = bool(ablation.get("selector", True)) if ablation else True
        use_evidence = bool(ablation.get("evidence", True)) if ablation else True
        use_level = bool(ablation.get("level", True)) if ablation else True
        started = time.perf_counter()
        intent = research_intent_from_understanding(
            session_id=session_id,
            query=query,
            understanding=understanding,
        )
        candidates = candidates_from_rows(candidate_rows)
        timings = StageTiming()

        stage = time.perf_counter()
        identities = {paper.canonical_paper_id: review_identity(paper) for paper in candidates}
        timings.identity_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        metadata = {paper.canonical_paper_id: validate_metadata(paper) for paper in candidates}
        timings.metadata_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        gates = {
            paper.canonical_paper_id: apply_hard_gate(
                paper,
                intent,
                identities[paper.canonical_paper_id],
                metadata[paper.canonical_paper_id],
            )
            for paper in candidates
        }
        timings.hard_gate_ms = (time.perf_counter() - stage) * 1000

        active = (
            candidates
            if include_rejected_in_scoring
            else [
                paper
                for paper in candidates
                if gates[paper.canonical_paper_id].decision != GateDecision.REJECT
            ]
        )
        stage = time.perf_counter()
        hybrid = HybridCandidateRanker(self.encoder, rrf_k=self.rrf_k)
        rrf_order, score_map, hybrid_meta = hybrid.rank(intent.raw_query, active)
        timings.hybrid_retrieval_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        rerank_rows = [
            {
                "id": paper.canonical_paper_id,
                "title": paper.title,
                "abstract": paper.abstract or "",
                "rrf_score": score_map[paper.canonical_paper_id].rrf_score,
            }
            for paper in rrf_order
        ]
        if use_reranker:
            reranked, rerank_meta = self.reranker.rerank(
                query=self._rerank_query(intent),
                papers=rerank_rows,
                top_k=len(rerank_rows),
                candidate_limit=(
                    self.evaluation_reranker_top_n
                    if fast_evaluation
                    else None
                ),
            )
        else:
            reranked = rerank_rows
            rerank_meta = {"applied": False}
        timings.cross_encoder_ms = (time.perf_counter() - stage) * 1000
        cross_order = [str(row["id"]) for row in reranked]
        cross_scored_rows = [row for row in reranked if row.get("cross_logit") is not None]
        cross_rank = {
            str(row["id"]): rank
            for rank, row in enumerate(cross_scored_rows, start=1)
        }
        for row in cross_scored_rows:
            paper_id = str(row["id"])
            scores = score_map[paper_id]
            scores.cross_logit = row.get("cross_logit")
            scores.cross_sigmoid = row.get("cross_sigmoid")
            scores.cross_rank = cross_rank[paper_id]

        # PaSa 7B Selector: an additional academic-relevance signal on the
        # cross-encoded head pool.  Its order is fused into the weighted RRF
        # below and its top positions participate in head protection (it is a
        # strictly better relevance judge than the reranker on the offline
        # benchmark).  Uses the raw query (not _rerank_query) to stay faithful
        # to the model's fine-tuning distribution.
        stage = time.perf_counter()
        sel_meta: dict[str, Any] = {
            "provider": "disabled",
            "applied": False,
            "error": None,
            "scored_count": 0,
        }
        selector_order: list[str] = []
        selector_weight = 0.0
        if (
            use_selector
            and self.selector is not None
            and getattr(self.selector, "enabled", False)
        ):
            sel_limit = (
                self.evaluation_selector_top_n if fast_evaluation else self.selector_top_n
            )
            sel_rows = [
                {
                    "id": str(row["id"]),
                    "title": row.get("title") or "",
                    "abstract": row.get("abstract") or "",
                }
                for row in reranked[:sel_limit]
            ]
            if sel_rows:
                sel_scores, sel_meta = self.selector.infer_scores(
                    intent.raw_query, sel_rows
                )
            else:
                sel_scores, sel_meta = None, dict(sel_meta)
            if sel_scores:
                sel_scored_rows = sorted(
                    [
                        dict(row, sel_score=score)
                        for row, score in zip(sel_rows, sel_scores, strict=True)
                    ],
                    key=lambda row: (-float(row["sel_score"]), str(row["id"])),
                )
                selector_order = [str(row["id"]) for row in sel_scored_rows]
                sel_rank = {
                    paper_id: rank
                    for rank, paper_id in enumerate(selector_order, start=1)
                }
                for row in sel_scored_rows:
                    paper_id = str(row["id"])
                    score_row = score_map[paper_id]
                    score_row.sel_score = row["sel_score"]
                    score_row.sel_rank = sel_rank[paper_id]
                selector_weight = self.selector_rank_weight
            if (
                getattr(self.selector, "required", False)
                and not sel_meta.get("applied")
            ):
                raise RuntimeError(
                    f"PaSa selector required but unavailable: "
                    f"{getattr(self.selector, '_load_error', None) or 'unknown error'}"
                )
        timings.selector_ms = (time.perf_counter() - stage) * 1000

        # Cross-Encoder scores are strong but can over-promote broadly similar
        # papers. Fuse its order with BGE-M3 dense+sparse RRF and a cheap exact
        # structured-concept signal. The latter is activated only for explicit
        # datasets, methods, relations and positive semantic constraints.
        concepts = self._structured_concepts(intent)
        if use_facet:
            facet_scores = {
                paper.canonical_paper_id: self._facet_match_score(paper, concepts)
                for paper in rrf_order
            }
            facet_order = sorted(
                [paper_id for paper_id, score in facet_scores.items() if score > 0.0],
                key=lambda paper_id: (
                    -facet_scores[paper_id],
                    score_map[paper_id].cross_rank or len(active) + 1,
                    paper_id,
                ),
            )
            lexical_anchor_scores = {
                paper.canonical_paper_id: self._lexical_anchor_score(
                    paper,
                    intent,
                    concepts,
                )
                for paper in rrf_order
            }
            lexical_anchor_order = [
                paper_id
                for paper_id in sorted(
                    lexical_anchor_scores,
                    key=lambda paper_id: (
                        -lexical_anchor_scores[paper_id],
                        score_map[paper_id].cross_rank or len(active) + 1,
                        paper_id,
                    ),
                )
                if lexical_anchor_scores[paper_id] > 0.0
            ]
        else:
            facet_scores = {}
            facet_order = []
            lexical_anchor_scores = {}
            lexical_anchor_order = []
        facet_ranks = {
            paper_id: rank
            for rank, paper_id in enumerate(facet_order, start=1)
        }
        structured_signal_active = bool(facet_order)
        facet_weight = self.facet_rank_weight if structured_signal_active else 0.0
        preference_mode = (
            self._explicit_preference_mode(intent.raw_query)
            if use_preference
            else None
        )
        preference_weight = 0.08 if preference_mode else 0.0
        preference_papers = {
            paper.canonical_paper_id: paper
            for paper in rrf_order
        }
        if preference_mode == "authority":
            preference_order = sorted(
                [paper.canonical_paper_id for paper in rrf_order],
                key=lambda paper_id: (
                    -math.log1p(
                        max(0, preference_papers[paper_id].citation_count)
                    ),
                    paper_id,
                ),
            )
        elif preference_mode == "recency":
            preference_order = sorted(
                [paper.canonical_paper_id for paper in rrf_order],
                key=lambda paper_id: (
                    -(preference_papers[paper_id].year or 0),
                    paper_id,
                ),
            )
        else:
            preference_order = []
        preference_ranks = {
            paper_id: rank
            for rank, paper_id in enumerate(preference_order, start=1)
        }
        hybrid_weight = max(
            0.0,
            1.0 - self.cross_rank_weight - facet_weight - preference_weight - selector_weight,
        )
        fusion_scores: dict[str, float] = {}
        for paper in rrf_order:
            paper_id = paper.canonical_paper_id
            scores = score_map[paper_id]
            cross_position = scores.cross_rank or (len(active) + 1)
            rrf_position = scores.rrf_rank or (len(active) + 1)
            facet_position = facet_ranks.get(paper_id)
            preference_position = preference_ranks.get(paper_id)
            sel_position = scores.sel_rank
            scores.facet_match_score = facet_scores.get(paper_id, 0.0)
            scores.facet_rank = facet_position
            fusion_scores[paper_id] = (
                self.cross_rank_weight / (self.rrf_k + cross_position)
                + hybrid_weight / (self.rrf_k + rrf_position)
                + (
                    facet_weight / (self.rrf_k + facet_position)
                    if facet_position is not None
                    else 0.0
                )
                + (
                    preference_weight / (self.rrf_k + preference_position)
                    if preference_position is not None
                    else 0.0
                )
                + (
                    selector_weight / (self.rrf_k + sel_position)
                    if sel_position is not None
                    else 0.0
                )
            )
        if selector_weight > 0.0 and self.selector_protect_head:
            # The PaSa Selector is the strongest relevance signal on the
            # offline benchmark; protect its head like the cross-encoder head.
            protected_count = min(self.head_protect, len(selector_order))
            protected_head = selector_order[:protected_count]
        else:
            protected_count = min(
                self.head_protect,
                len(cross_scored_rows),
                len(cross_order),
            )
            # Preserve the learned cross-encoder head. Exact anchors are
            # already represented by the structured-facet fusion and are
            # recorded below for diagnostics; they must not bypass the
            # learned top positions.
            protected_head = cross_order[:protected_count]
        protected_ids = set(protected_head)
        fused_tail = sorted(
            [paper_id for paper_id in cross_order if paper_id not in protected_ids],
            key=lambda paper_id: (
                -fusion_scores.get(paper_id, 0.0),
                paper_id,
            ),
        )
        fused_order = [*protected_head, *fused_tail]
        for fusion_position, paper_id in enumerate(fused_order, start=1):
            score_map[paper_id].fusion_score = fusion_scores[paper_id]
            score_map[paper_id].fusion_rank = fusion_position

        by_id: dict[str, CandidatePaper] = {
            paper.canonical_paper_id: paper
            for paper in candidates
        }
        ordered_ids = fused_order + [
            paper.canonical_paper_id
            for paper in candidates
            if gates[paper.canonical_paper_id].decision == GateDecision.REJECT
            and paper.canonical_paper_id not in fused_order
        ]

        stage = time.perf_counter()
        evidence: list[Any] = []
        evidence_by_paper: dict[str, list[Any]] = {}
        # LLM evidence judging is the most latency-sensitive remote stage.
        # Scale its pool with the requested UI result count, while keeping
        # headroom for verify/excluded replacements and respecting the
        # configured absolute ceiling.
        evidence_pool_limit = min(
            self.evidence_top_k,
            max(
                self.high_rank_cutoff,
                top_k * self.evidence_pool_multiplier,
            ),
        )
        evidence_candidates = (
            set(fused_order[:evidence_pool_limit]) if use_evidence else set()
        )
        evidence_judge = (
            self.metadata_evidence_judge
            if fast_evaluation
            else self.evidence_judge
        )
        reset_metrics = getattr(evidence_judge, "reset_metrics", None)
        if callable(reset_metrics):
            reset_metrics()
        batch_records: dict[str, list[Any]] = {}
        judge_batch = getattr(evidence_judge, "judge_batch", None)
        if callable(judge_batch) and evidence_candidates:
            evidence_papers = [
                by_id[paper_id]
                for paper_id in fused_order
                if paper_id in evidence_candidates
            ]
            batch_size = max(
                1,
                int(getattr(evidence_judge, "batch_size", len(evidence_papers))),
            )
            for start in range(0, len(evidence_papers), batch_size):
                paper_batch = evidence_papers[start:start + batch_size]
                result = judge_batch(
                    session_id=session_id,
                    papers=paper_batch,
                    intent=intent,
                )
                expected_ids = {paper.canonical_paper_id for paper in paper_batch}
                if not isinstance(result, dict) or set(result) != expected_ids:
                    raise ValueError(
                        "evidence judge batch output must contain every input paper_id exactly once"
                    )
                batch_records.update(result)
        for paper_id in ordered_ids:
            paper = by_id[paper_id]
            # Metadata constraints remain cheap for all papers. Semantic facets
            # (and therefore any LLM call) are evaluated only for the
            # configured evidence pool.
            if paper_id in evidence_candidates:
                if batch_records:
                    records = batch_records[paper_id]
                else:
                    records = evidence_judge.judge(
                        session_id=session_id,
                        paper=paper,
                        intent=intent,
                    )
            else:
                records = [
                    self.metadata_evidence_judge.judge_constraint(
                        session_id=session_id,
                        paper=paper,
                        constraint=constraint,
                    )
                    for constraint in intent.constraints
                    if constraint.facet.lower()
                    in {
                        "year",
                        "language",
                        "publication_type",
                        "paper_type",
                        "venue",
                        "paper_id",
                        "canonical_paper_id",
                        "retracted",
                    }
                ]
            evidence.extend(records)
            evidence_by_paper[paper_id] = records
        timings.evidence_judge_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        items: list[RankingItem] = []
        for paper_id in ordered_ids:
            paper = by_id[paper_id]
            records = evidence_by_paper.get(paper_id, [])
            labels, evidence_conflicts = self._effective_labels(records)
            scores = score_map.get(paper_id)
            if scores is None:
                from backend.app.models.evidrank import RankingScores

                scores = RankingScores(
                    provider_scores=paper.provider_scores,
                    provider_ranks=paper.provider_ranks,
                )
            item = RankingItem(
                paper_id=paper_id,
                preferred_version_id=paper.preferred_version_id,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                year=paper.year,
                venue=paper.venue,
                publication_type=paper.publication_type,
                identifiers=paper.identifiers,
                url=paper.url,
                citation_count=paper.citation_count,
                tags=paper.tags,
                identity_status=identities[paper_id].status,
                metadata_validation=metadata[paper_id],
                gate=gates[paper_id],
                scores=scores,
                evidence_ids=[record.evidence_id for record in records],
                facet_labels=labels,
                evidence_conflict_facets=evidence_conflicts,
                fulltext_available=bool(paper.pdf_url and paper.oa_status not in {"closed", "paywalled"}),
            )
            if use_level:
                level, reasons, critical_unknown = decide_level(
                    item,
                    intent,
                    high_rank_cutoff=self.high_rank_cutoff,
                    min_cross_logit=(
                        self.threshold_profile.min_cross_logit
                        if self.threshold_profile is not None
                        else None
                    ),
                    recall_first=self.decision_policy == "recall_first",
                )
            else:
                # Ablation: skip the level decision and treat everything as
                # high so the final order is purely score-driven.
                level = RankingLevel.HIGH
                reasons = ["ablation_no_level"]
                critical_unknown = []
            item.level = level
            item.level_reason_codes = reasons
            item.critical_unknown_facets = critical_unknown
            items.append(item)

        items.sort(key=self._final_sort_key)
        for rank, item in enumerate(items, start=1):
            item.final_rank = rank
        gaps = build_coverage_gaps(
            intent,
            items,
            default_target_count=self.default_facet_target,
        )
        uncertainty = build_uncertainty_queue(
            intent,
            items,
            gaps,
            high_rank_cutoff=self.high_rank_cutoff,
        )
        timings.level_decision_ms = (time.perf_counter() - stage) * 1000

        fulltext_ids: list[str] = []
        if fulltext_documents and fulltext_budget > 0:
            stage = time.perf_counter()
            validated_documents = {
                paper_id: (
                    value
                    if isinstance(value, FulltextDocument)
                    else FulltextDocument.model_validate(value)
                )
                for paper_id, value in fulltext_documents.items()
            }
            fulltext_records, fulltext_ids = verify_fulltext(
                session_id=session_id,
                intent=intent,
                papers=by_id,
                uncertainty_queue=uncertainty,
                documents=validated_documents,
                max_documents=fulltext_budget,
            )
            evidence.extend(fulltext_records)
            for record in fulltext_records:
                evidence_by_paper.setdefault(record.paper_id, []).append(record)
            for item in items:
                if item.paper_id not in fulltext_ids:
                    continue
                records = evidence_by_paper[item.paper_id]
                item.evidence_ids = [record.evidence_id for record in records]
                item.facet_labels, item.evidence_conflict_facets = self._effective_labels(records)
                level, reasons, critical_unknown = decide_level(
                    item,
                    intent,
                    high_rank_cutoff=self.high_rank_cutoff,
                    min_cross_logit=(
                        self.threshold_profile.min_cross_logit
                        if self.threshold_profile is not None
                        else None
                    ),
                    recall_first=self.decision_policy == "recall_first",
                )
                item.level = level
                item.level_reason_codes = reasons
                item.critical_unknown_facets = critical_unknown
            items.sort(key=self._final_sort_key)
            for rank, item in enumerate(items, start=1):
                item.final_rank = rank
            gaps = build_coverage_gaps(
                intent,
                items,
                default_target_count=self.default_facet_target,
            )
            uncertainty = build_uncertainty_queue(
                intent,
                items,
                gaps,
                high_rank_cutoff=self.high_rank_cutoff,
                fulltext_budget_exhausted=len(fulltext_ids) >= fulltext_budget,
            )
            timings.fulltext_verify_ms = (time.perf_counter() - stage) * 1000
        timings.total_ms = (time.perf_counter() - started) * 1000

        degraded_reasons: list[str] = []
        encoder_provider = str(hybrid_meta.get("provider") or "unknown")
        if "fallback" in encoder_provider or encoder_provider == "unavailable":
            degraded_reasons.append(f"EMBEDDING_PROVIDER:{encoder_provider}")
        if not rerank_meta.get("applied"):
            degraded_reasons.append("RERANKER_NOT_APPLIED")
        if (
            use_selector
            and self.selector is not None
            and getattr(self.selector, "enabled", False)
            and not sel_meta.get("applied")
        ):
            degraded_reasons.append("SELECTOR_NOT_APPLIED")
        evidence_metrics = getattr(evidence_judge, "get_metrics", lambda: {})()
        if evidence_metrics.get("errors"):
            degraded_reasons.append("EVIDENCE_LLM_FAILED")
        bundle = RankingBundle(
            ranking_run_id=uuid.uuid4().hex,
            session_id=session_id,
            intent_hash=intent.intent_hash,
            # The bundle is the complete auditable candidate-pool result.
            # ``top_k`` controls the public UI projection in search_service,
            # never which rejected/verification records survive for audit.
            items=items,
            evidence_matrix=evidence,
            coverage_gaps=gaps,
            uncertainty_queue=uncertainty,
            model_versions={
                "embedding": str(hybrid_meta.get("model_revision") or encoder_provider),
                "reranker": str(rerank_meta.get("model_revision") or rerank_meta.get("reranker") or "unknown"),
                "selector": str(sel_meta.get("provider") or "unknown"),
                "evidence_judge": str(getattr(evidence_judge, "provider", "unknown")),
                "intent_parser": intent.parser_version,
                "threshold_model": (
                    self.threshold_profile.model_revision
                    if self.threshold_profile is not None
                    else "rank-evidence-rules"
                ),
                "decision_policy": self.decision_policy,
            },
            threshold_version=self.threshold_version,
            bundle_hash="pending",
            timings=timings,
            stage_counts={
                "candidate_input": len(candidates),
                "gate_active": sum(
                    gate.decision != GateDecision.REJECT for gate in gates.values()
                ),
                "gate_rejected": sum(
                    gate.decision == GateDecision.REJECT for gate in gates.values()
                ),
                "scoring_input": len(active),
                "cross_scored": int(
                    rerank_meta.get("cross_input_count", len(cross_scored_rows))
                ),
                "selector_scored": int(sel_meta.get("scored_count") or 0),
                "abstract_evidence_papers": len(evidence_candidates),
                "fulltext_verified_papers": len(fulltext_ids),
            },
            execution_metadata={
                "hybrid": hybrid_meta,
                "reranker": rerank_meta,
                "selector": sel_meta,
                "rank_fusion": {
                    "policy": "weighted-cross-rrf-facet-preference-v3",
                    "cross_weight": self.cross_rank_weight,
                    "hybrid_rrf_weight": hybrid_weight,
                    "facet_rank_weight": facet_weight,
                    "preference_mode": preference_mode,
                    "preference_rank_weight": preference_weight,
                    "selector_rank_weight": selector_weight,
                    "protected_selector_head": (
                        protected_count if selector_weight > 0.0 else 0
                    ),
                    "structured_signal_active": structured_signal_active,
                    "structured_concepts": [value for value, _weight in concepts],
                    "lexical_anchor_candidates": lexical_anchor_order[:20],
                    "protected_cross_head": protected_count,
                },
                "evidence": evidence_metrics,
                "evidence_pool_limit": evidence_pool_limit,
                "fulltext_verified_ids": fulltext_ids,
                "fulltext_budget": fulltext_budget,
                "decision_policy": self.decision_policy,
                "fast_evaluation": fast_evaluation,
                "final_sort_policy": (
                    "hard-exclusion-then-relevance"
                    if self.decision_policy == "recall_first"
                    else "evidence-level-then-relevance"
                ),
            },
            costs=RunCosts(
                evidence_api_calls=int(evidence_metrics.get("api_calls") or 0),
                evidence_tokens=int(evidence_metrics.get("token_usage") or 0),
                fulltext_documents=len(fulltext_ids),
            ),
            degraded=bool(degraded_reasons),
            degradation_reasons=degraded_reasons,
            is_frozen=False,
        )
        return finalize_draft_hash(bundle)


@lru_cache
def get_evidrank_pipeline() -> EvidRankPipeline:
    return EvidRankPipeline()
