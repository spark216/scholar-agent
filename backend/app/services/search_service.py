"""Search orchestration with streaming agent progress events."""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from collections import OrderedDict, deque
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from backend.app.config import get_settings
from backend.app.models.evidrank import FulltextDocument, RankingBundle, RankingLevel
from backend.app.models.schemas import (
    AgentTraceStep,
    GraphEdge,
    GraphNode,
    PaperItem,
    QueryUnderstanding,
    RelationGraph,
    RelevanceLevel,
    SearchResponse,
    SearchSummary,
)
from backend.app.services.llm_intent import parse_query_intent
from backend.app.services.milvus_store import get_milvus_store
from backend.app.services.mock_data import build_mock_search_response
from backend.app.services.postgres_store import save_search_result
from backend.app.services.organizer import build_output_artifact
from backend.app.services.output_artifact_store import save_output_artifact
from backend.app.services.persistent_cache import (
    FrozenProviderCacheMiss,
    cache_payload_sha256,
    load_cached_json,
    save_cached_json,
)
from backend.app.services.ranking.adapters import research_intent_from_understanding
from backend.app.services.ranking.bundle_builder import freeze_bundle_for_controller
from backend.app.services.ranking.pipeline import get_evidrank_pipeline
from backend.app.services.reranker import get_bge_reranker


class InvalidPaperQueryError(ValueError):
    """Raised when a synchronous search request is not an academic-paper query."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def _get_paper_search_graph():
    """Load LangGraph only when a real non-mock search is executed."""
    from backend.app.agent.graph import paper_search_graph

    return paper_search_graph


def _trace(
    step: str,
    detail: str,
    *,
    status: str = "done",
    phase: str = "tool",
    tool: str | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    iteration: int | None = None,
    api_calls: int | None = None,
    token_usage: int | None = None,
    cumulative_api_calls: int | None = None,
    cumulative_token_usage: int | None = None,
) -> AgentTraceStep:
    return AgentTraceStep(
        step=step,
        detail=detail,
        status=status,
        phase=phase,
        tool=tool,
        input=input_text,
        output=output_text,
        iteration=iteration,
        api_calls=api_calls,
        token_usage=token_usage,
        cumulative_api_calls=cumulative_api_calls,
        cumulative_token_usage=cumulative_token_usage,
    )


def _build_graph(query: str, understanding: QueryUnderstanding, papers: list[PaperItem]) -> RelationGraph:
    nodes = [GraphNode(id="q", name="用户查询", category="query", value=16)]
    edges: list[GraphEdge] = []

    if understanding.domain:
        nodes.append(GraphNode(id="domain", name=understanding.domain, category="method", value=8))
        edges.append(GraphEdge(source="q", target="domain", relation="领域", weight=1.0))

    for i, sub in enumerate(understanding.sub_queries[:4]):
        nid = f"sq{i}"
        nodes.append(GraphNode(id=nid, name=sub[:24], category="subquery", value=9))
        edges.append(GraphEdge(source="q", target=nid, relation="分解", weight=0.9))

    facet_specs: list[tuple[str, str, str]] = []
    for category, relation, values in (
        ("dataset", "数据集约束", understanding.datasets),
        ("method", "方法约束", understanding.methods),
        ("topic", "研究主题", understanding.topics),
        ("entity", "关键实体", understanding.entities),
        ("venue", "发表来源", understanding.venues),
    ):
        for value in values:
            clean = " ".join(str(value or "").split()).strip()
            if clean and clean.casefold() not in {
                item[2].casefold() for item in facet_specs
            }:
                facet_specs.append((category, relation, clean))
    facet_specs = facet_specs[:10]
    for index, (category, relation, value) in enumerate(facet_specs):
        node_id = f"facet-{index}"
        nodes.append(GraphNode(id=node_id, name=value[:40], category=category, value=8))
        edges.append(GraphEdge(source="q", target=node_id, relation=relation, weight=1.0))

    for paper in papers:
        nodes.append(
            GraphNode(
                id=paper.id,
                name=paper.title[:28],
                category="paper",
                value=10 + paper.relevance_score * 8,
                paper_id=paper.id,
            )
        )
        edges.append(
            GraphEdge(
                source="q",
                target=paper.id,
                relation="语义相关",
                weight=paper.relevance_score,
            )
        )
        searchable = unicodedata.normalize(
            "NFKC",
            " ".join([paper.title, paper.abstract, *paper.tags]),
        ).casefold()
        for index, (_category, _relation, value) in enumerate(facet_specs):
            normalized = unicodedata.normalize("NFKC", value).casefold()
            if normalized and normalized in searchable:
                edges.append(
                    GraphEdge(
                        source=f"facet-{index}",
                        target=paper.id,
                        relation="证据涉及",
                        weight=max(0.5, paper.relevance_score),
                    )
                )

    # Add a bounded paper-to-paper layer for shared concepts. The UI therefore
    # shows an actual relationship map rather than a star-shaped result list.
    graph_papers = papers[:20]
    paper_tags = {
        paper.id: {
            unicodedata.normalize("NFKC", tag).casefold().strip()
            for tag in paper.tags
            if len(str(tag).strip()) > 2
        }
        for paper in graph_papers
    }
    relation_edges = 0
    for left_index, left in enumerate(graph_papers):
        for right in graph_papers[left_index + 1:]:
            shared = paper_tags[left.id] & paper_tags[right.id]
            if not shared:
                continue
            edges.append(
                GraphEdge(
                    source=left.id,
                    target=right.id,
                    relation=f"共享主题：{sorted(shared)[0][:24]}",
                    weight=min(1.0, 0.4 + 0.1 * len(shared)),
                )
            )
            relation_edges += 1
            if relation_edges >= 30:
                break
        if relation_edges >= 30:
            break
    return RelationGraph(nodes=nodes, edges=edges)


def _to_legacy_paper_items(rows: list[dict[str, Any]]) -> list[PaperItem]:
    papers: list[PaperItem] = []
    for row in rows:
        score = float(row.get("relevance_score") or 0.0)
        level = RelevanceLevel.HIGH if score >= 0.55 else RelevanceLevel.PARTIAL
        papers.append(
            PaperItem(
                id=str(row.get("id")),
                title=str(row.get("title") or "Untitled"),
                abstract=str(row.get("abstract") or ""),
                authors=list(row.get("authors") or []),
                year=row.get("year"),
                published_at=row.get("published_at"),
                venue=row.get("venue"),
                citation_count=int(row.get("citation_count") or 0),
                relevance=level,
                relevance_score=max(0.0, min(1.0, score)),
                url=row.get("url"),
                tags=list(row.get("tags") or []),
                source=row.get("source"),
                sources=list(row.get("sources") or []),
                doi=(row.get("external_ids") or {}).get("doi"),
                pmid=(row.get("external_ids") or {}).get("pmid"),
                arxiv_id=(row.get("external_ids") or {}).get("arxiv"),
                openalex_id=(row.get("external_ids") or {}).get("openalex"),
                semantic_scholar_id=(row.get("external_ids") or {}).get("semantic_scholar"),
                semantic_scholar_corpus_id=(row.get("external_ids") or {}).get("corpus_id"),
            )
        )
    return papers


def _to_paper_items(bundle: RankingBundle, max_papers: int) -> list[PaperItem]:
    """Project the auditable bundle to the backward-compatible UI list."""
    visible = [
        item
        for item in bundle.items
        if item.level != RankingLevel.EXCLUDED
    ][:max_papers]
    papers: list[PaperItem] = []
    for item in visible:
        display_score = item.scores.cross_sigmoid
        score_kind = "cross_sigmoid_uncalibrated"
        if display_score is None:
            display_score = 0.0
            score_kind = "not_available"
        papers.append(
            PaperItem(
                id=item.paper_id,
                title=item.title,
                abstract=item.abstract or "",
                authors=item.authors,
                year=item.year,
                published_at=str(item.year) if item.year else None,
                venue=item.venue,
                citation_count=item.citation_count,
                relevance=RelevanceLevel(item.level.value),
                relevance_score=max(0.0, min(1.0, float(display_score))),
                score_kind=score_kind,
                dense_score=item.scores.dense_score,
                sparse_score=item.scores.sparse_score,
                rrf_score=item.scores.rrf_score,
                cross_logit=item.scores.cross_logit,
                cross_sigmoid=item.scores.cross_sigmoid,
                evidence_status={key: value.value for key, value in item.facet_labels.items()},
                reason_codes=item.level_reason_codes,
                final_rank=item.final_rank,
                url=item.url,
                tags=item.tags,
                source="evidrank",
                doi=item.identifiers.get("doi"),
                pmid=item.identifiers.get("pmid"),
                arxiv_id=item.identifiers.get("arxiv"),
                openalex_id=item.identifiers.get("openalex"),
                semantic_scholar_id=item.identifiers.get("semantic_scholar"),
                semantic_scholar_corpus_id=item.identifiers.get("corpus_id"),
            )
        )
    return papers


def _normalize_title(title: str) -> str:
    return "".join(ch.lower() for ch in (title or "") if ch.isalnum())


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in candidates:
        key = _normalize_title(str(item.get("title") or ""))[:80]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _candidate_trace_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(candidate.get("canonical_paper_id") or candidate.get("id") or ""),
            "title": str(candidate.get("title") or ""),
            "external_ids": dict(candidate.get("external_ids") or {}),
            "source": str(candidate.get("source") or "unknown"),
            "source_iteration": int(candidate.get("source_iteration") or 0),
            "provider_ranks": {
                str(key): int(value)
                for key, value in sorted(
                    (candidate.get("provider_ranks") or {}).items()
                )
                if value is not None
            },
        }
        for candidate in candidates
    ]


def _candidate_stable_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    """Return a provider-order-independent identity used for budget ties."""
    external_ids = tuple(
        sorted(
            (str(key).casefold(), str(value).casefold())
            for key, value in (candidate.get("external_ids") or {}).items()
            if value
        )
    )
    return (
        str(candidate.get("canonical_paper_id") or candidate.get("id") or "").casefold(),
        _normalize_title(str(candidate.get("title") or "")),
        str(candidate.get("year") or ""),
        external_ids,
    )


def _ranking_trace_rows(bundle: RankingBundle) -> list[dict[str, Any]]:
    return [
        {
            "id": item.paper_id,
            "title": item.title,
            "external_ids": dict(item.identifiers),
            "rank": item.final_rank,
            "level": item.level.value,
            "reason_codes": list(item.level_reason_codes),
            "sel_score": item.scores.sel_score,
            "sel_rank": item.scores.sel_rank,
        }
        for item in bundle.items
    ]


def _limit_embedding_candidates(
    candidates: list[dict[str, Any]],
    limit: int,
    *,
    query: str = "",
    understanding: QueryUnderstanding | None = None,
) -> list[dict[str, Any]]:
    """Select a relevance-first pool with protected source/round diversity.

    Pure round-robin selection can discard an exact dataset or method match
    simply because it was returned lower by one provider.  A bounded cheap
    lexical/consensus pass reserves part of the BGE budget for strong matches;
    the remaining slots retain the previous source-and-round coverage.
    """
    if limit <= 0 or len(candidates) <= limit:
        return list(candidates)

    settings = get_settings()
    parsed = understanding or QueryUnderstanding()
    facet_values = [
        *parsed.datasets,
        *parsed.methods,
        *parsed.entities,
        *parsed.topics,
        *parsed.keywords,
    ]
    query_text = " ".join([query, *facet_values])

    def terms(value: str) -> set[str]:
        normalized = unicodedata.normalize("NFKC", value or "").casefold()
        return {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9+._-]*", normalized)
            if len(token) > 1
            and token not in {
                "about", "all", "and", "for", "from", "give", "paper", "papers",
                "provide", "research", "show", "that", "the", "their", "this", "to",
                "use", "using", "what", "which", "with",
            }
        }

    query_terms = terms(query_text)
    exact_facets = [
        unicodedata.normalize("NFKC", value).casefold().strip()
        for value in facet_values
        if len(str(value).strip()) >= 2
    ]

    def relevance_score(candidate: dict[str, Any]) -> tuple[float, ...]:
        title = str(candidate.get("title") or "")
        abstract = str(candidate.get("abstract") or "")
        title_terms = terms(title)
        abstract_terms = terms(abstract)
        title_overlap = len(query_terms & title_terms)
        abstract_overlap = len(query_terms & abstract_terms)
        searchable = f"{title}\n{abstract}".casefold()
        exact_matches = sum(facet in searchable for facet in exact_facets)
        sources = set(candidate.get("sources") or [candidate.get("source")])
        sources.discard(None)
        provider_ranks = [
            int(value)
            for value in (candidate.get("provider_ranks") or {}).values()
            if int(value) > 0
        ]
        best_provider_rank = min(provider_ranks, default=10**6)
        # Provider ranks are not comparable as raw scores, but reciprocal-rank
        # fusion is.  Reward papers independently surfaced by several sources
        # without allowing a single high-volume provider to dominate the cheap
        # pre-BGE budget selector.
        provider_rrf = sum(1.0 / (settings.rrf_k + rank) for rank in provider_ranks)
        return (
            float(exact_matches),
            float(title_overlap),
            float(abstract_overlap) / max(1.0, len(query_terms)),
            provider_rrf,
            float(min(len(sources), 4)),
            1.0 / best_provider_rank,
            -float(int(candidate.get("source_iteration") or 0)),
            float(candidate.get("score") or 0.0),
        )

    # Protect each provider's own head before the global relevance pass.  The
    # reservation itself is capped, so provider breadth cannot consume the
    # whole model budget when many sources are enabled.
    provider_head_fraction = max(
        0.0,
        min(1.0, settings.bge_candidate_provider_head_fraction),
    )
    provider_head_limit = max(0, settings.bge_candidate_provider_head_per_source)
    provider_slot_limit = min(limit, round(limit * provider_head_fraction))
    provider_rows: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        for source in {
            str(value)
            for value in [*(candidate.get("sources") or []), candidate.get("source")]
            if value
        }:
            provider_rows.setdefault(source, []).append(candidate)

    provider_heads: OrderedDict[str, deque[dict[str, Any]]] = OrderedDict()
    for source in sorted(provider_rows):
        rows = sorted(
            provider_rows[source],
            key=lambda candidate: (
                int((candidate.get("provider_ranks") or {}).get(source) or 10**9),
                _candidate_stable_key(candidate),
            ),
        )
        provider_heads[source] = deque(rows[:provider_head_limit])

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()
    active_sources = list(provider_heads)
    while active_sources and len(selected) < provider_slot_limit:
        next_sources: list[str] = []
        for source in active_sources:
            bucket = provider_heads[source]
            while bucket and len(selected) < provider_slot_limit:
                candidate = bucket.popleft()
                stable_key = _candidate_stable_key(candidate)
                if stable_key in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.add(stable_key)
                break
            if bucket:
                next_sources.append(source)
        active_sources = next_sources

    relevance_fraction = max(
        0.0,
        min(1.0, settings.bge_candidate_relevance_fraction),
    )
    anchor_fraction = max(
        0.0,
        min(0.5, settings.bge_candidate_anchor_fraction),
    )
    anchor_min_chars = max(2, settings.bge_candidate_anchor_min_chars)

    def anchor_score(candidate: dict[str, Any]) -> tuple[float, ...]:
        """Rank exact entity/dataset/method hits for a protected recall slice."""
        title = unicodedata.normalize(
            "NFKC", str(candidate.get("title") or "")
        ).casefold()
        abstract = unicodedata.normalize(
            "NFKC", str(candidate.get("abstract") or "")
        ).casefold()
        searchable = f"{title}\n{abstract}"
        exact_phrase_hits = sum(
            len(facet) >= anchor_min_chars and facet in searchable
            for facet in exact_facets
        )
        short_token_hits = sum(
            len(token) >= anchor_min_chars and token in title
            for token in query_terms
        )
        return (
            float(exact_phrase_hits),
            float(short_token_hits),
            -float(len(title.split())),
            _candidate_stable_key(candidate),
        )

    # Reserve a small exact-match slice before the generic relevance quota.
    # This prevents a semantically broad candidate pool from evicting a paper
    # whose title contains the only discriminating dataset or method token.
    anchor_slots = min(limit, max(0, round(limit * anchor_fraction)))
    anchor_first = sorted(
        candidates,
        key=lambda candidate: (
            tuple(-value if isinstance(value, (int, float)) else value for value in anchor_score(candidate)),
        ),
    )
    for candidate in anchor_first:
        if len(selected) >= min(limit, provider_slot_limit + anchor_slots):
            break
        stable_key = _candidate_stable_key(candidate)
        if stable_key in selected_keys:
            continue
        score = anchor_score(candidate)
        if score[0] <= 0 and score[1] <= 0:
            continue
        selected.append(candidate)
        selected_keys.add(stable_key)

    # ``relevance_fraction`` is an additional quota after provider-head
    # protection.  Previously it was treated as an absolute target, so a 25%
    # provider reservation silently reduced a configured 60% relevance pass
    # to only 35%.  That caused strong lower-ranked candidates to be replaced
    # by round-robin filler before BGE-M3 could score them.
    relevance_slots = min(
        limit,
        len(selected) + max(1, round(limit * relevance_fraction)),
    )
    relevance_first = sorted(
        candidates,
        key=lambda candidate: (
            tuple(-value for value in relevance_score(candidate)),
            _candidate_stable_key(candidate),
        ),
    )
    for candidate in relevance_first:
        if len(selected) >= relevance_slots:
            break
        stable_key = _candidate_stable_key(candidate)
        if stable_key in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(stable_key)

    bucket_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for candidate in candidates:
        stable_key = _candidate_stable_key(candidate)
        if stable_key in selected_keys:
            continue
        sources = {
            str(source)
            for source in [
                *(candidate.get("sources") or []),
                candidate.get("source") or "unknown",
            ]
            if source
        }
        iteration = int(candidate.get("source_iteration") or 0)
        for source in sources:
            bucket_rows.setdefault((source, iteration), []).append(candidate)

    buckets: OrderedDict[tuple[str, int], deque[dict[str, Any]]] = OrderedDict()
    for key in sorted(bucket_rows):
        source, _iteration = key
        rows = sorted(
            bucket_rows[key],
            key=lambda candidate: (
                int((candidate.get("provider_ranks") or {}).get(source) or 10**9),
                _candidate_stable_key(candidate),
            ),
        )
        buckets[key] = deque(rows)

    active_keys = list(buckets)
    while active_keys and len(selected) < limit:
        next_active: list[tuple[str, int]] = []
        for key in active_keys:
            bucket = buckets[key]
            while bucket and len(selected) < limit:
                candidate = bucket.popleft()
                stable_key = _candidate_stable_key(candidate)
                if stable_key in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.add(stable_key)
                break
            if bucket:
                next_active.append(key)
        active_keys = next_active
    return selected


def _embedding_candidate_limit(
    top_k: int,
    *,
    evaluation_fast_mode: bool = False,
) -> int:
    """Scale the model budget only for larger requested result windows."""
    settings = get_settings()
    base_limit = (
        settings.paper_test_bge_m3_candidate_limit
        if evaluation_fast_mode
        else settings.bge_m3_candidate_limit
    )
    maximum_limit = (
        settings.paper_test_bge_m3_max_candidate_limit
        if evaluation_fast_mode
        else settings.bge_m3_max_candidate_limit
    )
    return min(
        maximum_limit,
        max(base_limit, top_k * 3),
    )


def _calibrate_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    abs_scores = [float(r.get("relevance_score") or 0.0) for r in rows]
    max_abs = max(abs_scores) if abs_scores else 0.0
    n = len(rows)
    calibrated = []
    for idx, row in enumerate(rows):
        item = dict(row)
        abs_score = float(item.get("relevance_score") or 0.0)
        rel = (abs_score / max_abs) if max_abs > 1e-8 else 0.0
        rank_score = 1.0 - (idx / max(n - 1, 1))
        display = max(abs_score, 0.55 * rel + 0.25 * rank_score + 0.20 * abs_score)
        item["rerank_score"] = abs_score
        item["relevance_score"] = max(0.0, min(1.0, display))
        calibrated.append(item)
    return calibrated


def _empty_understanding() -> QueryUnderstanding:
    return QueryUnderstanding(intent="解析中…", language="zh+en")


def _benchmark_understanding(
    query: str,
    understanding: QueryUnderstanding,
) -> QueryUnderstanding:
    """Keep official benchmark queries even when a generic intent guard rejects them.

    Some IR benchmarks intentionally contain one- or two-word topics.  Only the
    paper-test path enables this override; normal interactive searches retain
    the stricter user-input validation.
    """
    if understanding.is_paper_search:
        return understanding
    topic = " ".join(str(query).split()).strip()
    return understanding.model_copy(
        update={
            "is_paper_search": True,
            "reject_reason": None,
            "intent": understanding.intent or f"检索与 {topic} 相关的学术论文",
            "topics": understanding.topics or [topic],
            "keywords": understanding.keywords or [topic],
            "sub_queries": understanding.sub_queries or [topic],
            "rewritten_queries": understanding.rewritten_queries or [topic],
        },
        deep=True,
    )


def _benchmark_fallback_understanding(
    query: str,
    error: Exception,
) -> tuple[QueryUnderstanding, dict[str, Any]]:
    """Build a conservative lexical intent when the intent LLM is unavailable.

    This path is restricted to benchmark mode.  It keeps retrieval evaluation
    focused on retrieval/ranking instead of silently scoring an LLM transport
    or JSON failure as a model miss.  Interactive search still fails closed.
    """
    topic = " ".join(str(query or "").split()).strip()
    if not topic:
        raise error
    understanding = QueryUnderstanding(
        intent=f"检索与 {topic} 相关的学术论文",
        is_paper_search=True,
        language="zh+en",
        topics=[topic],
        keywords=[topic],
        sub_queries=[topic],
        rewritten_queries=[topic, f"{topic} academic papers"],
        raw_json={
            "is_paper_search": True,
            "intent": f"检索与 {topic} 相关的学术论文",
            "topics": [topic],
            "keywords": [topic],
            "sub_queries": [topic],
            "rewritten_queries": [topic, f"{topic} academic papers"],
            "parser_version": "benchmark-lexical-fallback-v1",
            "fallback_reason": type(error).__name__,
        },
    )
    return understanding, {
        "intent_provider": "benchmark-lexical-fallback",
        "api_calls": 0,
        "token_usage": 0,
        "local_year": datetime.now().year,
        "is_paper_search": True,
        "benchmark_intent_fallback": True,
        "benchmark_intent_error": type(error).__name__,
    }


def iter_paper_search_events(
    query: str,
    max_papers: int = 5,
    use_mock: bool = False,
    controller_fulltext_documents: dict[str, FulltextDocument] | None = None,
    controller_fulltext_budget: int = 0,
    evaluation_fast_mode: bool = False,
    provider_cache_mode: str = "record",
    benchmark_mode: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield progressive agent events, then a final SearchResponse."""
    if use_mock:
        result = build_mock_search_response(query, max_papers=max_papers)
        for step in result.agent_trace:
            yield {"type": "step", "step": step.model_dump()}
        yield {"type": "understanding", "understanding": result.understanding.model_dump()}
        yield {"type": "result", "result": result.model_dump()}
        return

    settings = get_settings()
    if provider_cache_mode == "frozen" and (
        not evaluation_fast_mode or not settings.academic_provider_cache_enabled
    ):
        raise FrozenProviderCacheMiss(
            "冻结 Provider 快照要求开启 PAPER_TEST_FAST_MODE 和 "
            "ACADEMIC_PROVIDER_CACHE_ENABLED"
        )
    # Load the PaSa selector BEFORE any retrieval model is resident: its 4-bit
    # load materialises the bf16 checkpoint (RAM spike bounded by
    # PASA_SELECTOR_MAX_CPU_MEMORY_GB) and must happen while memory is free —
    # loading it after the encoder + faiss index are in memory segfaults on
    # 16GB machines. No-op when already loaded or disabled.
    if settings.pasa_selector_enabled:
        from backend.app.services.pasa_selector import get_pasa_selector

        get_pasa_selector()
    top_k = max_papers or settings.rag_top_k
    session_id = uuid.uuid4().hex
    trace: list[AgentTraceStep] = []
    meta: dict[str, Any] = {
        "mode": "intent+academic-multisource+bounded-citation+tavily+evidrank+scimap",
        "api_calls": 0,
        "token_usage": 0,
        "session_id": session_id,
        "milvus_collection": settings.milvus_collection,
        "evaluation_fast_mode": evaluation_fast_mode,
        "provider_cache_mode": provider_cache_mode,
        "benchmark_mode": benchmark_mode,
    }

    def emit(step: AgentTraceStep) -> dict[str, Any]:
        # Replace previous running step with same name, else append.
        replaced = False
        for i, old in enumerate(trace):
            if old.step == step.step and old.status == "running":
                trace[i] = step
                replaced = True
                break
        if not replaced:
            # also replace if updating same step id
            for i, old in enumerate(trace):
                if old.step == step.step and step.status != "running":
                    trace[i] = step
                    replaced = True
                    break
        if not replaced:
            trace.append(step)
        return {"type": "step", "step": step.model_dump(), "agent_trace": [t.model_dump() for t in trace]}

    # Think
    yield emit(
        _trace(
            "think",
            "分析用户科研问题，准备进行意图解析、多路检索与重排。",
            phase="think",
            input_text=query,
            status="running",
        )
    )
    yield emit(
        _trace(
            "think",
            "分析用户科研问题，准备进行意图解析、多路检索与重排。",
            phase="think",
            input_text=query,
            status="done",
        )
    )

    # Intent
    yield emit(
        _trace(
            "understand",
            "正在调用小模型进行意图识别…",
            phase="tool",
            tool="zhipu-intent",
            input_text=query,
            status="running",
        )
    )
    intent_cache_key = {
        "query": query,
        "model": settings.zhipu_intent_model,
        "parser": "query-understanding-v3-contract-aligned",
    }
    cached_intent = (
        load_cached_json(
            "intent",
            intent_cache_key,
            ttl_seconds=settings.academic_provider_cache_ttl_seconds,
            allow_expired=provider_cache_mode == "frozen",
        )
        if evaluation_fast_mode and settings.academic_provider_cache_enabled
        else None
    )
    if isinstance(cached_intent, dict) and isinstance(
        cached_intent.get("understanding"), dict
    ):
        understanding = QueryUnderstanding.model_validate(cached_intent["understanding"])
        original_meta = dict(cached_intent.get("meta") or {})
        intent_meta = {
            **original_meta,
            "api_calls": 0,
            "token_usage": 0,
            "cached_api_calls": int(original_meta.get("api_calls") or 0),
            "cached_token_usage": int(original_meta.get("token_usage") or 0),
            "cache_hit": True,
            "cache_entry_sha256": cache_payload_sha256(cached_intent),
        }
    else:
        if (
            evaluation_fast_mode
            and provider_cache_mode == "frozen"
            and settings.academic_provider_cache_enabled
        ):
            raise FrozenProviderCacheMiss(
                f"frozen provider snapshot missing intent for query={query!r}"
            )
        try:
            understanding, intent_meta = parse_query_intent(query)
        except Exception as exc:  # noqa: BLE001 - benchmark fallback is explicit
            if not benchmark_mode:
                raise
            understanding, intent_meta = _benchmark_fallback_understanding(query, exc)
        if evaluation_fast_mode and settings.academic_provider_cache_enabled:
            cached_intent_payload = {
                "understanding": understanding.model_dump(mode="json"),
                "meta": intent_meta,
            }
            save_cached_json(
                "intent",
                intent_cache_key,
                cached_intent_payload,
            )
        intent_meta = {
            **intent_meta,
            "cache_hit": False,
            "cache_entry_sha256": (
                cache_payload_sha256(cached_intent_payload)
                if evaluation_fast_mode and settings.academic_provider_cache_enabled
                else None
            ),
        }
    # Auxiliary query-variant fusion: pre-generated PaSa Crawler queries are
    # merged into the understanding as EXTRA variants (never a replacement of
    # the intent parser). Applies to both the cached and freshly-parsed
    # understanding. Opt-in via CRAWLER_QUERY_FUSION_ENABLED.
    if settings.crawler_query_fusion_enabled:
        from backend.app.services.query_variant_fusion import (
            crawler_fusion_digest,
            fuse_crawler_queries,
        )

        understanding = fuse_crawler_queries(query, understanding)
        fusion_digest = crawler_fusion_digest(query)
        if fusion_digest:
            meta["crawler_query_fusion"] = fusion_digest
    if benchmark_mode and not understanding.is_paper_search:
        understanding = _benchmark_understanding(query, understanding)
        intent_meta = {**intent_meta, "benchmark_intent_override": True}

    meta["api_calls"] += int(intent_meta.get("api_calls") or 0)
    meta["token_usage"] += int(intent_meta.get("token_usage") or 0)
    meta["intent_provider"] = intent_meta.get("intent_provider")
    meta["local_year"] = intent_meta.get("local_year")
    meta["is_paper_search"] = understanding.is_paper_search

    yield {"type": "understanding", "understanding": understanding.model_dump()}

    if not understanding.is_paper_search:
        reason = understanding.reject_reason or "请输入查找论文相关问题"
        yield emit(
            _trace(
                "validate",
                f"问题不合法：{reason}",
                phase="observe",
                tool="zhipu-intent",
                input_text=query,
                output_text=reason,
                status="error",
                api_calls=int(intent_meta.get("api_calls") or 0),
                token_usage=int(intent_meta.get("token_usage") or 0),
                cumulative_api_calls=meta["api_calls"],
                cumulative_token_usage=meta["token_usage"],
            )
        )
        yield {
            "type": "invalid_query",
            "message": "请输入查找论文相关问题",
            "reject_reason": reason,
            "understanding": understanding.model_dump(),
        }
        return

    yield emit(
        _trace(
            "plan",
            (
                f"计划：领域={understanding.domain or '-'}；时间={understanding.time_range or '-'}；"
                f"语言={understanding.language or '-'}；子查询={len(understanding.sub_queries)}。"
            ),
            phase="plan",
            tool="zhipu-intent",
            output_text=str(understanding.raw_json)[:500],
            status="done",
        )
    )
    yield emit(
        _trace(
            "understand",
            f"小模型意图识别完成（{meta['intent_provider']}）",
            phase="observe",
            tool="zhipu-intent",
            input_text=query,
            output_text=(
                f"time_range={understanding.time_range}, language={understanding.language}, "
                f"rewritten={understanding.rewritten_queries[:2]}"
            ),
            status="done",
            api_calls=int(intent_meta.get("api_calls") or 0),
            token_usage=int(intent_meta.get("token_usage") or 0),
            cumulative_api_calls=meta["api_calls"],
            cumulative_token_usage=meta["token_usage"],
        )
    )

    # Adaptive retrieval agent: plan -> retrieve -> analyze -> evolve/stop.
    agent_initial_state = {
        "raw_query": query,
        "understanding": understanding.model_dump(),
        "max_papers": top_k,
        "candidates": [],
        "round_history": [],
        "agent_trace": [],
        "api_calls": 0,
        "token_usage": 0,
        "target_candidates": settings.academic_min_candidates,
        "max_iterations": settings.agent_search_max_iterations,
        "round_candidate_target": settings.agent_search_round_target,
        "min_new_candidates": settings.agent_search_min_new_candidates,
        "min_coverage": settings.agent_search_min_coverage,
        "max_api_calls": settings.agent_search_max_api_calls,
        "llm_evolution_enabled": settings.agent_search_llm_evolution_enabled,
        "llm_evolution_calls": 0,
        "evaluation_fast_mode": evaluation_fast_mode,
        "provider_cache_mode": provider_cache_mode,
    }
    final_agent_state: dict[str, Any] = {}
    emitted_agent_steps = 0
    for agent_state in _get_paper_search_graph().stream(
        agent_initial_state,
        stream_mode="values",
    ):
        final_agent_state = dict(agent_state)
        agent_steps = list(agent_state.get("agent_trace") or [])
        for raw_step in agent_steps[emitted_agent_steps:]:
            yield emit(AgentTraceStep.model_validate(raw_step))
        emitted_agent_steps = len(agent_steps)

    candidates = list(final_agent_state.get("candidates") or [])
    merged_candidate_trace = _candidate_trace_rows(candidates)
    provider_candidate_trace = [
        candidate
        for round_info in final_agent_state.get("round_history") or []
        for candidate in round_info.get("provider_candidates") or []
    ]
    compact_round_history = [
        {key: value for key, value in round_info.items() if key != "provider_candidates"}
        for round_info in final_agent_state.get("round_history") or []
    ]
    agent_api_calls = int(final_agent_state.get("api_calls") or 0)
    agent_token_usage = int(final_agent_state.get("token_usage") or 0)
    meta["api_calls"] += agent_api_calls
    meta["token_usage"] += agent_token_usage
    meta["search_agent"] = {
        "iterations": int(final_agent_state.get("current_iteration") or 0) + 1,
        "executed_queries": list(final_agent_state.get("executed_queries") or []),
        "stop_reason": final_agent_state.get("stop_reason"),
        "coverage": final_agent_state.get("coverage") or {},
        "coverage_gaps": list(final_agent_state.get("coverage_gaps") or []),
        "round_history": compact_round_history,
        "api_calls": agent_api_calls,
        "token_usage": agent_token_usage,
        "budget": {
            "max_iterations": settings.agent_search_max_iterations,
            "max_api_calls": settings.agent_search_max_api_calls,
            "target_candidates": settings.academic_min_candidates,
            "round_candidate_target": settings.agent_search_round_target,
            "min_coverage": settings.agent_search_min_coverage,
            "min_new_candidates": settings.agent_search_min_new_candidates,
            "llm_evolution_enabled": settings.agent_search_llm_evolution_enabled,
            "llm_evolution_max_calls": settings.agent_search_llm_evolution_max_calls,
        },
    }
    provider_snapshot_entries = [
        str(value)
        for value in [
            intent_meta.get("cache_entry_sha256"),
            *[
                attempt.get("cache_entry_sha256")
                for round_info in compact_round_history
                for provider_meta in (round_info.get("academic") or {}).get(
                    "providers", {}
                ).values()
                for attempt in provider_meta.get("attempts") or []
            ],
            *[
                (round_info.get("tavily") or {}).get("cache_entry_sha256")
                for round_info in compact_round_history
            ],
            *[
                ((round_info.get("academic") or {}).get("citation_expansion") or {}).get(
                    "cache_entry_sha256"
                )
                for round_info in compact_round_history
            ],
        ]
        if value
    ]
    meta["provider_snapshot_entries"] = sorted(provider_snapshot_entries)
    # Fingerprint the actual retrieval input, not timing/API/cache-hit fields.
    # A live recording and a frozen replay therefore compare equal when they
    # produced the same candidates, while provider drift changes the hash.
    normalized_provider_candidates = sorted(
        provider_candidate_trace,
        key=lambda row: json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )
    meta["provider_snapshot_sha256"] = cache_payload_sha256(
        {
            "intent": understanding.model_dump(mode="json"),
            "executed_queries": list(final_agent_state.get("executed_queries") or []),
            "provider_candidates": normalized_provider_candidates,
        }
    )
    meta["cost_breakdown"] = {
        "intent_api_calls": int(intent_meta.get("api_calls") or 0),
        "intent_token_usage": int(intent_meta.get("token_usage") or 0),
        "agent_api_calls": agent_api_calls,
        "agent_token_usage": agent_token_usage,
        "citation_api_calls": sum(
            int(
                ((round_info.get("academic") or {}).get("citation_expansion") or {}).get(
                    "api_calls"
                )
                or 0
            )
            for round_info in compact_round_history
        ),
        "total_api_calls": meta["api_calls"],
        "total_token_usage": meta["token_usage"],
        "token_scope": "LLM API only; local BGE tokenization excluded",
    }
    meta["candidate_count_before_bge"] = len(candidates)
    meta["candidate_trace"] = {
        "provider": provider_candidate_trace,
        "merged": merged_candidate_trace,
    }
    meta["academic_target_met_after_fallback"] = (
        len(candidates) >= settings.academic_min_candidates
    )

    if not candidates:
        summary = SearchSummary(
            overview="学术 API 与网页检索均未返回可用论文，请尝试更具体的学术关键词。",
            key_findings=[],
            research_gaps=[],
        )
        response = SearchResponse(
            query=query,
            understanding=understanding,
            papers=[],
            graph=_build_graph(query, understanding, []),
            summary=summary,
            agent_trace=trace,
            meta=meta,
            session_id=session_id,
        )
        save_search_result(
            session_id=session_id,
            query=query,
            understanding=understanding.model_dump(),
            papers=[],
            meta=meta,
        )
        yield {"type": "result", "result": response.model_dump()}
        return

    if (
        settings.academic_require_min_candidates
        and len(candidates) < settings.academic_min_candidates
    ):
        raise RuntimeError(
            f"学术候选不足：跨源去重后仅 {len(candidates)} 篇，"
            f"要求至少 {settings.academic_min_candidates} 篇；未进入 BGE-M3 排序链路"
        )

    # Small interactive searches retain the proven 80-paper model budget.
    # Larger evaluation/result windows receive bounded extra headroom so a
    # relevant local-corpus paper at ranks 81-100 is not discarded before BGE.
    # The cap prevents a large custom K from growing inference without bound.
    embedding_limit = _embedding_candidate_limit(
        top_k,
        evaluation_fast_mode=evaluation_fast_mode,
    )
    raw_candidate_count = len(candidates)
    candidates = _limit_embedding_candidates(
        candidates,
        embedding_limit,
        query=query,
        understanding=understanding,
    )
    meta["candidate_trace"]["budget"] = _candidate_trace_rows(candidates)
    meta["embedding_candidate_limit"] = embedding_limit
    meta["embedding_candidate_count"] = len(candidates)
    meta["embedding_candidates_trimmed"] = raw_candidate_count - len(candidates)
    if raw_candidate_count > len(candidates):
        yield emit(
            _trace(
                "tool:candidate-budget",
                (
                    f"候选集从 {raw_candidate_count} 篇压缩到 {len(candidates)} 篇，"
                    "保留不同来源和检索轮次后再进行 BGE-M3 编码。"
                ),
                phase="plan",
                tool="source-round-diversity-selector",
                input_text=f"raw={raw_candidate_count}",
                output_text=f"selected={len(candidates)}",
                status="done",
            )
        )

    if settings.evidrank_enabled:
        yield emit(
            _trace(
                "tool:evidrank",
                "正在执行身份复核、元数据校验、非补偿门控、混合粗排、精排与证据矩阵…",
                phase="rank",
                tool="evidrank-v1",
                status="running",
            )
        )
        pipeline = get_evidrank_pipeline()
        meta["evidrank_decision_policy"] = pipeline.decision_policy
        ranking_bundle = pipeline.rank(
            session_id=session_id,
            query=query,
            understanding=understanding,
            candidate_rows=candidates,
            top_k=top_k,
            fulltext_documents=controller_fulltext_documents,
            fulltext_budget=controller_fulltext_budget,
            fast_evaluation=evaluation_fast_mode,
        )

        # Freezing belongs to the controller, never to EvidRank. The LangGraph
        # stream has reached END here; fail closed if a future graph returns a
        # state that still requests another search round.
        if bool(final_agent_state.get("should_continue")):
            raise RuntimeError("controller attempted to freeze while search should continue")
        controller_stop_reason = str(
            final_agent_state.get("stop_reason") or "controller_graph_completed"
        )
        final_ranking_bundle = freeze_bundle_for_controller(ranking_bundle)
        artifact_intent = research_intent_from_understanding(
            session_id=session_id,
            query=query,
            understanding=understanding,
        )
        output_artifact = build_output_artifact(final_ranking_bundle, artifact_intent)
        papers = _to_paper_items(ranking_bundle, top_k)
        meta["candidate_trace"]["ranking"] = _ranking_trace_rows(ranking_bundle)
        meta["candidate_trace"]["final"] = [
            {
                "id": paper.id,
                "title": paper.title,
                "arxiv_id": paper.arxiv_id,
                "rank": paper.final_rank,
                "level": paper.relevance.value,
            }
            for paper in papers
        ]

        _sel_meta = dict(
            ranking_bundle.execution_metadata.get("selector") or {}
        )
        meta["selector"] = _sel_meta.get("provider")
        meta["selector_applied"] = _sel_meta.get("applied")
        meta["selector_scored"] = _sel_meta.get("scored_count")
        if _sel_meta.get("error"):
            meta["selector_error"] = _sel_meta["error"]
        if _sel_meta.get("inference_ms") is not None:
            meta["selector_inference_ms"] = _sel_meta["inference_ms"]

        meta.update(
            {
                "controller_stop_reason": controller_stop_reason,
                "ranking_run_id": ranking_bundle.ranking_run_id,
                "ranking_bundle_hash": final_ranking_bundle.bundle_hash,
                "ranking_bundle_frozen": True,
                "scimap_artifact_hash": output_artifact.artifact_hash,
                "ranking_levels": {
                    "high": len(ranking_bundle.high),
                    "partial": len(ranking_bundle.partial),
                    "verify": len(ranking_bundle.verify),
                    "excluded": len(ranking_bundle.excluded),
                },
                "coverage_gap_count": len(ranking_bundle.coverage_gaps),
                "uncertainty_count": len(ranking_bundle.uncertainty_queue),
                "model_versions": ranking_bundle.model_versions,
                "evidrank_timings": ranking_bundle.timings.model_dump(),
                "evidrank_degraded": ranking_bundle.degraded,
                "evidrank_degradation_reasons": ranking_bundle.degradation_reasons,
            }
        )
        yield emit(
            _trace(
                "tool:evidrank",
                (
                    f"EvidRank 完成：high={len(ranking_bundle.high)}，"
                    f"verify={len(ranking_bundle.verify)}，partial={len(ranking_bundle.partial)}，"
                    f"excluded={len(ranking_bundle.excluded)}；"
                    f"总耗时={ranking_bundle.timings.total_ms / 1000:.1f}s，"
                    f"BGE={ranking_bundle.timings.hybrid_retrieval_ms / 1000:.1f}s，"
                    f"精排={ranking_bundle.timings.cross_encoder_ms / 1000:.1f}s，"
                    f"证据={ranking_bundle.timings.evidence_judge_ms / 1000:.1f}s"
                ),
                phase="rank",
                tool="evidrank-v1",
                input_text=f"candidate_registry_snapshot={len(candidates)}",
                output_text=(
                    f"ranking_run_id={ranking_bundle.ranking_run_id}, "
                    f"bundle_hash={final_ranking_bundle.bundle_hash}, frozen=true"
                ),
                status="done",
            )
        )

        valid_claims = [
            claim.text
            for claim in output_artifact.synthesis.claims
            if claim.validation_status == "valid"
        ]
        summary = SearchSummary(
            overview=(
                f"完成 {len(candidates)} 篇候选的 EvidRank 约束—证据排序，"
                f"输出 {len(papers)} 篇非排除论文；SciMap 已生成证据卡片、"
                "动态比较矩阵、保守时间线、证据图和分级结论。"
            ),
            key_findings=valid_claims[:5],
            research_gaps=[
                f"{gap.facet_id}: 支持 {gap.supported_count}/{gap.target_count}"
                for gap in ranking_bundle.coverage_gaps[:5]
            ],
        )
        yield emit(
            _trace(
                "tool:scimap",
                (
                    "SciMap 完成证据化归纳；"
                    f"claims={output_artifact.claim_validation.total_claims}，"
                    f"invalid={output_artifact.claim_validation.invalid_claims}"
                ),
                phase="observe",
                tool="scimap-v1",
                input_text=f"bundle_hash={final_ranking_bundle.bundle_hash}",
                output_text=f"artifact_hash={output_artifact.artifact_hash}",
                status="done",
            )
        )

        yield emit(
            _trace(
                "persist",
                "正在写入冻结排序与不可变归纳结果…",
                phase="persist",
                tool="postgres",
                status="running",
            )
        )
        try:
            save_search_result(
                session_id=session_id,
                query=query,
                understanding=understanding.model_dump(),
                papers=[paper.model_dump() for paper in papers],
                meta=meta,
                ranking_bundle=final_ranking_bundle.model_dump(mode="json"),
            )
            save_output_artifact(
                session_id=session_id,
                bundle=final_ranking_bundle,
                artifact=output_artifact,
            )
            meta["postgres"] = "ok"
            yield emit(
                _trace(
                    "persist",
                    "冻结 RankingBundle 与不可变 OutputArtifact 已写入 PostgreSQL",
                    phase="persist",
                    tool="postgres",
                    status="done",
                )
            )
        except Exception as exc:  # noqa: BLE001
            meta["postgres"] = f"error: {exc}"
            yield emit(
                _trace(
                    "persist",
                    f"PostgreSQL 写入失败：{exc}",
                    phase="persist",
                    tool="postgres",
                    status="error",
                )
            )

        response = SearchResponse(
            query=query,
            understanding=understanding,
            papers=papers,
            graph=_build_graph(query, understanding, papers),
            summary=summary,
            agent_trace=trace,
            meta=meta,
            session_id=session_id,
            ranking_bundle=final_ranking_bundle,
            output_artifact=output_artifact,
        )
        yield {"type": "result", "result": response.model_dump()}
        return

    # Index
    yield emit(
        _trace(
            "tool:bge-m3-index",
            "正在用 BGE-M3 生成稠密/稀疏向量并写入 Milvus…",
            phase="tool",
            tool="bge-m3.encode+milvus.upsert",
            status="running",
        )
    )
    store = get_milvus_store()
    upsert_meta = store.upsert_papers(session_id=session_id, papers=candidates)
    meta["milvus_upserted"] = upsert_meta.get("upserted")
    meta["embedding_provider"] = upsert_meta.get("embedding_provider")
    meta["dense_dim"] = upsert_meta.get("dense_dim")
    meta["encoding_seconds"] = upsert_meta.get("encoding_seconds")
    meta["milvus_write_seconds"] = upsert_meta.get("milvus_write_seconds")
    meta["total_upsert_seconds"] = upsert_meta.get("total_upsert_seconds")
    yield emit(
        _trace(
            "tool:bge-m3-index",
            f"BGE-M3 写入 Milvus 集合 {settings.milvus_collection}，共 {upsert_meta.get('upserted')} 篇（dense+sparse）",
            phase="tool",
            tool="bge-m3.encode+milvus.upsert",
            input_text=f"session_id={session_id}",
            output_text=str(upsert_meta),
            status="done",
        )
    )

    # RRF
    yield emit(
        _trace(
            "tool:rrf",
            "正在进行稠密+稀疏多路召回与 RRF 融合…",
            phase="rank",
            tool="milvus.hybrid_search+RRF",
            status="running",
        )
    )
    ranked_rows, search_meta = store.search_top_k(
        query=query,
        session_id=session_id,
        top_k=max(top_k, settings.hybrid_recall_limit),
    )
    meta["embedding_provider_query"] = search_meta.get("embedding_provider")
    meta["retriever"] = search_meta.get("retriever")
    meta["rrf_k"] = search_meta.get("rrf_k")
    if search_meta.get("error"):
        meta["retriever_error"] = search_meta["error"]

    by_id = {c["id"]: c for c in candidates}
    for row in ranked_rows:
        src = by_id.get(row["id"]) or {}
        row["tags"] = src.get("tags") or []
        row["venue"] = src.get("venue")
        row["authors"] = src.get("authors") or []
        row["citation_count"] = int(src.get("citation_count") or 0)
        row["source"] = src.get("source") or row.get("source")
        row["sources"] = list(src.get("sources") or [])
        row["external_ids"] = dict(src.get("external_ids") or {})
        row["tavily_score"] = float(src.get("score") or 0.0)
        if not row.get("published_at"):
            row["published_at"] = src.get("published_at")
        if not row.get("year"):
            row["year"] = src.get("year")

    yield emit(
        _trace(
            "tool:rrf",
            f"稠密语义召回 + 稀疏关键词召回，RRF(k={settings.rrf_k}) 融合得到 {len(ranked_rows)} 条",
            phase="rank",
            tool="milvus.hybrid_search+RRF",
            output_text=f"hits={len(ranked_rows)}",
            status="done",
        )
    )

    # Rerank
    yield emit(
        _trace(
            "tool:reranker",
            "正在进行 BGE Cross-Encoder 精排…",
            phase="rank",
            tool="bge-reranker",
            status="running",
        )
    )
    reranker = get_bge_reranker()
    ranked_rows, rerank_meta = reranker.rerank(query=query, papers=ranked_rows, top_k=top_k)
    ranked_rows = _calibrate_scores(ranked_rows)
    meta["reranker"] = rerank_meta.get("reranker")
    meta["reranker_applied"] = rerank_meta.get("applied")
    if rerank_meta.get("error"):
        meta["reranker_error"] = rerank_meta["error"]
    yield emit(
        _trace(
            "tool:reranker",
            (
                f"Cross-Encoder 精排（{rerank_meta.get('reranker')}, applied={rerank_meta.get('applied')}），"
                f"输出 Top-{len(ranked_rows)}"
            ),
            phase="rank",
            tool="bge-reranker",
            status="done" if rerank_meta.get("applied") else "skipped",
            output_text=f"top_k={len(ranked_rows)}",
        )
    )

    papers = _to_legacy_paper_items(ranked_rows)
    summary = SearchSummary(
        overview=(
            f"已完成 Serper Scholar、arXiv、OpenAlex、PubMed 与 Tavily 多源检索，"
            f"合并去重得到 {len(candidates)} 篇候选；随后完成 BGE-M3 稠密/稀疏召回、RRF 融合"
            f"{'与 Reranker 精排' if rerank_meta.get('applied') else ''}，返回 Top-{len(papers)}。"
            f" 意图领域：{understanding.domain or '未指定'}；"
            f"时间范围：{understanding.time_range or '未指定'}；"
            f"语言：{understanding.language or 'zh+en'}。"
        ),
        key_findings=[p.title for p in papers[:3]],
        research_gaps=[
            (
                f"本次候选池达到召回目标（{settings.academic_min_candidates} 篇）。"
                if len(candidates) >= settings.academic_min_candidates
                else f"本次可验证的唯一候选仅 {len(candidates)} 篇，低于目标 "
                f"{settings.academic_min_candidates}；仍基于现有候选完成排序，"
                "且未用重复或无关记录凑数。"
            ),
            "可通过来源计数、DOI/PMID/arXiv/OpenAlex ID 和检索错误元数据审计召回过程。",
        ],
    )
    yield emit(
        _trace(
            "synthesize",
            "整理结构化结果（列表/关系图/摘要）",
            phase="observe",
            status="done",
        )
    )

    # Persist
    yield emit(
        _trace(
            "persist",
            "正在写入 PostgreSQL…",
            phase="persist",
            tool="postgres",
            status="running",
        )
    )
    try:
        save_search_result(
            session_id=session_id,
            query=query,
            understanding=understanding.model_dump(),
            papers=[p.model_dump() for p in papers],
            meta=meta,
        )
        meta["postgres"] = "ok"
        yield emit(
            _trace(
                "persist",
                "检索结果已写入 PostgreSQL（search_sessions / search_papers）",
                phase="persist",
                tool="postgres",
                status="done",
            )
        )
    except Exception as exc:  # noqa: BLE001
        meta["postgres"] = f"error: {exc}"
        yield emit(
            _trace(
                "persist",
                f"PostgreSQL 写入失败：{exc}",
                phase="persist",
                tool="postgres",
                status="error",
            )
        )

    response = SearchResponse(
        query=query,
        understanding=understanding,
        papers=papers,
        graph=_build_graph(query, understanding, papers),
        summary=summary,
        agent_trace=trace,
        meta=meta,
        session_id=session_id,
    )
    yield {"type": "result", "result": response.model_dump()}


def run_paper_search(
    query: str,
    max_papers: int = 5,
    use_mock: bool = False,
    controller_fulltext_documents: dict[str, FulltextDocument] | None = None,
    controller_fulltext_budget: int = 0,
    evaluation_fast_mode: bool = False,
    provider_cache_mode: str = "record",
    benchmark_mode: bool = False,
) -> SearchResponse:
    """Non-streaming wrapper around the event iterator."""
    final: SearchResponse | None = None
    for event in iter_paper_search_events(
        query,
        max_papers=max_papers,
        use_mock=use_mock,
        controller_fulltext_documents=controller_fulltext_documents,
        controller_fulltext_budget=controller_fulltext_budget,
        evaluation_fast_mode=evaluation_fast_mode,
        provider_cache_mode=provider_cache_mode,
        benchmark_mode=benchmark_mode,
    ):
        if event.get("type") == "invalid_query":
            raise InvalidPaperQueryError(
                str(
                    event.get("reject_reason")
                    or event.get("message")
                    or "请输入查找论文相关问题"
                )
            )
        if event.get("type") == "result":
            final = SearchResponse.model_validate(event["result"])
    if final is None:
        raise RuntimeError("搜索未返回结果")
    return final
