"""Feature-4 (SciMap) component ablation.

Generates a frozen FinalRankingBundle for real PaSa queries (local FTS +
vector retrieval, offline), then builds the SciMap output artifact with each
organizer component disabled in turn, measuring automatic structural-quality
metrics (citation completeness, unknown-cell handling, graph edge legality,
claim citation precision).

Components: ranked_list, comparison_matrix, timeline, evidence_graph,
comparability, synthesis (+claim validation), artifact_hash.

Run: python -m backend.scripts.ablate_scimap [--limit N] [--queries QID,QID,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models.schemas import QueryUnderstanding  # noqa: E402
from backend.app.services.embeddings import get_bge_m3_encoder  # noqa: E402
from backend.app.services.llm_intent import parse_query_intent  # noqa: E402
from backend.app.services.local_pasa_corpus import search_local_pasa_corpus  # noqa: E402
from backend.app.services.local_vector_corpus import search_local_vector_corpus  # noqa: E402
from backend.app.services.organizer.comparison_matrix import build_comparison_matrix  # noqa: E402
from backend.app.services.organizer.comparability import build_comparability_assessments  # noqa: E402
from backend.app.services.organizer.evidence_graph import build_evidence_graph  # noqa: E402
from backend.app.services.organizer.ranked_list import build_ranked_list  # noqa: E402
from backend.app.services.organizer.synthesis import build_synthesis  # noqa: E402
from backend.app.services.organizer.timeline import build_timeline  # noqa: E402
from backend.app.services.ranking.adapters import research_intent_from_understanding  # noqa: E402
from backend.app.services.ranking.bundle_builder import finalize_draft_hash, freeze_bundle_for_controller  # noqa: E402
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge  # noqa: E402
from backend.app.services.ranking.pipeline import EvidRankPipeline  # noqa: E402
from backend.app.services.reranker import get_bge_reranker  # noqa: E402


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _candidate_rows(papers: list[dict]) -> list[dict]:
    rows = []
    for rank, paper in enumerate(papers, start=1):
        rows.append(
            {
                "id": paper.get("id") or paper.get("arxiv_id") or paper.get("title"),
                "title": paper.get("title") or "",
                "abstract": paper.get("abstract") or "",
                "year": paper.get("year"),
                "authors": paper.get("authors") or [],
                "external_ids": paper.get("external_ids") or {},
                "provider_ranks": {"local": rank},
                "provider_scores": {"local": float(paper.get("score") or 0)},
                "sources": [paper.get("source") or "local"],
                "source": paper.get("source") or "local",
                "citation_count": int(paper.get("citation_count") or 0),
                "url": paper.get("url"),
                "venue": paper.get("venue"),
            }
        )
    return rows


def _artifact_metrics(bundle, intent, *, include: dict[str, bool]) -> dict:
    """Compute SciMap structural-quality metrics for the enabled components."""
    metrics: dict = {}
    if include.get("ranked_list", True):
        cards = build_ranked_list(bundle)
        metrics["cards"] = len(cards)
        metrics["cards_with_evidence"] = sum(
            1 for card in cards if len(card.model_dump().get("evidence_ids") or [])
        )
    if include.get("comparison_matrix", True):
        matrix = build_comparison_matrix(bundle, intent).model_dump(mode="json")
        rows = matrix.get("rows") or []
        cell_total = 0
        unknown = 0
        for row in rows:
            for cell in (row.get("cells") or {}).values():
                cell_total += 1
                if str(cell.get("status") or "") in {"unknown", "Unknown", ""}:
                    unknown += 1
        metrics["matrix_rows"] = len(rows)
        metrics["matrix_cells"] = cell_total
        metrics["matrix_unknown"] = unknown
    if include.get("timeline", True):
        timeline = build_timeline(bundle)
        metrics["timeline_entries"] = len(timeline)
    if include.get("evidence_graph", True):
        graph = build_evidence_graph(bundle).model_dump(mode="json")
        edges = graph.get("edges") or []
        illegal = [
            e for e in edges
            if str(e.get("relation") or "") not in {"supports", "violates"}
        ]
        metrics["graph_nodes"] = len(graph.get("nodes") or [])
        metrics["graph_edges"] = len(edges)
        metrics["graph_illegal_edges"] = len(illegal)
    if include.get("comparability", True):
        comparisons = build_comparability_assessments(bundle, intent)
        metrics["comparability_pairs"] = len(comparisons)
    if include.get("synthesis", True):
        synthesis, validation = build_synthesis(bundle, intent)
        claims = synthesis.model_dump(mode="json").get("claims") or []
        cited = [
            c for c in claims
            if c.get("citation_ids") or c.get("evidence_ids") or c.get("citations")
        ]
        metrics["claims"] = len(claims)
        metrics["claims_cited"] = len(cited)
        metrics["claim_citation_precision"] = (len(cited) / len(claims)) if claims else 1.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--queries", type=str, default="")
    args = parser.parse_args()
    _safe_stdout()

    test_path = ROOT / "backend" / "paper-test" / "datasets" / "RealScholarQuery" / "test.jsonl"
    rows = [json.loads(line) for line in test_path.open(encoding="utf-8")]
    if args.queries:
        wanted = set(args.queries.split(","))
        rows = [r for r in rows if str(r.get("qid")) in wanted]
    elif args.limit:
        rows = rows[: args.limit]

    encoder = get_bge_m3_encoder()
    pipeline = EvidRankPipeline(
        encoder=encoder,
        reranker=get_bge_reranker(),
        evidence_judge=DeterministicEvidenceJudge(),
    )

    components = ["ranked_list", "comparison_matrix", "timeline", "evidence_graph", "comparability", "synthesis"]
    variants: list[tuple[str, dict[str, bool]]] = [("full", {c: True for c in components})]
    for component in components:
        include = {c: True for c in components}
        include[component] = False
        variants.append((f"no_{component}", include))

    report: dict = {"queries": [], "variants": {}}
    for row in rows:
        query = row.get("question") or ""
        qid = str(row.get("qid"))
        print(f"\n=== {qid}: {query[:60]} ===", flush=True)
        # Real intent parsing (LLM) so the bundle carries genuine constraints
        # and therefore real evidence records for SciMap to consume.
        understanding, _intent_meta = parse_query_intent(query)
        print(f"  intent: constraints={len(understanding.methods or [])} methods "
              f"datasets={len(understanding.datasets or [])} keywords={len(understanding.keywords or [])}",
              flush=True)
        # Local multi-source retrieval (offline).
        fts_papers, _ = search_local_pasa_corpus(query, understanding, limit=100, offset=0)
        vec_papers, _ = search_local_vector_corpus(query, understanding, limit=100, offset=0)
        seen = set()
        merged = []
        for paper in [*fts_papers, *vec_papers]:
            key = str(paper.get("id") or paper.get("arxiv_id") or paper.get("title"))
            if key not in seen:
                seen.add(key)
                merged.append(paper)
        if not merged:
            print("  no candidates", flush=True)
            continue
        bundle = pipeline.rank(
            session_id=f"scimap-{qid}",
            query=query,
            understanding=understanding,
            candidate_rows=_candidate_rows(merged),
            top_k=10,
            fast_evaluation=True,
        )
        bundle = finalize_draft_hash(bundle)
        frozen = freeze_bundle_for_controller(bundle)
        intent = research_intent_from_understanding(
            session_id=f"scimap-{qid}", query=query, understanding=understanding
        )
        report["queries"].append({"qid": qid, "candidates": len(merged)})

        for variant_name, include in variants:
            try:
                metrics = _artifact_metrics(frozen, intent, include=include)
            except Exception as exc:  # noqa: BLE001
                print(f"  {variant_name:<20} ERROR: {type(exc).__name__}: {exc}", flush=True)
                continue
            report["variants"].setdefault(variant_name, []).append(metrics)
            if variant_name == "full":
                print(
                    f"  full: cards={metrics.get('cards')} matrix={metrics.get('matrix_rows')}x{metrics.get('matrix_cells')} "
                    f"unknown={metrics.get('matrix_unknown')} timeline={metrics.get('timeline_entries')} "
                    f"graph={metrics.get('graph_nodes')}n/{metrics.get('graph_edges')}e "
                    f"claims={metrics.get('claims')} cited={metrics.get('claims_cited')}",
                    flush=True,
                )

    # Aggregate per variant.
    print("\n=== 功能4 SciMap 组件消融（结构质量指标，跨查询平均） ===")
    summary: dict[str, dict] = {}
    for variant_name, metrics_list in report["variants"].items():
        agg: dict = {}
        for metric in metrics_list:
            for key, value in metric.items():
                agg[key] = agg.get(key, 0.0) + value
        count = max(1, len(metrics_list))
        summary[variant_name] = {key: round(value / count, 3) for key, value in agg.items()}
        print(f"  {variant_name:<20} " + " ".join(f"{k}={v}" for k, v in summary[variant_name].items()))
    report["summary"] = summary

    out = ROOT / ".modeling-audit" / "scimap_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
