"""Feature-3 (EvidRank) component ablation on the fixed NFCorpus corpus.

Each variant runs the same FTS candidate pool through EvidRankPipeline.rank()
with one component disabled, isolating its contribution to retrieval metrics:

  full            : complete EvidRank pipeline
  no_reranker     : skip Cross-Encoder reranking (RRF order only)
  no_facet        : skip structured dataset/method/relation facet fusion
  no_preference   : skip authority/recency preference weighting
  no_evidence     : skip evidence judging (score-only ranking)
  no_level        : skip level decision (all HIGH, score-driven order)

All variants are offline: deterministic evidence judge, zero API/token.
Run: python -m backend.scripts.ablate_evidrank [--limit N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.models.schemas import QueryUnderstanding  # noqa: E402
from backend.app.services.embeddings import get_bge_m3_encoder  # noqa: E402
from backend.app.services.ranking.evidence_judge import DeterministicEvidenceJudge  # noqa: E402
from backend.app.services.ranking.pipeline import EvidRankPipeline  # noqa: E402
from backend.scripts.evaluate_fixed_corpus import (  # noqa: E402
    FixedCorpusFTS,
    _load_jsonl,
    _load_qrels,
)

VARIANTS: list[tuple[str, dict[str, bool] | None]] = [
    ("full", None),
    ("no_reranker", {"reranker": False}),
    ("no_facet", {"facet": False}),
    ("no_preference", {"preference": False}),
    ("no_selector", {"selector": False}),
    ("no_evidence", {"evidence": False}),
    ("no_level", {"level": False}),
]


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _dcg(ranked_ids: list[str], qrels: dict[str, int]) -> float:
    value = 0.0
    for rank, doc_id in enumerate(ranked_ids[:20], start=1):
        grade = qrels.get(doc_id, 0)
        if grade:
            value += (2**grade - 1) / math.log2(rank + 1)
    return value


def _metrics(ranked_ids: list[str], relevant: set[str], qrels: dict[str, int], k: int) -> dict:
    top = ranked_ids[:k]
    hits = [doc for doc in top if doc in relevant]
    recall = len(hits) / max(1, len(relevant))
    precision = len(hits) / k
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    dcg = _dcg(top, qrels)
    ideal = sorted(
        [(doc, grade) for doc, grade in qrels.items() if grade > 0],
        key=lambda item: -item[1],
    )[:
        k
    ]
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, (_, grade) in enumerate(ideal, start=1)
    )
    mrr = 0.0
    for rank, doc in enumerate(top, start=1):
        if doc in relevant:
            mrr = 1.0 / rank
            break
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "ndcg": dcg / idcg if idcg else 0.0,
        "mrr": mrr,
        "hits": len(hits),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all queries")
    args = parser.parse_args()
    _safe_stdout()
    data_dir = ROOT / "backend" / "paper-test" / "datasets" / "nfcorpus-fixed"
    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries-test.jsonl"
    qrels_path = data_dir / "qrels-test.tsv"

    corpus = _load_jsonl(corpus_path, id_field="_id")
    queries = _load_jsonl(queries_path, id_field="query_id")
    qrels = _load_qrels(qrels_path)

    encoder = get_bge_m3_encoder()
    from backend.app.services.reranker import get_bge_reranker

    pipeline = EvidRankPipeline(
        encoder=encoder,
        reranker=get_bge_reranker(),
        evidence_judge=DeterministicEvidenceJudge(),
    )

    fts = FixedCorpusFTS(corpus_path, ROOT / "backend" / "paper-test" / "index" / "_ablate_fts.sqlite3")

    results: dict[str, dict] = {}
    per_variant_queries: dict[str, list[dict]] = defaultdict(list)

    query_items = list(queries.items())
    if args.limit:
        query_items = query_items[: args.limit]

    for variant, ablation in VARIANTS:
        acc: dict[int, dict] = {k: defaultdict(float) for k in (5, 10)}
        counts = 0
        for qid, qrow in query_items:
            relevant = {doc for doc, rel in qrels.get(qid, {}).items() if rel > 0}
            if not relevant:
                continue
            query_text = str(qrow.get("query") or qrow.get("text") or "")
            fts_rows = fts.search(query_text, 200)
            if not fts_rows:
                continue
            candidates = [
                {
                    "id": row["doc_id"],
                    "title": row.get("title") or "",
                    "abstract": row.get("body") or row.get("text") or "",
                    "year": None,
                    "authors": [],
                    "external_ids": {},
                    "provider_ranks": {"fixed-corpus-fts": rank},
                    "provider_scores": {"fixed-corpus-fts": -float(row.get("score") or 0)},
                    "sources": ["fixed-corpus-fts"],
                    "source": "fixed-corpus-fts",
                }
                for rank, row in enumerate(fts_rows, start=1)
            ]
            understanding = QueryUnderstanding(
                intent=f"Retrieve papers relevant to {query_text}",
                topics=[query_text],
                keywords=[],
                sub_queries=[query_text],
                rewritten_queries=[query_text],
                raw_json={"parser_version": "ablate-evidrank-v1"},
            )
            try:
                bundle = pipeline.rank(
                    session_id=f"ablate-{qid}",
                    query=query_text,
                    understanding=understanding,
                    candidate_rows=candidates,
                    top_k=10,
                    fast_evaluation=True,
                    ablation=ablation,
                )
                ordered = [item.paper_id for item in bundle.items]
            except Exception as exc:  # noqa: BLE001
                print(f"  [{variant}] {qid} ERROR: {type(exc).__name__}: {exc}", flush=True)
                continue
            for k in (5, 10):
                m = _metrics(ordered, relevant, qrels[qid], k)
                for key, value in m.items():
                    acc[k][key] += value
            counts += 1
            per_variant_queries[variant].append(
                {
                    "qid": qid,
                    "hits5": sum(1 for d in ordered[:5] if d in relevant),
                    "hits10": sum(1 for d in ordered[:10] if d in relevant),
                    "mrr5": _metrics(ordered, relevant, qrels[qid], 5)["mrr"],
                }
            )
        results[variant] = {
            "query_count": counts,
            "metrics": {
                str(k): {key: value / max(1, counts) for key, value in acc[k].items()}
                for k in (5, 10)
            },
        }
        m5 = results[variant]["metrics"]["5"]
        print(
            f"{variant:<14} queries={counts} R@5={m5['recall']*100:.2f}% F1@5={m5['f1']*100:.2f}% "
            f"nDCG@5={m5['ndcg']*100:.2f}% MRR@5={m5['mrr']*100:.2f}% hits@5={m5['hits']:.0f}",
            flush=True,
        )

    out = ROOT / ".modeling-audit" / "evidrank_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "description": "Feature-3 EvidRank component ablation on NFCorpus test (same FTS pool)",
                "variants": results,
                "per_query": dict(per_variant_queries),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
