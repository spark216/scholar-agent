"""Independent recall validation on BEIR datasets (unrelated to the PaSa
corpus), comparing FTS vs BGE-M3 vector retrieval candidate recall.

The BEIR corpora are loaded from the local ir_datasets cache, so no network
is required and nothing here touches the PaSa evaluation data.

Run: python -m backend.scripts.eval_beir_vector_recall --dataset beir/arguana --pool 200
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.embeddings import get_bge_m3_encoder  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+._-]*", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "of", "on", "or", "paper", "papers", "research", "that",
    "the", "their", "this", "to", "using", "what", "which", "with",
}


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _rrf_merge(dense_ranks: list[int], sparse_ranks: list[int], k: int) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for rank, doc in enumerate(dense_ranks):
        scores[doc] += 1.0 / (k + rank + 1)
    for rank, doc in enumerate(sparse_ranks):
        scores[doc] += 1.0 / (k + rank + 1)
    return [doc for doc, _ in sorted(scores.items(), key=lambda item: -item[1])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beir/arguana")
    parser.add_argument("--pool", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=0, help="0 = all")
    args = parser.parse_args()
    _safe_stdout()

    import ir_datasets

    print(f"loading {args.dataset} from local ir_datasets cache ...", flush=True)
    dataset = ir_datasets.load(args.dataset)
    corpus: list[dict] = []
    for doc in dataset.docs_iter():
        text = " ".join(
            str(getattr(doc, "title", "") or "") + " " + str(doc.text or "")
        )
        corpus.append({"doc_id": str(doc.doc_id), "text": text.strip()})
    doc_index = {row["doc_id"]: i for i, row in enumerate(corpus)}
    print(f"corpus docs = {len(corpus)}", flush=True)

    qrels: dict[str, set[int]] = defaultdict(set)
    for qrel in dataset.qrels_iter():
        if int(qrel.relevance) > 0 and str(qrel.doc_id) in doc_index:
            qrels[str(qrel.query_id)].add(doc_index[str(qrel.doc_id)])

    queries: list[tuple[str, str]] = []
    for query in dataset.queries_iter():
        queries.append((str(query.query_id), str(query.text)))
    if args.max_queries:
        queries = queries[: args.max_queries]
    print(f"queries = {len(queries)} (with qrels: {sum(1 for q,_ in queries if q in qrels)})", flush=True)

    # FTS index (porter unicode61 over title+text).
    fts_conn = sqlite3.connect(":memory:")
    fts_conn.execute(
        "CREATE VIRTUAL TABLE documents_fts USING fts5("
        "doc_id UNINDEXED, body, tokenize='porter unicode61')"
    )
    for row in corpus:
        fts_conn.execute(
            "INSERT INTO documents_fts(doc_id,body) VALUES(?,?)",
            (row["doc_id"], row["text"]),
        )
    fts_conn.commit()

    def fts_top(query_text: str, k: int) -> set[int]:
        terms = [
            t.casefold() for t in _TOKEN_RE.findall(query_text)
            if t.casefold() not in _STOPWORDS
        ]
        terms = list(dict.fromkeys(terms))[:24]
        if not terms:
            return set()
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        fetched = fts_conn.execute(
            "SELECT doc_id FROM documents_fts WHERE documents_fts MATCH ? "
            "ORDER BY bm25(documents_fts,0.0,2.0,1.0) LIMIT ?",
            (fts_query, k),
        ).fetchall()
        return {doc_index[row[0]] for row in fetched if row[0] in doc_index}

    # BGE-M3 vector index (dense + sparse).
    encoder = get_bge_m3_encoder()
    import faiss

    dim = encoder.dim
    dense_index = faiss.IndexFlatIP(dim)
    sparse_postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
    texts = [row["text"] for row in corpus]
    started = time.perf_counter()
    batch = 128
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        dense, sparse, provider = encoder.encode(chunk)
        vectors = np.asarray(dense, dtype=np.float32)
        faiss.normalize_L2(vectors)
        dense_index.add(vectors)
        for local, sparse_vector in enumerate(sparse):
            doc = start + local
            for token_id, weight in (sparse_vector or {}).items():
                sparse_postings[int(token_id)].append((doc, float(weight)))
    print(
        f"corpus encoded in {time.perf_counter()-started:.0f}s "
        f"({len(texts)/(time.perf_counter()-started):.0f}/s) provider={provider}",
        flush=True,
    )

    recall_fts: dict[int, list[float]] = {k: [] for k in (10, 50, 100, 200)}
    recall_dense: dict[int, list[float]] = {k: [] for k in (10, 50, 100, 200)}
    recall_sparse: dict[int, list[float]] = {k: [] for k in (10, 50, 100, 200)}
    recall_hybrid: dict[int, list[float]] = {k: [] for k in (10, 50, 100, 200)}
    recall_fusion: dict[int, list[float]] = {k: [] for k in (10, 50, 100, 200)}

    for qid, query_text in queries:
        relevant = qrels.get(qid, set())
        if not relevant:
            continue
        dense_q, sparse_q, _ = encoder.encode_query(query_text)
        d_vector = np.asarray([dense_q], dtype=np.float32)
        faiss.normalize_L2(d_vector)
        distances, indices = dense_index.search(d_vector, args.pool)
        dense_ranks = [int(i) for i in indices[0] if i >= 0]
        sparse_scores: dict[int, float] = defaultdict(float)
        for token_id, query_weight in (sparse_q or {}).items():
            for doc, doc_weight in sparse_postings.get(int(token_id), []):
                sparse_scores[doc] += query_weight * doc_weight
        sparse_ranks = [d for d, _ in sorted(sparse_scores.items(), key=lambda item: -item[1])[: args.pool]]
        hybrid_ranks = _rrf_merge(dense_ranks, sparse_ranks, k=60)[: args.pool]
        fts_ranks = sorted(fts_top(query_text, args.pool))

        for k in recall_fts:
            recall_fts[k].append(len(relevant & set(fts_ranks[:k])) / len(relevant))
            recall_dense[k].append(len(relevant & set(dense_ranks[:k])) / len(relevant))
            recall_sparse[k].append(len(relevant & set(sparse_ranks[:k])) / len(relevant))
            recall_hybrid[k].append(len(relevant & set(hybrid_ranks[:k])) / len(relevant))
            recall_fusion[k].append(
                len(relevant & (set(hybrid_ranks[:k]) | set(fts_ranks[:k]))) / len(relevant)
            )

    mean = lambda values: sum(values) / max(1, len(values))
    print(f"\n=== {args.dataset} Candidate Recall (gold in top-K), {len(recall_fts[200])} queries ===")
    print(f"{'K':>6} | {'FTS':>8} | {'dense':>8} | {'sparse':>8} | {'hybrid':>8} | {'FTS+vec':>8}")
    for k in (10, 50, 100, 200):
        print(
            f"{k:>6} | {mean(recall_fts[k])*100:>7.2f}% | {mean(recall_dense[k])*100:>7.2f}% | "
            f"{mean(recall_sparse[k])*100:>7.2f}% | {mean(recall_hybrid[k])*100:>7.2f}% | "
            f"{mean(recall_fusion[k])*100:>7.2f}%"
        )

    out = ROOT / ".modeling-audit" / f"{args.dataset.replace('/','_')}_vector_recall.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "pool": args.pool,
                "queries": len(recall_fts[200]),
                "recall": {
                    "fts": {str(k): round(mean(v), 4) for k, v in recall_fts.items()},
                    "dense": {str(k): round(mean(v), 4) for k, v in recall_dense.items()},
                    "sparse": {str(k): round(mean(v), 4) for k, v in recall_sparse.items()},
                    "hybrid": {str(k): round(mean(v), 4) for k, v in recall_hybrid.items()},
                    "fusion": {str(k): round(mean(v), 4) for k, v in recall_fusion.items()},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
