"""Compare candidate recall of BGE-M3 vector retrieval vs the FTS baseline on
the fixed NFCorpus corpus.

Computes Candidate Recall@K = fraction of relevant (gold) documents that enter
the top-K candidate pool, using the same fixed NFCorpus corpus/queries/qrels as
``analyze_nfcorpus_recall.py``. Reports dense-only, sparse-only, hybrid (RRF)
and FTS+vector multi-fusion retrieval.

Run: python -m backend.scripts.eval_nfcorpus_vector_recall --split test --pool 200
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
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
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--pool", type=int, default=200)
    args = parser.parse_args()
    _safe_stdout()

    data_dir = ROOT / "backend" / "paper-test" / "datasets" / "nfcorpus-fixed"
    corpus: list[dict] = []
    with (data_dir / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus.append(
                {
                    "doc_id": row["_id"],
                    "title": str(row.get("title") or ""),
                    "body": str(row.get("text") or ""),
                    "text": f"{row.get('title') or ''}\n{row.get('text') or ''}",
                }
            )
    doc_index = {row["doc_id"]: i for i, row in enumerate(corpus)}

    # FTS5 index replicating the lexical baseline (porter unicode61).
    fts_path = data_dir / f"_fts_{args.split}.sqlite3"
    fts_conn = sqlite3.connect(fts_path)
    fts_conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5("
        "doc_id UNINDEXED, title, body, tokenize='porter unicode61')"
    )
    fts_conn.execute("DELETE FROM documents_fts")
    for row in corpus:
        fts_conn.execute(
            "INSERT INTO documents_fts(doc_id,title,body) VALUES(?,?,?)",
            (row["doc_id"], row["title"], str(row.get("body") or "")),
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
    queries: list[tuple[str, str]] = []
    with (data_dir / f"queries-{args.split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries.append((row["query_id"], row["query"]))
    qrels: dict[str, set[str]] = defaultdict(set)
    with (data_dir / f"qrels-{args.split}.tsv").open(encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3 or parts[2].casefold() in {"relevance", "score"}:
                continue
            if int(parts[2]) > 0:
                qrels[parts[0]].add(parts[1])

    print(f"corpus_docs={len(corpus)} queries={len(queries)}", flush=True)

    encoder = get_bge_m3_encoder()
    import faiss

    dim = encoder.dim
    dense_index = faiss.IndexFlatIP(dim)
    sparse_postings: dict[int, list[tuple[int, float]]] = defaultdict(list)

    # Encode the corpus in batches and build both indexes.
    texts = [row["text"] for row in corpus]
    batch = 64
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
        if start % 512 == 0:
            print(f"encoded {start + len(chunk)}/{len(texts)} provider={provider}", flush=True)
    print(f"corpus encoded (provider={provider}) dense_ntotal={dense_index.ntotal}", flush=True)

    recall_dense: dict[int, list[int]] = {k: [] for k in (10, 50, 100, 200)}
    recall_sparse: dict[int, list[int]] = {k: [] for k in (10, 50, 100, 200)}
    recall_hybrid: dict[int, list[int]] = {k: [] for k in (10, 50, 100, 200)}
    recall_fusion: dict[int, list[int]] = {k: [] for k in (10, 50, 100, 200)}

    for qid, query_text in queries:
        relevant = qrels.get(qid, set())
        if not relevant:
            continue
        relevant_idx = {doc_index[d] for d in relevant if d in doc_index}
        if not relevant_idx:
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

        for k in recall_dense:
            recall_dense[k].append(len(relevant_idx & set(dense_ranks[:k])) / len(relevant_idx))
            recall_sparse[k].append(len(relevant_idx & set(sparse_ranks[:k])) / len(relevant_idx))
            recall_hybrid[k].append(len(relevant_idx & set(hybrid_ranks[:k])) / len(relevant_idx))
            # Multi-fusion: FTS pool (lexical) unioned with the vector hybrid pool.
            fused = set(hybrid_ranks[:k]) | set(fts_ranks[:k])
            recall_fusion[k].append(len(relevant_idx & fused) / len(relevant_idx))

    print(f"\n=== NFCorpus {args.split} Candidate Recall (gold in top-K) ===")
    print(f"{'K':>6} | {'dense':>8} | {'sparse':>8} | {'hybrid':>8} | {'FTS+vec':>8}")
    for k in (10, 50, 100, 200):
        mean = lambda values: sum(values) / max(1, len(values))
        print(
            f"{k:>6} | {mean(recall_dense[k]) * 100:>7.2f}% | "
            f"{mean(recall_sparse[k]) * 100:>7.2f}% | {mean(recall_hybrid[k]) * 100:>7.2f}% | "
            f"{mean(recall_fusion[k]) * 100:>7.2f}%"
        )
    print(f"\n(FTS baseline from earlier audit: {args.split} candidate recall@200 = 27.93%)")
    print(f"query_count = {len(recall_fusion[200])}")

    out = ROOT / ".modeling-audit" / f"nfcorpus_{args.split}_vector_recall.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "split": args.split,
                "pool": args.pool,
                "queries": len(recall_hybrid[200]),
                "provider": provider,
                "recall": {
                    "dense": {str(k): round(sum(v) / len(v), 4) for k, v in recall_dense.items()},
                    "sparse": {str(k): round(sum(v) / len(v), 4) for k, v in recall_sparse.items()},
                    "hybrid": {str(k): round(sum(v) / len(v), 4) for k, v in recall_hybrid.items()},
                    "fusion": {str(k): round(sum(v) / len(v), 4) for k, v in recall_fusion.items()},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
