"""Build Selector (academic relevance) training pairs from the PaSa dataset.

Positives: (query, gold paper title+abstract) -> 1, from AutoScholar train.
Negatives: (query, retrieved-but-not-gold paper title+abstract) -> 0.
The negatives are hard negatives mined by the project's own BGE-M3 vector
retrieval (locally, zero API cost), so the model learns to distinguish
"similar but irrelevant" papers.

Run: python -m backend.scripts.build_selector_training_data [--limit N] [--negatives 3] [--pool 50]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.local_vector_corpus import PaSaVectorCorpusStore  # noqa: E402


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="ascii", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _arxiv_base(value: str) -> str:
    """Strip version suffix: '1909.11740v2' -> '1909.11740'."""
    return str(value or "").casefold().split("v")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all 33.5k")
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--pool", type=int, default=50)
    args = parser.parse_args()
    _safe_stdout()

    train_path = ROOT / "backend" / "paper-test" / "datasets" / "AutoScholarQuery" / "train.jsonl"
    rows = [json.loads(line) for line in train_path.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"queries = {len(rows)}", flush=True)

    store = PaSaVectorCorpusStore()
    store.ensure_collection()
    fb = store._faiss
    # Map arxiv id -> (title, abstract) from the vector index docs.
    doc_by_arxiv: dict[str, tuple[str, str]] = {}
    for row in fb.docs:
        aid = _arxiv_base(row.get("arxiv_id") or "")
        if aid and row.get("title"):
            doc_by_arxiv[aid] = (str(row["title"]), str(row.get("abstract") or ""))
    print(f"index docs = {len(fb.docs)}, mapped arxiv = {len(doc_by_arxiv)}", flush=True)

    import faiss

    encoder = store.encoder
    queries = [str(row.get("question") or "") for row in rows]
    # Batch encode all queries (zero API cost, local GPU).
    started = time.perf_counter()
    dense_all, _sparse, provider = encoder.encode(queries)
    print(f"encoded {len(queries)} queries in {time.perf_counter()-started:.0f}s ({provider})", flush=True)

    # Batch dense search: one matrix search against the full faiss index.
    query_matrix = np.asarray(dense_all, dtype=np.float32)
    faiss.normalize_L2(query_matrix)
    started = time.perf_counter()
    distances, indices = fb.dense.search(query_matrix, args.pool)
    print(f"searched {len(queries)} x {args.pool} in {time.perf_counter()-started:.0f}s", flush=True)

    pairs: list[dict] = []
    skipped_no_gold = 0
    skipped_no_abstract = 0
    for row_index, row in enumerate(rows):
        query = queries[row_index]
        gold_ids = {_arxiv_base(v) for v in (row.get("answer_arxiv_id") or [])}
        gold_ids = {g for g in gold_ids if g in doc_by_arxiv}
        if not gold_ids:
            skipped_no_gold += 1
            continue
        # Positives from the gold papers present in the corpus.
        pos_count = 0
        for gold_id in gold_ids:
            title, abstract = doc_by_arxiv[gold_id]
            if not abstract:
                skipped_no_abstract += 1
                continue
            pairs.append(
                {"query": query, "title": title, "abstract": abstract, "label": 1,
                 "qid": row.get("qid"), "kind": "positive"}
            )
            pos_count += 1
        # Moderately hard negatives: papers ranked 5..pool (avoid the top few
        # near-duplicates that are nearly indistinguishable from the positives,
        # which made the previous model over-conservative and demoted real
        # relevant papers). Randomly sample from that band.
        neg_pool = [
            fb.docs[int(doc_index)]
            for doc_index in indices[row_index][5:]
            if doc_index >= 0
            and _arxiv_base(fb.docs[int(doc_index)].get("arxiv_id") or "") not in gold_ids
            and fb.docs[int(doc_index)].get("abstract")
        ]
        if neg_pool:
            rng = __import__("random").Random(abs(hash(row.get("qid") or row_index)) % (2**32))
            for doc in rng.sample(neg_pool, min(args.negatives, len(neg_pool))):
                pairs.append(
                    {"query": query, "title": str(doc.get("title") or ""),
                     "abstract": str(doc.get("abstract") or ""), "label": 0,
                     "qid": row.get("qid"), "kind": "negative"}
                )
        if row_index and row_index % 5000 == 0:
            print(f"  processed {row_index}/{len(rows)} pairs={len(pairs)}", flush=True)

    out = ROOT / "backend" / "paper-test" / "datasets" / "selector_train.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    labels = {}
    for pair in pairs:
        labels[pair["label"]] = labels.get(pair["label"], 0) + 1
    print(f"pairs = {len(pairs)} | labels = {labels} | skipped_no_gold={skipped_no_gold} "
          f"no_abstract={skipped_no_abstract}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
