"""Build NFCorpus fixed-corpus evaluation assets from ir_datasets.

Writes, under an output directory:
  corpus.jsonl     {"_id", "title", "text"} for every NFCorpus document
  queries-dev.jsonl / queries-test.jsonl   {"query_id", "query"}
  qrels-dev.tsv / qrels-test.tsv           qid<TAB>docid<TAB>relevance

These are the exact inputs expected by evaluate_fixed_corpus.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ir_datasets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "backend" / "paper-test" / "datasets" / "nfcorpus-fixed"


def main() -> None:
    ds = ir_datasets.load("beir/nfcorpus")
    OUT.mkdir(parents=True, exist_ok=True)

    # Corpus: all documents
    with (OUT / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for doc in ds.docs_iter():
            f.write(
                json.dumps(
                    {
                        "_id": doc.doc_id,
                        "title": doc.title,
                        "text": doc.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print("corpus.jsonl written")

    for split in ("dev", "test"):
        split_ds = ir_datasets.load(f"beir/nfcorpus/{split}")
        with (OUT / f"queries-{split}.jsonl").open("w", encoding="utf-8") as f:
            for q in split_ds.queries_iter():
                f.write(
                    json.dumps(
                        {"query_id": q.query_id, "query": q.text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        with (OUT / f"qrels-{split}.tsv").open("w", encoding="utf-8") as f:
            f.write("qid\tdocid\trelevance\n")
            for qrel in split_ds.qrels_iter():
                f.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
        print(f"{split}: queries + qrels written")


if __name__ == "__main__":
    main()
