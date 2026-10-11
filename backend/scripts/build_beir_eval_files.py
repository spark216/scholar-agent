"""Build project eval/*.jsonl files from BEIR datasets via ir_datasets.

Run from the repository root:
    python -m backend.scripts.build_beir_eval_files

Writes backend/paper-test/datasets/eval/beir_*.jsonl in the format the
ScholarAgent paper test expects: one JSON object per line with
{"qid", "question", "answer": [relevant document titles]}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ir_datasets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = {
    "beir-nfcorpus": ("beir/nfcorpus/test", "beir_nfcorpus.jsonl"),
    "beir-scifact": ("beir/scifact/test", "beir_scifact.jsonl"),
    "beir-arguana": ("beir/arguana", "beir_arguana.jsonl"),
    "beir-trec-covid": ("beir/trec-covid", "beir_trec_covid.jsonl"),
    "beir-scidocs": ("beir/scidocs", "beir_scidocs.jsonl"),
}


def build(dataset_id: str, ir_name: str, out_name: str) -> None:
    print(f"[build] {dataset_id} <- {ir_name}")
    ds = ir_datasets.load(ir_name)
    docs: dict[str, str] = {}
    for doc in ds.docs_iter():
        doc_id = getattr(doc, "doc_id", None)
        title = getattr(doc, "title", None)
        if doc_id and title:
            docs[str(doc_id)] = str(title)

    queries: list[tuple[str, str]] = []
    for q in ds.queries_iter():
        qid = str(getattr(q, "query_id", ""))
        text = str(getattr(q, "query", None) or getattr(q, "text", "") or "")
        queries.append((qid, text))

    qrels: dict[str, list[str]] = {}
    for qrel in ds.qrels_iter():
        qid = str(getattr(qrel, "query_id", ""))
        doc_id = str(getattr(qrel, "doc_id", ""))
        grade = int(getattr(qrel, "relevance", 0) or 0)
        if grade > 0:
            qrels.setdefault(qid, []).append(doc_id)

    lines: list[str] = []
    skipped = 0
    for qid, text in queries:
        gold_ids = qrels.get(qid, [])
        titles = [docs[d] for d in gold_ids if d in docs]
        if not titles:
            skipped += 1
            continue
        lines.append(
            json.dumps(
                {"qid": qid, "question": text, "answer": titles},
                ensure_ascii=False,
            )
        )
    out_path = ROOT / "backend" / "paper-test" / "datasets" / "eval" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {out_path.name}: {len(lines)} queries ({skipped} skipped, no gold titles)")


def main() -> None:
    for dataset_id, (ir_name, out_name) in DATASETS.items():
        try:
            build(dataset_id, ir_name, out_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {dataset_id}: {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
