"""Build the optional full PaSa dense+sparse Milvus index in bounded batches."""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

from backend.app.services.local_pasa_corpus import (
    resolve_pasa_document_corpus,
    resolve_pasa_title_corpus,
)
from backend.app.services.local_vector_corpus import PaSaVectorCorpusStore, _arxiv_year


def _title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 569k-paper PaSa BGE-M3 dense+sparse Milvus index."
    )
    parser.add_argument("--titles", type=Path, help="Path to id2paper.json")
    parser.add_argument("--documents", type=Path, help="Path to cs_paper_2nd.zip")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 indexes all remaining entries")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20000,
        help="Persist the faiss index to disk every N indexed papers so an "
        "interrupted build only loses the current interval (default 20000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    title_source = args.titles or resolve_pasa_title_corpus()
    if title_source is None or not title_source.is_file():
        raise SystemExit("id2paper.json not found; set PAPER_TEST_DATASET_ROOT or --titles")
    documents = args.documents or resolve_pasa_document_corpus(title_source)
    if documents is None or not documents.is_file():
        raise SystemExit("cs_paper_2nd.zip not found; pass --documents")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    with title_source.open("r", encoding="utf-8") as handle:
        id_to_title = json.load(handle)
    title_to_ids: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for arxiv_id, title in id_to_title.items():
        clean_title = " ".join(str(title or "").split()).strip()
        if clean_title:
            title_to_ids[_title_key(clean_title)].append((str(arxiv_id), clean_title))

    store = PaSaVectorCorpusStore(ensure_collection=False)
    store.ensure_collection(rebuild=args.rebuild)
    started = time.perf_counter()
    processed = 0
    indexed = store._faiss_doc_count() if hasattr(store, "_faiss_doc_count") else 0
    last_checkpoint = indexed
    seen_ids: set[str] = set()
    batch: list[dict] = []

    # Persistent entry-position resume point so chunked/foreground runs do not
    # rescan the already-processed prefix of the zip on every restart.
    index_root = Path(store._faiss_index_dir()) if hasattr(store, "_faiss_index_dir") else None
    state_path = index_root / "build_state.json" if index_root else None
    resume_entries = args.start_offset
    if state_path and state_path.is_file():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            resume_entries = max(resume_entries, int(previous.get("processed_entries", 0)))
        except (OSError, ValueError, TypeError):
            pass

    def flush() -> None:
        nonlocal indexed, last_checkpoint
        if not batch:
            return
        result = store.upsert_documents(batch)
        indexed += int(result["upserted"])
        batch.clear()
        elapsed = max(time.perf_counter() - started, 0.001)
        print(f"processed={processed} indexed={indexed} rate={indexed / elapsed:.1f} papers/s", flush=True)
        if args.checkpoint_interval and indexed - last_checkpoint >= args.checkpoint_interval:
            store.flush()
            last_checkpoint = indexed
            if state_path is not None:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    json.dumps({"processed_entries": resume_entries + processed}),
                    encoding="utf-8",
                )
            print(
                f"[checkpoint] persisted {indexed} docs (entry {processed}) to "
                f"{index_root or 'disk'}",
                flush=True,
            )

    with zipfile.ZipFile(documents) as archive:
        for entry_index, info in enumerate(archive.infolist()):
            if entry_index < resume_entries:
                continue
            if args.limit and processed >= args.limit:
                break
            processed += 1
            try:
                record = json.loads(archive.read(info))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            matches = title_to_ids.get(_title_key(record.get("title") or "")) or []
            abstract = " ".join(str(record.get("abstract") or "").split())
            for arxiv_id, title in matches:
                seen_ids.add(arxiv_id)
                batch.append(
                    {
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "abstract": abstract,
                        "year": _arxiv_year(arxiv_id),
                    }
                )
                if len(batch) >= args.batch_size:
                    flush()
    # A small title-only tail has no document JSON. Include it during a full
    # build so the vector Provider covers the same 569k IDs as the FTS source.
    if args.start_offset == 0 and args.limit == 0:
        for arxiv_id, title in id_to_title.items():
            if str(arxiv_id) in seen_ids:
                continue
            batch.append(
                {
                    "arxiv_id": str(arxiv_id),
                    "title": " ".join(str(title or "").split()),
                    "abstract": "",
                    "year": _arxiv_year(str(arxiv_id)),
                }
            )
            if len(batch) >= args.batch_size:
                flush()
    flush()
    try:
        store.client.flush(store.collection)
        store.client.load_collection(store.collection)
    except Exception:  # Milvus client versions expose slightly different helpers.
        pass
    if state_path is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"processed_entries": resume_entries + processed}),
            encoding="utf-8",
        )
    print(f"done indexed={indexed} elapsed={time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
