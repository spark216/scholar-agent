"""Fast lexical retrieval over the PaSa title-and-abstract corpus.

The PaSa release contains ``paper_database/id2paper.json``: a mapping from
arXiv IDs to paper titles.  It is corpus metadata, not per-query qrels, so it
can safely be used as a retrieval source without leaking test answers.  The
JSON file and the optional paper JSON archive are converted once to a local
SQLite FTS5 index and reused by later searches.
"""

from __future__ import annotations

from contextlib import closing

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from threading import RLock
from typing import Any

from backend.app.config import ROOT_DIR, get_settings
from backend.app.models.schemas import QueryUnderstanding


_INDEX_LOCK = RLock()
PASA_INDEX_VERSION = "pasa-fts-title-abstract-porter-v3"
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+._-]*")
_STOPWORDS = {
    "a", "about", "an", "and", "are", "article", "articles", "be", "can",
    "find", "for", "from", "give", "have", "how", "in", "is", "me", "of",
    "on", "paper", "papers", "provide", "research", "show", "some", "studies",
    "study", "that", "the", "their", "these", "to", "using", "what", "which",
    "with", "work", "works", "you",
}


def _candidate_source_paths() -> list[Path]:
    settings = get_settings()
    roots: list[Path] = []
    if settings.academic_local_corpus_path:
        explicit = Path(settings.academic_local_corpus_path).expanduser()
        if explicit.suffix.casefold() == ".json":
            return [explicit]
        roots.append(explicit)
    if settings.paper_test_dataset_root:
        roots.append(Path(settings.paper_test_dataset_root).expanduser())
    roots.extend(
        [
            ROOT_DIR / "backend" / "paper-test" / "datasets",
            Path.home() / "Downloads" / "paper-dataset",
        ]
    )
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "paper_database" / "id2paper.json",
                root / "pasa-dataset" / "paper_database" / "id2paper.json",
            ]
        )
    return list(dict.fromkeys(candidates))


def resolve_pasa_title_corpus() -> Path | None:
    if not get_settings().academic_local_corpus_enabled:
        return None
    return next((path for path in _candidate_source_paths() if path.is_file()), None)


def _candidate_document_paths(source: Path) -> list[Path]:
    configured = get_settings().academic_local_documents_path
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(
        [
            source.parent / "cs_paper_2nd.zip",
            source.parent.parent / "paper_database" / "cs_paper_2nd.zip",
        ]
    )
    return list(dict.fromkeys(candidates))


def resolve_pasa_document_corpus(source: Path) -> Path | None:
    return next((path for path in _candidate_document_paths(source) if path.is_file()), None)


def _index_path() -> Path:
    configured = get_settings().academic_local_index_path
    if configured:
        return Path(configured).expanduser()
    return ROOT_DIR / "backend" / "paper-test" / "index" / "pasa_titles.sqlite3"


def _path_signature(path: Path | None) -> str:
    if path is None:
        return "none"
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def _source_signature(source: Path, documents: Path | None) -> str:
    return f"{PASA_INDEX_VERSION}|titles={_path_signature(source)}|documents={_path_signature(documents)}"


def _index_is_current(index_path: Path, signature: str) -> bool:
    if not index_path.is_file():
        return False
    try:
        with closing(sqlite3.connect(index_path)) as connection, connection:
            row = connection.execute(
                "SELECT value FROM index_meta WHERE key = 'source_signature'"
            ).fetchone()
            return bool(row and row[0] == signature)
    except sqlite3.Error:
        return False


def _title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _build_index(
    source: Path,
    documents: Path | None,
    index_path: Path,
    signature: str,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="pasa_titles_",
        suffix=".sqlite3",
        dir=index_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("PaSa id2paper.json must contain an object mapping IDs to titles")

        title_to_ids: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for arxiv_id, title in payload.items():
            clean_id = str(arxiv_id or "").strip()
            clean_title = " ".join(str(title or "").split()).strip()
            key = _title_key(clean_title)
            if clean_id and clean_title and key:
                title_to_ids[key].append((clean_id, clean_title))

        inserted_ids: set[str] = set()
        document_count = 0
        abstract_max_chars = max(0, get_settings().academic_local_abstract_max_chars)
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE VIRTUAL TABLE papers USING fts5("
                "arxiv_id UNINDEXED, title, abstract, "
                "tokenize='porter unicode61 remove_diacritics 2')"
            )
            batch: list[tuple[str, str, str]] = []

            def flush() -> None:
                if not batch:
                    return
                connection.executemany(
                    "INSERT INTO papers(arxiv_id, title, abstract) VALUES (?, ?, ?)",
                    batch,
                )
                batch.clear()

            if documents is not None:
                with zipfile.ZipFile(documents) as archive:
                    for info in archive.infolist():
                        try:
                            with archive.open(info) as handle:
                                record = json.loads(handle.read())
                        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if not isinstance(record, dict):
                            continue
                        record_title = " ".join(str(record.get("title") or "").split()).strip()
                        matches = title_to_ids.get(_title_key(record_title)) or []
                        if not matches:
                            continue
                        abstract = " ".join(str(record.get("abstract") or "").split())
                        if abstract_max_chars:
                            abstract = abstract[:abstract_max_chars]
                        for arxiv_id, canonical_title in matches:
                            if arxiv_id in inserted_ids:
                                continue
                            batch.append((arxiv_id, canonical_title, abstract))
                            inserted_ids.add(arxiv_id)
                            if abstract:
                                document_count += 1
                        if len(batch) >= 5_000:
                            flush()

            for matches in title_to_ids.values():
                for arxiv_id, title in matches:
                    if arxiv_id in inserted_ids:
                        continue
                    batch.append((arxiv_id, title, ""))
                    inserted_ids.add(arxiv_id)
                    if len(batch) >= 10_000:
                        flush()
            flush()
            connection.execute(
                "INSERT INTO index_meta(key, value) VALUES ('source_signature', ?)",
                (signature,),
            )
            connection.execute(
                "INSERT INTO index_meta(key, value) VALUES ('document_count', ?)",
                (str(document_count),),
            )
            connection.execute(
                "INSERT INTO index_meta(key, value) VALUES ('paper_count', ?)",
                (str(len(inserted_ids)),),
            )
            connection.execute("INSERT INTO papers(papers) VALUES ('optimize')")
        temporary.replace(index_path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_pasa_title_index(source: Path) -> tuple[Path, bool, float]:
    index_path = _index_path()
    documents = resolve_pasa_document_corpus(source)
    signature = _source_signature(source, documents)
    started = time.perf_counter()
    built = False
    with _INDEX_LOCK:
        if not _index_is_current(index_path, signature):
            _build_index(source, documents, index_path, signature)
            built = True
    return index_path, built, (time.perf_counter() - started) * 1000


def _meaningful_terms(value: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _WORD_RE.findall(value or ""):
        clean = token.strip("._-")
        key = clean.casefold()
        if len(clean) <= 1 or key in _STOPWORDS or key in seen:
            continue
        terms.append(clean)
        seen.add(key)
    return terms


def _technical_priority(term: str) -> tuple[int, int]:
    return (
        int(
            any(char.isdigit() for char in term)
            or any(char in term for char in "+._-")
            or (term.isupper() and len(term) >= 2)
        ),
        len(term),
    )


def _fts_phrase(value: str) -> str:
    clean = " ".join(_meaningful_terms(value))
    return f'"{clean.replace(chr(34), " ")}"' if clean else ""


def _fts_queries(query: str, understanding: QueryUnderstanding) -> list[str]:
    raw = understanding.raw_json or {}
    structured: list[str] = []
    for relation in raw.get("relations") or []:
        if isinstance(relation, dict):
            structured.append(
                " ".join(
                    str(relation.get(field) or "").strip()
                    for field in ("source", "relation", "target")
                ).strip()
            )
    for constraint in raw.get("constraints") or []:
        if isinstance(constraint, dict):
            structured.append(
                " ".join(
                    str(constraint.get(field) or "").strip()
                    for field in ("facet", "operator", "value")
                ).strip()
            )
    facets = [
        *understanding.datasets,
        *understanding.entities,
        *understanding.methods,
        *understanding.topics,
        *understanding.keywords,
        *(value for value in structured if value),
    ]
    result: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        clean = " ".join(value.split()).strip()
        if clean and clean.casefold() not in seen:
            result.append(clean)
            seen.add(clean.casefold())

    # Exact technical/facet phrases are high precision and especially useful
    # for datasets and acronyms such as HotPotQA, QAT, NER, RE and EE.
    precise = [_fts_phrase(value) for value in facets]
    precise = [value for value in precise if value]
    if precise:
        add(" OR ".join(precise[:6]))

    base_queries = list(
        dict.fromkeys(
            value.strip()
            for value in [query, *understanding.sub_queries]
            if value and value.strip()
        )
    )[:8]
    searchable = " ".join([*base_queries, *facets]).casefold()
    expansion_terms: list[str] = []
    if any(cue in searchable for cue in ("negative impact", "negatively", "harm", "degrad")):
        expansion_terms.extend(
            _meaningful_terms("performance degradation adverse effect reduced performance")
        )
    if any(cue in searchable for cue in ("cannot surpass", "underperform", "fall short", "compared to")):
        expansion_terms.extend(
            _meaningful_terms(
                "performance limitation underperformance comparative evaluation"
            )
        )
    if any(cue in searchable for cue in ("experiment", "evaluate", "benchmark", "dataset")):
        expansion_terms.extend(
            _meaningful_terms("benchmark evaluation experimental results")
        )
    expansion_terms = list(dict.fromkeys(expansion_terms))
    for base_query in base_queries:
        branch_terms = list(
            dict.fromkeys(
                [
                    *_meaningful_terms(" ".join([base_query, *facets])),
                    *expansion_terms,
                ]
            )
        )
        ranked = sorted(branch_terms, key=_technical_priority, reverse=True)
        if ranked:
            add(" OR ".join(_fts_phrase(term) for term in ranked[:8]))
            # FTS5 whitespace is AND. Keep one compact high-precision branch
            # per base query; otherwise a conjunction made only from the LLM
            # rewrite can silently discard decisive terms in the raw query.
            add(" ".join(_fts_phrase(term) for term in branch_terms[:5]))
    # At most seven small FTS branches are evaluated. This lets independently
    # decomposed subqueries participate in recall while remaining far cheaper
    # than a full-corpus dense/sparse model index.
    return result[:7]


def _arxiv_year(arxiv_id: str) -> int | None:
    match = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
    if not match:
        return None
    prefix = int(match.group(1))
    return 2000 + prefix if prefix <= 50 else 1900 + prefix


def _stable_arxiv_id(arxiv_id: str) -> str:
    return hashlib.sha256(f"arxiv:{arxiv_id.casefold()}".encode()).hexdigest()[:16]


def search_local_pasa_corpus(
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = resolve_pasa_title_corpus()
    if source is None:
        raise RuntimeError("PaSa 本地论文语料不可用")
    index_path, built, index_ms = ensure_pasa_title_index(source)
    queries = _fts_queries(query, understanding)
    if not queries:
        return [], {
            "provider": "pasa-local-corpus",
            "queries": [],
            "raw_count": 0,
            "returned_count": 0,
            "api_calls": 0,
            "index_built": built,
            "index_ms": index_ms,
        }

    started = time.perf_counter()
    fused: dict[str, dict[str, Any]] = {}
    branch_keys: list[list[str]] = []
    raw_count = 0
    page_limit = min(max(limit, 1), 100)
    with closing(sqlite3.connect(index_path)) as connection, connection:
        meta_rows = dict(connection.execute("SELECT key, value FROM index_meta").fetchall())
        for query_index, fts_query in enumerate(queries):
            rows = connection.execute(
                "SELECT arxiv_id, title, abstract FROM papers WHERE papers MATCH ? "
                "ORDER BY bm25(papers, 0.0, 8.0, 1.5) LIMIT ? OFFSET ?",
                (fts_query, page_limit, max(0, offset)),
            ).fetchall()
            raw_count += len(rows)
            current_branch: list[str] = []
            for rank, (arxiv_id, title, abstract) in enumerate(rows, start=1):
                key = str(arxiv_id).casefold()
                current_branch.append(key)
                reciprocal = 1.0 / (60 + rank)
                if key in fused:
                    fused[key]["score"] += reciprocal
                    continue
                year = _arxiv_year(str(arxiv_id))
                fused[key] = {
                    "id": _stable_arxiv_id(str(arxiv_id)),
                    "title": str(title),
                    "abstract": str(abstract or ""),
                    "authors": [],
                    "year": year,
                    "published_at": str(year) if year else None,
                    "venue": "arXiv",
                    "citation_count": 0,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "tags": [],
                    "source": "pasa-local-corpus",
                    "sources": ["pasa-local-corpus"],
                    "external_ids": {"arxiv": str(arxiv_id)},
                    # RRF should reward agreement between branches, not the
                    # order in which query rewrites were generated.  The old
                    # 0.001 branch-position bonus was large enough to suppress
                    # good raw-query matches returned by a later branch.
                    "score": reciprocal + (0.00001 * (len(queries) - query_index)),
                }
            branch_keys.append(current_branch)
    ranked = sorted(
        fused.values(),
        key=lambda item: (-float(item["score"]), str(item["id"])),
    )
    # Protect the top slice of every branch before filling by RRF. This keeps
    # an LLM rewrite from crowding the user's original wording out of the
    # bounded local result page. The later shared lexical/BGE budget still
    # decides which of these candidates deserve expensive model scoring.
    protected_ids: set[str] = set()
    branch_quota = max(1, math.ceil(page_limit / max(1, len(branch_keys))))
    for keys in branch_keys:
        protected_ids.update(keys[:branch_quota])
    protected = [
        item
        for item in ranked
        if str((item.get("external_ids") or {}).get("arxiv") or "").casefold()
        in protected_ids
    ]
    protected_objects = {id(item) for item in protected}
    papers = protected[:page_limit]
    if len(papers) < page_limit:
        papers.extend(
            item
            for item in ranked
            if id(item) not in protected_objects
        )
        papers = papers[:page_limit]
    return papers, {
        "provider": "pasa-local-corpus",
        "queries": queries,
        "raw_count": raw_count,
        "returned_count": len(papers),
        "api_calls": 0,
        "index_built": built,
        "index_ms": index_ms,
        "search_ms": (time.perf_counter() - started) * 1000,
        "source": str(source),
        "document_source": str(resolve_pasa_document_corpus(source) or ""),
        "paper_count": int(meta_rows.get("paper_count") or 0),
        "document_count": int(meta_rows.get("document_count") or 0),
    }
