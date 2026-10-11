"""Optional full-corpus PaSa retrieval with a persistent BGE-M3 index.

Backend is pluggable via ``academic_vector_backend``:

- ``faiss`` (default): a fully offline dense+sparse hybrid index built on
  faiss-cpu and numpy. No Docker/Milvus required. Persists under
  ``academic_local_vector_index_dir/{collection}/``.
- ``milvus``: the original Milvus-backed implementation (requires the Milvus
  service running).

The public surface (``PaSaVectorCorpusStore``, ``search_local_vector_corpus``,
``get_pasa_vector_corpus_store``, ``_time_filter``, ``_arxiv_year``) is kept
identical so callers such as ``build_pasa_vector_index.py``, the provider
registry in ``academic_search.py`` and the unit tests keep working.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from array import array
from collections import defaultdict
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from backend.app.config import ROOT_DIR, get_settings
from backend.app.models.schemas import QueryUnderstanding
from backend.app.services.embeddings import get_bge_m3_encoder

try:  # faiss is optional only for the legacy milvus backend path.
    import faiss
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore[assignment]

try:
    from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker
except Exception:  # noqa: BLE001
    AnnSearchRequest = DataType = MilvusClient = RRFRanker = None  # type: ignore[assignment]

_TITLE_BYTES = 1024
_ABSTRACT_BYTES = 4096
# Cap the number of sparse tokens kept per document. BGE-M3 lexical weights
# are long-tailed; keeping the top-64 weighted tokens preserves the retrieval
# signal while keeping the inverted index resident in RAM on 16GB machines.
_MAX_SPARSE_TOKENS_PER_DOC = 64
_STORE_LOCK = RLock()


def _stable_arxiv_id(arxiv_id: str) -> str:
    return hashlib.sha256(f"arxiv:{arxiv_id.casefold()}".encode()).hexdigest()[:16]


def _arxiv_year(arxiv_id: str) -> int | None:
    match = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
    if not match:
        return None
    prefix = int(match.group(1))
    return 2000 + prefix if prefix <= 50 else 1900 + prefix


def _time_filter(understanding: QueryUnderstanding) -> str:
    """Milvus-style filter expression (kept for API compatibility/tests)."""
    raw_range = understanding.time_range or ""
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", raw_range)]
    if not years:
        return ""
    if len(years) == 1:
        year = years[0]
        normalized = raw_range.casefold()
        if any(marker in normalized for marker in ("before", "earlier", "prior", "之前", "<")) and "<=" not in normalized:
            return f"year < {year}"
        if any(marker in normalized for marker in ("以前", "截至", "<=")):
            return f"year <= {year}"
        if any(marker in normalized for marker in ("after", "later", "之后", ">")) and ">=" not in normalized:
            return f"year > {year}"
        if any(marker in normalized for marker in ("since", "以后", ">=")):
            return f"year >= {year}"
        return f"year == {year}"
    start, end = min(years), max(years)
    return f"year >= {start} and year <= {end}"


def _time_bounds(understanding: QueryUnderstanding) -> tuple[int | None, int | None]:
    """Parse the time filter into inclusive (min_year, max_year) bounds."""
    raw_range = understanding.time_range or ""
    years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", raw_range)]
    if not years:
        return None, None
    if len(years) == 1:
        year = years[0]
        normalized = raw_range.casefold()
        if any(marker in normalized for marker in ("after", "later", "之后", ">", "since", "以后", ">=")):
            return year, None
        if any(marker in normalized for marker in ("before", "earlier", "prior", "之前", "<", "以前", "截至", "<=")):
            return None, year
        return year, year
    return min(years), max(years)


def _index_root() -> Path:
    settings = get_settings()
    configured = settings.academic_local_vector_index_dir
    if configured:
        return Path(configured).expanduser()
    return ROOT_DIR / "backend" / "paper-test" / "index" / "vector"


# Cap on how many query variants are encoded per search. Encoding is local
# (zero API tokens), but keeps the per-query latency bounded.
_MAX_QUERY_VARIANTS = 4


def _query_variants(query: str, understanding: QueryUnderstanding) -> list[str]:
    """Build search variants from the intent parser's already-paid outputs.

    The original query plus the parser's rewritten queries (LLM or lexical
    fallback) and sub-queries give several angles on the same need, all at
    zero additional token cost because the parser already produced them.
    """
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = " ".join(text.casefold().split())
        if key not in seen:
            seen.add(key)
            variants.append(text)

    add(query)
    for rewritten in (understanding.rewritten_queries or []):
        add(rewritten)
    for sub_query in (understanding.sub_queries or []):
        add(sub_query)
    if understanding.intent:
        add(understanding.intent)
    if not variants:
        variants = [str(query or "")]
    return variants[:_MAX_QUERY_VARIANTS]


def _rrf_merge_variant_rankings(
    per_variant: list[list[tuple[int, float]]], k: int
) -> list[tuple[int, float]]:
    """Fuse per-variant dense+sparse rankings with a second RRF pass."""
    fused: dict[int, float] = defaultdict(float)
    for ranked in per_variant:
        for rank, (doc_index, _score) in enumerate(ranked):
            fused[doc_index] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda item: -item[1])


class _FaissVectorBackend:
    """Dense (faiss) + sparse (inverted index) hybrid backend."""

    def __init__(self, collection: str, dim: int) -> None:
        self.collection = collection
        self.dim = dim
        self.root = _index_root() / collection
        self.root.mkdir(parents=True, exist_ok=True)
        self.dense: Any = None  # faiss IndexFlatIP over L2-normalized vectors
        self.sparse_postings: dict[int, list[tuple[int, float]]] = {}
        self.docs: list[dict[str, Any]] = []
        self._doc_index: dict[str, int] = {}
        self.meta: dict[str, Any] = {}

    # -- persistence -----------------------------------------------------
    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def dense_path(self) -> Path:
        return self.root / "dense.index"

    @property
    def sparse_path(self) -> Path:
        return self.root / "sparse.npz"

    @property
    def docs_path(self) -> Path:
        return self.root / "docs.jsonl"

    def is_built(self) -> bool:
        return self.meta_path.is_file() and self.dense_path.is_file()

    def save(self) -> None:
        """Persist the index atomically: every artifact is written to a
        sibling temporary file and moved into place with ``replace``, so an
        interrupted save never leaves a half-written file behind."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta.update(
            {
                "collection": self.collection,
                "dim": self.dim,
                "count": len(self.docs),
                "backend": "faiss",
            }
        )

        def _atomic_write(target: Path, writer) -> None:
            # Keep the real suffix: numpy's _savez silently appends ".npz" to
            # any filename that does not already end in ".npz".
            temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
            try:
                writer(temporary)
                temporary.replace(target)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

        if self.dense is not None and self.dense.ntotal:
            _atomic_write(
                self.dense_path,
                lambda path: faiss.write_index(self.dense, str(path)),
            )
        # Persist the sparse inverted index as flat arrays.
        token_ids = sorted(self.sparse_postings)
        if token_ids:
            lengths = np.asarray(
                [len(self.sparse_postings[t][0]) for t in token_ids], dtype=np.int32
            )
            flat_docs = np.concatenate(
                [np.asarray(self.sparse_postings[t][0], dtype=np.int32) for t in token_ids]
            )
            flat_w = np.concatenate(
                [np.asarray(self.sparse_postings[t][1], dtype=np.float32) for t in token_ids]
            )

            def _write_sparse(path: Path) -> None:
                np.savez_compressed(
                    path,
                    token_ids=np.asarray(token_ids, dtype=np.int64),
                    lengths=lengths,
                    docs=flat_docs,
                    weights=flat_w,
                )

            _atomic_write(self.sparse_path, _write_sparse)
        elif self.sparse_path.exists():
            try:
                self.sparse_path.unlink()
            except OSError:
                pass

        def _write_docs(path: Path) -> None:
            with path.open("w", encoding="utf-8") as handle:
                for row in self.docs:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        _atomic_write(self.docs_path, _write_docs)
        # meta.json is written last as the commit marker: if it exists, the
        # preceding artifacts are complete (dense/sparse/docs are the same
        # generation). An interrupted save leaves the old meta.json behind.
        _atomic_write(
            self.meta_path,
            lambda path: path.write_text(
                json.dumps(self.meta, ensure_ascii=True), encoding="utf-8"
            ),
        )

    def load(self) -> bool:
        """Load the persisted index from disk, self-healing interrupted
        checkpoints. docs.jsonl is the source of truth; dense vectors and
        sparse postings that reference newer (never-committed) documents are
        truncated to match, so a save interrupted at any step converges to the
        last consistent generation instead of wiping the whole index."""
        if not self.is_built():
            return False
        try:
            meta_raw = self.meta_path.read_text(encoding="utf-8")
            self.meta = json.loads(meta_raw)
            if not isinstance(self.meta, dict):
                raise ValueError("meta.json is not a JSON object")
            dense = faiss.read_index(str(self.dense_path))
            docs: list[dict[str, Any]] = []
            with self.docs_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        if not isinstance(row, dict) or not row.get("arxiv_id"):
                            raise ValueError("docs.jsonl has an invalid row")
                        docs.append(row)
            if dense.ntotal > len(docs):
                # Dense index is newer than docs (save cut before docs): drop
                # the trailing vectors that have no committed document row.
                keep = len(docs)
                if keep > 0:
                    dense.remove_ids(np.arange(keep, dense.ntotal, dtype=np.int64))
                else:
                    dense = None
            postings: dict[int, tuple[array, array]] = {}
            if self.sparse_path.exists():
                with np.load(self.sparse_path, allow_pickle=False) as data:
                    token_ids = data["token_ids"]
                    lengths = data["lengths"]
                    sparse_docs = data["docs"]
                    weights = data["weights"]
                offset = 0
                for token_id, length in zip(token_ids, lengths, strict=True):
                    window = sparse_docs[offset : offset + length]
                    weights_window = weights[offset : offset + length]
                    # Filter out postings that reference uncommitted docs.
                    kept_docs = array("i")
                    kept_weights = array("f")
                    for doc_value, weight_value in zip(window, weights_window, strict=True):
                        if int(doc_value) < len(docs):
                            kept_docs.append(int(doc_value))
                            kept_weights.append(float(weight_value))
                    if kept_docs:
                        postings[int(token_id)] = (kept_docs, kept_weights)
                    offset += length
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ImportError):
            self.dense = None
            self.sparse_postings = {}
            self.docs = []
            self._doc_index = {}
            return False
        self.dense = dense
        self.sparse_postings = postings
        self.docs = docs
        self._doc_index = {
            str(row.get("arxiv_id")): index for index, row in enumerate(docs)
        }
        return True

    # -- build -----------------------------------------------------------
    def reset(self) -> None:
        self.dense = None
        self.sparse_postings = {}
        self.docs = []
        self._doc_index = {}
        self.meta = {}
        if self.root.exists():
            for path in self.root.iterdir():
                try:
                    path.unlink()
                except OSError:
                    pass

    def add(self, dense: list[list[float]], sparse: list[dict[int, float]], documents: list[dict[str, Any]]) -> int:
        # Filter to genuinely new documents first so vectors, sparse postings
        # and rows stay aligned when resuming an interrupted build.
        new_dense: list[list[float]] = []
        new_sparse: list[dict[int, float]] = []
        new_documents: list[dict[str, Any]] = []
        for vector, sparse_vector, row in zip(dense, sparse, documents, strict=True):
            if str(row["arxiv_id"]) in self._doc_index:
                continue
            new_dense.append(vector)
            new_sparse.append(sparse_vector)
            new_documents.append(row)
        if not new_documents:
            return 0
        if self.dense is None:
            self.dense = faiss.IndexFlatIP(self.dim)
        vectors = np.asarray(new_dense, dtype=np.float32)
        faiss.normalize_L2(vectors)
        start = len(self.docs)
        self.dense.add(vectors)
        for local_index, sparse_vector in enumerate(new_sparse):
            doc_index = start + local_index
            # Keep only the highest-weighted tokens so the inverted index
            # stays compact (flat parallel lists per token, no tuple objects).
            items = sorted(
                (sparse_vector or {}).items(), key=lambda kv: -abs(kv[1])
            )[:_MAX_SPARSE_TOKENS_PER_DOC]
            for token_id, weight in items:
                # array('i')/array('f') keep the inverted index resident in
                # RAM on 16GB machines (~4 bytes per entry vs ~40 for lists).
                doc_indices, doc_weights = self.sparse_postings.setdefault(
                    int(token_id), (array("i"), array("f"))
                )
                doc_indices.append(doc_index)
                doc_weights.append(float(weight))
        for row in new_documents:
            arxiv_id = str(row["arxiv_id"])
            self._doc_index[arxiv_id] = len(self.docs)
            self.docs.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": str(row.get("title") or ""),
                    "abstract": str(row.get("abstract") or ""),
                    "year": int(row.get("year") or 0) or None,
                }
            )
        return len(new_documents)

    # -- query -----------------------------------------------------------
    def search(
        self,
        dense_query: list[float],
        sparse_query: dict[int, float],
        recall: int,
        min_year: int | None,
        max_year: int | None,
        rrf_k: int,
    ) -> list[tuple[int, float]]:
        count = len(self.docs)
        if count == 0:
            return []
        scores: dict[int, float] = defaultdict(float)

        # Dense: cosine via normalized IP search.
        vector = np.asarray([dense_query], dtype=np.float32)
        faiss.normalize_L2(vector)
        distances, indices = self.dense.search(vector, min(recall, self.dense.ntotal))
        for rank, doc_index in enumerate(indices[0]):
            if doc_index < 0:
                continue
            scores[int(doc_index)] += 1.0 / (rrf_k + rank + 1)

        # Sparse: exact dot product over the inverted index.
        sparse_scores: dict[int, float] = defaultdict(float)
        for token_id, query_weight in (sparse_query or {}).items():
            entry = self.sparse_postings.get(int(token_id))
            if entry:
                for doc_index, document_weight in zip(entry[0], entry[1], strict=True):
                    sparse_scores[doc_index] += query_weight * document_weight
        if sparse_scores:
            top_sparse = sorted(sparse_scores.items(), key=lambda item: -item[1])[:recall]
            for rank, (doc_index, _score) in enumerate(top_sparse):
                scores[doc_index] += 1.0 / (rrf_k + rank + 1)

        merged = sorted(scores.items(), key=lambda item: -item[1])
        if min_year is not None or max_year is not None:
            merged = [
                (doc_index, score)
                for doc_index, score in merged
                if (
                    (min_year is None or (self.docs[doc_index].get("year") or 0) >= min_year)
                    and (max_year is None or (self.docs[doc_index].get("year") or 0) <= max_year)
                )
            ]
        return merged[:recall]


class PaSaVectorCorpusStore:
    """Persistent corpus collection, separate from session-scoped candidates.

    Backend is selected by ``academic_vector_backend``; the default ``faiss``
    backend needs no external service.
    """

    # Class-level defaults so ``__new__``-constructed test doubles (which set
    # their own Milvus-style ``client``) keep following the legacy path.
    backend_name: str = "milvus"
    _faiss: _FaissVectorBackend | None = None

    def __init__(self, *, ensure_collection: bool = False) -> None:
        settings = get_settings()
        self.collection = settings.academic_local_vector_collection
        self.dim = settings.rag_embedding_dim
        self.rrf_k = settings.rrf_k
        self.recall_limit = settings.academic_local_vector_recall_limit
        self.backend_name = (settings.academic_vector_backend or "faiss").lower()
        self.encoder = get_bge_m3_encoder()
        if self.backend_name == "milvus":
            if MilvusClient is None:
                raise RuntimeError(
                    "academic_vector_backend=milvus but pymilvus is not installed"
                )
            self.client = MilvusClient(uri=settings.milvus_uri)
            self._faiss = None
        else:
            self.client = _MilvusCompatClient(self)
            self._faiss = _FaissVectorBackend(self.collection, self.dim)
        if ensure_collection:
            self.ensure_collection()

    def is_ready(self) -> bool:
        if self.backend_name == "milvus":
            return bool(self.client.has_collection(self.collection))
        return bool(self._faiss and self._faiss.is_built())

    def ensure_collection(self, *, rebuild: bool = False) -> None:
        if self.backend_name == "milvus":
            self._ensure_milvus_collection(rebuild=rebuild)
            return
        if rebuild:
            self._faiss.reset()
        elif not self._faiss.load():
            self._faiss.reset()

    def _ensure_milvus_collection(self, *, rebuild: bool) -> None:
        if self.client.has_collection(self.collection):
            if not rebuild:
                return
            self.client.drop_collection(self.collection)
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("arxiv_id", DataType.VARCHAR, max_length=64)
        schema.add_field("title", DataType.VARCHAR, max_length=_TITLE_BYTES)
        schema.add_field("abstract", DataType.VARCHAR, max_length=_ABSTRACT_BYTES)
        schema.add_field("year", DataType.INT64)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        indexes = self.client.prepare_index_params()
        indexes.add_index(field_name="dense", metric_type="COSINE", index_type="AUTOINDEX")
        indexes.add_index(
            field_name="sparse",
            metric_type="IP",
            index_type="SPARSE_INVERTED_INDEX",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=indexes,
        )

    def upsert_documents(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        if not documents:
            return {"upserted": 0, "embedding_provider": None}
        texts = [f"{row.get('title', '')}\n{row.get('abstract', '')}" for row in documents]
        dense, sparse, provider = self.encoder.encode(texts)
        if dense and len(dense[0]) != self.dim:
            raise RuntimeError(
                f"full-corpus vector dimension mismatch: collection={self.dim}, model={len(dense[0])}"
            )
        if self.backend_name == "milvus":
            rows = []
            for document, dense_vector, sparse_vector in zip(documents, dense, sparse, strict=True):
                arxiv_id = str(document["arxiv_id"])
                rows.append(
                    {
                        "id": _stable_arxiv_id(arxiv_id),
                        "arxiv_id": arxiv_id,
                        "title": str(document.get("title") or ""),
                        "abstract": str(document.get("abstract") or ""),
                        "year": int(document.get("year") or _arxiv_year(arxiv_id) or 0),
                        "dense": dense_vector,
                        "sparse": sparse_vector,
                    }
                )
            self.client.upsert(collection_name=self.collection, data=rows)
            return {"upserted": len(rows), "embedding_provider": provider}
        inserted = self._faiss.add(dense, sparse, documents)
        return {"upserted": inserted, "embedding_provider": provider}

    def flush(self) -> None:
        if self.backend_name != "milvus":
            self._faiss.save()

    def _faiss_doc_count(self) -> int:
        """Number of documents currently loaded/indexed by the faiss backend."""
        if self.backend_name == "milvus" or self._faiss is None:
            return 0
        return len(self._faiss.docs)

    def _faiss_index_dir(self) -> str:
        """Filesystem directory the faiss backend persists to (for progress)."""
        if self.backend_name == "milvus" or self._faiss is None:
            return ""
        return str(self._faiss.root)

    def load_collection(self, collection: str | None = None) -> None:
        # No-op for the faiss backend (data is already in memory/on disk).
        if self.backend_name == "milvus" and self.client is not None:
            try:
                self.client.load_collection(collection or self.collection)
            except Exception:  # noqa: BLE001
                pass

    def search(
        self,
        query: str,
        understanding: QueryUnderstanding,
        *,
        limit: int,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError(
                "PaSa full-corpus vector index is not built; run build_pasa_vector_index.py first"
            )
        recall = max(offset + limit, self.recall_limit)
        if self.backend_name == "milvus":
            dense, sparse, provider = self.encoder.encode_query(query)
            return self._search_milvus(query, dense, sparse, provider, understanding, recall, limit, offset)
        # Load the persisted index into memory on first use (the store is
        # constructed lazily, without the corpus resident in RAM).
        if self._faiss is not None and not self._faiss.docs and self._faiss.is_built():
            with _STORE_LOCK:
                self._faiss.load()
        min_year, max_year = _time_bounds(understanding)
        # Multi-variant retrieval: encode the intent parser's already-produced
        # rewrites locally (zero API tokens) and fuse per-variant rankings.
        # Can be disabled to isolate reranking effects from query variants.
        if get_settings().academic_local_vector_multivariant:
            variants = _query_variants(query, understanding)
        else:
            variants = [query]
        provider = None
        with _STORE_LOCK:
            per_variant: list[list[tuple[int, float]]] = []
            for variant in variants:
                dense, sparse, provider = self.encoder.encode_query(variant)
                per_variant.append(
                    self._faiss.search(
                        dense, sparse, recall, min_year, max_year, self.rrf_k
                    )
                )
        ranked = _rrf_merge_variant_rankings(per_variant, self.rrf_k)[:recall]
        raw_scores = [float(score) for _, score in ranked]
        maximum = max(raw_scores, default=1.0)
        papers: list[dict[str, Any]] = []
        for doc_index, raw_score in ranked[offset : offset + limit]:
            row = self._faiss.docs[doc_index]
            arxiv_id = str(row.get("arxiv_id") or "")
            if not arxiv_id or not row.get("title"):
                continue
            year = row.get("year") or None
            papers.append(
                {
                    "id": _stable_arxiv_id(arxiv_id),
                    "title": str(row.get("title") or ""),
                    "abstract": str(row.get("abstract") or ""),
                    "authors": [],
                    "year": year,
                    "published_at": str(year) if year else None,
                    "venue": "arXiv",
                    "citation_count": 0,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "tags": [],
                    "source": "pasa-local-vector",
                    "sources": ["pasa-local-vector"],
                    "external_ids": {"arxiv": arxiv_id},
                    "score": raw_score / maximum if maximum > 0 else 0.0,
                }
            )
        return papers, {
            "provider": "pasa-local-vector",
            "query": query,
            "raw_count": len(ranked),
            "returned_count": len(papers),
            "api_calls": 0,
            "embedding_provider": provider,
            "collection": self.collection,
            "filter": _time_filter(understanding) or None,
            "backend": self.backend_name,
        }

    def _search_milvus(
        self,
        query: str,
        dense: list[float],
        sparse: dict[int, float],
        provider: str,
        understanding: QueryUnderstanding,
        recall: int,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        filter_expression = _time_filter(understanding)
        common: dict[str, Any] = {"limit": recall}
        if filter_expression:
            common["filter"] = filter_expression
        dense_request = AnnSearchRequest(
            data=[dense],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            **common,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse],
            anns_field="sparse",
            param={"metric_type": "IP", "params": {}},
            **common,
        )
        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(self.rrf_k),
            limit=recall,
            output_fields=["arxiv_id", "title", "abstract", "year"],
        )
        hits = (results[0] if results else [])[offset : offset + limit]
        raw_scores = [float(hit.get("distance") or 0.0) for hit in hits]
        maximum = max(raw_scores, default=1.0)
        papers: list[dict[str, Any]] = []
        for hit, raw_score in zip(hits, raw_scores, strict=True):
            entity = hit.get("entity") or {}
            arxiv_id = str(entity.get("arxiv_id") or "")
            if not arxiv_id or not entity.get("title"):
                continue
            year = int(entity.get("year") or 0) or None
            papers.append(
                {
                    "id": _stable_arxiv_id(arxiv_id),
                    "title": str(entity.get("title") or ""),
                    "abstract": str(entity.get("abstract") or ""),
                    "authors": [],
                    "year": year,
                    "published_at": str(year) if year else None,
                    "venue": "arXiv",
                    "citation_count": 0,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "tags": [],
                    "source": "pasa-local-vector",
                    "sources": ["pasa-local-vector"],
                    "external_ids": {"arxiv": arxiv_id},
                    "score": raw_score / maximum if maximum > 0 else 0.0,
                }
            )
        return papers, {
            "provider": "pasa-local-vector",
            "query": query,
            "raw_count": len(hits),
            "returned_count": len(papers),
            "api_calls": 0,
            "embedding_provider": provider,
            "collection": self.collection,
            "filter": filter_expression or None,
        }


class _MilvusCompatClient:
    """Small shim so ``build_pasa_vector_index.py``'s
    ``store.client.flush(...)`` / ``load_collection(...)`` calls keep working
    on the faiss backend."""

    def __init__(self, store: PaSaVectorCorpusStore) -> None:
        self._store = store

    def flush(self, collection: str | None = None) -> None:
        self._store.flush()

    def load_collection(self, collection: str | None = None) -> None:
        self._store.load_collection(collection)


_store: PaSaVectorCorpusStore | None = None


def get_pasa_vector_corpus_store() -> PaSaVectorCorpusStore:
    global _store
    if _store is None:
        _store = PaSaVectorCorpusStore()
    return _store


def search_local_vector_corpus(
    query: str,
    understanding: QueryUnderstanding,
    *,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = get_settings()
    return get_pasa_vector_corpus_store().search(
        query,
        understanding,
        limit=min(max(1, limit), settings.academic_local_vector_provider_limit),
        offset=max(0, offset),
    )
