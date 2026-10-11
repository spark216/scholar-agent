"""Milvus hybrid store: BGE-M3 dense + sparse with RRF fusion."""

from __future__ import annotations

import logging
import time
from typing import Any

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from backend.app.config import get_settings
from backend.app.services.embeddings import get_bge_m3_encoder

logger = logging.getLogger(__name__)

_VARCHAR_LIMITS = {
    "id": 64,
    "session_id": 64,
    "title": 1024,
    "abstract": 4096,
    "url": 1024,
    "published_at": 64,
}


def _truncate_utf8(value: Any, max_bytes: int) -> str:
    """Return valid UTF-8 whose encoded size fits a Milvus VARCHAR field."""
    text = str(value or "")
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class MilvusPaperStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.uri = settings.milvus_uri
        self.collection = settings.milvus_collection
        self.dim = settings.rag_embedding_dim
        self.rrf_k = settings.rrf_k
        self.recall_limit = settings.hybrid_recall_limit
        self.client = MilvusClient(uri=self.uri)
        self.encoder = get_bge_m3_encoder()
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            # If old schema (no sparse), recreate for BGE hybrid.
            try:
                info = self.client.describe_collection(self.collection)
                fields = {f.get("name") for f in info.get("fields", [])}
                if "dense" in fields and "sparse" in fields:
                    return
                logger.warning("Dropping incompatible collection %s for BGE-M3 hybrid schema", self.collection)
                self.client.drop_collection(self.collection)
            except Exception:
                # If describe fails, keep existing and hope schema matches.
                return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            "id",
            DataType.VARCHAR,
            is_primary=True,
            max_length=_VARCHAR_LIMITS["id"],
        )
        schema.add_field(
            "session_id",
            DataType.VARCHAR,
            max_length=_VARCHAR_LIMITS["session_id"],
        )
        schema.add_field(
            "title",
            DataType.VARCHAR,
            max_length=_VARCHAR_LIMITS["title"],
        )
        schema.add_field(
            "abstract",
            DataType.VARCHAR,
            max_length=_VARCHAR_LIMITS["abstract"],
        )
        schema.add_field(
            "url",
            DataType.VARCHAR,
            max_length=_VARCHAR_LIMITS["url"],
        )
        schema.add_field(
            "published_at",
            DataType.VARCHAR,
            max_length=_VARCHAR_LIMITS["published_at"],
        )
        schema.add_field("year", DataType.INT64)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            metric_type="COSINE",
            index_type="AUTOINDEX",
        )
        index_params.add_index(
            field_name="sparse",
            metric_type="IP",
            index_type="SPARSE_INVERTED_INDEX",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    def upsert_papers(
        self,
        session_id: str,
        papers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not papers:
            return {"upserted": 0, "embedding_provider": None}

        started_at = time.perf_counter()
        texts = [f"{p.get('title', '')}\n{p.get('abstract', '')}" for p in papers]
        dense_vecs, sparse_vecs, provider = self.encoder.encode(texts)
        encoding_seconds = time.perf_counter() - started_at

        # Align dense dim with collection if model dim differs.
        if dense_vecs and len(dense_vecs[0]) != self.dim:
            # Update runtime dim and recreate collection once.
            self.dim = len(dense_vecs[0])
            if self.client.has_collection(self.collection):
                self.client.drop_collection(self.collection)
            self.ensure_collection()

        rows = []
        truncated_fields = 0
        for paper, dense, sparse in zip(papers, dense_vecs, sparse_vecs, strict=True):
            raw_fields = {
                "id": f"{session_id}:{paper['id']}",
                "session_id": session_id,
                "title": paper.get("title") or "",
                "abstract": paper.get("abstract") or "",
                "url": paper.get("url") or "",
                "published_at": paper.get("published_at") or paper.get("year") or "",
            }
            varchar_fields = {
                field: _truncate_utf8(value, _VARCHAR_LIMITS[field])
                for field, value in raw_fields.items()
            }
            truncated_fields += sum(
                len(str(raw_fields[field]).encode("utf-8"))
                > _VARCHAR_LIMITS[field]
                for field in raw_fields
            )
            rows.append(
                {
                    **varchar_fields,
                    "year": int(paper["year"]) if paper.get("year") else 0,
                    "dense": dense,
                    "sparse": sparse,
                }
            )
        write_started_at = time.perf_counter()
        self.client.upsert(collection_name=self.collection, data=rows)
        try:
            self.client.flush(self.collection)
        except Exception:
            # older/newer clients may use different flush APIs; ignore if unavailable
            pass
        try:
            self.client.load_collection(self.collection)
        except Exception:
            pass
        milvus_write_seconds = time.perf_counter() - write_started_at
        return {
            "upserted": len(rows),
            "embedding_provider": provider,
            "dense_dim": self.dim,
            "has_sparse": True,
            "truncated_varchar_fields": truncated_fields,
            "encoding_seconds": round(encoding_seconds, 3),
            "milvus_write_seconds": round(milvus_write_seconds, 3),
            "total_upsert_seconds": round(time.perf_counter() - started_at, 3),
        }

    def _hit_to_paper(self, hit: dict[str, Any], score: float) -> dict[str, Any]:
        entity = hit.get("entity") or {}
        raw_id = str(hit.get("id") or "")
        paper_id = raw_id.split(":", 1)[-1] if ":" in raw_id else raw_id
        year = entity.get("year") or None
        if year == 0:
            year = None
        return {
            "id": paper_id,
            "title": entity.get("title") or "Untitled",
            "abstract": entity.get("abstract") or "",
            "url": entity.get("url") or None,
            "published_at": entity.get("published_at") or (str(year) if year else None),
            "year": year,
            "relevance_score": max(0.0, min(1.0, float(score))),
            "source": "milvus-hybrid-rrf",
            "authors": [],
            "tags": [],
            "venue": None,
            "citation_count": 0,
        }

    def search_top_k(
        self,
        query: str,
        session_id: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        dense_q, sparse_q, provider = self.encoder.encode_query(query)
        expr = f'session_id == "{session_id}"'
        recall = max(top_k, self.recall_limit)

        dense_req = AnnSearchRequest(
            data=[dense_q],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=recall,
            filter=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_q],
            anns_field="sparse",
            param={"metric_type": "IP", "params": {}},
            limit=recall,
            filter=expr,
        )

        output_fields = ["title", "abstract", "url", "published_at", "year", "session_id"]

        def dense_fallback(reason: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            """Keep the frontend useful when hybrid/RRF returns no current-session hits."""
            results = self.client.search(
                collection_name=self.collection,
                data=[dense_q],
                anns_field="dense",
                limit=recall,
                filter=expr,
                output_fields=output_fields,
            )
            hits = results[0] if results else []
            # Do not leak results from previous searches even if a server ignores the filter.
            hits = [
                h
                for h in hits
                if (h.get("entity") or {}).get("session_id") == session_id
                or str(h.get("id", "")).startswith(f"{session_id}:")
            ]
            papers = []
            for hit in hits:
                distance = float(hit.get("distance") or 0.0)
                score = max(
                    0.0,
                    min(1.0, distance if distance <= 1.5 else 1.0 / (1.0 + distance)),
                )
                papers.append(self._hit_to_paper(hit, score))
            return papers, {
                "embedding_provider": provider,
                "retriever": "dense-only-fallback",
                "hit_count": len(papers),
                "fallback_reason": reason,
            }

        try:
            results = self.client.hybrid_search(
                collection_name=self.collection,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(self.rrf_k),
                limit=recall,
                output_fields=output_fields,
            )
            hits = results[0] if results else []
            # Keep only current session (belt-and-suspenders if server filter unsupported).
            hits = [
                h
                for h in hits
                if (h.get("entity") or {}).get("session_id") == session_id
                or str(h.get("id", "")).startswith(f"{session_id}:")
            ]
            if not hits:
                logger.warning(
                    "hybrid_search returned no hits for session %s; falling back to dense-only",
                    session_id,
                )
                return dense_fallback("hybrid_search_empty")

            raw_scores = [float(h.get("distance") or 0.0) for h in hits]
            max_score = max(raw_scores) if raw_scores else 1.0
            papers: list[dict[str, Any]] = []
            for hit, raw in zip(hits, raw_scores, strict=True):
                # Normalize RRF scores within this result set to [0,1] for UI.
                score = (raw / max_score) if max_score > 0 else 0.0
                papers.append(self._hit_to_paper(hit, score))
            return papers, {
                "embedding_provider": provider,
                "retriever": "dense+sparse+rrf",
                "hit_count": len(papers),
                "rrf_k": self.rrf_k,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("hybrid_search failed, fallback to dense-only: %s", exc)
            return dense_fallback(f"hybrid_search_error: {exc}")


_store: MilvusPaperStore | None = None


def get_milvus_store() -> MilvusPaperStore:
    global _store
    if _store is None:
        _store = MilvusPaperStore()
    return _store
