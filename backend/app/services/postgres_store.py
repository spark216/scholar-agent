"""PostgreSQL persistence for search sessions and ranked papers."""

from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.rows import dict_row

from backend.app.config import get_settings
from backend.app.models.evidrank import FinalRankingBundle
from backend.app.services.ranking.bundle_builder import compute_bundle_hash

_SCHEMA_READY = False

DDL = """
CREATE TABLE IF NOT EXISTS search_sessions (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    understanding JSONB NOT NULL DEFAULT '{}'::jsonb,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS search_papers (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES search_sessions(id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    url TEXT,
    published_at TEXT,
    year INT,
    relevance_score DOUBLE PRECISION,
    relevance_level TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_papers_session ON search_papers(session_id);
CREATE INDEX IF NOT EXISTS idx_search_sessions_created ON search_sessions(created_at DESC);

CREATE TABLE IF NOT EXISTS ranking_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES search_sessions(id) ON DELETE CASCADE,
    intent_hash TEXT NOT NULL,
    bundle_hash TEXT NOT NULL,
    threshold_version TEXT NOT NULL,
    model_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    timings JSONB NOT NULL DEFAULT '{}'::jsonb,
    degraded BOOLEAN NOT NULL DEFAULT FALSE,
    degradation_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_frozen BOOLEAN NOT NULL DEFAULT FALSE,
    frozen_at TIMESTAMPTZ,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ranking_items (
    ranking_run_id TEXT NOT NULL REFERENCES ranking_runs(id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL,
    final_rank INT,
    level TEXT NOT NULL,
    gate_decision TEXT NOT NULL,
    dense_score DOUBLE PRECISION,
    dense_rank INT,
    sparse_score DOUBLE PRECISION,
    sparse_rank INT,
    rrf_score DOUBLE PRECISION,
    rrf_rank INT,
    cross_logit DOUBLE PRECISION,
    cross_sigmoid DOUBLE PRECISION,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (ranking_run_id, paper_id)
);

CREATE TABLE IF NOT EXISTS paper_evidence (
    evidence_id TEXT NOT NULL,
    ranking_run_id TEXT NOT NULL REFERENCES ranking_runs(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES search_sessions(id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL,
    paper_version_id TEXT NOT NULL,
    facet_id TEXT NOT NULL,
    label TEXT NOT NULL,
    source_type TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ranking_run_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_ranking_runs_session ON ranking_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_ranking_items_run_rank ON ranking_items(ranking_run_id, final_rank);
CREATE INDEX IF NOT EXISTS idx_paper_evidence_run_paper ON paper_evidence(ranking_run_id, paper_id);
"""


def _connect():
    settings = get_settings()
    if not settings.postgres_uri:
        raise RuntimeError("未配置 POSTGRES_URI")
    return psycopg.connect(
        settings.postgres_uri,
        row_factory=dict_row,
        connect_timeout=settings.postgres_connect_timeout,
    )


def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
    _SCHEMA_READY = True


def validate_ranking_bundle_envelope(
    *,
    session_id: str,
    ranking_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Reject draft, cross-session, or tampered bundles before persistence."""
    bundle = FinalRankingBundle.model_validate(ranking_bundle)
    if bundle.session_id != session_id:
        raise ValueError("ranking bundle session_id does not match search session")
    if compute_bundle_hash(bundle) != bundle.bundle_hash:
        raise ValueError("ranking bundle hash verification failed")
    return bundle.model_dump(mode="json")


def save_search_result(
    session_id: str,
    query: str,
    understanding: dict[str, Any],
    papers: list[dict[str, Any]],
    meta: dict[str, Any],
    ranking_bundle: dict[str, Any] | None = None,
) -> None:
    if ranking_bundle is not None:
        ranking_bundle = validate_ranking_bundle_envelope(
            session_id=session_id,
            ranking_bundle=ranking_bundle,
        )
    ensure_schema()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO search_sessions (id, query, understanding, meta)
                VALUES (%s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                SET query = EXCLUDED.query,
                    understanding = EXCLUDED.understanding,
                    meta = EXCLUDED.meta
                """,
                (
                    session_id,
                    query,
                    json.dumps(understanding, ensure_ascii=False),
                    json.dumps(meta, ensure_ascii=False),
                ),
            )
            cur.execute("DELETE FROM search_papers WHERE session_id = %s", (session_id,))
            for paper in papers:
                cur.execute(
                    """
                    INSERT INTO search_papers (
                        session_id, paper_id, title, abstract, url,
                        published_at, year, relevance_score, relevance_level, payload
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    """,
                    (
                        session_id,
                        paper.get("id"),
                        paper.get("title"),
                        paper.get("abstract"),
                        paper.get("url"),
                        paper.get("published_at"),
                        paper.get("year"),
                        paper.get("relevance_score"),
                        paper.get("relevance"),
                        json.dumps(paper, ensure_ascii=False),
                    ),
                )
            if ranking_bundle:
                run_id = str(ranking_bundle["ranking_run_id"])
                cur.execute(
                    "SELECT is_frozen, bundle_hash FROM ranking_runs WHERE id = %s FOR UPDATE",
                    (run_id,),
                )
                existing_run = cur.fetchone()
                if existing_run:
                    if isinstance(existing_run, dict):
                        existing_frozen = bool(existing_run["is_frozen"])
                        existing_hash = str(existing_run["bundle_hash"])
                    else:
                        existing_frozen = bool(existing_run[0])
                        existing_hash = str(existing_run[1])
                    if existing_frozen and existing_hash != ranking_bundle["bundle_hash"]:
                        raise RuntimeError(
                            "cannot overwrite frozen ranking run "
                            f"{run_id} (bundle_hash={existing_hash})"
                        )
                    if existing_frozen and existing_hash == ranking_bundle["bundle_hash"]:
                        ranking_bundle = None

            if ranking_bundle:
                run_id = str(ranking_bundle["ranking_run_id"])
                cur.execute(
                    """
                    INSERT INTO ranking_runs (
                        id, session_id, intent_hash, bundle_hash, threshold_version,
                        model_versions, timings, degraded, degradation_reasons,
                        is_frozen, frozen_at, payload
                    ) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb,%s,%s,%s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        bundle_hash = EXCLUDED.bundle_hash,
                        threshold_version = EXCLUDED.threshold_version,
                        model_versions = EXCLUDED.model_versions,
                        timings = EXCLUDED.timings,
                        degraded = EXCLUDED.degraded,
                        degradation_reasons = EXCLUDED.degradation_reasons,
                        is_frozen = EXCLUDED.is_frozen,
                        frozen_at = EXCLUDED.frozen_at,
                        payload = EXCLUDED.payload
                    """,
                    (
                        run_id,
                        session_id,
                        ranking_bundle["intent_hash"],
                        ranking_bundle["bundle_hash"],
                        ranking_bundle["threshold_version"],
                        json.dumps(ranking_bundle.get("model_versions") or {}, ensure_ascii=False),
                        json.dumps(ranking_bundle.get("timings") or {}, ensure_ascii=False),
                        bool(ranking_bundle.get("degraded", False)),
                        json.dumps(ranking_bundle.get("degradation_reasons") or [], ensure_ascii=False),
                        bool(ranking_bundle.get("is_frozen", False)),
                        ranking_bundle.get("frozen_at"),
                        json.dumps(ranking_bundle, ensure_ascii=False),
                    ),
                )
                cur.execute("DELETE FROM ranking_items WHERE ranking_run_id = %s", (run_id,))
                cur.execute("DELETE FROM paper_evidence WHERE ranking_run_id = %s", (run_id,))
                for item in ranking_bundle.get("items") or []:
                    scores = item.get("scores") or {}
                    gate = item.get("gate") or {}
                    cur.execute(
                        """
                        INSERT INTO ranking_items (
                            ranking_run_id, paper_id, final_rank, level, gate_decision,
                            dense_score, dense_rank, sparse_score, sparse_rank,
                            rrf_score, rrf_rank, cross_logit, cross_sigmoid, payload
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        """,
                        (
                            run_id,
                            item["paper_id"],
                            item.get("final_rank"),
                            item["level"],
                            gate.get("decision"),
                            scores.get("dense_score"),
                            scores.get("dense_rank"),
                            scores.get("sparse_score"),
                            scores.get("sparse_rank"),
                            scores.get("rrf_score"),
                            scores.get("rrf_rank"),
                            scores.get("cross_logit"),
                            scores.get("cross_sigmoid"),
                            json.dumps(item, ensure_ascii=False),
                        ),
                    )
                for evidence in ranking_bundle.get("evidence_matrix") or []:
                    cur.execute(
                        """
                        INSERT INTO paper_evidence (
                            evidence_id, ranking_run_id, session_id, paper_id,
                            paper_version_id, facet_id, label, source_type,
                            validation_status, payload
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        """,
                        (
                            evidence["evidence_id"],
                            run_id,
                            session_id,
                            evidence["paper_id"],
                            evidence["paper_version_id"],
                            evidence["facet_id"],
                            evidence["label"],
                            evidence["source_type"],
                            evidence["validation_status"],
                            json.dumps(evidence, ensure_ascii=False),
                        ),
                    )
        conn.commit()


def healthcheck() -> str:
    try:
        ensure_schema()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"
