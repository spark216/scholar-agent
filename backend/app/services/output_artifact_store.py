"""Immutable PostgreSQL persistence for frozen SciMap artifacts."""

from __future__ import annotations

import json

import psycopg

from backend.app.config import get_settings
from backend.app.models.evidrank import FinalRankingBundle
from backend.app.models.scimap import OutputArtifact
from backend.app.services.organizer.scimap import verify_output_artifact_hash


OUTPUT_ARTIFACT_DDL = """
CREATE TABLE IF NOT EXISTS output_artifacts (
    artifact_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES search_sessions(id) ON DELETE CASCADE,
    ranking_run_id TEXT NOT NULL REFERENCES ranking_runs(id) ON DELETE CASCADE,
    bundle_hash TEXT NOT NULL,
    intent_hash TEXT NOT NULL,
    artifact_hash TEXT NOT NULL UNIQUE,
    artifact_version TEXT NOT NULL,
    ranked_list JSONB NOT NULL,
    comparison_matrix JSONB NOT NULL,
    timeline JSONB NOT NULL,
    evidence_graph JSONB NOT NULL,
    comparability_assessments JSONB NOT NULL DEFAULT '[]'::jsonb,
    synthesis JSONB NOT NULL,
    claim_validation JSONB NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ranking_run_id, artifact_version)
);

CREATE INDEX IF NOT EXISTS idx_output_artifacts_session
    ON output_artifacts(session_id, created_at DESC);
"""


def _connect():
    settings = get_settings()
    if not settings.postgres_uri:
        raise RuntimeError("POSTGRES_URI is required to persist OutputArtifact")
    return psycopg.connect(
        settings.postgres_uri,
        connect_timeout=settings.postgres_connect_timeout,
    )


def validate_artifact_envelope(
    *,
    session_id: str,
    bundle: FinalRankingBundle,
    artifact: OutputArtifact,
) -> None:
    if not bundle.is_frozen or bundle.frozen_at is None:
        raise ValueError("only a frozen FinalRankingBundle may be persisted")
    if session_id != bundle.session_id:
        raise ValueError("session_id does not match FinalRankingBundle")
    if artifact.ranking_run_id != bundle.ranking_run_id:
        raise ValueError("OutputArtifact ranking_run_id does not match bundle")
    if artifact.bundle_hash != bundle.bundle_hash:
        raise ValueError("OutputArtifact bundle_hash does not match bundle")
    if artifact.intent_hash != bundle.intent_hash:
        raise ValueError("OutputArtifact intent_hash does not match bundle")
    if not verify_output_artifact_hash(artifact):
        raise ValueError("OutputArtifact hash verification failed")


def save_output_artifact(
    *,
    session_id: str,
    bundle: FinalRankingBundle,
    artifact: OutputArtifact,
) -> None:
    validate_artifact_envelope(
        session_id=session_id,
        bundle=bundle,
        artifact=artifact,
    )
    payload = artifact.model_dump(mode="json")
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(OUTPUT_ARTIFACT_DDL)
            cur.execute(
                """
                INSERT INTO output_artifacts (
                    artifact_id, session_id, ranking_run_id, bundle_hash,
                    intent_hash, artifact_hash, artifact_version, ranked_list,
                    comparison_matrix, timeline, evidence_graph,
                    comparability_assessments, synthesis, claim_validation, payload
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,
                    %s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb
                )
                ON CONFLICT (ranking_run_id, artifact_version) DO NOTHING
                RETURNING artifact_hash
                """,
                (
                    artifact.artifact_hash,
                    session_id,
                    artifact.ranking_run_id,
                    artifact.bundle_hash,
                    artifact.intent_hash,
                    artifact.artifact_hash,
                    artifact.artifact_version,
                    json.dumps(payload["ranked_list"], ensure_ascii=False),
                    json.dumps(payload["comparison_matrix"], ensure_ascii=False),
                    json.dumps(payload["timeline"], ensure_ascii=False),
                    json.dumps(payload["evidence_graph"], ensure_ascii=False),
                    json.dumps(payload["comparability_assessments"], ensure_ascii=False),
                    json.dumps(payload["synthesis"], ensure_ascii=False),
                    json.dumps(payload["claim_validation"], ensure_ascii=False),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            inserted = cur.fetchone()
            if inserted:
                conn.commit()
                return

            # A concurrent writer may have inserted the immutable version while
            # this transaction waited on the unique index. Under READ COMMITTED,
            # this statement sees that committed row.
            cur.execute(
                """
                SELECT artifact_hash
                FROM output_artifacts
                WHERE ranking_run_id = %s AND artifact_version = %s
                """,
                (artifact.ranking_run_id, artifact.artifact_version),
            )
            existing = cur.fetchone()
            if existing is None:
                raise RuntimeError("OutputArtifact conflict row disappeared")
            if str(existing[0]) != artifact.artifact_hash:
                raise RuntimeError(
                    "cannot overwrite immutable OutputArtifact "
                    f"for ranking_run_id={artifact.ranking_run_id}, "
                    f"artifact_version={artifact.artifact_version}"
                )
        conn.commit()
