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
