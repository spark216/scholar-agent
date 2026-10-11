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
