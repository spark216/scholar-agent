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
