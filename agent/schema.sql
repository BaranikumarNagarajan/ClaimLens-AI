CREATE TABLE IF NOT EXISTS claimlens_search_cache (
    cache_key TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    market TEXT,
    domain TEXT,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS claimlens_search_cache_expires_at_idx
    ON claimlens_search_cache (expires_at);

CREATE TABLE IF NOT EXISTS claimlens_evidence_library (
    evidence_hash TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    first_seen_market TEXT,
    first_seen_domain TEXT,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS claimlens_evidence_library_url_idx
    ON claimlens_evidence_library (url);
