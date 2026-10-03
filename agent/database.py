import hashlib

import psycopg2
from psycopg2.extras import Json

from .config import settings


def get_connection():
    if not settings.supabase_db_url:
        raise RuntimeError("SUPABASE_DB_URL is required.")

    conn = psycopg2.connect(settings.supabase_db_url)
    conn.autocommit = False
    return conn


def database_configured():
    return bool(settings.supabase_db_url)


def load_search_cache(cache_key):
    if not database_configured():
        return None

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT evidence, fetched_at
                FROM claimlens_search_cache
                WHERE cache_key = %s AND expires_at > NOW();
                """,
                (cache_key,),
            )
            row = cur.fetchone()

    if row is None:
        return None
    evidence = row[0]
    if isinstance(evidence, str):
        import json

        evidence = json.loads(evidence)
    return evidence, row[1]


def save_search_cache(cache_key, query, market, domain, evidence):
    if not database_configured():
        return False

    with get_connection() as conn:
        with conn.cursor() as cur:
            for item in evidence:
                url = (item.get("url") or "").strip().lower().rstrip("/")
                identity = url or " ".join(
                    (item.get("title", "") + " " + item.get("content", "")).lower().split()
                )
                evidence_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                cur.execute(
                    """
                    INSERT INTO claimlens_evidence_library
                        (evidence_hash, title, url, source, content, first_seen_market, first_seen_domain)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (evidence_hash) DO UPDATE SET
                        title = EXCLUDED.title,
                        url = EXCLUDED.url,
                        source = EXCLUDED.source,
                        content = EXCLUDED.content,
                        last_seen_at = NOW();
                    """,
                    (
                        evidence_hash,
                        item.get("title", ""),
                        item.get("url", ""),
                        item.get("source", ""),
                        item.get("content", ""),
                        market,
                        domain,
                    ),
                )

            cur.execute(
                """
                INSERT INTO claimlens_search_cache
                    (cache_key, query_text, market, domain, evidence, fetched_at, expires_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW() + (%s * INTERVAL '1 day'))
                ON CONFLICT (cache_key) DO UPDATE SET
                    query_text = EXCLUDED.query_text,
                    market = EXCLUDED.market,
                    domain = EXCLUDED.domain,
                    evidence = EXCLUDED.evidence,
                    fetched_at = NOW(),
                    expires_at = EXCLUDED.expires_at;
                """,
                (
                    cache_key,
                    query,
                    market,
                    domain,
                    Json(evidence),
                    settings.evidence_cache_ttl_days,
                ),
            )
    return True
