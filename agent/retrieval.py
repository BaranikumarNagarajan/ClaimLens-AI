import hashlib
import re
from urllib.parse import urlparse

import pandas as pd

from .config import settings
from .database import database_configured, get_connection, load_search_cache, save_search_cache


COUNTRY_NAMES = {
    "australia": "Australia",
    "au": "Australia",
    "canada": "Canada",
    "ca": "Canada",
    "france": "France",
    "fr": "France",
    "germany": "Germany",
    "de": "Germany",
    "india": "India",
    "in": "India",
    "ireland": "Ireland",
    "ie": "Ireland",
    "japan": "Japan",
    "jp": "Japan",
    "malaysia": "Malaysia",
    "my": "Malaysia",
    "netherlands": "Netherlands",
    "netherland": "Netherlands",
    "holland": "Netherlands",
    "nl": "Netherlands",
    "singapore": "Singapore",
    "sg": "Singapore",
    "united arab emirates": "United Arab Emirates",
    "uae": "United Arab Emirates",
    "ae": "United Arab Emirates",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "gb": "United Kingdom",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
}


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(text):
    text = safe_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9%+\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_float(value):
    if value is None:
        return None
    try:
        parsed = float(str(value).replace(",", "").replace("%", ""))
        if parsed != parsed:
            return None
        return parsed
    except Exception:
        return None


def get_domain(url):
    url = safe_text(url)
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        return domain.replace("www.", "")
    except Exception:
        return ""


def build_search_query(question, market=None, domain=None):
    parts = [safe_text(question)]
    if market and safe_text(market):
        parts.append(f"Market or country: {safe_text(market)}")
    if domain and safe_text(domain):
        parts.append(f"Domain: {safe_text(domain)}")
    return " ".join(parts)


def _search_cache_key(query, market=None, domain=None):
    identity = "\n".join(
        [normalize_text(query), normalize_text(market), normalize_text(domain)]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _tavily_country(market):
    return COUNTRY_NAMES.get(normalize_text(market))


def fetch_tavily_results(query, max_results=None, country=None):
    if not settings.tavily_api_key:
        return []

    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    search_options = {
        "query": query,
        "max_results": max_results or settings.evidence_max_results,
        "search_depth": "advanced",
    }
    if country:
        search_options["country"] = country
    response = client.search(
        **search_options,
    )
    return response.get("results", [])


def first_existing_column(dataframe, candidates):
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def standardize_evidence_dataframe(df_source):
    if df_source is None or df_source.empty:
        return pd.DataFrame(
            columns=["evidence_id", "title", "url", "raw_evidence", "source", "analysis_text"]
        )

    title_column = first_existing_column(df_source, ["title", "product_title", "name"])
    url_column = first_existing_column(df_source, ["url", "source_url", "link"])
    content_column = first_existing_column(
        df_source,
        ["content", "evidence", "raw_evidence", "excerpt", "snippet", "text"],
    )
    source_column = first_existing_column(df_source, ["source", "domain"])

    df_evidence = pd.DataFrame(index=df_source.index)
    df_evidence["evidence_id"] = df_source["id"] if "id" in df_source.columns else df_source.index
    df_evidence["title"] = df_source[title_column] if title_column else ""
    df_evidence["url"] = df_source[url_column] if url_column else ""
    df_evidence["raw_evidence"] = df_source[content_column] if content_column else ""

    if source_column:
        df_evidence["source"] = df_source[source_column].fillna("").astype(str)
    else:
        df_evidence["source"] = df_evidence["url"].apply(get_domain)

    df_evidence["analysis_text"] = (
        df_evidence["title"].fillna("").astype(str)
        + " "
        + df_evidence["raw_evidence"].fillna("").astype(str)
    )

    return df_evidence


def _normalize_search_results(results):
    normalized = []
    seen = set()
    for item in results or []:
        if not isinstance(item, dict):
            continue
        title = safe_text(item.get("title"))
        url = safe_text(item.get("url"))
        content = safe_text(item.get("content") or item.get("snippet"))
        if not title and not content:
            continue
        identity = url.lower().rstrip("/") or " ".join((title + " " + content).lower().split())
        evidence_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if evidence_hash in seen:
            continue
        seen.add(evidence_hash)
        normalized.append(
            {
                "id": evidence_hash,
                "title": title,
                "url": url,
                "source": safe_text(item.get("source")) or get_domain(url),
                "content": content,
            }
        )
    return normalized


def _empty_retrieval_details(query, cache_key, database_status):
    return {
        "search_query": query,
        "cache_key": cache_key,
        "source": "none",
        "search_status": "not_started",
        "database_status": database_status,
        "cache_hit": False,
        "web_search_performed": False,
        "fetched_count": 0,
        "deduplicated_count": 0,
        "persisted": False,
    }


def _load_legacy_investigation(investigation_id):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, url, source, content, extracted_numbers, created_at
                    FROM claimlens_evidence
                    WHERE investigation_id = %s
                    ORDER BY id;
                    """,
                    (investigation_id,),
                )
                rows = cur.fetchall()
        return pd.DataFrame(
            rows,
            columns=["id", "title", "url", "source", "content", "extracted_numbers", "created_at"],
        )
    except Exception:
        return pd.DataFrame()


def retrieve_evidence(question, market=None, domain=None, investigation_id=None, candidate_df=None):
    if candidate_df is not None and isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
        df = standardize_evidence_dataframe(candidate_df.copy())
        details = _empty_retrieval_details(question, "provided-input", "not_used")
        details.update(
            {
                "source": "provided_input",
                "search_status": "provided_input",
                "fetched_count": int(len(df)),
                "deduplicated_count": int(len(df)),
            }
        )
        return df, details

    if isinstance(candidate_df, list) and candidate_df:
        try:
            records = _normalize_search_results(candidate_df)
            df = standardize_evidence_dataframe(pd.DataFrame(records))
            details = _empty_retrieval_details(question, "provided-input", "not_used")
            details.update(
                {
                    "source": "provided_input",
                    "search_status": "provided_input",
                    "fetched_count": len(candidate_df),
                    "deduplicated_count": len(records),
                }
            )
            return df, details
        except Exception:
            pass

    search_query = build_search_query(question, market=market, domain=domain)
    cache_key = _search_cache_key(search_query, market=market, domain=domain)
    database_status = "disabled" if not database_configured() else "miss"
    details = _empty_retrieval_details(search_query, cache_key, database_status)
    details["country_filter"] = _tavily_country(market)

    if database_configured():
        try:
            cached = load_search_cache(cache_key)
            if cached is not None:
                cached_records, fetched_at = cached
                normalized = _normalize_search_results(cached_records)
                details.update(
                    {
                        "source": "database_cache",
                        "search_status": "cached_results" if normalized else "cached_no_results",
                        "database_status": "hit",
                        "cache_hit": True,
                        "fetched_count": len(normalized),
                        "deduplicated_count": len(normalized),
                        "fetched_at": fetched_at.isoformat() if fetched_at else None,
                    }
                )
                return standardize_evidence_dataframe(pd.DataFrame(normalized)), details
        except Exception as error:
            details["database_status"] = "unavailable"
            details["database_error"] = type(error).__name__

    if not settings.tavily_api_key:
        details["search_status"] = "missing_api_key"
        records = []
    else:
        details["web_search_performed"] = True
        try:
            fetched = fetch_tavily_results(search_query, country=details["country_filter"])
            records = _normalize_search_results(fetched)
            details["source"] = "tavily"
            details["search_status"] = "success" if records else "no_results"
            details["fetched_count"] = len(fetched)
            details["deduplicated_count"] = len(records)
        except Exception as error:
            records = []
            details["search_status"] = "search_failed"
            details["search_error"] = type(error).__name__

    if details["search_status"] in {"success", "no_results"} and database_configured():
        try:
            save_search_cache(
                cache_key,
                search_query,
                market,
                domain,
                records,
            )
            details["database_status"] = "saved"
            details["persisted"] = True
        except Exception as error:
            details["database_status"] = "save_failed"
            details["database_error"] = type(error).__name__

    if not records and investigation_id:
        legacy = _load_legacy_investigation(investigation_id)
        if not legacy.empty:
            details["source"] = "legacy_investigation"
            details["search_status"] = "legacy_database_fallback"
            details["fetched_count"] = len(legacy)
            details["deduplicated_count"] = len(legacy)
            return standardize_evidence_dataframe(legacy), details

    return standardize_evidence_dataframe(pd.DataFrame(records)), details


def recover_evidence_dataframe(question, investigation_id=None, candidate_df=None):
    df, _ = retrieve_evidence(
        question,
        investigation_id=investigation_id,
        candidate_df=candidate_df,
    )
    return df
