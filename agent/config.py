import os
from dataclasses import dataclass
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()


def _database_url():
    connection_url = os.getenv("SUPABASE_DB_URL", "").strip()
    if connection_url:
        return connection_url

    host = os.getenv("SUPABASE_HOST", "").strip()
    user = os.getenv("SUPABASE_USER", "").strip()
    password = os.getenv("SUPABASE_PASSWORD", "").strip()
    if not all((host, user, password)):
        return ""

    port = os.getenv("SUPABASE_PORT", "5432").strip() or "5432"
    database = os.getenv("SUPABASE_DATABASE", "postgres").strip() or "postgres"
    return (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(database, safe='')}"
    )


def _int_setting(name, default, minimum=1, maximum=100):
    try:
        return max(minimum, min(maximum, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    hf_token: str = os.getenv("HF_TOKEN", "").strip()
    hf_model: str = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct").strip()
    hf_provider: str = os.getenv("HF_PROVIDER", "auto").strip() or "auto"
    hf_timeout_seconds: int = _int_setting("HF_TIMEOUT_SECONDS", 45, 5, 120)
    supabase_db_url: str = _database_url()
    evidence_cache_ttl_days: int = _int_setting("EVIDENCE_CACHE_TTL_DAYS", 30, 1, 365)
    evidence_max_results: int = _int_setting("EVIDENCE_MAX_RESULTS", 8, 1, 20)
    streamlit_title: str = os.getenv("STREAMLIT_TITLE", "ClaimLens AI")


settings = Settings()
