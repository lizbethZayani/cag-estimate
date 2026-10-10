from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings

DEFAULT_SESSION_MAX_TURNS = 6
DEFAULT_SESSION_MAX_SESSIONS = 1000
DEFAULT_STRUCTURED_MAX_TOKENS = 8192


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str | None = None
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5-20251001"
    fallback_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 60
    llm_num_retries: int = 2
    # Output budget for structured (Instructor) calls; too low truncates large estimates.
    llm_structured_max_tokens: int = DEFAULT_STRUCTURED_MAX_TOKENS
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86400
    embedding_model: str = "text-embedding-3-small"
    # Needed only for embedding models missing from the built-in dimensions map.
    embedding_dims: int | None = None
    # Single default for the semantic cache: cosine similarity needed to reuse an answer.
    semantic_cache_threshold: float = 0.90
    semantic_cache_ttl: int = 86400
    # Log would-be hits without serving them (to calibrate the threshold).
    semantic_cache_log_only: bool = False
    semantic_cache_enabled: bool = True
    # Sliding window: one turn = one user + one assistant message.
    session_max_turns: int = DEFAULT_SESSION_MAX_TURNS
    # Safety cap for the in-memory session store (LRU eviction beyond it).
    session_max_sessions: int = DEFAULT_SESSION_MAX_SESSIONS
    # Seconds of inactivity after which a session expires; 0 disables expiry.
    session_idle_ttl_seconds: int = 0
    # Compress evicted history: pin anchor turns and summarize the rest (False = plain drop).
    session_compression_enabled: bool = True
    # Most anchor (hard-constraint) turns pinned per session; older ones fold into the summary.
    session_max_anchors: int = 5
    # Longest cumulative summary of evicted turns, in characters.
    session_summary_max_chars: int = 2000
    # "llm" folds evicted turns with one short LLM call (extractive fallback on failure).
    session_summary_mode: Literal["llm", "extractive"] = "llm"
    # Model for those calls (use a cheap one); None = the router's primary model.
    session_summary_model: str | None = None
    # Attachment limits: file count, bytes per file, chars of transcript + attachments.
    attachment_max_files: int = 5
    attachment_max_bytes: int = 5_000_000
    attachment_max_chars_total: int = 80_000
    # Longest transcript accepted by the session estimate endpoint, in characters.
    transcript_max_chars: int = 80_000
    app_env: str = "development"
    log_level: str = "DEBUG"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get and cache settings instance."""
    return Settings()
