from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str | None = None
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5-20251001"
    fallback_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 60
    llm_num_retries: int = 2
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
