from pydantic_settings import BaseSettings
from functools import lru_cache


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
    app_env: str = "development"
    log_level: str = "DEBUG"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get and cache settings instance."""
    return Settings()
