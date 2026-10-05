"""FastAPI dependency factories (overridable via ``app.dependency_overrides``)."""

from functools import lru_cache
from typing import Any

import redis
import structlog
from openai import OpenAI

from cag_estimate.cache.semantic import EstimationSemanticCache
from cag_estimate.cache.vectorizer import LazyOpenAIVectorizer
from cag_estimate.config import get_settings
from cag_estimate.prompts import render_system_prompt
from cag_estimate.services.cache import get_cache
from cag_estimate.services.estimation import EstimationService
from cag_estimate.services.llm_wrapper import get_llm_wrapper
from cag_estimate.services.sessions import SessionStore

log = structlog.get_logger()


def build_moderation_client() -> Any | None:
    """OpenAI client for moderation, or None when no key is set or it cannot be built."""
    api_key = get_settings().openai_api_key
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:  # noqa: BLE001 - moderation is optional, fail soft
        log.warning("moderation_client_unavailable", error_type=type(exc).__name__)
        return None


def build_semantic_cache() -> EstimationSemanticCache | None:
    """Semantic cache, or None when disabled, keyless, or Redis Stack is unavailable."""
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        log.warning("semantic_cache_disabled", reason="disabled_by_settings")
        return None
    if not settings.openai_api_key:
        log.warning("semantic_cache_disabled", reason="no_openai_key")
        return None
    try:
        vectorizer = LazyOpenAIVectorizer(
            model=settings.embedding_model,
            client=OpenAI(api_key=settings.openai_api_key),
            dims=settings.embedding_dims,
        )
        return EstimationSemanticCache(
            redis_client=redis.from_url(settings.redis_url, decode_responses=False),
            vectorizer=vectorizer,
            threshold=settings.semantic_cache_threshold,
            ttl=settings.semantic_cache_ttl,
            log_only=settings.semantic_cache_log_only,
        )
    except Exception as exc:  # noqa: BLE001 - optional layer, fail soft
        log.warning("semantic_cache_disabled", reason="setup_failed", error_type=type(exc).__name__)
        return None


@lru_cache
def get_semantic_cache() -> EstimationSemanticCache | None:
    """Process-wide semantic cache (None when unavailable)."""
    return build_semantic_cache()


@lru_cache
def get_session_store() -> SessionStore:
    """Process-wide in-memory session store (volatile: lost on restart)."""
    settings = get_settings()
    return SessionStore(
        system_prompt_provider=render_system_prompt,
        max_turns=settings.session_max_turns,
        max_sessions=settings.session_max_sessions,
    )


@lru_cache
def get_estimation_service() -> EstimationService:
    """Process-wide estimation service built from the shared singletons."""
    return EstimationService(
        get_llm_wrapper(), get_cache(), build_moderation_client(), get_semantic_cache()
    )
