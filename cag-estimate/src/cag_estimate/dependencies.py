"""FastAPI dependency factories (overridable via ``app.dependency_overrides``)."""

from functools import lru_cache, partial
from typing import Any

import redis
import structlog
from openai import OpenAI

from cag_estimate.cache.semantic import EstimationSemanticCache
from cag_estimate.cache.vectorizer import LazyOpenAIVectorizer
from cag_estimate.config import get_settings
from cag_estimate.db.session import get_async_session_factory
from cag_estimate.embedding_pipeline.chunker import JSONStructuralChunker
from cag_estimate.embedding_pipeline.embedder import OpenAIEmbedder
from cag_estimate.embedding_pipeline.ingest_service import RagIngestService
from cag_estimate.embedding_pipeline.retriever import SemanticRetriever
from cag_estimate.prompts import render_system_prompt
from cag_estimate.services.cache import get_cache
from cag_estimate.services.critic import Critic
from cag_estimate.services.estimation import EstimationService
from cag_estimate.services.llm_summarizer import LlmSummarizer, TextCompleter
from cag_estimate.services.llm_wrapper import get_llm_wrapper
from cag_estimate.services.session_compression import (
    AnchorDetector,
    ExtractiveSummarizer,
    HistoryCompactor,
    Summarizer,
)
from cag_estimate.services.session_estimation import (
    SESSION_PROMPT_VERSION,
    MeteredLLM,
    SessionEstimationService,
)
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


def build_embedder() -> OpenAIEmbedder | None:
    """Embedder, or None when no OpenAI key is set or the client cannot be built.

    The client disables the SDK's own retries (``max_retries=0``) so the retry policy
    of ``OpenAIEmbedder`` is the only one and waits are not multiplied.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        log.warning("embedding_service_unavailable", reason="no_openai_key")
        return None
    try:
        client = OpenAI(api_key=settings.openai_api_key, max_retries=0)
    except Exception as exc:  # noqa: BLE001 - the endpoints answer 503, fail soft
        log.warning("embedding_service_unavailable", error_type=type(exc).__name__)
        return None
    return OpenAIEmbedder(client, model=settings.embedding_model)


@lru_cache
def get_embedder() -> OpenAIEmbedder | None:
    """Process-wide embedder (None when not configured)."""
    return build_embedder()


def _get_embedder_with_database() -> OpenAIEmbedder | None:
    """The embedder, or None when it or ``DATABASE_URL`` is missing (RAG services need both)."""
    embedder = get_embedder()
    if embedder is None:
        return None
    if get_settings().database_url is None:
        log.warning("embedding_service_unavailable", reason="no_database_url")
        return None
    return embedder


def build_rag_ingest_service() -> RagIngestService | None:
    """Persisting ingest service, or None without an embedder or ``DATABASE_URL``."""
    embedder = _get_embedder_with_database()
    if embedder is None:
        return None
    return RagIngestService(get_async_session_factory(), JSONStructuralChunker(), embedder)


@lru_cache
def get_rag_ingest_service() -> RagIngestService | None:
    """Process-wide ingest service (None when not configured)."""
    return build_rag_ingest_service()


def build_semantic_retriever() -> SemanticRetriever | None:
    """Semantic retriever, or None without an embedder or ``DATABASE_URL``."""
    embedder = _get_embedder_with_database()
    if embedder is None:
        return None
    return SemanticRetriever(get_async_session_factory(), embedder)


@lru_cache
def get_semantic_retriever() -> SemanticRetriever | None:
    """Process-wide semantic retriever (None when not configured)."""
    return build_semantic_retriever()


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
        system_prompt_provider=partial(render_system_prompt, version=SESSION_PROMPT_VERSION),
        max_turns=settings.session_max_turns,
        max_sessions=settings.session_max_sessions,
        idle_ttl_seconds=settings.session_idle_ttl_seconds,
    )


@lru_cache
def get_estimation_service() -> EstimationService:
    """Process-wide estimation service built from the shared singletons."""
    return EstimationService(
        get_llm_wrapper(), get_cache(), build_moderation_client(), get_semantic_cache()
    )


def build_history_compactor(wrapper: TextCompleter | None = None) -> HistoryCompactor | None:
    """Compactor, or None when compression is disabled.

    Summaries come from the LLM in ``llm`` mode when a wrapper is given, otherwise
    from the extractive summarizer.
    """
    settings = get_settings()
    if not settings.session_compression_enabled:
        return None
    summarizer: Summarizer = ExtractiveSummarizer(settings.session_summary_max_chars)
    if settings.session_summary_mode == "llm" and wrapper is not None:
        summarizer = LlmSummarizer(
            wrapper, settings.session_summary_max_chars, settings.session_summary_model
        )
    return HistoryCompactor(
        AnchorDetector(),
        summarizer,
        settings.session_max_anchors,
        settings.session_summary_max_chars,
    )


@lru_cache
def get_session_estimation_service() -> SessionEstimationService:
    """Process-wide session estimation service (no cache: it bypasses both caches)."""
    # One metered proxy shared by the actor and the summarizer so a turn's cost adds up.
    wrapper = MeteredLLM(get_llm_wrapper())
    settings = get_settings()
    return SessionEstimationService(
        wrapper,
        build_moderation_client(),
        build_history_compactor(wrapper),
        Critic(wrapper, settings.critic_model),
        settings.acb_max_iterations,
    )
