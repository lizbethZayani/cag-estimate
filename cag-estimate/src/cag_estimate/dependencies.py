"""FastAPI dependency factories (overridable via ``app.dependency_overrides``)."""

from functools import lru_cache
from typing import Any

import structlog
from openai import OpenAI

from cag_estimate.config import get_settings
from cag_estimate.services.cache import get_cache
from cag_estimate.services.estimation import EstimationService
from cag_estimate.services.llm_wrapper import get_llm_wrapper

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


@lru_cache
def get_estimation_service() -> EstimationService:
    """Process-wide estimation service built from the shared singletons."""
    return EstimationService(get_llm_wrapper(), get_cache(), build_moderation_client())
