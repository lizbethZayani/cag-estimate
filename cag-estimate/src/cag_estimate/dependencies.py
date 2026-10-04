"""FastAPI dependency factories (overridable via ``app.dependency_overrides``)."""

from functools import lru_cache

from cag_estimate.services.cache import get_cache
from cag_estimate.services.estimation import EstimationService
from cag_estimate.services.llm_wrapper import get_llm_wrapper


@lru_cache
def get_estimation_service() -> EstimationService:
    """Process-wide estimation service built from the shared singletons."""
    return EstimationService(get_llm_wrapper(), get_cache())
