"""HTTP layer of the cited RAG estimate: ``POST /api/v1/estimate/rag``."""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException

from cag_estimate.dependencies import get_rag_estimation_service
from cag_estimate.embedding_pipeline.retriever import SearchEmbeddingError
from cag_estimate.embedding_pipeline.router import (
    DATABASE_ERRORS,
    DATABASE_UNAVAILABLE,
    EMBEDDINGS_FAILED,
)
from cag_estimate.schemas.rag_estimation_api import RagEstimateRequest, RagEstimateResponse
from cag_estimate.services.rag_errors import DanglingCitationError
from cag_estimate.services.rag_estimation import RagEstimationService

log = structlog.get_logger()

router = APIRouter(prefix="/api/v1", tags=["estimations"])

RAG_NOT_CONFIGURED = "The RAG estimation service is not configured."
CITATION_VERIFICATION_FAILED = "The estimate cited sources that were not retrieved. Please retry."
GENERATION_FAILED = "The estimate could not be generated. Please try again later."


def _detail(reason: str, message: str) -> dict[str, str]:
    return {"reason": reason, "message": message}


def _require_service(
    service: Annotated[RagEstimationService | None, Depends(get_rag_estimation_service)],
) -> RagEstimationService:
    if service is None:
        raise HTTPException(status_code=503, detail=_detail("rag_not_configured", RAG_NOT_CONFIGURED))
    return service


Service = Annotated[RagEstimationService, Depends(_require_service)]


def _to_http_error(exc: Exception) -> HTTPException:
    """Map a failure to a generic HTTP error; the exception text never reaches the client."""
    if isinstance(exc, DanglingCitationError):
        detail = _detail("citation_verification_failed", CITATION_VERIFICATION_FAILED)
        return HTTPException(status_code=502, detail=detail)
    if isinstance(exc, SearchEmbeddingError):
        return HTTPException(status_code=502, detail=_detail("embeddings_failed", EMBEDDINGS_FAILED))
    if isinstance(exc, DATABASE_ERRORS):
        detail = _detail("database_unavailable", DATABASE_UNAVAILABLE)
        return HTTPException(status_code=503, detail=detail)
    return HTTPException(status_code=502, detail=_detail("generation_failed", GENERATION_FAILED))


@router.post("/estimate/rag", response_model=RagEstimateResponse)
async def estimate_rag(request: RagEstimateRequest, service: Service) -> RagEstimateResponse:
    """Estimate a project from the retrieved history, with per-line verifiable citations.

    Responses:
        200: the estimate, its citation report, the prompt version and the retrieved chunk ids.
        422: invalid body (FastAPI validation).
        502: generation failed, the answer cited sources that were not retrieved, or the
            embeddings API failed; the body is generic, details go to the logs.
        503: the service is not configured or the database is not available.
    """
    try:
        result = await service.estimate(request.query)
    except Exception as exc:
        log.error("rag_estimate_failed", error_type=type(exc).__name__)
        raise _to_http_error(exc) from exc
    return RagEstimateResponse(
        estimate=result.estimate,
        citation_report=result.citation_report,
        prompt_version=result.prompt_version,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in result.chunks],
    )
