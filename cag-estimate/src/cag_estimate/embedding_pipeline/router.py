"""HTTP layer of the embedding pipeline.

The route is ``/embeddings/ingest`` (no ``/api/v1`` prefix) and an embeddings API
failure is a 500, both on purpose: they follow the exercise brief, not this
project's usual conventions.
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException

from cag_estimate.dependencies import get_embedding_service
from cag_estimate.embedding_pipeline.schemas import IngestRequest, IngestResponse
from cag_estimate.embedding_pipeline.service import EmbeddingIngestService

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

EMBEDDINGS_FAILED = "The embeddings service failed. Please try again later."
EMBEDDINGS_NOT_CONFIGURED = "The embeddings service is not configured."


def _require_service(
    service: Annotated[EmbeddingIngestService | None, Depends(get_embedding_service)],
) -> EmbeddingIngestService:
    if service is None:
        raise HTTPException(status_code=503, detail=EMBEDDINGS_NOT_CONFIGURED)
    return service


Service = Annotated[EmbeddingIngestService, Depends(_require_service)]


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, service: Service) -> IngestResponse:
    """Chunk the budgets, embed every chunk and return the vectors with run stats.

    Responses:
        200: ``chunks`` (id, text, metadata, token_count, embedding) and ``stats``
            (total_budgets, total_chunks, total_tokens, estimated_cost_usd).
        422: invalid body (FastAPI validation), e.g. an empty ``budgets`` list.
        500: the embeddings API failed; the body is generic, details go to the logs.
        503: the service is not configured (no OpenAI API key).

    The vectors are returned, NOT persisted: storage comes in a later session.
    """
    try:
        return service.ingest(request.budgets)
    except Exception as exc:
        log.error("ingest_failed", error_type=type(exc).__name__, error=str(exc))
        raise HTTPException(status_code=500, detail=EMBEDDINGS_FAILED) from exc
