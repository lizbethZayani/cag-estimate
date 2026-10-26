"""HTTP layer of the embedding pipeline.

The route is ``/embeddings/ingest`` (no ``/api/v1`` prefix), a duplicate is a 409 with
a top-level ``document_id`` and an embeddings API failure is a 500, all on purpose:
they follow the exercise brief, not this project's usual conventions.
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from cag_estimate.dependencies import get_rag_ingest_service
from cag_estimate.embedding_pipeline.ingest_service import (
    DuplicateDocumentError,
    RagIngestService,
)
from cag_estimate.embedding_pipeline.schemas import IngestRequest, IngestResponse

log = structlog.get_logger()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

EMBEDDINGS_FAILED = "The embeddings service failed. Please try again later."
EMBEDDINGS_NOT_CONFIGURED = "The embeddings service is not configured."
DATABASE_UNAVAILABLE = "The database is not available."
ALREADY_INGESTED = "Document already ingested"
# SQLAlchemyError: query/connection failures; RuntimeError: DATABASE_URL not set;
# OSError: the database host is unreachable.
DATABASE_ERRORS = (SQLAlchemyError, RuntimeError, OSError)


def _require_service(
    service: Annotated[RagIngestService | None, Depends(get_rag_ingest_service)],
) -> RagIngestService:
    if service is None:
        raise HTTPException(status_code=503, detail=EMBEDDINGS_NOT_CONFIGURED)
    return service


Service = Annotated[RagIngestService, Depends(_require_service)]


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, service: Service) -> IngestResponse | JSONResponse:
    """Chunk one budget, embed its chunks and store document and chunks in one transaction.

    Responses:
        200: ``document_id``, ``chunks_created``, ``embedding_dimension``,
            ``ingestion_time_ms``.
        409: a document with that ``source_path`` exists; the body is exactly
            ``{"detail": "Document already ingested", "document_id": <id>}``.
        422: invalid body (FastAPI validation), e.g. an invalid budget.
        500: the embeddings API failed; the body is generic, details go to the logs.
        503: the service is not configured (no OpenAI key or ``DATABASE_URL``) or the
            database is not available.
    """
    try:
        return await service.ingest(request)
    except DuplicateDocumentError as exc:
        return JSONResponse(
            status_code=409, content={"detail": ALREADY_INGESTED, "document_id": exc.document_id}
        )
    except Exception as exc:
        log.error("ingest_failed", error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, DATABASE_ERRORS):
            raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE) from exc
        raise HTTPException(status_code=500, detail=EMBEDDINGS_FAILED) from exc
