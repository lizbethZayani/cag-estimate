"""HTTP layer of the semantic search: ``POST /search`` (no ``/api/v1`` prefix, per the brief)."""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException

from cag_estimate.dependencies import get_semantic_retriever
from cag_estimate.embedding_pipeline.retriever import SemanticRetriever
from cag_estimate.embedding_pipeline.router import (
    DATABASE_ERRORS,
    DATABASE_UNAVAILABLE,
    EMBEDDINGS_FAILED,
)
from cag_estimate.embedding_pipeline.schemas import SearchRequest, SearchResponse

log = structlog.get_logger()

router = APIRouter(tags=["search"])

SEARCH_NOT_CONFIGURED = "The search service is not configured."


def _require_retriever(
    retriever: Annotated[SemanticRetriever | None, Depends(get_semantic_retriever)],
) -> SemanticRetriever:
    if retriever is None:
        raise HTTPException(status_code=503, detail=SEARCH_NOT_CONFIGURED)
    return retriever


Retriever = Annotated[SemanticRetriever, Depends(_require_retriever)]


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, retriever: Retriever) -> SearchResponse:
    """Embed the query and return the ``k`` chunks nearest by cosine distance.

    Responses:
        200: ``query``, ``k``, ``search_time_ms`` and ``results`` ordered by ascending
            distance (empty when nothing is stored).
        422: invalid body (FastAPI validation).
        502: the embeddings API failed (or any other unexpected error, as this repo
            maps unexpected failures to 502); the body is generic, details go to the logs.
        503: the service is not configured or the database is not available.
    """
    try:
        return await retriever.search(request)
    except Exception as exc:
        log.error("search_failed", error_type=type(exc).__name__, error=str(exc))
        if isinstance(exc, DATABASE_ERRORS):
            raise HTTPException(status_code=503, detail=DATABASE_UNAVAILABLE) from exc
        raise HTTPException(status_code=502, detail=EMBEDDINGS_FAILED) from exc
