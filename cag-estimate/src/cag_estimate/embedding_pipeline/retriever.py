"""Semantic search: embed the query, then rank stored chunks by cosine distance.

The query text is never logged, only its length.
"""

import asyncio
import time
from collections.abc import Callable

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from cag_estimate.embedding_pipeline.embedder import OpenAIEmbedder
from cag_estimate.embedding_pipeline.repository import ChunkHit, ChunkStore
from cag_estimate.embedding_pipeline.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

log = structlog.get_logger()

MS_PER_SECOND = 1000


class SearchEmbeddingError(Exception):
    """Embedding the query failed; the cause is chained and never shown to clients."""


def _to_item(hit: ChunkHit) -> SearchResultItem:
    return SearchResultItem(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        chunk_type=hit.chunk_type,
        content=hit.content,
        distance=hit.distance,
        metadata=hit.metadata,
    )


class SemanticRetriever:
    """Embeds a query and returns its ``k`` nearest chunks."""

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        embedder: OpenAIEmbedder,
        store: type[ChunkStore] = ChunkStore,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._store = store
        self._clock = clock

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Return the nearest chunks; raises ``SearchEmbeddingError`` if embedding fails."""
        started = self._clock()
        vector = await self._embed(request.query)
        async with self._session_factory() as session:
            hits = await self._store.search(session, vector, request.k)
        latency_ms = round((self._clock() - started) * MS_PER_SECOND)
        log.info(
            "rag_search_completed",
            k=request.k,
            results=len(hits),
            query_length=len(request.query),
            latency_ms=latency_ms,
        )
        return SearchResponse(
            query=request.query,
            k=request.k,
            search_time_ms=latency_ms,
            results=[_to_item(hit) for hit in hits],
        )

    async def _embed(self, query: str) -> list[float]:
        # The embedder is synchronous: run it off the event loop, before any DB session.
        try:
            return await asyncio.to_thread(self._embedder.embed_one, query)
        except Exception as exc:
            raise SearchEmbeddingError("the query could not be embedded") from exc
