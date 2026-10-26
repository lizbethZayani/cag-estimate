"""Persisting ingest: one budget -> document row -> chunks -> embeddings -> chunk rows.

Everything happens in ONE transaction, so a failure at any step (the embeddings API
included) rolls the document row back and leaves no orphans. Texts and API keys are
never logged: only ids, counts, tokens, cost and latency.
"""

import asyncio
import time
from collections.abc import Callable

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from cag_estimate.db.models import EMBEDDING_DIMENSIONS
from cag_estimate.embedding_pipeline.chunker import JSONStructuralChunker
from cag_estimate.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from cag_estimate.embedding_pipeline.repository import ChunkStore
from cag_estimate.embedding_pipeline.schemas import Budget, IngestRequest, IngestResponse

log = structlog.get_logger()

CHUNK_TYPE = "budget_component"
MS_PER_SECOND = 1000


class DuplicateDocumentError(Exception):
    """A document with the same ``source_path`` is already stored."""

    def __init__(self, document_id: int) -> None:
        super().__init__(f"document {document_id} is already ingested")
        self.document_id = document_id


def _document_metadata(budget: Budget) -> dict[str, str | int]:
    return {
        "budget_id": budget.budget_id,
        "client_sector": budget.client_metadata.sector,
        "year": budget.year,
    }


class RagIngestService:
    """Chunks one budget, embeds it in one batched call and stores it atomically."""

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        chunker: JSONStructuralChunker,
        embedder: OpenAIEmbedder,
        store: type[ChunkStore] = ChunkStore,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._session_factory = session_factory
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._clock = clock

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        """Persist the budget; raises ``DuplicateDocumentError`` for a known ``source_path``."""
        started = self._clock()
        async with self._session_factory() as session, session.begin():
            document_id = await self._add_document(session, request)
            chunks_created, tokens = await self._add_chunks(session, document_id, request.content)
        latency_ms = round((self._clock() - started) * MS_PER_SECOND)
        log.info(
            "rag_ingest_persisted",
            document_id=document_id,
            chunks=chunks_created,
            tokens=tokens,
            estimated_cost_usd=estimate_cost_usd(tokens),
            latency_ms=latency_ms,
        )
        return IngestResponse(
            document_id=document_id,
            chunks_created=chunks_created,
            embedding_dimension=EMBEDDING_DIMENSIONS,
            ingestion_time_ms=latency_ms,
        )

    async def _add_document(self, session: AsyncSession, request: IngestRequest) -> int:
        existing_id = await self._store.find_document_id(session, request.source_path)
        if existing_id is not None:
            log.info("embeddings_ingest_duplicate", document_id=existing_id)
            raise DuplicateDocumentError(existing_id)
        return await self._store.add_document(
            session,
            source_path=request.source_path,
            document_type=request.document_type,
            metadata=_document_metadata(request.content),
        )

    async def _add_chunks(
        self, session: AsyncSession, document_id: int, budget: Budget
    ) -> tuple[int, int]:
        chunks = self._chunker.chunk([budget])
        # The embedder is synchronous: run it off the event loop, once for all chunks.
        embedded = await asyncio.to_thread(self._embedder.embed_many, chunks)
        await self._store.add_chunks(
            session, document_id=document_id, chunks=embedded, chunk_type=CHUNK_TYPE
        )
        return len(embedded), sum(chunk.token_count for chunk in embedded)
