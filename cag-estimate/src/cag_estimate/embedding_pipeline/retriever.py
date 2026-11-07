"""Semantic search: embed the query, recall chunks by vector and/or keywords, optionally rerank.

The query text and the documents are never logged, only the query length.
"""

import asyncio
import time
from collections.abc import Callable

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from cag_estimate.embedding_pipeline.embedder import OpenAIEmbedder
from cag_estimate.embedding_pipeline.fusion import reciprocal_rank_fusion
from cag_estimate.embedding_pipeline.repository import ChunkHit, ChunkStore
from cag_estimate.embedding_pipeline.reranker import CrossEncoderReranker
from cag_estimate.embedding_pipeline.schemas import (
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

log = structlog.get_logger()

MS_PER_SECOND = 1000
DEFAULT_RECALL_K = 50
DEFAULT_RRF_K = 60

# A hit and the score of the step that ranked it (None for a plain vector search).
ScoredHit = tuple[ChunkHit, float | None]


class SearchEmbeddingError(Exception):
    """Embedding the query failed; the cause is chained and never shown to clients."""


def _to_item(hit: ChunkHit, score: float | None) -> SearchResultItem:
    return SearchResultItem(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        chunk_type=hit.chunk_type,
        content=hit.content,
        distance=hit.distance,
        score=score,
        metadata=hit.metadata,
    )


class SemanticRetriever:
    """Embeds a query and returns its ``k`` best chunks.

    The request picks the strategy; a missing request field falls back to the configured
    default. Reranking fails soft: the pre-rerank order is returned and ``reranked`` is False.
    """

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        embedder: OpenAIEmbedder,
        store: type[ChunkStore] = ChunkStore,
        clock: Callable[[], float] = time.perf_counter,
        *,
        default_search_mode: SearchMode = "vector",
        default_rerank: bool = False,
        recall_k: int = DEFAULT_RECALL_K,
        rrf_k: int = DEFAULT_RRF_K,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._store = store
        self._clock = clock
        self._default_search_mode = default_search_mode
        self._default_rerank = default_rerank
        self._recall_k = recall_k
        self._rrf_k = rrf_k
        self._reranker = reranker

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Return the best chunks; raises ``SearchEmbeddingError`` if embedding fails."""
        started = self._clock()
        mode = request.search_mode or self._default_search_mode
        rerank = self._default_rerank if request.rerank is None else request.rerank
        vector = await self._embed(request.query)
        async with self._session_factory() as session:
            candidates = await self._recall(session, request, vector, mode, rerank)
        ranked, reranked = await self._finalize(request, candidates, rerank)
        latency_ms = round((self._clock() - started) * MS_PER_SECOND)
        log.info(
            "rag_search_completed",
            k=request.k,
            search_mode=mode,
            reranked=reranked,
            candidates=len(candidates),
            results=len(ranked),
            query_length=len(request.query),
            latency_ms=latency_ms,
        )
        return SearchResponse(
            query=request.query,
            k=request.k,
            search_time_ms=latency_ms,
            search_mode=mode,
            reranked=reranked,
            results=[_to_item(hit, score) for hit, score in ranked],
        )

    async def _embed(self, query: str) -> list[float]:
        # The embedder is synchronous: run it off the event loop, before any DB session.
        try:
            return await asyncio.to_thread(self._embedder.embed_one, query)
        except Exception as exc:
            raise SearchEmbeddingError("the query could not be embedded") from exc

    async def _recall(
        self,
        session: AsyncSession,
        request: SearchRequest,
        vector: list[float],
        mode: SearchMode,
        rerank: bool,
    ) -> list[ScoredHit]:
        """Candidates in pre-rerank order: ``k`` of them, or a larger pool to rerank."""
        pool = max(self._recall_k, request.k)
        wanted = pool if rerank else request.k
        if mode == "vector":
            hits = await self._store.search(session, vector, wanted)
            return [(hit, None) for hit in hits]
        # One AsyncSession cannot run concurrent operations: the branches run in turn.
        vector_hits = await self._store.search(session, vector, pool)
        lexical_hits = await self._store.search_lexical(session, request.query, pool)
        return self._fuse(vector_hits, lexical_hits)[:wanted]

    def _fuse(self, vector_hits: list[ChunkHit], lexical_hits: list[ChunkHit]) -> list[ScoredHit]:
        # The vector hit wins on a shared chunk: only it carries the cosine distance.
        by_id = {hit.chunk_id: hit for hit in [*lexical_hits, *vector_hits]}
        fused = reciprocal_rank_fusion(
            [[hit.chunk_id for hit in vector_hits], [hit.chunk_id for hit in lexical_hits]],
            k=self._rrf_k,
        )
        return [(by_id[chunk_id], score) for chunk_id, score in fused]

    async def _finalize(
        self, request: SearchRequest, candidates: list[ScoredHit], rerank: bool
    ) -> tuple[list[ScoredHit], bool]:
        """The final ``k`` results and whether they were reranked."""
        if rerank:
            reranked = await self._rerank(request, candidates)
            if reranked is not None:
                return reranked, True
        return candidates[: request.k], False

    async def _rerank(
        self, request: SearchRequest, candidates: list[ScoredHit]
    ) -> list[ScoredHit] | None:
        """Rerank by cross-encoder score, or None (after a warning) when it cannot."""
        try:
            if self._reranker is None:
                raise RuntimeError("no reranker is configured")
            # The cross-encoder is CPU bound: keep it off the event loop.
            return await asyncio.to_thread(
                self._reranker.rerank,
                request.query,
                [hit for hit, _ in candidates],
                top_n=request.k,
                text_of=lambda hit: hit.content,
            )
        except Exception as exc:  # noqa: BLE001 - reranking is optional, fail soft
            log.warning("rerank_failed", error_type=type(exc).__name__)
            return None
