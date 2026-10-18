"""Ingestion orchestration: budgets -> chunks -> embeddings -> stats."""

from cag_estimate.embedding_pipeline.chunker import JSONStructuralChunker
from cag_estimate.embedding_pipeline.embedder import OpenAIEmbedder, build_ingest_stats
from cag_estimate.embedding_pipeline.schemas import Budget, IngestResponse


class EmbeddingIngestService:
    """Chunks historical budgets and embeds every chunk; nothing is persisted."""

    def __init__(self, chunker: JSONStructuralChunker, embedder: OpenAIEmbedder) -> None:
        self._chunker = chunker
        self._embedder = embedder

    def ingest(self, budgets: list[Budget]) -> IngestResponse:
        """Embedded chunks in input order plus the run's totals; embedder errors propagate."""
        chunks = self._embedder.embed_many(self._chunker.chunk(budgets))
        return IngestResponse(chunks=chunks, stats=build_ingest_stats(len(budgets), chunks))
