"""Embedding pipeline: turns historical budgets into embedded, stored and searchable chunks."""

from cag_estimate.embedding_pipeline.chunker import (
    JSONStructuralChunker,
    TiktokenCounter,
    TokenCounter,
)
from cag_estimate.embedding_pipeline.embedder import (
    EMBEDDING_PRICE_PER_M_TOKENS_USD,
    OpenAIEmbedder,
    cosine_similarity,
    estimate_cost_usd,
)
from cag_estimate.embedding_pipeline.ingest_service import (
    DuplicateDocumentError,
    RagIngestService,
)
from cag_estimate.embedding_pipeline.retriever import SearchEmbeddingError, SemanticRetriever
from cag_estimate.embedding_pipeline.schemas import (
    Budget,
    BudgetComponent,
    Chunk,
    ClientMetadata,
    Complexity,
    EmbeddedChunk,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    Sector,
)

__all__ = [
    "EMBEDDING_PRICE_PER_M_TOKENS_USD",
    "Budget",
    "BudgetComponent",
    "Chunk",
    "ClientMetadata",
    "Complexity",
    "DuplicateDocumentError",
    "EmbeddedChunk",
    "IngestRequest",
    "IngestResponse",
    "JSONStructuralChunker",
    "OpenAIEmbedder",
    "RagIngestService",
    "SearchEmbeddingError",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "Sector",
    "SemanticRetriever",
    "TiktokenCounter",
    "TokenCounter",
    "cosine_similarity",
    "estimate_cost_usd",
]
