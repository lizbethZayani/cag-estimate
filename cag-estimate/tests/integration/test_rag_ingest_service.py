"""Live tests of the persisting ingest against the compose pgvector container."""

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from cag_estimate.db.models import EMBEDDING_DIMENSIONS, ChunkRow, DocumentRow
from cag_estimate.embedding_pipeline.chunker import JSONStructuralChunker
from cag_estimate.embedding_pipeline.ingest_service import (
    DuplicateDocumentError,
    RagIngestService,
)
from cag_estimate.embedding_pipeline.schemas import Chunk, EmbeddedChunk, IngestRequest
from tests.integration.conftest import SessionFactory, scalar

pytestmark = pytest.mark.integration

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "budgets_sample.json"
DOCUMENT_TYPE = "historical_budget"
CHUNK_METADATA_KEYS = {
    "budget_id",
    "component_id",
    "client_sector",
    "main_technology",
    "year",
    "complexity",
    "estimated_hours",
}


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class DeterministicEmbedder:
    """1536-d vectors derived from the chunk position; no network."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if self.error:
            raise self.error
        return [
            EmbeddedChunk(
                **chunk.model_dump(),
                embedding=[(position + 1) / 10, *([0.0] * (EMBEDDING_DIMENSIONS - 1))],
            )
            for position, chunk in enumerate(chunks)
        ]


def load_budgets() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def make_request(budget: dict[str, Any]) -> IngestRequest:
    return IngestRequest(
        source_path=f"data/budgets/{budget['budget_id']}.json",
        document_type=DOCUMENT_TYPE,
        content=budget,
    )


def make_service(
    factory: SessionFactory, embedder: DeterministicEmbedder | None = None
) -> RagIngestService:
    return RagIngestService(
        factory,
        JSONStructuralChunker(token_counter=WordCounter()),
        embedder or DeterministicEmbedder(),
    )


async def test_ingest_stores_the_document_and_one_chunk_per_component(
    session_factory: SessionFactory,
) -> None:
    budget = load_budgets()[0]
    request = make_request(budget)

    response = await make_service(session_factory).ingest(request)

    assert response.chunks_created == len(budget["components"])
    assert response.embedding_dimension == EMBEDDING_DIMENSIONS
    async with session_factory() as session:
        document = await session.get(DocumentRow, response.document_id)
        chunks = (await session.scalars(select(ChunkRow).order_by(ChunkRow.id))).all()
    assert document is not None
    assert document.source_path == request.source_path
    assert document.document_type == DOCUMENT_TYPE
    assert document.metadata_ == {
        "budget_id": budget["budget_id"],
        "client_sector": budget["client_metadata"]["sector"],
        "year": budget["year"],
    }
    assert await scalar(session_factory, "SELECT count(*) FROM documents") == 1
    assert len(chunks) == len(budget["components"])
    assert {chunk.document_id for chunk in chunks} == {response.document_id}
    assert {chunk.chunk_type for chunk in chunks} == {"budget_component"}
    assert all(set(chunk.metadata_) == CHUNK_METADATA_KEYS for chunk in chunks)
    assert all(len(chunk.embedding) == EMBEDDING_DIMENSIONS for chunk in chunks)


async def test_ingesting_the_same_source_path_again_is_a_duplicate(
    session_factory: SessionFactory,
) -> None:
    request = make_request(load_budgets()[0])
    service = make_service(session_factory)
    first = await service.ingest(request)

    with pytest.raises(DuplicateDocumentError) as caught:
        await service.ingest(request)

    assert caught.value.document_id == first.document_id
    assert await scalar(session_factory, "SELECT count(*) FROM documents") == 1
    assert await scalar(session_factory, "SELECT count(*) FROM chunks") == first.chunks_created


async def test_embedder_failure_leaves_no_document_and_no_chunks(
    session_factory: SessionFactory,
) -> None:
    service = make_service(session_factory, DeterministicEmbedder(error=ValueError("boom")))

    with pytest.raises(ValueError, match="boom"):
        await service.ingest(make_request(load_budgets()[0]))

    assert await scalar(session_factory, "SELECT count(*) FROM documents") == 0
    assert await scalar(session_factory, "SELECT count(*) FROM chunks") == 0


async def test_ingesting_the_whole_sample_dataset(session_factory: SessionFactory) -> None:
    service = make_service(session_factory)

    for budget in load_budgets():
        await service.ingest(make_request(budget))

    assert await scalar(session_factory, "SELECT count(*) FROM documents") == 15
    assert await scalar(session_factory, "SELECT count(*) FROM chunks") == 52
