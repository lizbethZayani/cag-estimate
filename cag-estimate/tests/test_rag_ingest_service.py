"""Offline tests for the persisting ingest orchestration (fake session, store and embedder)."""

from typing import Any, Self

import pytest
from structlog.testing import capture_logs

from cag_estimate.db.models import EMBEDDING_DIMENSIONS
from cag_estimate.embedding_pipeline.chunker import JSONStructuralChunker
from cag_estimate.embedding_pipeline.ingest_service import (
    DuplicateDocumentError,
    RagIngestService,
)
from cag_estimate.embedding_pipeline.schemas import Chunk, EmbeddedChunk, IngestRequest

SECRET_TEXT = "Auth backend"


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class FakeTransaction:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    async def __aenter__(self) -> Self:
        self._journal.append("begin")
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        self._journal.append("rollback" if exc_type else "commit")


class FakeSession:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self._journal)


class FakeStore:
    def __init__(self, journal: list[str], existing_id: int | None = None) -> None:
        self.journal = journal
        self.existing_id = existing_id
        self.document_args: dict[str, Any] = {}
        self.chunk_args: dict[str, Any] = {}

    async def find_document_id(self, session: Any, source_path: str) -> int | None:
        self.journal.append("exists")
        return self.existing_id

    async def add_document(self, session: Any, **kwargs: Any) -> int:
        self.journal.append("add_document")
        self.document_args = kwargs
        return 7

    async def add_chunks(self, session: Any, **kwargs: Any) -> int:
        self.journal.append("add_chunks")
        self.chunk_args = kwargs
        return len(kwargs["chunks"])


class RecordingChunker:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal
        self._real = JSONStructuralChunker(token_counter=WordCounter())

    def chunk(self, budgets: list[Any]) -> list[Chunk]:
        self._journal.append("chunk")
        return self._real.chunk(budgets)


class FakeEmbedder:
    def __init__(self, journal: list[str], error: Exception | None = None) -> None:
        self._journal = journal
        self.error = error
        self.batches: list[list[Chunk]] = []

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self._journal.append("embed")
        self.batches.append(chunks)
        if self.error:
            raise self.error
        return [EmbeddedChunk(**c.model_dump(), embedding=[0.5] * 3) for c in chunks]


def make_request(component_ids: tuple[str, ...] = ("A", "B", "C")) -> IngestRequest:
    return IngestRequest.model_validate(
        {
            "source_path": "data/budgets/b1.json",
            "document_type": "historical_budget",
            "content": {
                "budget_id": "BUD-1",
                "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
                "project_summary": "Mobile banking API",
                "main_technology": "ruby_on_rails",
                "year": 2024,
                "total_estimated_hours": 30,
                "components": [
                    {
                        "component_id": component_id,
                        "name": SECRET_TEXT,
                        "description": "OAuth flows.",
                        "estimated_hours": 10,
                        "complexity": "high",
                    }
                    for component_id in component_ids
                ],
            },
        }
    )


class Harness:
    def __init__(
        self, existing_id: int | None = None, embed_error: Exception | None = None
    ) -> None:
        self.journal: list[str] = []
        self.store = FakeStore(self.journal, existing_id)
        self.embedder = FakeEmbedder(self.journal, embed_error)
        ticks = iter([10.0, 11.24])
        self.service = RagIngestService(
            lambda: FakeSession(self.journal),
            RecordingChunker(self.journal),
            self.embedder,
            store=self.store,
            clock=lambda: next(ticks),
        )


async def test_ingest_runs_the_steps_in_order_inside_one_transaction():
    harness = Harness()

    await harness.service.ingest(make_request())

    assert harness.journal == [
        "begin",
        "exists",
        "add_document",
        "chunk",
        "embed",
        "add_chunks",
        "commit",
    ]


async def test_all_chunks_are_embedded_in_a_single_batched_call():
    harness = Harness()

    await harness.service.ingest(make_request())

    assert len(harness.embedder.batches) == 1
    assert [c.chunk_id for c in harness.embedder.batches[0]] == ["BUD-1::A", "BUD-1::B", "BUD-1::C"]


async def test_document_and_chunks_are_persisted_with_their_metadata():
    harness = Harness()

    await harness.service.ingest(make_request())

    assert harness.store.document_args == {
        "source_path": "data/budgets/b1.json",
        "document_type": "historical_budget",
        "metadata": {"budget_id": "BUD-1", "client_sector": "finance", "year": 2024},
    }
    assert harness.store.chunk_args["document_id"] == 7
    assert harness.store.chunk_args["chunk_type"] == "budget_component"
    assert len(harness.store.chunk_args["chunks"]) == 3


async def test_response_reports_the_document_chunks_dimension_and_elapsed_time():
    harness = Harness()

    response = await harness.service.ingest(make_request())

    assert response.model_dump() == {
        "document_id": 7,
        "chunks_created": 3,
        "embedding_dimension": EMBEDDING_DIMENSIONS,
        "ingestion_time_ms": 1240,
    }


async def test_duplicate_source_path_raises_and_touches_nothing_else():
    harness = Harness(existing_id=42)

    with capture_logs() as logs, pytest.raises(DuplicateDocumentError) as caught:
        await harness.service.ingest(make_request())

    assert caught.value.document_id == 42
    assert harness.journal == ["begin", "exists", "rollback"]
    assert any(entry["event"] == "embeddings_ingest_duplicate" for entry in logs)


async def test_embedder_failure_propagates_and_rolls_back_the_transaction():
    harness = Harness(embed_error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await harness.service.ingest(make_request())

    assert harness.journal[-1] == "rollback"
    assert "commit" not in harness.journal
    assert "add_chunks" not in harness.journal


async def test_persisted_log_has_counts_and_cost_but_no_text():
    harness = Harness()

    with capture_logs() as logs:
        await harness.service.ingest(make_request())

    event = next(entry for entry in logs if entry["event"] == "rag_ingest_persisted")
    assert event["document_id"] == 7
    assert event["chunks"] == 3
    assert event["tokens"] == sum(c.token_count for c in harness.embedder.batches[0])
    assert event["estimated_cost_usd"] == pytest.approx(event["tokens"] / 1_000_000 * 0.02)
    assert event["latency_ms"] == 1240
    assert SECRET_TEXT not in repr(logs)
