"""Offline tests for the semantic retriever (fake embedder, session factory and store)."""

from typing import Any, Self

import pytest
from structlog.testing import capture_logs

from cag_estimate.embedding_pipeline.repository import ChunkHit
from cag_estimate.embedding_pipeline.retriever import SearchEmbeddingError, SemanticRetriever
from cag_estimate.embedding_pipeline.schemas import SearchRequest

QUERY = "oauth banking backend"
VECTOR = [0.1, 0.2, 0.3]


class FakeSession:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    async def __aenter__(self) -> Self:
        self._journal.append("open")
        return self

    async def __aexit__(self, *_: object) -> None:
        self._journal.append("close")


class FakeEmbedder:
    def __init__(self, journal: list[str], error: Exception | None = None) -> None:
        self._journal = journal
        self.error = error
        self.queries: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self._journal.append("embed")
        self.queries.append(text)
        if self.error:
            raise self.error
        return VECTOR


class FakeStore:
    def __init__(self, journal: list[str], hits: list[ChunkHit]) -> None:
        self._journal = journal
        self.hits = hits
        self.calls: list[tuple[Any, list[float], int]] = []

    async def search(self, session: Any, query_vector: list[float], k: int) -> list[ChunkHit]:
        self._journal.append("search")
        self.calls.append((session, query_vector, k))
        return self.hits


def make_hit(chunk_id: int, distance: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        document_id=3,
        chunk_type="budget_component",
        content=f"content {chunk_id}",
        distance=distance,
        metadata={"year": 2024},
    )


class Harness:
    def __init__(
        self, hits: list[ChunkHit] | None = None, embed_error: Exception | None = None
    ) -> None:
        self.journal: list[str] = []
        self.embedder = FakeEmbedder(self.journal, embed_error)
        self.store = FakeStore(self.journal, hits if hits is not None else [])
        ticks = iter([10.0, 10.123])
        self.retriever = SemanticRetriever(
            lambda: FakeSession(self.journal),
            self.embedder,
            store=self.store,
            clock=lambda: next(ticks),
        )


async def test_the_query_is_embedded_once_before_the_database_is_opened():
    harness = Harness()

    await harness.retriever.search(SearchRequest(query=QUERY))

    assert harness.journal == ["embed", "open", "search", "close"]
    assert harness.embedder.queries == [QUERY]


async def test_the_vector_and_k_are_passed_to_the_store():
    harness = Harness()

    await harness.retriever.search(SearchRequest(query=QUERY, k=7))

    _, vector, k = harness.store.calls[0]
    assert vector == VECTOR
    assert k == 7


async def test_response_keeps_the_store_order_and_reports_the_elapsed_time():
    harness = Harness(hits=[make_hit(5, 0.1), make_hit(2, 0.4)])

    response = await harness.retriever.search(SearchRequest(query=QUERY, k=2))

    assert response.query == QUERY
    assert response.k == 2
    assert response.search_time_ms == 123
    assert [(r.chunk_id, r.distance) for r in response.results] == [(5, 0.1), (2, 0.4)]
    assert response.results[0].model_dump() == {
        "chunk_id": 5,
        "document_id": 3,
        "chunk_type": "budget_component",
        "content": "content 5",
        "distance": 0.1,
        "metadata": {"year": 2024},
    }


async def test_an_empty_corpus_gives_an_empty_result_list():
    response = await Harness(hits=[]).retriever.search(SearchRequest(query=QUERY))

    assert response.results == []


async def test_embedding_failure_raises_search_embedding_error_and_never_opens_the_database():
    harness = Harness(embed_error=ValueError("boom"))

    with pytest.raises(SearchEmbeddingError) as caught:
        await harness.retriever.search(SearchRequest(query=QUERY))

    assert isinstance(caught.value.__cause__, ValueError)
    assert harness.journal == ["embed"]


async def test_completed_log_has_counts_and_latency_but_not_the_query_text():
    harness = Harness(hits=[make_hit(1, 0.1)])

    with capture_logs() as logs:
        await harness.retriever.search(SearchRequest(query=QUERY, k=3))

    event = next(entry for entry in logs if entry["event"] == "rag_search_completed")
    assert event["k"] == 3
    assert event["results"] == 1
    assert event["latency_ms"] == 123
    assert event["query_length"] == len(QUERY)
    assert QUERY not in repr(logs)
