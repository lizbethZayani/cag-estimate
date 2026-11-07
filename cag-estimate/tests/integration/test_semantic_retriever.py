"""Live tests of the semantic retriever against the compose pgvector container."""

import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from cag_estimate.db.models import EMBEDDING_DIMENSIONS
from cag_estimate.embedding_pipeline.repository import ChunkStore
from cag_estimate.embedding_pipeline.retriever import SemanticRetriever
from cag_estimate.embedding_pipeline.schemas import (
    EmbeddedChunk,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from tests.integration.conftest import SessionFactory

pytestmark = pytest.mark.integration

QUERY_VECTOR = [1.0, 0.0]
# name -> (first two components, hand-computed cosine distance to QUERY_VECTOR)
CHUNK_VECTORS = {
    "same": ([1.0, 0.0], 0.0),
    "diagonal": ([1.0, 1.0], 1 - 1 / math.sqrt(2)),
    "orthogonal": ([0.0, 1.0], 1.0),
    "opposite": ([-1.0, 0.0], 2.0),
}
TOLERANCE = 1e-6


def pad(vector: list[float]) -> list[float]:
    return [*vector, *([0.0] * (EMBEDDING_DIMENSIONS - len(vector)))]


class FixedEmbedder:
    """Maps every query to the same vector; no network."""

    def embed_one(self, text: str) -> list[float]:
        return pad(QUERY_VECTOR)


def make_chunk(name: str, vector: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=f"BUD-1::{name}",
        text=f"component {name}",
        metadata={"component_id": name, "year": 2024, "client_sector": "finance"},
        token_count=2,
        embedding=pad(vector),
    )


async def store_chunks(factory: SessionFactory) -> dict[str, int]:
    """Insert the chunks in a scrambled order; return the chunk id by name."""
    async with factory() as session, session.begin():
        document_id = await ChunkStore.add_document(
            session, source_path="data/budgets/b1.json", document_type="t", metadata={}
        )
        names = ["orthogonal", "same", "opposite", "diagonal"]
        await ChunkStore.add_chunks(
            session,
            document_id=document_id,
            chunks=[make_chunk(name, CHUNK_VECTORS[name][0]) for name in names],
            chunk_type="budget_component",
        )
    hits = await search(factory, k=50)
    return {hit.metadata["component_id"]: hit.chunk_id for hit in hits}


async def search(factory: SessionFactory, k: int) -> list[SearchResultItem]:
    retriever = SemanticRetriever(factory, FixedEmbedder())
    response = await retriever.search(SearchRequest(query="payments", k=k))
    return response.results


async def test_results_are_ordered_by_ascending_cosine_distance(
    session_factory: SessionFactory,
) -> None:
    await store_chunks(session_factory)

    results = await search(session_factory, k=10)

    assert [r.metadata["component_id"] for r in results] == [
        "same",
        "diagonal",
        "orthogonal",
        "opposite",
    ]
    for result in results:
        expected = CHUNK_VECTORS[result.metadata["component_id"]][1]
        assert result.distance == pytest.approx(expected, abs=TOLERANCE)


async def test_k_limits_the_number_of_results(session_factory: SessionFactory) -> None:
    await store_chunks(session_factory)

    results = await search(session_factory, k=2)

    assert [r.metadata["component_id"] for r in results] == ["same", "diagonal"]


async def test_result_fields_and_metadata_round_trip(session_factory: SessionFactory) -> None:
    ids = await store_chunks(session_factory)

    top = (await search(session_factory, k=1))[0]

    assert top.chunk_id == ids["same"]
    assert top.chunk_type == "budget_component"
    assert top.content == "component same"
    assert top.metadata == {"component_id": "same", "year": 2024, "client_sector": "finance"}
    assert top.document_id >= 1


async def test_an_empty_table_returns_no_results(session_factory: SessionFactory) -> None:
    results = await search(session_factory, k=5)

    assert results == []


class FakeReranker:
    """Scores a candidate by its component name; no model, no torch."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def rerank(
        self, query: str, candidates: Sequence[Any], *, top_n: int, text_of: Callable[[Any], str]
    ) -> list[tuple[Any, float]]:
        scored = [(c, self._scores[c.metadata["component_id"]]) for c in candidates]
        return sorted(scored, key=lambda pair: -pair[1])[:top_n]


async def store_keyword_chunk(factory: SessionFactory) -> None:
    """The vector branch ranks 'opposite' last; only it contains the query keyword."""
    keyword_chunk = make_chunk("opposite", CHUNK_VECTORS["opposite"][0])
    keyword_chunk.text = "payments gateway integration"
    async with factory() as session, session.begin():
        document_id = await ChunkStore.add_document(
            session, source_path="data/budgets/b1.json", document_type="t", metadata={}
        )
        others = [make_chunk(n, CHUNK_VECTORS[n][0]) for n in ("orthogonal", "same", "diagonal")]
        await ChunkStore.add_chunks(
            session,
            document_id=document_id,
            chunks=[*others, keyword_chunk],
            chunk_type="budget_component",
        )


async def search_with(
    factory: SessionFactory, reranker: FakeReranker | None = None, **fields: Any
) -> SearchResponse:
    retriever = SemanticRetriever(factory, FixedEmbedder(), reranker=reranker)
    return await retriever.search(SearchRequest(query="payments", **fields))


def components(response: SearchResponse) -> list[str]:
    return [r.metadata["component_id"] for r in response.results]


async def test_hybrid_surfaces_a_keyword_only_chunk_the_vector_branch_ranks_last(
    session_factory: SessionFactory,
) -> None:
    await store_keyword_chunk(session_factory)

    vector = await search_with(session_factory, k=1)
    hybrid = await search_with(session_factory, k=1, search_mode="hybrid")

    assert components(vector) == ["same"]
    assert components(hybrid) == ["opposite"]
    assert hybrid.results[0].distance == pytest.approx(2.0, abs=TOLERANCE)
    assert hybrid.results[0].score is not None
    assert (hybrid.search_mode, hybrid.reranked) == ("hybrid", False)


async def test_reranking_reorders_by_the_fake_scores(session_factory: SessionFactory) -> None:
    await store_keyword_chunk(session_factory)
    reranker = FakeReranker({"same": 0.1, "diagonal": 0.2, "orthogonal": 0.9, "opposite": 0.5})

    response = await search_with(session_factory, reranker, k=2, search_mode="hybrid", rerank=True)

    assert components(response) == ["orthogonal", "opposite"]
    assert [r.score for r in response.results] == [0.9, 0.5]
    assert response.reranked is True


async def test_the_vector_only_default_is_unchanged(session_factory: SessionFactory) -> None:
    await store_keyword_chunk(session_factory)

    response = await search_with(session_factory, k=4)

    assert components(response) == ["same", "diagonal", "orthogonal", "opposite"]
    assert all(r.score is None for r in response.results)
    assert (response.search_mode, response.reranked) == ("vector", False)
