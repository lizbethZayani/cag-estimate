"""Offline tests for POST /search (fake retriever injected via dependency override)."""

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from sqlalchemy.exc import SQLAlchemyError
from structlog.testing import capture_logs

from cag_estimate import dependencies
from cag_estimate.config import Settings
from cag_estimate.dependencies import get_semantic_retriever
from cag_estimate.embedding_pipeline.retriever import SearchEmbeddingError
from cag_estimate.embedding_pipeline.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from cag_estimate.main import app

SECRET = "sk-secret-failure-detail"


class FakeRetriever:
    def __init__(self, error: Exception | None = None, results: int = 1) -> None:
        self.error = error
        self.results = results
        self.received: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchResponse:
        if self.error:
            raise self.error
        self.received.append(request)
        items = [
            SearchResultItem(
                chunk_id=17 + position,
                document_id=3,
                chunk_type="budget_component",
                content="OAuth backend",
                distance=0.2314 + position,
                metadata={"year": 2024},
            )
            for position in range(self.results)
        ]
        return SearchResponse(query=request.query, k=request.k, search_time_ms=123, results=items)


@pytest.fixture
def retriever() -> FakeRetriever:
    return FakeRetriever()


@pytest.fixture(autouse=True)
def overrides(retriever: FakeRetriever) -> Iterator[None]:
    app.dependency_overrides[get_semantic_retriever] = lambda: retriever
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


async def test_search_returns_the_documented_shape(client, retriever):
    response = await client.post("/search", json={"query": "oauth", "k": 3})

    assert response.status_code == 200
    assert response.json() == {
        "query": "oauth",
        "k": 3,
        "search_time_ms": 123,
        "results": [
            {
                "chunk_id": 17,
                "document_id": 3,
                "chunk_type": "budget_component",
                "content": "OAuth backend",
                "distance": 0.2314,
                "metadata": {"year": 2024},
            }
        ],
    }
    assert retriever.received[0].k == 3


async def test_k_defaults_to_five(client, retriever):
    response = await client.post("/search", json={"query": "oauth"})

    assert response.status_code == 200
    assert response.json()["k"] == 5
    assert retriever.received[0].k == 5


async def test_an_empty_corpus_returns_200_with_no_results(client, retriever):
    retriever.results = 0

    response = await client.post("/search", json={"query": "oauth"})

    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"query": "oauth", "k": 0},
        {"query": "oauth", "k": 51},
        {"k": 3},
        {"query": ""},
        {"query": "   "},
        ["not", "an", "object"],
    ],
)
async def test_invalid_bodies_are_rejected_with_422(client, retriever, body):
    response = await client.post("/search", json=body)

    assert response.status_code == 422
    assert retriever.received == []


async def test_embedding_failure_returns_a_generic_502(client, retriever):
    retriever.error = SearchEmbeddingError(SECRET)

    with capture_logs() as logs:
        response = await client.post("/search", json={"query": "oauth"})

    assert response.status_code == 502
    assert SECRET not in response.text
    assert response.json() == {"detail": "The embeddings service failed. Please try again later."}
    event = next(entry for entry in logs if entry["event"] == "search_failed")
    assert event["error_type"] == "SearchEmbeddingError"
    assert event["error"] == SECRET


@pytest.mark.parametrize("error", [SQLAlchemyError(SECRET), RuntimeError(SECRET), OSError(SECRET)])
async def test_database_failures_return_a_generic_503(client, retriever, error):
    retriever.error = error

    response = await client.post("/search", json={"query": "oauth"})

    assert response.status_code == 503
    assert SECRET not in response.text
    assert response.json() == {"detail": "The database is not available."}


async def test_unexpected_failures_return_a_generic_502(client, retriever):
    retriever.error = ValueError(SECRET)

    response = await client.post("/search", json={"query": "oauth"})

    assert response.status_code == 502
    assert SECRET not in response.text


async def test_missing_retriever_returns_503(client, monkeypatch):
    app.dependency_overrides.clear()
    settings = Settings(anthropic_api_key="test", openai_api_key=None)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    dependencies.get_embedder.cache_clear()
    get_semantic_retriever.cache_clear()

    try:
        response = await client.post("/search", json={"query": "oauth"})
    finally:
        dependencies.get_embedder.cache_clear()
        get_semantic_retriever.cache_clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "The search service is not configured."}


def test_search_route_is_in_the_openapi_schema():
    operation = app.openapi()["paths"]["/search"]["post"]

    assert (
        "SearchResponse"
        in operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    )
