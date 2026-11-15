"""Offline tests for POST /api/v1/estimate/rag (fake service injected via dependency override)."""

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from sqlalchemy.exc import SQLAlchemyError

from cag_estimate.dependencies import get_rag_estimation_service
from cag_estimate.embedding_pipeline.retriever import SearchEmbeddingError
from cag_estimate.main import app
from cag_estimate.services.citations import verify_citations
from cag_estimate.services.rag_context import to_retrieved_chunks
from cag_estimate.services.rag_errors import (
    DanglingCitationError,
    RagGenerationError,
    RagOutputInvalidError,
)
from cag_estimate.services.rag_estimation import RagEstimationResult
from tests.rag_fakes import make_estimate, make_item

URL = "/api/v1/estimate/rag"
SECRET = "sk-secret-failure-detail"
CHUNK_ID = "BUD-1::CART-001"


class FakeService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.queries: list[str] = []

    async def estimate(self, query: str, request_id: str | None = None) -> RagEstimationResult:
        if self.error:
            raise self.error
        self.queries.append(query)
        estimate = make_estimate(CHUNK_ID)
        return RagEstimationResult(
            estimate=estimate,
            citation_report=verify_citations(estimate, {CHUNK_ID}),
            chunks=to_retrieved_chunks([make_item()]),
            prompt_version="v1",
        )


def _dangling_error() -> DanglingCitationError:
    report = verify_citations(make_estimate("BUD-2099-999::FAKE-001"), {CHUNK_ID})
    return DanglingCitationError(report)


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture(autouse=True)
def overrides(service: FakeService) -> Iterator[None]:
    app.dependency_overrides[get_rag_estimation_service] = lambda: service
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


async def test_returns_the_cited_estimate(client, service):
    response = await client.post(URL, json={"query": "  An online store  "})

    assert response.status_code == 200
    body = response.json()
    line = body["estimate"]["lines"][0]
    assert line["grounded"] is True
    assert line["sources"][0]["chunk_id"] == CHUNK_ID
    assert body["estimate"]["total_hours"] == 140.0
    assert body["citation_report"]["grounded_lines"] == 1
    assert body["citation_report"]["dangling_lines"] == 0
    assert body["prompt_version"] == "v1"
    assert body["retrieved_chunk_ids"] == [CHUNK_ID]
    assert service.queries == ["An online store"]
    assert response.headers["X-Request-ID"]


async def test_a_client_request_id_is_echoed(client):
    response = await client.post(URL, json={"query": "store"}, headers={"X-Request-ID": "abc-123"})

    assert response.headers["X-Request-ID"] == "abc-123"


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "   "}, {"query": "x" * 20001}, {}])
async def test_invalid_bodies_are_rejected_with_422(client, service, body):
    response = await client.post(URL, json=body)

    assert response.status_code == 422
    assert service.queries == []


async def test_a_missing_body_is_rejected_with_422(client, service):
    response = await client.post(URL)

    assert response.status_code == 422
    assert service.queries == []


async def test_an_unconfigured_service_returns_503(client):
    app.dependency_overrides[get_rag_estimation_service] = lambda: None

    response = await client.post(URL, json={"query": "store"})

    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "rag_not_configured"


@pytest.mark.parametrize(
    ("error", "status", "reason"),
    [
        (_dangling_error(), 502, "citation_verification_failed"),
        (RagOutputInvalidError(SECRET), 502, "generation_failed"),
        (RagGenerationError(SECRET), 502, "generation_failed"),
        (SearchEmbeddingError(SECRET), 502, "embeddings_failed"),
        (SQLAlchemyError(SECRET), 503, "database_unavailable"),
        (OSError(SECRET), 503, "database_unavailable"),
    ],
)
async def test_failures_map_to_a_status_and_reason_without_leaking(
    client, service, error, status, reason
):
    service.error = error

    response = await client.post(URL, json={"query": "store"})

    assert response.status_code == status
    assert response.json()["detail"]["reason"] == reason
    assert response.json()["detail"]["message"]
    assert SECRET not in response.text
    assert str(error) not in response.text
    assert "FAKE-001" not in response.text


def test_the_route_is_in_the_openapi_schema():
    operation = app.openapi()["paths"][URL]["post"]

    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert "RagEstimateResponse" in schema["$ref"]


def test_the_legacy_estimate_route_is_still_registered():
    assert "/api/v1/estimate" in app.openapi()["paths"]
