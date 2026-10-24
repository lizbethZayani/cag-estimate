"""Offline tests for POST /embeddings/ingest (fake service injected via dependency override)."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from sqlalchemy.exc import SQLAlchemyError
from structlog.testing import capture_logs

from cag_estimate import dependencies
from cag_estimate.config import Settings
from cag_estimate.dependencies import get_rag_ingest_service
from cag_estimate.embedding_pipeline.ingest_service import DuplicateDocumentError
from cag_estimate.embedding_pipeline.schemas import IngestRequest, IngestResponse
from cag_estimate.main import app

SECRET = "sk-secret-failure-detail"
SOURCE_PATH = "data/budgets/budget_2024_q1_fintech.json"


def make_budget_payload(budget_id: str = "BUD-1") -> dict[str, Any]:
    return {
        "budget_id": budget_id,
        "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
        "project_summary": "Mobile banking API",
        "main_technology": "ruby_on_rails",
        "year": 2024,
        "total_estimated_hours": 480,
        "components": [
            {
                "component_id": "AUTH-001",
                "name": "OAuth backend",
                "description": "OAuth 2.0 flows.",
                "estimated_hours": 120,
                "complexity": "high",
            }
        ],
    }


def make_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "source_path": SOURCE_PATH,
        "document_type": "historical_budget",
        "content": make_budget_payload(),
    }
    body.update(overrides)
    return body


class FakeService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.received: list[IngestRequest] = []

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        if self.error:
            raise self.error
        self.received.append(request)
        return IngestResponse(
            document_id=42, chunks_created=17, embedding_dimension=1536, ingestion_time_ms=1240
        )


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture(autouse=True)
def overrides(service: FakeService) -> Iterator[None]:
    app.dependency_overrides[get_rag_ingest_service] = lambda: service
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


async def test_ingest_returns_the_documented_shape(client, service):
    response = await client.post("/embeddings/ingest", json=make_body())

    assert response.status_code == 200
    assert response.json() == {
        "document_id": 42,
        "chunks_created": 17,
        "embedding_dimension": 1536,
        "ingestion_time_ms": 1240,
    }
    assert service.received[0].source_path == SOURCE_PATH
    assert service.received[0].content.budget_id == "BUD-1"


async def test_duplicate_returns_409_with_the_literal_top_level_body(client, service):
    service.error = DuplicateDocumentError(42)

    response = await client.post("/embeddings/ingest", json=make_body())

    assert response.status_code == 409
    assert response.json() == {"detail": "Document already ingested", "document_id": 42}


def invalid_budget(**changes: Any) -> dict[str, Any]:
    return {**make_budget_payload(), **changes}


@pytest.mark.parametrize(
    "body",
    [
        {"source_path": SOURCE_PATH, "document_type": "historical_budget"},
        make_body(content=invalid_budget(components=[])),
        make_body(
            content=invalid_budget(
                client_metadata={"name": "X", "sector": "retail", "country": "ES"}
            )
        ),
        make_body(source_path=""),
        make_body(document_type=""),
        {"budgets": [make_budget_payload()]},
        ["not", "an", "object"],
    ],
)
async def test_invalid_bodies_are_rejected_with_422(client, body):
    response = await client.post("/embeddings/ingest", json=body)

    assert response.status_code == 422


async def test_embeddings_failure_returns_a_generic_500(client, service):
    service.error = ValueError(SECRET)

    with capture_logs() as logs:
        response = await client.post("/embeddings/ingest", json=make_body())

    assert response.status_code == 500
    assert SECRET not in response.text
    assert response.json() == {"detail": "The embeddings service failed. Please try again later."}
    event = next(entry for entry in logs if entry["event"] == "ingest_failed")
    assert event["error_type"] == "ValueError"
    assert event["error"] == SECRET


@pytest.mark.parametrize("error", [SQLAlchemyError(SECRET), OSError(SECRET)])
async def test_database_failures_return_a_generic_503(client, service, error):
    service.error = error

    response = await client.post("/embeddings/ingest", json=make_body())

    assert response.status_code == 503
    assert SECRET not in response.text
    assert response.json() == {"detail": "The database is not available."}


async def test_unconfigured_database_url_returns_a_generic_503(client, service):
    service.error = RuntimeError("DATABASE_URL is not set; the database is not configured.")

    response = await client.post("/embeddings/ingest", json=make_body())

    assert response.status_code == 503
    assert response.json() == {"detail": "The database is not available."}


async def test_missing_service_returns_503(client, monkeypatch):
    app.dependency_overrides.clear()
    settings = Settings(anthropic_api_key="test", openai_api_key=None)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    dependencies.get_embedder.cache_clear()
    get_rag_ingest_service.cache_clear()

    try:
        response = await client.post("/embeddings/ingest", json=make_body())
    finally:
        dependencies.get_embedder.cache_clear()
        get_rag_ingest_service.cache_clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "The embeddings service is not configured."}


def test_build_embedder_uses_a_client_without_sdk_retries(monkeypatch):
    settings = Settings(
        anthropic_api_key="test", openai_api_key="sk-test", embedding_model="embed-model"
    )
    created: dict[str, Any] = {}

    def fake_openai(**kwargs: Any) -> object:
        created.update(kwargs)
        return object()

    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "OpenAI", fake_openai)

    assert dependencies.build_embedder() is not None
    assert created == {"api_key": "sk-test", "max_retries": 0}


def test_build_embedder_is_none_without_a_key(monkeypatch):
    settings = Settings(anthropic_api_key="test", openai_api_key=None)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)

    assert dependencies.build_embedder() is None


def test_build_embedder_fails_soft_when_the_client_cannot_be_built(monkeypatch):
    settings = Settings(anthropic_api_key="test", openai_api_key="sk-test")

    def broken_openai(**_kwargs: Any) -> object:
        raise RuntimeError("cannot build")

    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "OpenAI", broken_openai)

    assert dependencies.build_embedder() is None


def test_build_rag_ingest_service_needs_a_database_url(monkeypatch):
    settings = Settings(anthropic_api_key="test", openai_api_key="sk-test", database_url=None)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "get_embedder", lambda: object())

    assert dependencies.build_rag_ingest_service() is None


def test_build_rag_ingest_service_needs_an_embedder(monkeypatch):
    settings = Settings(
        anthropic_api_key="test",
        openai_api_key=None,
        database_url="postgresql+asyncpg://u:p@localhost:5433/db",
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "get_embedder", lambda: None)

    assert dependencies.build_rag_ingest_service() is None


def test_build_rag_ingest_service_is_built_when_configured(monkeypatch):
    settings = Settings(
        anthropic_api_key="test",
        openai_api_key="sk-test",
        database_url="postgresql+asyncpg://u:p@localhost:5433/db",
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "get_embedder", lambda: object())
    monkeypatch.setattr(dependencies, "get_async_session_factory", lambda: object())

    assert dependencies.build_rag_ingest_service() is not None


def test_ingest_route_is_in_the_openapi_schema():
    operation = app.openapi()["paths"]["/embeddings/ingest"]["post"]

    assert "200" in operation["responses"]
    assert (
        "IngestResponse"
        in operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    )


async def test_existing_routes_still_work(client):
    response = await client.get("/health")

    assert response.status_code == 200
