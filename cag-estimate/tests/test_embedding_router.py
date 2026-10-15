"""Offline tests for POST /embeddings/ingest (fake service injected via dependency override)."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from structlog.testing import capture_logs

from cag_estimate import dependencies
from cag_estimate.config import Settings
from cag_estimate.dependencies import get_embedding_service
from cag_estimate.embedding_pipeline.schemas import (
    Budget,
    EmbeddedChunk,
    IngestResponse,
    IngestStats,
)
from cag_estimate.main import app

SECRET = "sk-secret-failure-detail"


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


class FakeService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.received: list[Budget] = []

    def ingest(self, budgets: list[Budget]) -> IngestResponse:
        if self.error:
            raise self.error
        self.received = budgets
        chunk = EmbeddedChunk(
            chunk_id="BUD-1_AUTH-001",
            text="text",
            metadata={"budget_id": "BUD-1"},
            token_count=12,
            embedding=[0.1, 0.2],
        )
        stats = IngestStats(
            total_budgets=len(budgets),
            total_chunks=1,
            total_tokens=12,
            estimated_cost_usd=0.00000024,
        )
        return IngestResponse(chunks=[chunk], stats=stats)


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture(autouse=True)
def overrides(service: FakeService) -> Iterator[None]:
    app.dependency_overrides[get_embedding_service] = lambda: service
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


async def test_ingest_returns_the_documented_shape(client, service):
    response = await client.post("/embeddings/ingest", json={"budgets": [make_budget_payload()]})

    assert response.status_code == 200
    body = response.json()
    assert set(body["chunks"][0]) == {"chunk_id", "text", "metadata", "token_count", "embedding"}
    assert set(body["stats"]) == {
        "total_budgets",
        "total_chunks",
        "total_tokens",
        "estimated_cost_usd",
    }
    assert body["stats"]["total_budgets"] == 1
    assert [budget.budget_id for budget in service.received] == ["BUD-1"]


@pytest.mark.parametrize(
    "payload",
    [
        {"budgets": []},
        {},
        {"budgets": [{"budget_id": "BUD-1"}]},
        ["not", "an", "object"],
    ],
)
async def test_invalid_bodies_are_rejected_with_422(client, payload):
    response = await client.post("/embeddings/ingest", json=payload)

    assert response.status_code == 422


async def test_service_failure_returns_a_generic_500(client, service):
    service.error = RuntimeError(SECRET)

    with capture_logs() as logs:
        response = await client.post(
            "/embeddings/ingest", json={"budgets": [make_budget_payload()]}
        )

    assert response.status_code == 500
    assert SECRET not in response.text
    assert response.json() == {"detail": "The embeddings service failed. Please try again later."}
    event = next(entry for entry in logs if entry["event"] == "ingest_failed")
    assert event["error_type"] == "RuntimeError"
    assert event["error"] == SECRET


async def test_missing_openai_key_returns_503(client, monkeypatch):
    app.dependency_overrides.clear()
    settings = Settings(anthropic_api_key="test", openai_api_key=None)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    get_embedding_service.cache_clear()

    try:
        response = await client.post(
            "/embeddings/ingest", json={"budgets": [make_budget_payload()]}
        )
    finally:
        get_embedding_service.cache_clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "The embeddings service is not configured."}


def test_build_embedding_service_uses_a_client_without_sdk_retries(monkeypatch):
    settings = Settings(anthropic_api_key="test", openai_api_key="sk-test")
    created: dict[str, Any] = {}

    def fake_openai(**kwargs: Any) -> object:
        created.update(kwargs)
        return object()

    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "OpenAI", fake_openai)

    assert dependencies.build_embedding_service() is not None
    assert created == {"api_key": "sk-test", "max_retries": 0}


def test_build_embedding_service_fails_soft_when_the_client_cannot_be_built(monkeypatch):
    settings = Settings(anthropic_api_key="test", openai_api_key="sk-test")

    def broken_openai(**_kwargs: Any) -> object:
        raise RuntimeError("cannot build")

    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "OpenAI", broken_openai)

    assert dependencies.build_embedding_service() is None


def test_ingest_route_is_in_the_openapi_schema():
    assert "post" in app.openapi()["paths"]["/embeddings/ingest"]


async def test_existing_routes_still_work(client):
    response = await client.get("/health")

    assert response.status_code == 200
