"""Offline tests for the embedding pipeline schemas."""

from typing import Any

import pytest
from pydantic import ValidationError

from cag_estimate.embedding_pipeline.schemas import (
    Budget,
    Chunk,
    EmbeddedChunk,
    IngestRequest,
    IngestResponse,
    IngestStats,
)


def budget_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "budget_id": "BUD-2024-014",
        "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
        "project_summary": "Mobile banking API with OAuth 2.0 authentication",
        "main_technology": "ruby_on_rails",
        "year": 2024,
        "total_estimated_hours": 480,
        "components": [
            {
                "component_id": "AUTH-001",
                "name": "OAuth 2.0 authentication backend",
                "description": "Implementation of OAuth 2.0 flows.",
                "tech_stack": ["ruby_on_rails", "postgresql", "redis"],
                "estimated_hours": 120,
                "complexity": "high",
                "dependencies": [],
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_valid_budget_parses():
    budget = Budget.model_validate(budget_payload())

    assert budget.budget_id == "BUD-2024-014"
    assert budget.client_metadata.sector == "finance"
    assert budget.components[0].tech_stack == ["ruby_on_rails", "postgresql", "redis"]
    assert budget.components[0].dependencies == []


def test_dependencies_default_to_empty():
    payload = budget_payload()
    del payload["components"][0]["dependencies"]

    assert Budget.model_validate(payload).components[0].dependencies == []


@pytest.mark.parametrize("field", ["budget_id", "project_summary", "main_technology"])
def test_empty_budget_strings_are_rejected(field):
    with pytest.raises(ValidationError):
        Budget.model_validate(budget_payload(**{field: ""}))


def test_missing_budget_id_is_rejected():
    payload = budget_payload()
    del payload["budget_id"]

    with pytest.raises(ValidationError):
        Budget.model_validate(payload)


def test_empty_component_id_is_rejected():
    payload = budget_payload()
    payload["components"][0]["component_id"] = ""

    with pytest.raises(ValidationError):
        Budget.model_validate(payload)


def test_empty_sector_is_rejected():
    payload = budget_payload()
    payload["client_metadata"]["sector"] = ""

    with pytest.raises(ValidationError):
        Budget.model_validate(payload)


def test_unknown_sector_is_rejected():
    payload = budget_payload()
    payload["client_metadata"]["sector"] = "retail"

    with pytest.raises(ValidationError, match="sector"):
        Budget.model_validate(payload)


def test_unknown_complexity_is_rejected():
    payload = budget_payload()
    payload["components"][0]["complexity"] = "extreme"

    with pytest.raises(ValidationError, match="complexity"):
        Budget.model_validate(payload)


@pytest.mark.parametrize("year", [1999, 2101])
def test_year_outside_bounds_is_rejected(year):
    with pytest.raises(ValidationError, match="year"):
        Budget.model_validate(budget_payload(year=year))


@pytest.mark.parametrize("year", [2000, 2100])
def test_year_on_the_bounds_is_accepted(year):
    assert Budget.model_validate(budget_payload(year=year)).year == year


def test_budget_without_components_is_rejected():
    with pytest.raises(ValidationError, match="at least one component"):
        Budget.model_validate(budget_payload(components=[]))


def test_negative_hours_are_rejected():
    payload = budget_payload()
    payload["components"][0]["estimated_hours"] = -1

    with pytest.raises(ValidationError):
        Budget.model_validate(payload)


def test_extra_fields_are_ignored():
    payload = budget_payload(unknown_field="x")
    payload["components"][0]["owner"] = "team-a"

    budget = Budget.model_validate(payload)

    assert not hasattr(budget, "unknown_field")
    assert not hasattr(budget.components[0], "owner")


def test_ingest_request_needs_at_least_one_budget():
    with pytest.raises(ValidationError, match="at least one budget"):
        IngestRequest(budgets=[])


def test_ingest_request_accepts_budgets():
    request = IngestRequest.model_validate({"budgets": [budget_payload()]})

    assert len(request.budgets) == 1


def test_negative_token_count_is_rejected():
    with pytest.raises(ValidationError):
        Chunk(chunk_id="A::B", text="t", metadata={}, token_count=-1)


def test_embedded_chunk_carries_the_embedding():
    chunk = EmbeddedChunk(
        chunk_id="A::B", text="t", metadata={"k": 1}, token_count=3, embedding=[0.1, 0.2]
    )

    assert chunk.embedding == [0.1, 0.2]
    assert isinstance(chunk, Chunk)


def test_ingest_response_wraps_chunks_and_stats():
    chunk = EmbeddedChunk(chunk_id="A::B", text="t", metadata={}, token_count=3, embedding=[0.1])
    stats = IngestStats(
        total_budgets=1, total_chunks=1, total_tokens=3, estimated_cost_usd=0.00000006
    )

    response = IngestResponse(chunks=[chunk], stats=stats)

    assert response.stats.total_chunks == 1
    assert response.chunks[0].embedding == [0.1]
