"""HTTP-level tests with a FakeEstimationService injected via dependency_overrides."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from instructor.core.exceptions import InstructorRetryException

from cag_estimate.dependencies import get_estimation_service
from cag_estimate.main import app
from cag_estimate.schemas.estimation import EstimationRequest, EstimationResponse, ProjectEstimation

SECRET = "secret-upstream-detail"


def _response() -> EstimationResponse:
    tasks = [
        {
            "task_id": i,
            "name": f"T{i}",
            "description": "d",
            "estimated_hours": 10,
            "estimated_cost_usd": 400,
            "complexity": "Medium",
            "includes": [],
        }
        for i in range(1, 4)
    ]
    result = ProjectEstimation.model_validate(
        {
            "project_name": "Demo",
            "meeting_summary": "s",
            "tasks": tasks,
            "summary": {
                "total_hours": 30,
                "total_cost_usd": 1200,
                "team_size": "2",
                "estimated_duration_weeks": 2,
                "hourly_rate": 40,
            },
        }
    )
    return EstimationResponse(result=result, prompt_version="v1", cached=False)


class FakeService:
    def __init__(self, error: Exception | None = None, stream_error: bool = False) -> None:
        self.error = error
        self.stream_error = stream_error

    def estimate(self, request: EstimationRequest) -> EstimationResponse:
        if self.error:
            raise self.error
        return _response()

    def stream(self, request: EstimationRequest, result: dict[str, Any]) -> Iterator[str]:
        yield "hello "
        if self.stream_error:
            raise RuntimeError(SECRET)
        yield "world"
        result.update(model="m", provider="p", usage={"total_tokens": 3})


def _client(service: FakeService) -> TestClient:
    app.dependency_overrides[get_estimation_service] = lambda: service
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def _events(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_estimate_ok():
    res = _client(FakeService()).post("/api/v1/estimate", json={"transcription": "x"})
    body = res.json()
    assert res.status_code == 200
    assert body["cached"] is False
    assert body["prompt_version"] == "v1"
    assert body["result"]["project_name"] == "Demo"


def test_estimate_bad_body_is_422():
    res = _client(FakeService()).post("/api/v1/estimate", json={"hourly_rate": 40})
    assert res.status_code == 422


@pytest.mark.parametrize(
    "error",
    [
        InstructorRetryException(
            SECRET, n_attempts=3, total_usage=0, messages=[], last_completion=None
        ),
        RuntimeError(SECRET),
    ],
)
def test_upstream_failure_is_502_without_leak(error):
    res = _client(FakeService(error=error)).post(
        "/api/v1/estimate", json={"transcription": "x"}
    )
    assert res.status_code == 502
    assert SECRET not in res.text


def test_stream_events():
    res = _client(FakeService()).post("/api/v1/estimate/stream", json={"transcription": "x"})
    events = _events(res.text)
    assert [e["type"] for e in events] == ["token", "token", "done"]
    assert events[0]["content"] == "hello "
    assert events[-1]["tokens_used"]["total_tokens"] == 3
    assert "cost_breakdown" in events[-1]


def test_stream_error_event_is_generic():
    res = _client(FakeService(stream_error=True)).post(
        "/api/v1/estimate/stream", json={"transcription": "x"}
    )
    events = _events(res.text)
    assert events[-1]["type"] == "error"
    assert SECRET not in res.text
