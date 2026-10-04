"""Offline tests for EstimationService (fake wrapper + fakeredis cache)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import fakeredis
import pytest

from cag_estimate.schemas.estimation import (
    EstimationRequest,
    ProjectEstimation,
)
from cag_estimate.services.cache import EstimationCache
from cag_estimate.services.estimation import EstimationService, build_cache_key


def _estimation() -> ProjectEstimation:
    tasks = [
        {
            "task_id": i,
            "name": f"Task {i}",
            "description": "desc",
            "estimated_hours": 10,
            "estimated_cost_usd": 400,
            "complexity": "Medium",
            "includes": ["code"],
        }
        for i in range(1, 4)
    ]
    return ProjectEstimation.model_validate(
        {
            "project_name": "Demo",
            "meeting_summary": "summary",
            "tasks": tasks,
            "summary": {
                "total_hours": 30,
                "total_cost_usd": 1200,
                "team_size": "2 developers",
                "estimated_duration_weeks": 2,
                "hourly_rate": 40,
            },
        }
    )


class FakeWrapper:
    primary_model = "fake-model"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.structured_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[ProjectEstimation, dict[str, Any]]:
        self.structured_calls.append(kwargs)
        if self.error:
            raise self.error
        return _estimation(), {"model": "fake-model"}

    def complete_stream(self, **kwargs: Any) -> Iterator[str]:
        self.stream_calls.append(kwargs)
        yield "a"
        yield "b"


@pytest.fixture
def cache() -> EstimationCache:
    return EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)


@pytest.fixture
def request_() -> EstimationRequest:
    return EstimationRequest(transcription="Build an app", hourly_rate=40)


def test_cache_miss_calls_llm_and_stores(cache, request_):
    wrapper = FakeWrapper()
    response = EstimationService(wrapper, cache).estimate(request_)

    assert response.cached is False
    assert response.prompt_version == "v1"
    assert response.result.project_name == "Demo"
    assert len(wrapper.structured_calls) == 1
    assert wrapper.structured_calls[0]["response_model"] is ProjectEstimation
    assert cache.get(build_cache_key(request_, "v1", "fake-model")) is not None


def test_cache_hit_skips_llm(cache, request_):
    EstimationService(FakeWrapper(), cache).estimate(request_)
    wrapper = FakeWrapper()
    response = EstimationService(wrapper, cache).estimate(request_)

    assert response.cached is True
    assert response.result.project_name == "Demo"
    assert wrapper.structured_calls == []


def test_corrupt_cache_payload_is_a_miss(cache, request_):
    cache.set(build_cache_key(request_, "v1", "fake-model"), {"result": {"bad": 1}})
    wrapper = FakeWrapper()
    response = EstimationService(wrapper, cache).estimate(request_)

    assert response.cached is False
    assert len(wrapper.structured_calls) == 1


def test_llm_exception_propagates_and_nothing_cached(cache, request_):
    wrapper = FakeWrapper(error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        EstimationService(wrapper, cache).estimate(request_)
    assert cache.get(build_cache_key(request_, "v1", "fake-model")) is None


def test_cache_key_depends_on_inputs_not_prompt_text(request_):
    base = build_cache_key(request_, "v1", "m")
    assert base.startswith("estimation:v2:")
    assert base == build_cache_key(request_.model_copy(), "v1", "m")
    assert base != build_cache_key(request_.model_copy(update={"hourly_rate": 50}), "v1", "m")
    assert base != build_cache_key(request_, "v2", "m")
    assert base != build_cache_key(request_, "v1", "other")


def test_stream_uses_markdown_prompt(cache, request_):
    wrapper = FakeWrapper()
    result: dict[str, Any] = {}
    chunks = list(EstimationService(wrapper, cache).stream(request_, result))

    assert chunks == ["a", "b"]
    call = wrapper.stream_calls[0]
    assert "Markdown" in call["system_prompt"] or "markdown" in call["system_prompt"].lower()
    assert call["result"] is result
