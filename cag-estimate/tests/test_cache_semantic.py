"""Offline tests for the semantic cache: redisvl SearchIndex and vectorizer are faked."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from structlog.testing import capture_logs

from cag_estimate.cache import semantic
from cag_estimate.cache.semantic import EstimationSemanticCache
from cag_estimate.schemas.estimation import EstimationRequest, ProjectEstimation


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


class FakeIndex:
    """Records what the cache calls on redisvl's SearchIndex."""

    instances: ClassVar[list[FakeIndex]] = []

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema
        self.client: Any = None
        self.created = False
        self.hits: list[dict[str, Any]] = []
        self.queries: list[Any] = []
        self.loaded: list[tuple[list[dict[str, Any]], int | None]] = []
        self.query_error: Exception | None = None
        self.load_error: Exception | None = None
        FakeIndex.instances.append(self)

    @classmethod
    def from_dict(cls, schema: dict[str, Any]) -> FakeIndex:
        return cls(schema)

    def set_client(self, client: Any) -> None:
        self.client = client

    def create(self, overwrite: bool = False) -> None:
        self.created = True

    def query(self, query: Any) -> list[dict[str, Any]]:
        if self.query_error:
            raise self.query_error
        self.queries.append(query)
        return self.hits

    def load(self, data: list[dict[str, Any]], ttl: int | None = None) -> None:
        if self.load_error:
            raise self.load_error
        self.loaded.append((data, ttl))


class FakeVectorizer:
    dims = 4

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture(autouse=True)
def fake_index_class(monkeypatch):
    FakeIndex.instances = []
    monkeypatch.setattr(semantic, "SearchIndex", FakeIndex)


def _cache(**kwargs: Any) -> tuple[EstimationSemanticCache, FakeIndex]:
    cache = EstimationSemanticCache(
        redis_client=object(),
        vectorizer=FakeVectorizer(),
        threshold=kwargs.pop("threshold", 0.9),
        ttl=kwargs.pop("ttl", 60),
        **kwargs,
    )
    return cache, FakeIndex.instances[-1]


@pytest.fixture
def request_() -> EstimationRequest:
    return EstimationRequest(transcription="Build an app", hourly_rate=40)


def _hit(distance: float, payload: str | None = None) -> dict[str, Any]:
    result_json = payload if payload is not None else _estimation().model_dump_json()
    return {"result_json": result_json, "vector_distance": str(distance)}


def test_index_is_created_with_dims_from_vectorizer():
    _, index = _cache(index_name="custom")

    assert index.created is True
    assert index.schema["index"]["name"] == "custom"
    assert index.schema["index"]["prefix"] == "estimation:semantic"
    vector_field = next(f for f in index.schema["fields"] if f["name"] == "embedding")
    assert vector_field["attrs"]["dims"] == 4
    assert vector_field["attrs"]["distance_metric"] == "cosine"


def test_index_creation_failure_is_not_fatal(monkeypatch):
    class ExistingIndex(FakeIndex):
        def create(self, overwrite: bool = False) -> None:
            raise RuntimeError("already exists")

    monkeypatch.setattr(semantic, "SearchIndex", ExistingIndex)
    EstimationSemanticCache(
        redis_client=object(), vectorizer=FakeVectorizer(), threshold=0.9, ttl=60
    )


def test_bucket_includes_prompt_version_and_rate(request_):
    base = EstimationSemanticCache.bucket_for(request_, "v1")

    assert base == EstimationSemanticCache.bucket_for(request_.model_copy(), "v1")
    assert base != EstimationSemanticCache.bucket_for(request_, "v2")
    other_rate = request_.model_copy(update={"hourly_rate": 50})
    assert base != EstimationSemanticCache.bucket_for(other_rate, "v1")


def test_empty_index_is_a_miss(request_):
    cache, _ = _cache()
    assert cache.lookup(request_, "v1") is None


def test_lookup_filters_by_bucket(request_):
    cache, index = _cache()
    cache.lookup(request_, "v1")

    query = index.queries[0]
    assert "@bucket:{v1\\:40}" in query.query_string()


def test_below_threshold_is_a_miss(request_):
    cache, index = _cache(threshold=0.9)
    index.hits = [_hit(0.2)]  # similarity 0.8

    assert cache.lookup(request_, "v1") is None


def test_above_threshold_is_a_hit(request_):
    cache, index = _cache(threshold=0.9)
    index.hits = [_hit(0.05)]  # similarity 0.95

    result = cache.lookup(request_, "v1")

    assert result is not None
    assert result.project_name == "Demo"


def test_log_only_logs_would_be_hit_but_returns_none(request_):
    cache, index = _cache(threshold=0.9, log_only=True)
    index.hits = [_hit(0.05)]

    with capture_logs() as logs:
        assert cache.lookup(request_, "v1") is None

    entry = next(e for e in logs if e["event"] == "semantic_cache_hit_log_only")
    assert entry["similarity"] == 0.95


def test_corrupt_payload_is_a_miss(request_):
    cache, index = _cache()
    index.hits = [_hit(0.0, payload="not json")]

    assert cache.lookup(request_, "v1") is None


def test_lookup_error_is_swallowed(request_):
    cache, index = _cache()
    index.query_error = RuntimeError("redis down")

    assert cache.lookup(request_, "v1") is None


def test_store_loads_with_ttl_and_float32_embedding(request_):
    cache, index = _cache(ttl=123)
    cache.store(request_, _estimation(), "v1")

    (rows, ttl) = index.loaded[0]
    assert ttl == 123
    assert rows[0]["bucket"] == EstimationSemanticCache.bucket_for(request_, "v1")
    assert ProjectEstimation.model_validate_json(rows[0]["result_json"]).project_name == "Demo"
    assert np.frombuffer(rows[0]["embedding"], dtype=np.float32).tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )


def test_store_error_is_swallowed_and_logged(request_):
    cache, index = _cache()
    index.load_error = RuntimeError("boom")

    with capture_logs() as logs:
        cache.store(request_, _estimation(), "v1")

    assert any(e["event"] == "semantic_cache_store_failed" for e in logs)

