"""Offline tests for EstimationCache fail-soft behavior."""

from __future__ import annotations

from typing import Any

import fakeredis
import pytest
import redis
from structlog.testing import capture_logs

from cag_estimate.services.cache import EstimationCache

KEY = "estimation:abc"


class BrokenRedis:
    """Redis client whose every command raises a connection error."""

    def get(self, key: str) -> Any:
        raise redis.ConnectionError("down")

    def setex(self, key: str, ttl: int, value: str) -> Any:
        raise redis.ConnectionError("down")


@pytest.fixture
def client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def cache(client: fakeredis.FakeRedis) -> EstimationCache:
    return EstimationCache(client, ttl=60)


def test_round_trip(cache: EstimationCache) -> None:
    cache.set(KEY, {"total": 3})
    assert cache.get(KEY) == {"total": 3}


def test_get_missing_key_is_a_miss(cache: EstimationCache) -> None:
    assert cache.get(KEY) is None


def test_corrupt_json_is_a_logged_miss(cache: EstimationCache, client: fakeredis.FakeRedis) -> None:
    client.set(KEY, "{not json")
    with capture_logs() as logs:
        assert cache.get(KEY) is None
    assert any(entry["event"] == "cache_get_corrupt" for entry in logs)


def test_unserializable_value_is_skipped_without_raising(
    cache: EstimationCache,
) -> None:
    with capture_logs() as logs:
        cache.set(KEY, {"bad": object()})
    assert cache.get(KEY) is None
    assert any(entry["event"] == "cache_set_unserializable" for entry in logs)


def test_circular_value_is_skipped_without_raising(cache: EstimationCache) -> None:
    value: dict[str, Any] = {}
    value["self"] = value
    cache.set(KEY, value)
    assert cache.get(KEY) is None


def test_redis_errors_are_swallowed() -> None:
    broken = EstimationCache(BrokenRedis(), ttl=60)  # type: ignore[arg-type]
    assert broken.get(KEY) is None
    broken.set(KEY, {"total": 3})
