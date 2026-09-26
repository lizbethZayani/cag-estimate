"""
Exact-match Redis cache for LLM responses.

The cache key is a SHA-256 hash of the full system prompt, the user message,
and the generation knobs (model, max_tokens, thinking_budget). Any change to
those inputs — e.g. editing the CAG reference examples, which are embedded in
the system prompt — implicitly invalidates the cache without needing a manual
flush, because it changes the prompt text and therefore the hash.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import redis
import structlog

from cag_estimate.config import get_settings

log = structlog.get_logger()


class EstimationCache:
    """Thin wrapper around redis-py with deterministic keying and TTL."""

    def __init__(self, redis_client: redis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: int = 86400) -> "EstimationCache":
        return cls(redis.from_url(url, decode_responses=True), ttl=ttl)

    @staticmethod
    def make_key(
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        thinking_budget: int | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "model": model,
                "max_tokens": max_tokens,
                "thinking_budget": thinking_budget,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"estimation:{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            cached = self.redis.get(key)
        except redis.RedisError as exc:
            # Fail open: a Redis outage should degrade to "always call the LLM",
            # not take down the estimation endpoints.
            log.warning("cache_get_failed", error=str(exc))
            return None
        if cached:
            log.info("cache_hit", key_prefix=key[:24])
            return json.loads(cached)
        log.info("cache_miss", key_prefix=key[:24])
        return None

    def set(self, key: str, response: dict[str, Any]) -> None:
        try:
            self.redis.setex(key, self.ttl, json.dumps(response))
            log.info("cache_stored", key_prefix=key[:24], ttl=self.ttl)
        except redis.RedisError as exc:
            log.warning("cache_set_failed", error=str(exc))


@lru_cache()
def get_cache() -> EstimationCache:
    """Process-wide singleton cache, configured from Settings."""
    settings = get_settings()
    return EstimationCache.from_url(settings.redis_url, ttl=settings.cache_ttl_seconds)
