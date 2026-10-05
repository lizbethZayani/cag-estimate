"""Semantic cache for estimations on Redis Stack (RediSearch via redisvl).

A stored estimation is reused only when both conditions hold:

1. The **bucket** matches exactly. It is composed of ``prompt_version`` and
   ``hourly_rate``, so entries made under another prompt or rate are never served.
2. The cosine similarity of the transcription embeddings is at least ``threshold``.

With ``log_only=True`` the lookup still runs and logs the similarity but never
returns a hit, to calibrate the threshold against real traffic.

Vanilla ``redis:7-alpine`` lacks RediSearch: use ``redis/redis-stack``.
All failures are soft: a broken cache degrades to a miss, never to an error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from pydantic import ValidationError
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

from cag_estimate.schemas.estimation import EstimationRequest, ProjectEstimation

log = structlog.get_logger()

INDEX_PREFIX = "estimation:semantic"


def _to_bytes(vector: list[float]) -> bytes:
    """RediSearch stores vectors as float32 bytes; redisvl rejects plain lists."""
    return np.array(vector, dtype=np.float32).tobytes()


def _build_schema(index_name: str, dims: int) -> dict[str, Any]:
    return {
        "index": {"name": index_name, "prefix": INDEX_PREFIX, "storage_type": "hash"},
        "fields": [
            {"name": "bucket", "type": "tag"},
            {"name": "result_json", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {"dims": dims, "distance_metric": "cosine", "algorithm": "flat"},
            },
        ],
    }


class EstimationSemanticCache:
    """Vector-similarity cache of ``ProjectEstimation`` results."""

    def __init__(
        self,
        *,
        redis_client: Any,
        vectorizer: Any,
        threshold: float,
        ttl: int,
        log_only: bool = False,
        index_name: str = "estimations",
    ) -> None:
        self.vectorizer = vectorizer
        self.threshold = threshold
        self.ttl = ttl
        self.log_only = log_only

        self.index = SearchIndex.from_dict(_build_schema(index_name, vectorizer.dims))
        self.index.set_client(redis_client)
        try:
            self.index.create(overwrite=False)
        except Exception as exc:  # noqa: BLE001 - an existing index is fine
            log.debug("semantic_index_create_skipped", error_type=type(exc).__name__)

    @staticmethod
    def bucket_for(request: EstimationRequest, prompt_version: str) -> str:
        return f"{prompt_version}:{request.hourly_rate}"

    def lookup(self, request: EstimationRequest, prompt_version: str) -> ProjectEstimation | None:
        bucket = self.bucket_for(request, prompt_version)
        try:
            hit = self._nearest(request, bucket)
        except Exception as exc:  # noqa: BLE001 - fail soft: a broken cache is a miss
            log.warning("semantic_cache_lookup_failed", error_type=type(exc).__name__)
            return None
        if hit is None:
            log.info("semantic_cache_miss", bucket=bucket, reason="empty_index")
            return None

        # redisvl returns cosine *distance* (0 = identical, 2 = opposite).
        similarity = round(1.0 - float(hit.get("vector_distance", 1.0)), 4)
        if similarity < self.threshold:
            log.info("semantic_cache_miss", bucket=bucket, similarity=similarity)
            return None
        if self.log_only:
            log.info("semantic_cache_hit_log_only", bucket=bucket, similarity=similarity)
            return None
        return self._parse(hit, bucket, similarity)

    def store(
        self, request: EstimationRequest, result: ProjectEstimation, prompt_version: str
    ) -> None:
        bucket = self.bucket_for(request, prompt_version)
        try:
            row = {
                "bucket": bucket,
                "result_json": result.model_dump_json(),
                "embedding": _to_bytes(self.vectorizer.embed(request.transcription)),
            }
            self.index.load([row], ttl=self.ttl)
        except Exception as exc:  # noqa: BLE001 - fail soft: caching is best effort
            log.warning("semantic_cache_store_failed", error_type=type(exc).__name__)
            return
        log.info("semantic_cache_stored", bucket=bucket, ttl=self.ttl)

    def _nearest(self, request: EstimationRequest, bucket: str) -> dict[str, Any] | None:
        query = VectorQuery(
            vector=_to_bytes(self.vectorizer.embed(request.transcription)),
            vector_field_name="embedding",
            return_fields=["result_json", "bucket"],
            num_results=1,
            return_score=True,
            filter_expression=Tag("bucket") == bucket,
        )
        results = self.index.query(query)
        return results[0] if results else None

    @staticmethod
    def _parse(hit: dict[str, Any], bucket: str, similarity: float) -> ProjectEstimation | None:
        try:
            result = ProjectEstimation.model_validate_json(hit["result_json"])
        except (KeyError, ValidationError):
            log.warning("semantic_cache_corrupt", bucket=bucket)
            return None
        log.info("semantic_cache_hit", bucket=bucket, similarity=similarity)
        return result
