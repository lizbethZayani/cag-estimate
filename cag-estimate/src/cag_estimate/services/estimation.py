"""
Estimation orchestration: input guardrail, exact + semantic cache lookup, prompt rendering,
structured LLM call, output guardrail, cache store.

``estimate()`` is a short pipeline of single-purpose steps; the semantic
cache is optional and the pipeline works unchanged without it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

import structlog
from pydantic import ValidationError

from cag_estimate.cache.semantic import EstimationSemanticCache
from cag_estimate.guardrails.input import check_input
from cag_estimate.guardrails.output import enforce_output
from cag_estimate.prompts import render_estimation_prompt
from cag_estimate.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    ProjectEstimation,
)
from cag_estimate.services.cache import EstimationCache
from cag_estimate.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()

PROMPT_VERSION = "v1"
MAX_TOKENS = 4096


def build_cache_key(request: EstimationRequest, prompt_version: str, model: str) -> str:
    """Key on the inputs, not the rendered prompt, so wording edits keep entries."""
    payload = json.dumps(
        {
            "description": request.transcription,
            "hourly_rate": request.hourly_rate,
            "prompt_version": prompt_version,
            "model": model,
        },
        sort_keys=True,
    )
    return f"estimation:v2:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class EstimationService:
    """Produces validated project estimations."""

    def __init__(
        self,
        wrapper: LLMWrapper,
        cache: EstimationCache,
        moderation_client: Any | None = None,
        semantic_cache: EstimationSemanticCache | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.cache = cache
        self.moderation_client = moderation_client
        self.semantic_cache = semantic_cache

    def estimate(self, request: EstimationRequest) -> EstimationResponse:
        # Input check first: a rejected input must never be served from cache.
        self._check_input(request)
        key = build_cache_key(request, PROMPT_VERSION, self.wrapper.primary_model)
        cached = self._lookup(key) or self._lookup_semantic(request)
        if cached is not None:
            return EstimationResponse(result=cached, prompt_version=PROMPT_VERSION, cached=True)

        result = self._check_output(self._generate(request), request)
        self._store(key, request, result)
        return EstimationResponse(result=result, prompt_version=PROMPT_VERSION, cached=False)

    def stream(self, request: EstimationRequest, result: dict[str, Any]) -> Iterator[str]:
        """Stream the markdown estimation; ``result`` is filled when it ends.

        The input guardrail runs eagerly (this is not a generator), so a
        rejected input raises before any token is produced.
        """
        self._check_input(request)
        system, user = render_estimation_prompt(
            request, version=PROMPT_VERSION, output_format="markdown"
        )
        return self.wrapper.complete_stream(
            system_prompt=system, user_message=user, max_tokens=MAX_TOKENS, result=result
        )

    def _check_input(self, request: EstimationRequest) -> None:
        check_input(request.transcription, openai_client=self.moderation_client)

    def _check_output(
        self, result: ProjectEstimation, request: EstimationRequest
    ) -> ProjectEstimation:
        return enforce_output(result, request.hourly_rate)

    def _lookup(self, key: str) -> ProjectEstimation | None:
        payload = self.cache.get(key)
        if payload is None:
            return None
        try:
            return ProjectEstimation.model_validate(payload["result"])
        except (KeyError, TypeError, ValidationError):
            log.warning("estimation_cache_corrupt", key_prefix=key[:24])
            return None

    def _lookup_semantic(self, request: EstimationRequest) -> ProjectEstimation | None:
        if self.semantic_cache is None:
            return None
        return self.semantic_cache.lookup(request, PROMPT_VERSION)

    def _generate(self, request: EstimationRequest) -> ProjectEstimation:
        system, user = render_estimation_prompt(
            request, version=PROMPT_VERSION, output_format="json"
        )
        result, _meta = self.wrapper.complete_structured(
            system_prompt=system,
            user_message=user,
            response_model=ProjectEstimation,
            max_tokens=MAX_TOKENS,
        )
        return result

    def _store(self, key: str, request: EstimationRequest, result: ProjectEstimation) -> None:
        self.cache.set(key, {"result": result.model_dump(mode="json")})
        if self.semantic_cache is not None:
            self.semantic_cache.store(request, result, PROMPT_VERSION)
