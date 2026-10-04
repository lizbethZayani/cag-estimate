"""
Estimation orchestration: cache lookup, prompt rendering, structured LLM call.

``estimate()`` is a short pipeline of single-purpose steps so later stages
(input/output guardrails, semantic cache) can be added as extra steps around
``_generate`` without rewriting the flow.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

import structlog
from pydantic import ValidationError

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

    def __init__(self, wrapper: LLMWrapper, cache: EstimationCache) -> None:
        self.wrapper = wrapper
        self.cache = cache

    def estimate(self, request: EstimationRequest) -> EstimationResponse:
        key = build_cache_key(request, PROMPT_VERSION, self.wrapper.primary_model)
        cached = self._lookup(key)
        if cached is not None:
            return EstimationResponse(result=cached, prompt_version=PROMPT_VERSION, cached=True)

        result = self._generate(request)
        self._store(key, result)
        return EstimationResponse(result=result, prompt_version=PROMPT_VERSION, cached=False)

    def stream(self, request: EstimationRequest, result: dict[str, Any]) -> Iterator[str]:
        """Stream the markdown estimation; ``result`` is filled when it ends."""
        system, user = render_estimation_prompt(
            request, version=PROMPT_VERSION, output_format="markdown"
        )
        return self.wrapper.complete_stream(
            system_prompt=system, user_message=user, max_tokens=MAX_TOKENS, result=result
        )

    def _lookup(self, key: str) -> ProjectEstimation | None:
        payload = self.cache.get(key)
        if payload is None:
            return None
        try:
            return ProjectEstimation.model_validate(payload["result"])
        except (KeyError, TypeError, ValidationError):
            log.warning("estimation_cache_corrupt", key_prefix=key[:24])
            return None

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

    def _store(self, key: str, result: ProjectEstimation) -> None:
        self.cache.set(key, {"result": result.model_dump(mode="json")})
