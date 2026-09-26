"""
LiteLLM-backed wrapper that adds provider fallback, exact-match caching, and
cost tracking to every LLM call in the estimator.

Design notes
------------
- The wrapper exposes two primitives: ``complete()`` (blocking, full response) and
  ``complete_stream()`` (yields text chunks). Prompt building and JSON validation
  stay in ``routers/estimations.py`` — this module only knows how to reach an LLM.
- The Router is configured with two deployments under the same ``model_name``
  ("estimator") so LiteLLM can switch from the primary (Anthropic) to the
  fallback (OpenAI) transparently if the primary errors out or times out. The
  business logic (and the FastAPI endpoint) never sees which provider actually
  answered until it reads ``result["provider"]`` back.
- When the caller overrides the model explicitly, we bypass the Router and call
  ``litellm.completion`` directly with the matching provider's API key — that
  path has no fallback by design, since the caller asked for a specific model.
- The cache key includes the full system prompt and generation knobs, so any
  change to the CAG reference examples (which live in the system prompt)
  implicitly invalidates the cache without a manual flush.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Iterator

import litellm
import structlog
from litellm import Router

from cag_estimate.config import get_settings
from cag_estimate.services.cache import EstimationCache, get_cache

log = structlog.get_logger()


# Cost per 1M tokens (USD). Add an entry here whenever PRIMARY_MODEL /
# FALLBACK_MODEL changes to a model not already listed, otherwise cost
# reporting silently falls back to $0 for the unknown model.
MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def _cost_breakdown(model: str, input_tokens: int, output_tokens: int) -> dict[str, float]:
    base = _normalise_model_name(model)
    costs = MODEL_COSTS.get(base) or MODEL_COSTS.get(model) or {"input": 0.0, "output": 0.0}
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def _normalise_model_name(model: str) -> str:
    """Strip provider prefixes like ``anthropic/`` that LiteLLM may emit."""
    return model.split("/", 1)[1] if "/" in model else model


def _provider_from_model(model: str) -> str:
    name = _normalise_model_name(model).lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return "openai"
    return "unknown"


class LLMWrapper:
    """Unified LLM client with cache, provider fallback, and cost tracking."""

    def __init__(
        self,
        *,
        anthropic_api_key: str | None,
        openai_api_key: str | None,
        primary_model: str,
        fallback_model: str,
        timeout: int,
        num_retries: int,
        cache: EstimationCache,
    ):
        self.anthropic_api_key = anthropic_api_key
        self.openai_api_key = openai_api_key
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.num_retries = num_retries
        self.cache = cache

        # Two distinct model_name groups, not two deployments under the same
        # name: LiteLLM's Router load-balances across deployments that share
        # a model_name, so a shared name gives no primary/fallback ordering
        # (it may pick either one first). Distinct names + `fallbacks` is the
        # idiom that actually guarantees "try primary, only use fallback on
        # failure".
        self.router = Router(
            model_list=[
                {
                    "model_name": "estimator-primary",
                    "litellm_params": {
                        "model": primary_model,
                        "api_key": anthropic_api_key,
                        "timeout": timeout,
                    },
                },
                {
                    "model_name": "estimator-fallback",
                    "litellm_params": {
                        "model": fallback_model,
                        "api_key": openai_api_key,
                        "timeout": timeout,
                    },
                },
            ],
            fallbacks=[{"estimator-primary": ["estimator-fallback"]}],
            num_retries=num_retries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_override: str | None = None,
        max_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Single blocking LLM call with cache + provider fallback.

        Returns a dict with ``estimation`` (raw text), ``model``, ``provider``,
        ``finish_reason``, ``usage``, ``cost_breakdown``, ``latency_ms``, and
        ``cache_hit``.
        """
        cache_key_model = model_override or self.primary_model
        cache_key = EstimationCache.make_key(
            system_prompt=system_prompt,
            user_message=user_message,
            model=cache_key_model,
            max_tokens=max_tokens,
        )
        cached = self.cache.get(cache_key)
        if cached:
            return {**cached, "cache_hit": True}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        log.info("llm_call_started", mode="blocking", model=cache_key_model)
        t0 = time.perf_counter()
        try:
            response = self._dispatch(
                model_override=model_override, messages=messages, max_tokens=max_tokens
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.error(
                "llm_call_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        result = self._normalise_response(response, latency_ms=latency_ms)
        log.info(
            "llm_call_completed",
            model=result["model"],
            provider=result["provider"],
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            cost_usd=result["cost_breakdown"]["total_cost_usd"],
            latency_ms=latency_ms,
            finish_reason=result["finish_reason"],
        )
        self.cache.set(cache_key, result)
        return {**result, "cache_hit": False}

    def complete_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model_override: str | None = None,
        max_tokens: int = 4000,
        result: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive from the model.

        If `result` is provided, it's populated in place with `model`,
        `provider`, `finish_reason`, `usage`, and `cost_breakdown` once the
        stream ends — mirroring the shape `complete()` returns, so callers
        (e.g. the SSE endpoint) can report accurate metrics after streaming
        finishes. Token usage during streaming is provider-dependent; when the
        provider doesn't report it, these fields default to zero.

        Cache hits replay the cached estimation as a single chunk so the
        client UX stays consistent. Cache misses stream live and the full
        text is cached once the stream finishes.
        """
        cache_key_model = model_override or self.primary_model
        cache_key = EstimationCache.make_key(
            system_prompt=system_prompt,
            user_message=user_message,
            model=cache_key_model,
            max_tokens=max_tokens,
        )
        cached = self.cache.get(cache_key)
        if cached:
            log.info("stream_cache_hit", chars=len(cached.get("estimation", "")))
            if result is not None:
                result.update(
                    model=cached.get("model", cache_key_model),
                    provider=cached.get("provider", _provider_from_model(cache_key_model)),
                    finish_reason=cached.get("finish_reason", "stop"),
                    usage=cached.get("usage", _empty_usage()),
                    cost_breakdown=cached.get("cost_breakdown", _cost_breakdown(cache_key_model, 0, 0)),
                )
            yield cached.get("estimation", "")
            return

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        log.info("llm_stream_started", model=cache_key_model)
        t0 = time.perf_counter()
        full_text: list[str] = []
        model_seen = cache_key_model
        finish_reason = "stop"
        usage = _empty_usage()

        try:
            response = self._dispatch(
                model_override=model_override,
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response:
                delta, chunk_finish_reason, chunk_usage = _inspect_chunk(chunk)
                if delta:
                    full_text.append(delta)
                    yield delta
                if chunk_finish_reason:
                    finish_reason = chunk_finish_reason
                if chunk_usage:
                    usage = chunk_usage
                model_seen = getattr(chunk, "model", None) or model_seen
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.error(
                "llm_stream_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                latency_ms=latency_ms,
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        rendered = "".join(full_text)
        model = _normalise_model_name(model_seen)
        provider = _provider_from_model(model)
        cost_breakdown = _cost_breakdown(model, usage["input_tokens"], usage["output_tokens"])
        log.info(
            "llm_stream_completed",
            latency_ms=latency_ms,
            chars=len(rendered),
            model=model,
            provider=provider,
        )

        if result is not None:
            result.update(
                model=model,
                provider=provider,
                finish_reason=finish_reason,
                usage=usage,
                cost_breakdown=cost_breakdown,
            )

        self.cache.set(
            cache_key,
            {
                "estimation": rendered,
                "model": model,
                "provider": provider,
                "finish_reason": finish_reason,
                "usage": usage,
                "cost_breakdown": cost_breakdown,
                "latency_ms": latency_ms,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(
        self, *, model_override: str | None, messages: list[dict], max_tokens: int, stream: bool = False
    ) -> Any:
        """Call the Router (with fallback) or LiteLLM directly when the caller
        wants a specific model."""
        kwargs: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
        if stream:
            kwargs["stream"] = True

        if model_override:
            api_key = (
                self.anthropic_api_key
                if _provider_from_model(model_override) == "anthropic"
                else self.openai_api_key
            )
            return litellm.completion(
                model=model_override,
                api_key=api_key,
                timeout=self.timeout,
                num_retries=self.num_retries,
                **kwargs,
            )
        return self.router.completion(model="estimator-primary", **kwargs)

    @staticmethod
    def _normalise_response(response: Any, *, latency_ms: int) -> dict[str, Any]:
        choice = response.choices[0]
        finish_reason = (choice.finish_reason or "stop").lower()
        usage_obj = response.usage
        input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
        total_tokens = getattr(usage_obj, "total_tokens", input_tokens + output_tokens) or (
            input_tokens + output_tokens
        )

        model = _normalise_model_name(response.model)
        return {
            "estimation": choice.message.content or "",
            "model": model,
            "provider": _provider_from_model(model),
            "finish_reason": finish_reason,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            "latency_ms": latency_ms,
            "cost_breakdown": _cost_breakdown(model, input_tokens, output_tokens),
        }


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _inspect_chunk(chunk: Any) -> tuple[str, str | None, dict[str, int] | None]:
    """Pull the text delta, finish_reason, and (if present) usage out of a
    LiteLLM streaming chunk. Usage is only populated on the final chunk, and
    only for providers that report it mid-stream via LiteLLM."""
    try:
        choice = chunk.choices[0]
    except (AttributeError, IndexError):
        return "", None, None

    delta_obj = getattr(choice, "delta", None)
    content = getattr(delta_obj, "content", None) or ""
    finish_reason = getattr(choice, "finish_reason", None)
    finish_reason = finish_reason.lower() if finish_reason else None

    usage = None
    usage_obj = getattr(chunk, "usage", None)
    if usage_obj is not None:
        input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": getattr(usage_obj, "total_tokens", input_tokens + output_tokens)
            or (input_tokens + output_tokens),
        }

    return content, finish_reason, usage


@lru_cache()
def get_llm_wrapper() -> LLMWrapper:
    """Process-wide singleton wrapper, configured from Settings."""
    settings = get_settings()
    return LLMWrapper(
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
        primary_model=settings.llm_model,
        fallback_model=settings.fallback_model,
        timeout=settings.llm_timeout_seconds,
        num_retries=settings.llm_num_retries,
        cache=get_cache(),
    )
