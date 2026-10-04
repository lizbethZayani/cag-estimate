"""
LiteLLM-backed wrapper that adds provider fallback, exact-match caching, and
cost tracking to every LLM call in the estimator.

Design notes
------------
- The wrapper exposes three primitives: ``complete()`` (blocking, full
  response), ``complete_stream()`` (yields text chunks) and
  ``complete_structured()`` (Instructor-validated Pydantic model). Prompt
  building stays in ``prompts/`` and the business rules in services; this module
  only knows how to reach an LLM.
- The Router is configured with two ``model_name`` groups, ``estimator-primary``
  (Anthropic) and ``estimator-fallback`` (OpenAI), linked through ``fallbacks``
  so LiteLLM switches to the fallback only if the primary errors out or times
  out. Callers never see which provider answered until they read
  ``result["provider"]``.
- When the caller overrides the model explicitly, we bypass the Router and call
  ``litellm.completion`` directly with the matching provider's API key; that
  path has no fallback by design, since the caller asked for a specific model.
- ``complete_structured()`` hands the same dispatch function to Instructor, so
  structured calls keep the primary -> fallback behaviour. It does not cache:
  the service layer caches the validated result.
- The cache key includes the full system prompt and generation knobs, so any
  change to the CAG reference examples (which live in the system prompt)
  implicitly invalidates the cache without a manual flush.
- Cache failures fail soft: ``EstimationCache.get/set`` swallow ``RedisError``
  themselves (a Redis outage degrades to "always call the LLM"), so the wrapper
  does not wrap them again.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, TypeVar

import instructor
import litellm
import structlog
from litellm import Router
from pydantic import BaseModel

from cag_estimate.config import get_settings
from cag_estimate.services.cache import EstimationCache, get_cache

log = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


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
    if name.startswith(("gpt", "o1", "o3")):
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
    ) -> None:
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
        requested_model = model_override or self.primary_model
        cache_key = self._build_cache_key(system_prompt, user_message, requested_model, max_tokens)
        cached = self.cache.get(cache_key)
        if cached:
            return {**cached, "cache_hit": True}

        call_logger = log.bind(model=requested_model)
        call_logger.info("llm_call_started", mode="blocking")
        t0 = time.perf_counter()
        try:
            response = self._dispatch(
                model_override=model_override,
                messages=_build_messages(system_prompt, user_message),
                max_tokens=max_tokens,
            )
        except Exception as exc:
            _log_failure(call_logger, exc, t0)
            raise

        result = self._normalise_response(response, latency_ms=_elapsed_ms(t0))
        _log_completed(call_logger, result, cache_hit=False)
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
        requested_model = model_override or self.primary_model
        cache_key = self._build_cache_key(system_prompt, user_message, requested_model, max_tokens)
        cached = self.cache.get(cache_key)
        if cached:
            yield from _replay_cached(cached, requested_model, result)
            return

        call_logger = log.bind(model=requested_model)
        call_logger.info("llm_call_started", mode="stream")
        t0 = time.perf_counter()
        state = _StreamState(model_seen=requested_model)
        try:
            response = self._dispatch(
                model_override=model_override,
                messages=_build_messages(system_prompt, user_message),
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response:
                delta = state.feed(chunk)
                if delta:
                    yield delta
        except Exception as exc:
            _log_failure(call_logger, exc, t0)
            raise

        final = state.to_result(latency_ms=_elapsed_ms(t0))
        _log_completed(call_logger, final, cache_hit=False, chars=len(final["estimation"]))
        if result is not None:
            result.update({k: v for k, v in final.items() if k not in _STREAM_RESULT_SKIP})
        self.cache.set(cache_key, final)

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        response_model: type[T],
        model_override: str | None = None,
        max_tokens: int = 4000,
        max_retries: int = 3,
    ) -> tuple[T, dict[str, Any]]:
        """Run the LLM through Instructor and return ``(model_instance, meta)``.

        Instructor re-prompts the model, feeding validator errors back, up to
        ``max_retries`` times; when retries are exhausted its exception
        propagates. The call goes through the same dispatch as ``complete()``,
        so the Router's primary -> fallback behaviour is preserved. Results are
        not cached here; the service layer caches the validated model.

        ``meta`` has ``model``, ``provider``, ``latency_ms``, ``usage`` and
        ``cost_breakdown``, shaped like the ``complete()`` result.
        """
        requested_model = model_override or self.primary_model
        call_logger = log.bind(model=requested_model, response_model=response_model.__name__)
        call_logger.info("llm_call_started", mode="structured")
        client = instructor.from_litellm(self._structured_completion(model_override))
        t0 = time.perf_counter()
        try:
            parsed, raw = client.chat.completions.create_with_completion(
                messages=_build_messages(system_prompt, user_message),
                response_model=response_model,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        except Exception as exc:
            _log_failure(call_logger, exc, t0)
            raise

        meta = _response_meta(raw, latency_ms=_elapsed_ms(t0))
        _log_completed(call_logger, meta, cache_hit=False)
        return parsed, meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cache_key(system_prompt: str, user_message: str, model: str, max_tokens: int) -> str:
        return EstimationCache.make_key(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            max_tokens=max_tokens,
        )

    def _structured_completion(self, model_override: str | None) -> Any:
        """Completion callable for Instructor that reuses ``_dispatch``."""

        def completion(**kwargs: Any) -> Any:
            return self._dispatch(model_override=model_override, **kwargs)

        return completion

    def _dispatch(
        self,
        *,
        model_override: str | None,
        messages: list[dict[str, str]],
        max_tokens: int,
        stream: bool = False,
        **extra: Any,
    ) -> Any:
        """Call the Router (with fallback) or LiteLLM directly when the caller
        wants a specific model. ``extra`` carries Instructor's tool arguments."""
        kwargs: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, **extra}
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
    def _normalise_response(response: Any, *, latency_ms: float) -> dict[str, Any]:
        choice = response.choices[0]
        return {
            "estimation": choice.message.content or "",
            "finish_reason": (choice.finish_reason or "stop").lower(),
            **_response_meta(response, latency_ms=latency_ms),
        }


_STREAM_RESULT_SKIP = frozenset({"estimation", "latency_ms"})


@dataclass
class _StreamState:
    """Accumulates text, finish reason, usage and model across stream chunks."""

    model_seen: str
    full_text: list[str] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=lambda: _empty_usage())

    def feed(self, chunk: Any) -> str:
        """Absorb one chunk and return its text delta (possibly empty)."""
        delta, finish_reason, usage = _inspect_chunk(chunk)
        if delta:
            self.full_text.append(delta)
        if finish_reason:
            self.finish_reason = finish_reason
        if usage:
            self.usage = usage
        self.model_seen = getattr(chunk, "model", None) or self.model_seen
        return delta

    def to_result(self, *, latency_ms: float) -> dict[str, Any]:
        model = _normalise_model_name(self.model_seen)
        return {
            "estimation": "".join(self.full_text),
            "model": model,
            "provider": _provider_from_model(model),
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "cost_breakdown": _cost_breakdown(
                model, self.usage["input_tokens"], self.usage["output_tokens"]
            ),
            "latency_ms": latency_ms,
        }


def _build_messages(system_prompt: str, user_message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _log_failure(call_logger: Any, exc: Exception, t0: float) -> None:
    call_logger.error(
        "llm_call_failed",
        error_type=type(exc).__name__,
        error_msg=str(exc),
        latency_ms=_elapsed_ms(t0),
    )


def _log_completed(call_logger: Any, result: dict[str, Any], **extra: Any) -> None:
    call_logger.info(
        "llm_call_completed",
        model=result["model"],  # may differ from the requested model if fallback fired
        provider=result["provider"],
        tokens_in=result["usage"]["input_tokens"],
        tokens_out=result["usage"]["output_tokens"],
        finish_reason=result.get("finish_reason"),
        cost_usd=result["cost_breakdown"]["total_cost_usd"],
        latency_ms=result["latency_ms"],
        **extra,
    )


def _replay_cached(
    cached: dict[str, Any], requested_model: str, result: dict[str, Any] | None
) -> Iterator[str]:
    """Replay a cached estimation as a single chunk, filling ``result``."""
    log.info("stream_cache_hit", chars=len(cached.get("estimation", "")))
    if result is not None:
        result.update(
            model=cached.get("model", requested_model),
            provider=cached.get("provider", _provider_from_model(requested_model)),
            finish_reason=cached.get("finish_reason", "stop"),
            usage=cached.get("usage", _empty_usage()),
            cost_breakdown=cached.get("cost_breakdown", _cost_breakdown(requested_model, 0, 0)),
        )
    yield cached.get("estimation", "")


def _response_meta(response: Any, *, latency_ms: float) -> dict[str, Any]:
    """Model, provider, usage and cost shared by every non-streaming result."""
    usage = _extract_usage(response.usage) or _empty_usage()
    model = _normalise_model_name(response.model)
    return {
        "model": model,
        "provider": _provider_from_model(model),
        "usage": usage,
        "latency_ms": latency_ms,
        "cost_breakdown": _cost_breakdown(model, usage["input_tokens"], usage["output_tokens"]),
    }


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _extract_usage(usage_obj: Any) -> dict[str, int] | None:
    """Normalise a LiteLLM usage object; ``None`` when the provider sent none."""
    if usage_obj is None:
        return None
    input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) or (input_tokens + output_tokens),
    }


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
    return content, finish_reason, _extract_usage(getattr(chunk, "usage", None))


@lru_cache
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
