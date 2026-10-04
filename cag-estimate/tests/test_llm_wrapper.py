"""Offline tests for LLMWrapper: caching, streaming, and structured extraction."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import fakeredis
import litellm
import pytest
import redis
from instructor.core.exceptions import InstructorRetryException
from pydantic import BaseModel, field_validator

from cag_estimate.services.cache import EstimationCache
from cag_estimate.services.llm_wrapper import LLMWrapper

PRIMARY = "claude-haiku-4-5-20251001"
FALLBACK = "gpt-4o-mini"


def _wrapper(cache: EstimationCache) -> LLMWrapper:
    return LLMWrapper(
        anthropic_api_key="a",
        openai_api_key="o",
        primary_model=PRIMARY,
        fallback_model=FALLBACK,
        timeout=5,
        num_retries=0,
        cache=cache,
    )


@pytest.fixture
def cache() -> EstimationCache:
    return EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)


def _text_response(text: str = "hello", model: str = PRIMARY) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model=model,
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _tool_response(arguments: dict[str, Any], model: str = PRIMARY) -> litellm.ModelResponse:
    return litellm.ModelResponse(
        model=model,
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Out", "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    )


def _chunk(text: str = "", finish: str | None = None, usage: Any = None, model: str = PRIMARY):
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=usage, model=model
    )


class FakeRouter:
    """Records calls and replays scripted responses (or raises scripted errors)."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _BrokenRedis:
    def get(self, key: str) -> None:
        raise redis.RedisError("down")

    def setex(self, *args: Any) -> None:
        raise redis.RedisError("down")


# ---------------------------------------------------------------- complete()


def test_complete_miss_calls_router_and_caches(cache: EstimationCache) -> None:
    wrapper = _wrapper(cache)
    wrapper.router = FakeRouter([_text_response("42")])

    first = wrapper.complete(system_prompt="s", user_message="u")
    second = wrapper.complete(system_prompt="s", user_message="u")

    assert first["estimation"] == "42" and first["cache_hit"] is False
    assert first["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert second["estimation"] == "42" and second["cache_hit"] is True
    assert len(wrapper.router.calls) == 1
    assert wrapper.router.calls[0]["messages"][0] == {"role": "system", "content": "s"}


def test_complete_failure_is_reraised(cache: EstimationCache) -> None:
    wrapper = _wrapper(cache)
    wrapper.router = FakeRouter([RuntimeError("boom")])

    with pytest.raises(RuntimeError):
        wrapper.complete(system_prompt="s", user_message="u")


def test_complete_survives_cache_outage() -> None:
    wrapper = _wrapper(EstimationCache(_BrokenRedis(), ttl=60))  # type: ignore[arg-type]
    wrapper.router = FakeRouter([_text_response("ok")])

    result = wrapper.complete(system_prompt="s", user_message="u")

    assert result["estimation"] == "ok" and result["cache_hit"] is False


# ----------------------------------------------------------- complete_stream()


def test_stream_live_accumulates_and_caches(cache: EstimationCache) -> None:
    wrapper = _wrapper(cache)
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    wrapper.router = FakeRouter([iter([_chunk("a"), _chunk("b"), _chunk("", "stop", usage)])])
    meta: dict[str, Any] = {}

    chunks = list(wrapper.complete_stream(system_prompt="s", user_message="u", result=meta))

    assert chunks == ["a", "b"]
    assert meta["usage"] == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    assert meta["finish_reason"] == "stop" and meta["provider"] == "anthropic"
    replay = wrapper.complete(system_prompt="s", user_message="u")
    assert replay["estimation"] == "ab" and replay["cache_hit"] is True


def test_stream_cache_hit_replays_single_chunk(cache: EstimationCache) -> None:
    wrapper = _wrapper(cache)
    wrapper.router = FakeRouter([_text_response("cached text")])
    wrapper.complete(system_prompt="s", user_message="u")
    wrapper.router = FakeRouter([])
    meta: dict[str, Any] = {}

    chunks = list(wrapper.complete_stream(system_prompt="s", user_message="u", result=meta))

    assert chunks == ["cached text"]
    assert meta["model"] == PRIMARY and wrapper.router.calls == []


def test_stream_failure_is_reraised(cache: EstimationCache) -> None:
    wrapper = _wrapper(cache)
    wrapper.router = FakeRouter([RuntimeError("boom")])

    with pytest.raises(RuntimeError):
        list(wrapper.complete_stream(system_prompt="s", user_message="u"))


def test_stream_survives_cache_outage() -> None:
    wrapper = _wrapper(EstimationCache(_BrokenRedis(), ttl=60))  # type: ignore[arg-type]
    wrapper.router = FakeRouter([iter([_chunk("x", "stop")])])

    assert list(wrapper.complete_stream(system_prompt="s", user_message="u")) == ["x"]


# ----------------------------------------------------------- complete_structured


class Out(BaseModel):
    n: int

    @field_validator("n")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n must be positive")
        return value


def test_structured_success_returns_model_and_meta(cache: EstimationCache) -> None:
    wrapper = _wrapper(cache)
    wrapper.router = FakeRouter([_tool_response({"n": 3})])

    parsed, meta = wrapper.complete_structured(
        system_prompt="s", user_message="u", response_model=Out
    )

    assert isinstance(parsed, Out) and parsed.n == 3
    assert meta["model"] == PRIMARY and meta["provider"] == "anthropic"
    assert meta["usage"] == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    assert meta["cost_breakdown"]["total_cost_usd"] > 0
    assert meta["latency_ms"] >= 0
    assert wrapper.router.calls[0]["model"] == "estimator-primary"


def test_structured_retries_on_validation_failure(cache: EstimationCache) -> None:
    wrapper = _wrapper(cache)
    wrapper.router = FakeRouter([_tool_response({"n": -1}), _tool_response({"n": 2})])

    parsed, _ = wrapper.complete_structured(system_prompt="s", user_message="u", response_model=Out)

    assert parsed.n == 2
    assert len(wrapper.router.calls) == 2


def test_structured_exhausted_retries_propagate(cache: EstimationCache) -> None:
    wrapper = _wrapper(cache)
    wrapper.router = FakeRouter([_tool_response({"n": -1}), _tool_response({"n": -1})])

    with pytest.raises(Exception) as exc_info:
        wrapper.complete_structured(
            system_prompt="s", user_message="u", response_model=Out, max_retries=2
        )

    assert "n must be positive" in str(exc_info.value)


def test_structured_goes_through_router_fallback(cache: EstimationCache) -> None:
    """The Router owns primary->fallback; the structured call must use it."""
    wrapper = _wrapper(cache)
    wrapper.router = FakeRouter([_tool_response({"n": 1}, model=FALLBACK)])

    _, meta = wrapper.complete_structured(system_prompt="s", user_message="u", response_model=Out)

    assert meta["model"] == FALLBACK and meta["provider"] == "openai"
    assert len(wrapper.router.calls) == 1


def test_structured_provider_failure_propagates(cache: EstimationCache) -> None:
    wrapper = _wrapper(cache)
    wrapper.router = FakeRouter([RuntimeError("boom")])

    with pytest.raises(InstructorRetryException) as exc_info:
        wrapper.complete_structured(system_prompt="s", user_message="u", response_model=Out)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
