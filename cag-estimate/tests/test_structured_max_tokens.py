"""The configured structured max_tokens must reach the provider-level call.

These tests run the real services with a real ``LLMWrapper`` whose router is
faked, so any explicit limit a service passes would show up in the recorded
provider kwargs.
"""

from __future__ import annotations

from typing import Any

import fakeredis
import litellm
import pytest

from cag_estimate.prompts import render_system_prompt
from cag_estimate.schemas.estimation import EstimationRequest
from cag_estimate.services.cache import EstimationCache
from cag_estimate.services.estimation import EstimationService
from cag_estimate.services.llm_wrapper import LLMWrapper
from cag_estimate.services.session_estimation import SessionEstimationService
from cag_estimate.services.sessions import Session
from tests.conftest import build_estimation

PRIMARY = "claude-haiku-4-5-20251001"


class RecordingRouter:
    """Replies with a valid estimation and records every provider call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def completion(self, **kwargs: Any) -> litellm.ModelResponse:
        self.calls.append(kwargs)
        arguments = build_estimation().model_dump_json()
        return litellm.ModelResponse(
            model=PRIMARY,
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
                                "function": {"name": "ProjectEstimation", "arguments": arguments},
                            }
                        ],
                    },
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


@pytest.fixture
def cache() -> EstimationCache:
    return EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)


def _wrapper(cache: EstimationCache, structured_max_tokens: int) -> LLMWrapper:
    wrapper = LLMWrapper(
        anthropic_api_key="a",
        openai_api_key="o",
        primary_model=PRIMARY,
        fallback_model="gpt-4o-mini",
        timeout=5,
        num_retries=0,
        cache=cache,
        structured_max_tokens=structured_max_tokens,
    )
    wrapper.router = RecordingRouter()
    return wrapper


@pytest.mark.parametrize("configured", [8192, 12345])
def test_estimation_service_uses_configured_structured_max_tokens(
    cache: EstimationCache, configured: int
) -> None:
    wrapper = _wrapper(cache, configured)
    request = EstimationRequest(transcription="Build an app", hourly_rate=40)

    EstimationService(wrapper, cache).estimate(request)

    assert [call["max_tokens"] for call in wrapper.router.calls] == [configured]


@pytest.mark.parametrize("configured", [8192, 12345])
def test_session_estimation_service_uses_configured_structured_max_tokens(
    cache: EstimationCache, configured: int
) -> None:
    wrapper = _wrapper(cache, configured)
    session = Session("s1", render_system_prompt, max_turns=6)

    SessionEstimationService(wrapper).estimate(session, "Build a portal", [], 40)

    assert [call["max_tokens"] for call in wrapper.router.calls] == [configured]
