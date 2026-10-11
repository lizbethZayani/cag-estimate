"""A wrong total from the model must be repaired, not retried or turned into a 502."""

from __future__ import annotations

import json
from typing import Any

import fakeredis
import litellm
import pytest

from cag_estimate.prompts import render_system_prompt
from cag_estimate.schemas.estimation import EstimationRequest, ProjectEstimation
from cag_estimate.services.cache import EstimationCache
from cag_estimate.services.estimation import EstimationService
from cag_estimate.services.llm_wrapper import LLMWrapper
from cag_estimate.services.session_estimation import SessionEstimationService
from cag_estimate.services.sessions import Session
from tests.conftest import build_estimation

PRIMARY = "claude-haiku-4-5-20251001"
WRONG_SUMMARY = {
    "total_hours": 29,
    "total_cost_usd": 1,
    "team_size": "2 developers",
    "estimated_duration_weeks": 2,
    "hourly_rate": 40,
}


def _wrong_total_payload() -> dict[str, Any]:
    return build_estimation().model_dump() | {"summary": WRONG_SUMMARY}


class WrongTotalRouter:
    """Replies with task hours summing to 30 but a summary that says 29."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def completion(self, **kwargs: Any) -> litellm.ModelResponse:
        self.calls.append(kwargs)
        arguments = json.dumps(_wrong_total_payload())
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


class WrongTotalWrapper:
    """Fake wrapper whose result comes from a payload with a wrong total."""

    primary_model = "fake-model"

    def complete_structured(self, **kwargs: Any) -> tuple[ProjectEstimation, dict[str, Any]]:
        return ProjectEstimation.model_validate(_wrong_total_payload()), {"model": "fake-model"}


@pytest.fixture
def cache() -> EstimationCache:
    return EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)


def test_wrong_total_is_accepted_on_the_first_call(cache: EstimationCache) -> None:
    wrapper = LLMWrapper(
        anthropic_api_key="a",
        openai_api_key="o",
        primary_model=PRIMARY,
        fallback_model="gpt-4o-mini",
        timeout=5,
        num_retries=0,
        cache=cache,
    )
    wrapper.router = WrongTotalRouter()

    parsed, _ = wrapper.complete_structured(
        system_prompt="s", user_message="u", response_model=ProjectEstimation
    )

    assert len(wrapper.router.calls) == 1
    assert parsed.summary.total_hours == 30


def test_estimation_service_returns_the_derived_total_and_cost(cache: EstimationCache) -> None:
    request = EstimationRequest(transcription="Build an app", hourly_rate=40)

    response = EstimationService(WrongTotalWrapper(), cache).estimate(request)

    assert response.result.summary.total_hours == 30
    assert response.result.summary.total_cost_usd == 30 * 40


def test_session_service_returns_the_derived_total_and_cost() -> None:
    session = Session("s1", render_system_prompt, max_turns=6)

    response = SessionEstimationService(WrongTotalWrapper()).estimate(
        session, "Build a portal", [], 40
    )

    assert response.result.summary.total_hours == 30
    assert response.result.summary.total_cost_usd == 30 * 40
