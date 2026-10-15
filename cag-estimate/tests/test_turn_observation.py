"""Offline tests for the per-turn observation (fake wrapper, real session state)."""

from typing import Any

import pytest
import structlog

from cag_estimate.prompts import render_system_prompt
from cag_estimate.schemas.estimation import ProjectEstimation
from cag_estimate.schemas.session import SessionEstimationResponse, TurnObservation
from cag_estimate.services.attachments import ExtractedAttachment
from cag_estimate.services.llm_summarizer import LlmSummarizer
from cag_estimate.services.session_compression import (
    AnchorDetector,
    ExtractiveSummarizer,
    HistoryCompactor,
)
from cag_estimate.services.session_estimation import MeteredLLM, SessionEstimationService
from cag_estimate.services.sessions import Session
from tests.conftest import build_estimation


def _meta(tokens_in: int, tokens_out: int, cost: float, latency: float) -> dict[str, Any]:
    return {
        "model": "fake-model",
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
        "cost_breakdown": {"total_cost_usd": cost},
        "latency_ms": latency,
    }


class MeteredFakeWrapper:
    primary_model = "fake-model"

    def __init__(self, meta: dict[str, Any] | None) -> None:
        self.meta = meta

    def complete_structured(self, **_kwargs: Any) -> tuple[ProjectEstimation, dict[str, Any]]:
        return build_estimation(), self.meta if self.meta is not None else {"model": "fake"}

    def complete(self, **_kwargs: Any) -> dict[str, Any]:
        return {"estimation": "- summarised", **_meta(10, 5, 0.001, 20.0)}


def _session(max_turns: int = 6) -> Session:
    return Session("s1", render_system_prompt, max_turns=max_turns)


def test_observation_aggregates_the_actor_call():
    service = SessionEstimationService(MeteredFakeWrapper(_meta(100, 40, 0.01, 250.5)))
    attachment = ExtractedAttachment("a.txt", "x" * 30)

    response = service.estimate(_session(), "Build a portal", [attachment], 40)

    obs = response.observation
    assert isinstance(obs, TurnObservation)
    assert obs.turn_index == 1
    assert obs.session_id == "s1"
    assert obs.attachments_total_chars == 30
    assert obs.enriched_transcript_chars > len("Build a portal")
    assert obs.messages_in_window == 2
    assert (obs.tokens_in, obs.tokens_out) == (100, 40)
    assert obs.cost_usd == pytest.approx(0.01)
    assert obs.latency_ms == pytest.approx(250.5)
    assert obs.cache_hit_kind == "none"
    assert obs.last_resolved_tier in {"executive", "pm", "developer", "default"}


def test_turn_index_counts_beyond_the_window():
    service = SessionEstimationService(MeteredFakeWrapper(_meta(1, 1, 0.0, 1.0)))
    session = _session(max_turns=1)

    indexes = [
        service.estimate(session, f"turn {i}", [], 40).observation.turn_index for i in range(3)
    ]

    assert indexes == [1, 2, 3]


def test_fake_meta_without_usage_counts_as_zero():
    service = SessionEstimationService(MeteredFakeWrapper(None))

    obs = service.estimate(_session(), "Build a portal", [], 40).observation

    assert (obs.tokens_in, obs.tokens_out, obs.cost_usd, obs.latency_ms) == (0, 0, 0.0, 0.0)


def test_summarizer_calls_are_added_when_compaction_runs():
    inner = MeteredFakeWrapper(_meta(100, 40, 0.01, 100.0))
    llm = MeteredLLM(inner)
    compactor = HistoryCompactor(AnchorDetector(), LlmSummarizer(llm, 500), 3, 500)
    service = SessionEstimationService(llm, compactor=compactor)
    session = _session(max_turns=1)

    first = service.estimate(session, "first request", [], 40).observation
    second = service.estimate(session, "second request", [], 40).observation

    assert (first.tokens_in, first.cost_usd) == (100, pytest.approx(0.01))
    assert second.tokens_in == 110
    assert second.tokens_out == 45
    assert second.cost_usd == pytest.approx(0.011)
    assert second.latency_ms == pytest.approx(120.0)
    assert second.summary_chars == len("- summarised")


def test_extractive_compaction_adds_no_llm_cost_and_reports_context_sizes():
    compactor = HistoryCompactor(AnchorDetector(), ExtractiveSummarizer(500), 3, 500)
    service = SessionEstimationService(
        MeteredFakeWrapper(_meta(5, 5, 0.5, 5.0)), compactor=compactor
    )
    session = _session(max_turns=1)

    service.estimate(session, "first request", [], 40)
    obs = service.estimate(session, "second request", [], 40).observation

    assert obs.cost_usd == pytest.approx(0.5)
    assert obs.summary_chars == len("- first request")


def test_turn_observed_event_is_logged_once_with_the_observation():
    service = SessionEstimationService(MeteredFakeWrapper(_meta(7, 3, 0.002, 9.0)))

    with structlog.testing.capture_logs() as logs:
        response = service.estimate(_session(), "Build a portal", [], 40)

    events = [entry for entry in logs if entry["event"] == "turn_observed"]
    assert len(events) == 1
    assert events[0]["tokens_in"] == 7
    assert events[0]["session_id"] == response.observation.session_id


def test_observation_is_optional_on_the_response_schema():
    assert SessionEstimationResponse.model_fields["observation"].default is None
