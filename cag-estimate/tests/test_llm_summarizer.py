"""Offline tests for LlmSummarizer (fake wrapper, no network)."""

from typing import Any

import pytest

from cag_estimate.services.llm_summarizer import LlmSummarizer
from cag_estimate.services.session_compression import (
    AnchorDetector,
    HistoryCompactor,
)
from cag_estimate.services.sessions import ConversationHistory, ProjectMetadata
from tests.conftest import build_estimation


class FakeWrapper:
    def __init__(self, text: str = "- web portal agreed", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"estimation": self.text, "usage": {}, "cost_breakdown": {}}


ESTIMATE_JSON = build_estimation().model_dump_json()


def summarize(wrapper: FakeWrapper, evicted, previous=None, max_chars=2000, model=None) -> str:
    return LlmSummarizer(wrapper, max_chars, model).summarize(previous, evicted)


def test_call_bypasses_the_cache_and_uses_a_short_budget():
    wrapper = FakeWrapper()

    summarize(wrapper, [("Add a portal", ESTIMATE_JSON)])

    call = wrapper.calls[0]
    assert call["use_cache"] is False
    assert call["max_tokens"] == 600
    assert call["model_override"] is None


def test_configured_model_is_passed_as_override():
    wrapper = FakeWrapper()

    summarize(wrapper, [("Add a portal", ESTIMATE_JSON)], model="claude-cheap")

    assert wrapper.calls[0]["model_override"] == "claude-cheap"


def test_previous_summary_and_evicted_user_texts_reach_the_prompt():
    wrapper = FakeWrapper()

    summarize(wrapper, [("Add a portal", ESTIMATE_JSON), ("Use React", ESTIMATE_JSON)], "- old")

    call = wrapper.calls[0]
    assert "- old" in call["user_message"]
    assert "Add a portal" in call["user_message"]
    assert "Use React" in call["user_message"]
    assert "2000 characters" in call["system_prompt"]


def test_assistant_json_is_reduced_to_a_digest():
    wrapper = FakeWrapper()

    summarize(wrapper, [("Add a portal", ESTIMATE_JSON)])

    message = wrapper.calls[0]["user_message"]
    assert "Demo" in message
    assert "30" in message and "1200" in message and "3 tasks" in message
    assert "meeting_summary" not in message
    assert "estimated_hours" not in message


def test_unparseable_assistant_content_falls_back_to_its_first_characters():
    wrapper = FakeWrapper()

    summarize(wrapper, [("hello", "not json " + "x" * 1000)])

    message = wrapper.calls[0]["user_message"]
    assert "not json" in message
    assert "x" * 400 not in message


def test_empty_unparseable_assistant_content_is_tolerated():
    wrapper = FakeWrapper()

    assert summarize(wrapper, [("hello", "   ")]) == "- web portal agreed"


def test_anchor_overflow_pair_renders_as_pinned_constraint():
    wrapper = FakeWrapper()

    summarize(wrapper, [("Under NDA", "")])

    assert "pinned constraint" in wrapper.calls[0]["user_message"]
    assert "<assistant_digest>" not in wrapper.calls[0]["user_message"]


def test_result_is_collapsed_and_clipped_at_a_word_boundary():
    wrapper = FakeWrapper("alpha   beta\ngamma delta epsilon")

    result = summarize(wrapper, [("hi", ESTIMATE_JSON)], max_chars=18)

    assert result == "alpha beta gamma"
    assert len(result) <= 18


def test_single_overlong_word_is_hard_cut():
    result = summarize(FakeWrapper("x" * 50), [("hi", ESTIMATE_JSON)], max_chars=10)

    assert result == "x" * 10


@pytest.mark.parametrize("text", ["", "   \n "])
def test_empty_result_raises(text):
    with pytest.raises(ValueError, match="empty"):
        summarize(FakeWrapper(text), [("hi", ESTIMATE_JSON)])


def test_wrapper_failure_propagates():
    with pytest.raises(RuntimeError):
        summarize(FakeWrapper(error=RuntimeError("down")), [("hi", ESTIMATE_JSON)])


def test_compactor_falls_back_to_extractive_when_the_llm_summarizer_fails():
    summarizer = LlmSummarizer(FakeWrapper(error=RuntimeError("down")), 2000)
    compactor = HistoryCompactor(AnchorDetector(), summarizer, 5, 2000)
    history = ConversationHistory(ProjectMetadata(), lambda _m: "SYS", max_turns=6)

    compactor.compact(history, [("plain request", ESTIMATE_JSON)])

    assert history.summary == "- plain request"
