"""Settings and wiring of the summary mode, plus the end-to-end eviction flow."""

from typing import Any

import pytest

from cag_estimate import dependencies
from cag_estimate.config import Settings
from cag_estimate.prompts import render_system_prompt
from cag_estimate.services.llm_summarizer import LlmSummarizer
from cag_estimate.services.session_compression import (
    AnchorDetector,
    HistoryCompactor,
)
from cag_estimate.services.session_estimation import SessionEstimationService
from cag_estimate.services.sessions import ConversationHistory, ProjectMetadata, Session
from tests.conftest import build_estimation

LLM_SUMMARY = "- LLM summary: portal agreed"


class FakeWrapper:
    primary_model = "fake-model"

    def __init__(self) -> None:
        self.summary_calls: list[dict[str, Any]] = []
        self.estimate_calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        self.summary_calls.append(kwargs)
        return {"estimation": LLM_SUMMARY}

    def complete_structured(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.estimate_calls.append(kwargs)
        return build_estimation(), {"model": "fake-model"}


def use_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    settings = Settings(anthropic_api_key="test", **overrides)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)


def test_summary_settings_defaults():
    settings = Settings(anthropic_api_key="test")

    assert settings.session_summary_mode == "llm"
    assert settings.session_summary_model is None


def test_summary_mode_rejects_unknown_values():
    with pytest.raises(ValueError):
        Settings(anthropic_api_key="test", session_summary_mode="magic")


def test_llm_mode_with_a_wrapper_summarizes_through_the_llm(monkeypatch):
    use_settings(monkeypatch, session_summary_model="claude-cheap")
    wrapper = FakeWrapper()

    compactor = dependencies.build_history_compactor(wrapper)
    history = ConversationHistory(ProjectMetadata(), lambda _m: "SYS", max_turns=6)
    compactor.compact(history, [("plain request", "")])

    assert history.summary == LLM_SUMMARY
    assert wrapper.summary_calls[0]["model_override"] == "claude-cheap"


def test_extractive_mode_never_calls_the_llm(monkeypatch):
    use_settings(monkeypatch, session_summary_mode="extractive")
    wrapper = FakeWrapper()

    compactor = dependencies.build_history_compactor(wrapper)
    history = ConversationHistory(ProjectMetadata(), lambda _m: "SYS", max_turns=6)
    compactor.compact(history, [("plain request", "")])

    assert history.summary == "- plain request"
    assert wrapper.summary_calls == []


def test_llm_mode_without_a_wrapper_falls_back_to_extractive(monkeypatch):
    use_settings(monkeypatch)

    compactor = dependencies.build_history_compactor(None)
    history = ConversationHistory(ProjectMetadata(), lambda _m: "SYS", max_turns=6)
    compactor.compact(history, [("plain request", "")])

    assert history.summary == "- plain request"


def test_disabled_compression_builds_no_compactor(monkeypatch):
    use_settings(monkeypatch, session_compression_enabled=False)

    assert dependencies.build_history_compactor(FakeWrapper()) is None


def test_nine_turns_make_one_summary_call_per_evicting_turn():
    wrapper = FakeWrapper()
    compactor = HistoryCompactor(AnchorDetector(), LlmSummarizer(wrapper, 2000), 5, 2000)
    session = Session("s1", render_system_prompt, max_turns=6)
    service = SessionEstimationService(wrapper, compactor=compactor)

    for index in range(9):
        service.estimate(session, f"request number {index}", [], 40)

    # Turns 7, 8 and 9 evict one pair each; earlier turns never summarize.
    assert len(wrapper.summary_calls) == 3
    assert all(call["use_cache"] is False for call in wrapper.summary_calls)
    first_content = wrapper.estimate_calls[-1]["history"][0]["content"]
    assert LLM_SUMMARY in first_content
