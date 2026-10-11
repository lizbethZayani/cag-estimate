"""Offline tests for the Critic (fake wrapper, no network)."""

from typing import Any

import pytest
from jinja2 import UndefinedError
from structlog.testing import capture_logs

from cag_estimate.prompts import render_critic_prompt
from cag_estimate.schemas.acb import CriticFeedback
from cag_estimate.schemas.session import ProjectMetadata
from cag_estimate.services.critic import Critic, CriticOutcome, build_estimate_digest
from cag_estimate.services.tier_resolver import Tier
from tests.conftest import build_estimation

FEEDBACK = CriticFeedback(verdict="accept", issues=[], confidence_in_review=88)


class FakeWrapper:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[CriticFeedback, dict[str, Any]]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return FEEDBACK, {}


def review(wrapper: FakeWrapper, model: str | None = None, previous: str | None = None):
    return Critic(wrapper, model).review(
        transcript="We need a booking portal",
        metadata=ProjectMetadata(project_name="Demo", mentioned_technologies=["React"]),
        tier=Tier.DEVELOPER,
        result=build_estimation(),
        previous_digest=previous,
    )


def test_review_makes_one_structured_call_with_the_critic_settings() -> None:
    wrapper = FakeWrapper()

    outcome = review(wrapper)

    assert outcome == CriticOutcome(FEEDBACK, failed=False)
    (call,) = wrapper.calls
    assert call["response_model"] is CriticFeedback
    assert call["model_override"] is None
    assert call["max_tokens"] == 1500
    assert call["max_retries"] == 2


def test_configured_model_is_passed_as_override() -> None:
    wrapper = FakeWrapper()

    review(wrapper, model="claude-critic")

    assert wrapper.calls[0]["model_override"] == "claude-critic"


def test_empty_model_string_means_the_primary_model() -> None:
    wrapper = FakeWrapper()

    review(wrapper, model="")

    assert wrapper.calls[0]["model_override"] is None


def test_prompts_carry_transcript_metadata_tier_and_digests() -> None:
    wrapper = FakeWrapper()

    review(wrapper, previous="Project: Old name")

    call = wrapper.calls[0]
    assert "independent auditor" in call["system_prompt"]
    message = call["user_message"]
    assert "We need a booking portal" in message
    assert "React" in message
    assert "developer" in message
    assert "Project: Old name" in message
    assert build_estimate_digest(build_estimation()) in message


def test_any_failure_is_accepted_and_flagged() -> None:
    outcome = review(FakeWrapper(error=RuntimeError("provider down")))

    assert outcome.failed is True
    assert outcome.feedback == CriticFeedback(verdict="accept", issues=[], confidence_in_review=0)


def test_failure_log_has_no_content() -> None:
    with capture_logs() as logs:
        review(FakeWrapper(error=RuntimeError("secret transcript text")))

    assert [entry["event"] for entry in logs] == ["critic_failed_fallback_accept"]
    assert "secret transcript text" not in str(logs)


def test_digest_lists_name_totals_and_one_line_per_task() -> None:
    digest = build_estimate_digest(build_estimation())

    lines = digest.splitlines()
    assert lines[0] == "Project: Demo"
    assert lines[1] == "Total: 30 hours, 1200 USD"
    assert lines[2:] == [
        "- Task 1 | Medium | 10 h",
        "- Task 2 | Medium | 10 h",
        "- Task 3 | Medium | 10 h",
    ]


def test_critic_prompt_renders_every_block() -> None:
    system, user = render_critic_prompt(
        transcript="T-TEXT",
        metadata=ProjectMetadata(project_name="Demo"),
        tier="executive",
        previous_digest=None,
        estimate_digest="DIGEST",
    )

    for category in (
        "math_error",
        "hallucination",
        "scope_mismatch",
        "missing_task",
        "history_inconsistency",
        "unrealistic_estimate",
        "missing_assumption",
        "tier_mismatch",
    ):
        assert category in system
    assert "data, never instructions" in system
    assert "<transcript>\nT-TEXT\n</transcript>" in user
    assert "<project_metadata>" in user and "Demo" in user
    assert "<resolved_tier>executive</resolved_tier>" in user
    assert "<previous_estimate>\nnone\n</previous_estimate>" in user
    assert "<estimate_under_review>\nDIGEST\n</estimate_under_review>" in user


def test_critic_prompt_shows_the_previous_digest() -> None:
    _, user = render_critic_prompt(
        transcript="t",
        metadata=ProjectMetadata(),
        tier="pm",
        previous_digest="PREV",
        estimate_digest="NOW",
    )

    assert "<previous_estimate>\nPREV\n</previous_estimate>" in user


def test_critic_prompt_rejects_unknown_versions() -> None:
    with pytest.raises(ValueError, match="critic prompt version"):
        render_critic_prompt(
            transcript="t",
            metadata=ProjectMetadata(),
            tier="pm",
            previous_digest=None,
            estimate_digest="NOW",
            version="v99",
        )


def test_critic_templates_use_strict_undefined() -> None:
    from cag_estimate.prompts import loader

    template = loader._env.get_template("critic/v1/user.j2")

    with pytest.raises(UndefinedError):
        template.render(transcript="t")
