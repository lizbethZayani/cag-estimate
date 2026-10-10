"""Offline tests for the actor-critic-boss schemas."""

from typing import Any

import pytest
from pydantic import ValidationError

from cag_estimate.config import Settings
from cag_estimate.schemas.acb import (
    ACBIteration,
    ACBSessionResponse,
    BossTrace,
    CriticFeedback,
    CriticIssue,
)
from cag_estimate.schemas.session import ProjectMetadata
from tests.conftest import build_estimation


def issue(severity: str = "major", **overrides: Any) -> dict[str, Any]:
    data = {
        "category": "math_error",
        "severity": severity,
        "field_path": "summary.total_hours",
        "description": "Hours do not add up.",
    }
    return {**data, **overrides}


def test_accept_without_issues_is_valid() -> None:
    feedback = CriticFeedback(verdict="accept", issues=[], confidence_in_review=80)

    assert feedback.issues == []


def test_accept_may_carry_minor_issues() -> None:
    feedback = CriticFeedback(verdict="accept", issues=[issue("minor")], confidence_in_review=70)

    assert feedback.issues[0].severity == "minor"


def test_needs_iteration_with_only_minor_issues_raises() -> None:
    with pytest.raises(ValidationError, match="critical, major"):
        CriticFeedback(verdict="needs_iteration", issues=[issue("minor")], confidence_in_review=70)


def test_needs_iteration_without_issues_raises() -> None:
    with pytest.raises(ValidationError):
        CriticFeedback(verdict="needs_iteration", issues=[], confidence_in_review=70)


@pytest.mark.parametrize("severity", ["major", "critical"])
def test_needs_iteration_with_a_blocking_issue_is_valid(severity: str) -> None:
    feedback = CriticFeedback(
        verdict="needs_iteration", issues=[issue(severity)], confidence_in_review=70
    )

    assert feedback.verdict == "needs_iteration"


def test_reject_without_issues_raises() -> None:
    with pytest.raises(ValidationError, match="reject"):
        CriticFeedback(verdict="reject", issues=[], confidence_in_review=70)


def test_reject_with_an_issue_is_valid() -> None:
    feedback = CriticFeedback(verdict="reject", issues=[issue("critical")], confidence_in_review=70)

    assert feedback.verdict == "reject"


def test_at_most_twelve_issues() -> None:
    with pytest.raises(ValidationError):
        CriticFeedback(verdict="accept", issues=[issue("minor")] * 13, confidence_in_review=70)


@pytest.mark.parametrize("confidence", [-1, 101])
def test_confidence_must_be_a_percentage(confidence: int) -> None:
    with pytest.raises(ValidationError):
        CriticFeedback(verdict="accept", issues=[], confidence_in_review=confidence)


@pytest.mark.parametrize(
    "overrides",
    [
        {"field_path": ""},
        {"field_path": "x" * 121},
        {"description": "four"},
        {"description": "x" * 501},
        {"suggested_fix": "x" * 301},
        {"category": "style"},
        {"severity": "blocker"},
    ],
)
def test_issue_field_limits(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CriticIssue(**issue(**overrides))


def test_issue_suggested_fix_is_optional() -> None:
    assert CriticIssue(**issue()).suggested_fix is None


def test_iteration_defaults_and_bounds() -> None:
    iteration = ACBIteration(
        iteration=0, decision_after="accept", critic_verdict="accept", critic_confidence=90
    )

    assert iteration.critic_failed is False
    assert iteration.issue_summary == []
    with pytest.raises(ValidationError):
        ACBIteration(
            iteration=-1, decision_after="accept", critic_verdict="accept", critic_confidence=90
        )
    with pytest.raises(ValidationError):
        ACBIteration(
            iteration=0, decision_after="accept", critic_verdict="accept", critic_confidence=101
        )


def test_boss_trace_caveats_default_to_empty() -> None:
    trace = BossTrace(iterations=[], final_decision="accept", iterations_run=0)

    assert trace.caveats == []


def test_acb_session_response_extends_the_session_response() -> None:
    response = ACBSessionResponse(
        result=build_estimation(),
        session_id="s1",
        project_metadata=ProjectMetadata(),
        turns=1,
        resolved_tier="default",
        tier_rule="default",
        acb=BossTrace(iterations=[], final_decision="accept", iterations_run=1),
    )

    assert response.acb.final_decision == "accept"
    assert response.attachments == []


def test_acb_settings_defaults() -> None:
    settings = Settings(anthropic_api_key="test")

    assert settings.acb_max_iterations == 2
    assert settings.critic_model is None


def test_acb_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACB_MAX_ITERATIONS", "3")
    monkeypatch.setenv("CRITIC_MODEL", "claude-critic")

    settings = Settings(anthropic_api_key="test")

    assert (settings.acb_max_iterations, settings.critic_model) == (3, "claude-critic")
