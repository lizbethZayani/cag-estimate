"""Offline tests for the Boss orchestrator (scripted actor and critic)."""

from typing import Any

import pytest

from cag_estimate.schemas.acb import CriticFeedback, CriticIssue
from cag_estimate.schemas.estimation import ProjectEstimation
from cag_estimate.services.boss import Boss
from cag_estimate.services.critic import CriticOutcome
from tests.conftest import build_estimation


def make_issue(severity: str = "major", index: int = 0, **overrides: Any) -> CriticIssue:
    data = {
        "category": "math_error",
        "severity": severity,
        "field_path": f"tasks[{index}].estimated_hours",
        "description": "Hours look wrong.",
    }
    return CriticIssue(**{**data, **overrides})


def feedback(verdict: str, issues: list[CriticIssue] | None = None) -> CriticFeedback:
    return CriticFeedback(verdict=verdict, issues=issues or [], confidence_in_review=80)


ACCEPT = CriticOutcome(feedback("accept"), failed=False)
ITERATE = CriticOutcome(feedback("needs_iteration", [make_issue()]), failed=False)
REJECT = CriticOutcome(feedback("reject", [make_issue("critical")]), failed=False)


class Scripted:
    """Actor and critic with canned answers; records what each one received."""

    def __init__(self, outcomes: list[CriticOutcome]) -> None:
        self.outcomes = list(outcomes)
        self.drafts: list[ProjectEstimation] = []
        self.feedbacks: list[CriticFeedback | None] = []
        self.reviewed: list[ProjectEstimation] = []

    def actor(self, received: CriticFeedback | None) -> ProjectEstimation:
        self.feedbacks.append(received)
        draft = build_estimation(project_name=f"Draft {len(self.drafts) + 1}")
        self.drafts.append(draft)
        return draft

    def critic(self, draft: ProjectEstimation) -> CriticOutcome:
        self.reviewed.append(draft)
        return self.outcomes[len(self.reviewed) - 1]


def run(outcomes: list[CriticOutcome], max_iterations: int = 2):
    script = Scripted(outcomes)
    return script, Boss(max_iterations).run(script.actor, script.critic)


def test_accept_on_the_first_pass_calls_the_actor_once_without_feedback() -> None:
    script, outcome = run([ACCEPT])

    assert script.feedbacks == [None]
    assert outcome.result is script.drafts[0]
    assert outcome.trace.final_decision == "accept"
    assert outcome.trace.iterations_run == 1
    assert outcome.trace.caveats == []


def test_iterate_then_accept_feeds_the_first_review_to_the_second_draft() -> None:
    script, outcome = run([ITERATE, ACCEPT])

    assert script.feedbacks == [None, ITERATE.feedback]
    assert outcome.result is script.drafts[1]
    assert [i.decision_after for i in outcome.trace.iterations] == ["iterate", "accept"]
    assert outcome.trace.final_decision == "accept"
    assert outcome.trace.iterations_run == 2


def test_only_the_latest_review_is_fed_back() -> None:
    second = CriticOutcome(feedback("needs_iteration", [make_issue(index=7)]), failed=False)
    script, _ = run([ITERATE, second, ACCEPT], max_iterations=3)

    assert script.feedbacks == [None, ITERATE.feedback, second.feedback]


def test_exhausted_budget_synthesizes_the_last_draft_unmodified() -> None:
    script, outcome = run([ITERATE, ITERATE])

    assert outcome.result is script.drafts[-1]
    assert outcome.result == script.drafts[-1].model_copy()
    assert outcome.trace.final_decision == "synthesize"
    assert outcome.trace.iterations_run == 2
    assert [i.decision_after for i in outcome.trace.iterations] == ["iterate", "synthesize"]
    assert outcome.trace.caveats == [
        "[major] math_error (tasks[0].estimated_hours): Hours look wrong."
    ]


def test_caveats_are_capped_at_five_issues_and_two_hundred_characters() -> None:
    issues = [make_issue(index=n, description="d" * 400) for n in range(8)]
    final = CriticOutcome(feedback("needs_iteration", issues), failed=False)

    _, outcome = run([final], max_iterations=1)

    assert len(outcome.trace.caveats) == 5
    assert all(len(caveat) <= 200 for caveat in outcome.trace.caveats)
    assert len(outcome.trace.iterations[0].issue_summary) == 5


def test_reject_synthesizes_immediately_even_with_budget_left() -> None:
    script, outcome = run([REJECT], max_iterations=3)

    assert len(script.drafts) == 1
    assert outcome.result is script.drafts[0]
    assert outcome.trace.final_decision == "synthesize"
    assert outcome.trace.iterations_run == 1
    assert outcome.trace.caveats[0].startswith("[critical] math_error")


def test_a_failed_critic_is_accepted_and_flagged() -> None:
    failed = CriticOutcome(feedback("accept"), failed=True)

    script, outcome = run([failed])

    assert outcome.result is script.drafts[0]
    assert outcome.trace.final_decision == "accept"
    only = outcome.trace.iterations[0]
    assert (only.critic_failed, only.critic_confidence) == (True, 0)


def test_a_single_iteration_budget_synthesizes_on_needs_iteration() -> None:
    script, outcome = run([ITERATE], max_iterations=1)

    assert len(script.drafts) == 1
    assert outcome.trace.final_decision == "synthesize"


def test_iteration_records_carry_the_critic_view() -> None:
    _, outcome = run([ITERATE, ACCEPT])

    first = outcome.trace.iterations[0]
    assert (first.iteration, first.critic_verdict, first.critic_confidence) == (
        0,
        "needs_iteration",
        80,
    )
    assert first.issue_summary == ["[major] math_error @ tasks[0].estimated_hours"]


@pytest.mark.parametrize("budget", [0, -1])
def test_a_budget_below_one_is_rejected(budget: int) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        Boss(max_iterations=budget)
