"""Boss: decides, after each critic review, whether to accept, retry or give up.

The boss makes no LLM call of its own, so every decision is traceable in
``BossTrace``. The actor is a callable that receives only the LATEST review
(``None`` on the first draft). When the boss cannot accept, the last draft is
returned UNMODIFIED and the open issues travel in ``BossTrace.caveats``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import structlog

from cag_estimate.schemas.acb import (
    ACBIteration,
    BossDecision,
    BossTrace,
    CriticFeedback,
    CriticIssue,
    FinalDecision,
)
from cag_estimate.schemas.estimation import ProjectEstimation
from cag_estimate.services.critic import CriticOutcome

log = structlog.get_logger()

MAX_REPORTED_ISSUES = 5
MAX_CAVEAT_CHARS = 200

ActorCallable = Callable[[CriticFeedback | None], ProjectEstimation]
CriticCallable = Callable[[ProjectEstimation], CriticOutcome]


@dataclass(frozen=True)
class BossOutcome:
    """The estimate to return and the audit trail of how it was reached."""

    result: ProjectEstimation
    trace: BossTrace


def _decide(review: CriticFeedback, *, iterations_left: int) -> BossDecision:
    if review.verdict == "accept":
        return "accept"
    if review.verdict == "needs_iteration" and iterations_left > 0:
        return "iterate"
    return "synthesize"


def _issue_summary(issue: CriticIssue) -> str:
    return f"[{issue.severity}] {issue.category} @ {issue.field_path}"


def _caveat(issue: CriticIssue) -> str:
    text = f"[{issue.severity}] {issue.category} ({issue.field_path}): {issue.description}"
    return text[:MAX_CAVEAT_CHARS]


class Boss:
    """Stateless orchestrator: one ``run`` per estimate."""

    def __init__(self, max_iterations: int = 2) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations = max_iterations

    def run(self, actor: ActorCallable, critic: CriticCallable) -> BossOutcome:
        iterations: list[ACBIteration] = []
        feedback: CriticFeedback | None = None
        while True:
            draft = actor(feedback)
            review = critic(draft)
            decision = _decide(
                review.feedback, iterations_left=self.max_iterations - len(iterations) - 1
            )
            iterations.append(self._record(len(iterations), decision, review))
            log.info("boss_iteration", iteration=len(iterations) - 1, decision=decision)
            if decision != "iterate":
                return BossOutcome(draft, self._trace(iterations, decision, review.feedback))
            feedback = review.feedback

    @staticmethod
    def _record(index: int, decision: BossDecision, review: CriticOutcome) -> ACBIteration:
        feedback = review.feedback
        return ACBIteration(
            iteration=index,
            decision_after=decision,
            critic_verdict=feedback.verdict,
            critic_confidence=0 if review.failed else feedback.confidence_in_review,
            critic_failed=review.failed,
            issue_summary=[_issue_summary(i) for i in feedback.issues[:MAX_REPORTED_ISSUES]],
        )

    @staticmethod
    def _trace(
        iterations: list[ACBIteration], decision: FinalDecision, last_review: CriticFeedback
    ) -> BossTrace:
        caveats = (
            [_caveat(i) for i in last_review.issues[:MAX_REPORTED_ISSUES]]
            if decision == "synthesize"
            else []
        )
        return BossTrace(
            iterations=iterations,
            final_decision=decision,
            iterations_run=len(iterations),
            caveats=caveats,
        )
