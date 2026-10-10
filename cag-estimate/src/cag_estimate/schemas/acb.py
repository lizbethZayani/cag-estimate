"""Actor-critic-boss schemas: the critic's structured review and the boss's trace.

``CriticFeedback`` is the contract between the critic and the boss: the boss
reads each issue's severity and the verdict instead of parsing prose.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from cag_estimate.schemas.session import SessionEstimationResponse

CriticIssueCategory = Literal[
    "math_error",
    "hallucination",
    "scope_mismatch",
    "missing_task",
    "history_inconsistency",
    "unrealistic_estimate",
    "missing_assumption",
    "tier_mismatch",
]
CriticIssueSeverity = Literal["critical", "major", "minor"]
CriticVerdict = Literal["accept", "needs_iteration", "reject"]
BossDecision = Literal["accept", "iterate", "synthesize"]
FinalDecision = Literal["accept", "synthesize"]

BLOCKING_SEVERITIES = frozenset({"critical", "major"})


class CriticIssue(BaseModel):
    """One defect found in the estimate under review.

    ``field_path`` uses dotted/bracket notation on the estimate, for example
    ``tasks[2].estimated_hours`` or ``summary.total_hours``.
    """

    category: CriticIssueCategory
    severity: CriticIssueSeverity
    field_path: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=5, max_length=500)
    suggested_fix: str | None = Field(default=None, max_length=300)


class CriticFeedback(BaseModel):
    """The critic's review. Minor issues alone never justify another draft."""

    verdict: CriticVerdict
    issues: list[CriticIssue] = Field(default_factory=list, max_length=12)
    confidence_in_review: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_verdict_matches_issues(self) -> "CriticFeedback":
        blocking = any(issue.severity in BLOCKING_SEVERITIES for issue in self.issues)
        if self.verdict == "needs_iteration" and not blocking:
            raise ValueError(
                "verdict 'needs_iteration' requires at least one issue with severity "
                "in {critical, major}; minor-only issues should accept"
            )
        if self.verdict == "reject" and not self.issues:
            raise ValueError("verdict 'reject' requires at least one issue describing why")
        return self


class ACBIteration(BaseModel):
    """What happened in one actor + critic round."""

    iteration: int = Field(ge=0)
    decision_after: BossDecision
    critic_verdict: CriticVerdict
    critic_confidence: int = Field(ge=0, le=100)
    critic_failed: bool = False
    issue_summary: list[str] = Field(default_factory=list)


class BossTrace(BaseModel):
    """Audit trail of one reviewed estimate; ``caveats`` are the issues left open."""

    iterations: list[ACBIteration]
    final_decision: FinalDecision
    iterations_run: int
    caveats: list[str] = Field(default_factory=list)


class ACBSessionResponse(SessionEstimationResponse):
    """A session estimate plus the trace of the review that produced it."""

    acb: BossTrace
