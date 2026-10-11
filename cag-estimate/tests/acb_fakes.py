"""Offline fakes for actor-critic tests: one wrapper that answers drafts and reviews."""

from typing import Any

from cag_estimate.schemas.acb import CriticFeedback, CriticIssue
from cag_estimate.schemas.estimation import ProjectEstimation
from tests.conftest import build_estimation


def accept() -> CriticFeedback:
    return CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)


def needs_iteration(description: str = "Task hours look inflated") -> CriticFeedback:
    issue = CriticIssue(
        category="unrealistic_estimate",
        severity="major",
        field_path="tasks[0].estimated_hours",
        description=description,
    )
    return CriticFeedback(verdict="needs_iteration", issues=[issue], confidence_in_review=70)


class ACBWrapper:
    """Serves queued drafts and reviews, recording every call by kind."""

    primary_model = "fake-model"

    def __init__(
        self,
        reviews: list[CriticFeedback | Exception],
        drafts: list[ProjectEstimation] | None = None,
        draft_error: Exception | None = None,
    ) -> None:
        self.reviews = list(reviews)
        self.drafts = drafts or [build_estimation(project_name=f"Draft {n}") for n in range(1, 5)]
        self.draft_error = draft_error
        self.draft_calls: list[dict[str, Any]] = []
        self.review_calls: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if kwargs["response_model"] is CriticFeedback:
            return self._next_review(kwargs), {}
        self.draft_calls.append(kwargs)
        if self.draft_error is not None and len(self.draft_calls) == 2:
            raise self.draft_error
        return self.drafts[len(self.draft_calls) - 1], {}

    def _next_review(self, kwargs: dict[str, Any]) -> CriticFeedback:
        self.review_calls.append(kwargs)
        review = self.reviews[len(self.review_calls) - 1]
        if isinstance(review, Exception):
            raise review
        return review
