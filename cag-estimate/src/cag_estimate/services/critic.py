"""Independent critic: audits a draft estimate and returns structured feedback.

The critic fails open. If the review call fails for any reason the draft is
accepted and the outcome is flagged, because an unavailable reviewer must not
turn a good estimate into an error.
"""

from dataclasses import dataclass
from typing import Any

import structlog

from cag_estimate.prompts import render_critic_prompt
from cag_estimate.schemas.acb import CriticFeedback
from cag_estimate.schemas.estimation import ProjectEstimation
from cag_estimate.schemas.session import ProjectMetadata

log = structlog.get_logger()

CRITIC_MAX_TOKENS = 1500
CRITIC_MAX_RETRIES = 2


@dataclass(frozen=True)
class CriticOutcome:
    """A review plus whether the critic itself failed (fail-open accept)."""

    feedback: CriticFeedback
    failed: bool


def build_estimate_digest(result: ProjectEstimation) -> str:
    """Compact text view of an estimate: name, totals and one line per task."""
    summary = result.summary
    lines = [
        f"Project: {result.project_name}",
        f"Total: {summary.total_hours} hours, {summary.total_cost_usd} USD",
    ]
    lines.extend(
        f"- {task.name} | {task.complexity} | {task.estimated_hours} h" for task in result.tasks
    )
    return "\n".join(lines)


class Critic:
    """Reviews one draft per call through the shared ``LLMWrapper``."""

    def __init__(self, wrapper: Any, model: str | None = None) -> None:
        self._wrapper = wrapper
        self._model = model or None

    def review(
        self,
        *,
        transcript: str,
        metadata: ProjectMetadata,
        tier: str,
        result: ProjectEstimation,
        previous_digest: str | None,
    ) -> CriticOutcome:
        system, user = render_critic_prompt(
            transcript=transcript,
            metadata=metadata,
            tier=tier,
            previous_digest=previous_digest,
            estimate_digest=build_estimate_digest(result),
        )
        try:
            feedback, _meta = self._wrapper.complete_structured(
                system_prompt=system,
                user_message=user,
                response_model=CriticFeedback,
                model_override=self._model,
                max_tokens=CRITIC_MAX_TOKENS,
                max_retries=CRITIC_MAX_RETRIES,
            )
        except Exception:  # noqa: BLE001 - any critic failure must fail open
            log.warning("critic_failed_fallback_accept")
            return CriticOutcome(
                CriticFeedback(verdict="accept", issues=[], confidence_in_review=0), failed=True
            )
        return CriticOutcome(feedback, failed=False)
