"""
Output guardrail applied after schema validation and before caching.

Fixable issues are corrected silently (and logged): the prompt tells the model
that ``total_cost_usd`` is total hours x rate but the schema does not enforce
it, so we recompute it. Unfixable issues (system-prompt leakage) raise
``OutputGuardrailViolation``.
"""

from __future__ import annotations

import re

import structlog

from cag_estimate.guardrails.errors import GuardrailError
from cag_estimate.schemas.estimation import ProjectEstimation

log = structlog.get_logger()

_LEAKAGE_RE = re.compile(r"</?\s*(rules|output_schema|scope)\s*>|system\s+prompt", re.IGNORECASE)


class OutputGuardrailViolation(GuardrailError):
    """Raised when a result cannot be repaired and must not reach the client."""


def enforce_output(result: ProjectEstimation, hourly_rate: int) -> ProjectEstimation:
    """Reject leaked prompt text, then fix the total cost if it disagrees."""
    _reject_leakage(result)
    return _fix_total_cost(result, hourly_rate)


def _text_fields(result: ProjectEstimation) -> list[str]:
    texts = [result.project_name, result.meeting_summary, result.summary.team_size]
    texts += result.summary.assumptions or []
    for task in result.tasks:
        texts += [task.name, task.description, *task.includes]
    return texts


def _reject_leakage(result: ProjectEstimation) -> None:
    if any(_LEAKAGE_RE.search(text) for text in _text_fields(result)):
        log.warning("output_leakage_detected")
        raise OutputGuardrailViolation("Output contains system prompt markers.")


def _fix_total_cost(result: ProjectEstimation, hourly_rate: int) -> ProjectEstimation:
    expected = result.summary.total_hours * hourly_rate
    if result.summary.total_cost_usd == expected:
        return result
    log.info(
        "output_total_cost_corrected",
        model_value=result.summary.total_cost_usd,
        corrected_value=expected,
    )
    summary = result.summary.model_copy(update={"total_cost_usd": expected})
    return result.model_copy(update={"summary": summary})
