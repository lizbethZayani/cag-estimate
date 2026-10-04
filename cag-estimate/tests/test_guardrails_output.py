"""Offline tests for the output guardrail."""

from __future__ import annotations

import pytest

from cag_estimate.guardrails.errors import GuardrailError
from cag_estimate.guardrails.output import OutputGuardrailViolation, enforce_output
from cag_estimate.schemas.estimation import ProjectEstimation


def _estimation(total_cost: int = 1200, **overrides: str) -> ProjectEstimation:
    tasks = [
        {
            "task_id": i,
            "name": f"Task {i}",
            "description": "desc",
            "estimated_hours": 10,
            "estimated_cost_usd": 400,
            "complexity": "Medium",
            "includes": ["code"],
        }
        for i in range(1, 4)
    ]
    data = {
        "project_name": "Demo",
        "meeting_summary": "summary",
        "tasks": tasks,
        "summary": {
            "total_hours": 30,
            "total_cost_usd": total_cost,
            "team_size": "2 developers",
            "estimated_duration_weeks": 2,
            "hourly_rate": 40,
        },
    }
    data.update(overrides)
    return ProjectEstimation.model_validate(data)


def test_consistent_result_passes_through_unchanged():
    result = _estimation()
    assert enforce_output(result, 40) == result


def test_cost_is_recomputed_from_hours_and_rate():
    fixed = enforce_output(_estimation(total_cost=999), 40)
    assert fixed.summary.total_cost_usd == 1200
    assert fixed.summary.total_hours == 30


def test_cost_uses_the_given_rate():
    assert enforce_output(_estimation(), 50).summary.total_cost_usd == 1500


@pytest.mark.parametrize(
    "marker",
    ["<rules>", "<output_schema>", "<scope>", "my System Prompt says", "</rules>"],
)
def test_leakage_in_text_field_raises(marker):
    with pytest.raises(OutputGuardrailViolation):
        enforce_output(_estimation(meeting_summary=f"see {marker}"), 40)


def test_leakage_in_task_text_raises():
    result = _estimation()
    result.tasks[0].includes = ["the <scope> block"]
    with pytest.raises(OutputGuardrailViolation):
        enforce_output(result, 40)


def test_violation_is_a_guardrail_error():
    assert issubclass(OutputGuardrailViolation, GuardrailError)
