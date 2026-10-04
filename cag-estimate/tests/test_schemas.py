"""Offline tests for the validated estimation schemas."""

import pytest
from pydantic import ValidationError

from cag_estimate.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
    ProjectEstimation,
    ProjectSummary,
    TaskEstimate,
)


def make_task(task_id: int = 1, hours: int = 16, complexity: str = "Medium") -> dict:
    return {
        "task_id": task_id,
        "name": f"Task {task_id}",
        "description": "Implement the feature end to end",
        "estimated_hours": hours,
        "estimated_cost_usd": hours * 40,
        "complexity": complexity,
        "includes": ["Backend", "Tests"],
    }


def make_estimation(
    task_hours: list[int] | None = None,
    total_hours: int | None = None,
    hourly_rate: int = 40,
) -> dict:
    task_hours = task_hours if task_hours is not None else [16, 16, 16]
    total = sum(task_hours) if total_hours is None else total_hours
    return {
        "project_name": "Demo Project",
        "meeting_summary": "A demo meeting summary",
        "tasks": [make_task(i + 1, h) for i, h in enumerate(task_hours)],
        "summary": {
            "total_hours": total,
            "total_cost_usd": total * hourly_rate,
            "team_size": "2 developers",
            "estimated_duration_weeks": 2,
            "hourly_rate": hourly_rate,
            "assumptions": ["Stable requirements"],
        },
    }


def test_valid_estimation_parses():
    estimation = ProjectEstimation(**make_estimation())
    assert estimation.summary.total_hours == 48
    assert len(estimation.tasks) == 3


def test_request_keeps_default_hourly_rate():
    assert EstimationRequest(transcription="hello").hourly_rate == 40


def test_sum_of_task_hours_must_match_total():
    with pytest.raises(ValidationError, match="sum of task hours"):
        ProjectEstimation(**make_estimation(total_hours=100))


@pytest.mark.parametrize("count", [2, 21])
def test_task_count_out_of_range(count):
    with pytest.raises(ValidationError, match="between 3 and 20 tasks"):
        ProjectEstimation(**make_estimation(task_hours=[16] * count))


@pytest.mark.parametrize("count", [3, 20])
def test_task_count_boundaries_accepted(count):
    ProjectEstimation(**make_estimation(task_hours=[16] * count))


@pytest.mark.parametrize(
    ("complexity", "hours"),
    [("Simple", 3), ("Simple", 25), ("Medium", 9), ("Medium", 65), ("High", 15), ("High", 113)],
)
def test_task_hours_outside_complexity_range(complexity, hours):
    with pytest.raises(ValidationError, match=complexity):
        TaskEstimate(**make_task(hours=hours, complexity=complexity))


@pytest.mark.parametrize(
    ("complexity", "hours"),
    [("Simple", 4), ("Simple", 24), ("Medium", 10), ("Medium", 64), ("High", 16), ("High", 112)],
)
def test_task_hours_complexity_boundaries_accepted(complexity, hours):
    TaskEstimate(**make_task(hours=hours, complexity=complexity))


def test_unknown_complexity_rejected():
    with pytest.raises(ValidationError):
        TaskEstimate(**make_task(complexity="Huge"))


@pytest.mark.parametrize("total", [7, 501])
def test_total_hours_out_of_range(total):
    summary = make_estimation()["summary"] | {"total_hours": total}
    with pytest.raises(ValidationError, match="between 8 and 500"):
        ProjectSummary(**summary)


@pytest.mark.parametrize("rate", [29, 151])
def test_hourly_rate_out_of_range(rate):
    summary = make_estimation()["summary"] | {"hourly_rate": rate}
    with pytest.raises(ValidationError, match="between 30 and 150"):
        ProjectSummary(**summary)


@pytest.mark.parametrize("rate", [30, 150])
def test_hourly_rate_boundaries_accepted(rate):
    ProjectSummary(**make_estimation(hourly_rate=rate)["summary"])


def test_response_defaults():
    response = EstimationResponse(result=ProjectEstimation(**make_estimation()))
    assert response.prompt_version == "v1"
    assert response.cached is False
