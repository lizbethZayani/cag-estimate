"""Shared test helpers."""

from __future__ import annotations

from typing import Any

from cag_estimate.schemas.estimation import ProjectEstimation


def build_estimation(total_cost: int = 1200, **overrides: Any) -> ProjectEstimation:
    """A valid 3-task, 30-hour estimation at 40 USD/h (top-level fields overridable)."""
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
    data: dict[str, Any] = {
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
