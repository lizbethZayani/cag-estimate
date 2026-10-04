"""
Estimation schemas with semantic validation.

Validation messages are written to be actionable: structured extraction
feeds them back to the LLM when it has to correct an invalid answer.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Complexity = Literal["Simple", "Medium", "High"]

# Allowed task hours per complexity: (min, max). Mirrors the original
# verification rules, which allow a 2x buffer over the nominal maximum.
COMPLEXITY_HOURS: dict[str, tuple[int, int]] = {
    "Simple": (4, 24),
    "Medium": (10, 64),
    "High": (16, 112),
}
MIN_PROJECT_HOURS, MAX_PROJECT_HOURS = 8, 500
MIN_HOURLY_RATE, MAX_HOURLY_RATE = 30, 150
MIN_TASKS, MAX_TASKS = 3, 20


class EstimationRequest(BaseModel):
    """Request model for project estimation."""

    transcription: str = Field(..., description="Meeting transcription to estimate")
    hourly_rate: int = Field(
        default=40, description="Hourly rate for cost calculation (default: $40)"
    )


class TaskEstimate(BaseModel):
    """Individual task estimation."""

    task_id: int
    name: str
    description: str
    estimated_hours: int
    estimated_cost_usd: int
    complexity: Complexity
    includes: list[str]

    @model_validator(mode="after")
    def check_hours_match_complexity(self) -> "TaskEstimate":
        min_hours, max_hours = COMPLEXITY_HOURS[self.complexity]
        if not min_hours <= self.estimated_hours <= max_hours:
            raise ValueError(
                f"Task {self.task_id} is marked {self.complexity} but has "
                f"{self.estimated_hours} hours; {self.complexity} tasks must have "
                f"between {min_hours} and {max_hours} hours. Adjust the hours "
                "or the complexity."
            )
        return self


class ProjectSummary(BaseModel):
    """Project summary with totals."""

    total_hours: int
    total_cost_usd: int
    team_size: str
    estimated_duration_weeks: int
    hourly_rate: int
    assumptions: list[str] | None = None

    @field_validator("total_hours")
    @classmethod
    def check_total_hours_range(cls, value: int) -> int:
        if not MIN_PROJECT_HOURS <= value <= MAX_PROJECT_HOURS:
            raise ValueError(
                f"total_hours is {value}; it must be between {MIN_PROJECT_HOURS} "
                f"and {MAX_PROJECT_HOURS}. Rescope the tasks to fit this range."
            )
        return value

    @field_validator("hourly_rate")
    @classmethod
    def check_hourly_rate_range(cls, value: int) -> int:
        if not MIN_HOURLY_RATE <= value <= MAX_HOURLY_RATE:
            raise ValueError(
                f"hourly_rate is {value}; it must be between {MIN_HOURLY_RATE} "
                f"and {MAX_HOURLY_RATE} USD per hour."
            )
        return value


class ProjectEstimation(BaseModel):
    """Project estimation result. Tasks come before the totals derived from them."""

    project_name: str
    meeting_summary: str
    tasks: list[TaskEstimate]
    summary: ProjectSummary

    @field_validator("tasks")
    @classmethod
    def check_task_count(cls, tasks: list[TaskEstimate]) -> list[TaskEstimate]:
        if not MIN_TASKS <= len(tasks) <= MAX_TASKS:
            raise ValueError(
                f"The estimation has {len(tasks)} tasks; it must have between "
                f"{MIN_TASKS} and {MAX_TASKS} tasks."
            )
        return tasks

    @model_validator(mode="after")
    def check_total_matches_tasks(self) -> "ProjectEstimation":
        task_hours = sum(task.estimated_hours for task in self.tasks)
        if task_hours != self.summary.total_hours:
            raise ValueError(
                f"The sum of task hours is {task_hours} but summary.total_hours "
                f"is {self.summary.total_hours}; set total_hours to {task_hours}."
            )
        return self


class EstimationResponse(BaseModel):
    """Response model for the structured estimation endpoint."""

    result: ProjectEstimation
    prompt_version: str = "v1"
    cached: bool = False
