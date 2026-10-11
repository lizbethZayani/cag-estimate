"""The golden dataset: scripted conversations with loose, regression-oriented expectations."""

import json
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from cag_estimate.guardrails.input import Reason
from cag_estimate.schemas.estimation import (
    MAX_HOURLY_RATE,
    MAX_PROJECT_HOURS,
    MAX_TASKS,
    MIN_HOURLY_RATE,
    MIN_PROJECT_HOURS,
    MIN_TASKS,
)

DATASET_PACKAGE = "cag_estimate.evals"
DATASET_FILE = "golden_dataset.json"
MIN_TURN_CHARS, MAX_TURN_CHARS = 20, 80_000

Turn = Annotated[str, Field(min_length=MIN_TURN_CHARS, max_length=MAX_TURN_CHARS)]
Tier = Literal["auto", "executive", "pm", "developer"]
ResolvedTier = Literal["executive", "pm", "developer", "default"]


def _check_range(name: str, bounds: tuple[int, int], floor: int, ceiling: int) -> None:
    low, high = bounds
    if not floor <= low <= high <= ceiling:
        raise ValueError(f"{name} {bounds} must satisfy {floor} <= low <= high <= {ceiling}")


class GoldenCase(BaseModel):
    """One scripted conversation (all turns go to ONE session) and what to expect.

    Expectations are evaluated on the FINAL turn unless a metric says otherwise.
    They are loose on purpose: they catch regressions, not exact values.
    """

    id: str = Field(min_length=3, max_length=64)
    description: str
    turns: list[Turn] = Field(min_length=1)
    hourly_rate: int = Field(default=40, ge=MIN_HOURLY_RATE, le=MAX_HOURLY_RATE)
    tier: Tier = "auto"
    expect_guardrail_reason: Reason | None = None
    expected_in_text: list[str] = Field(default_factory=list)
    expected_technologies_any_of: list[str] = Field(default_factory=list)
    expected_task_count_range: tuple[int, int] | None = None
    expected_hours_range: tuple[int, int] | None = None
    expected_resolved_tier: ResolvedTier | None = None
    check_memory: bool = False

    @model_validator(mode="after")
    def check_ranges(self) -> "GoldenCase":
        if self.expected_task_count_range:
            _check_range(
                "expected_task_count_range", self.expected_task_count_range, MIN_TASKS, MAX_TASKS
            )
        if self.expected_hours_range:
            _check_range(
                "expected_hours_range",
                self.expected_hours_range,
                MIN_PROJECT_HOURS,
                MAX_PROJECT_HOURS,
            )
        return self


def _read(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    return resources.files(DATASET_PACKAGE).joinpath(DATASET_FILE).read_text(encoding="utf-8")


def load_dataset(path: Path | None = None) -> list[GoldenCase]:
    """Load the cases from ``path`` (default: the JSON packaged with the library)."""
    cases = [GoldenCase.model_validate(item) for item in json.loads(_read(path))]
    duplicates = sorted(
        case_id for case_id, count in Counter(c.id for c in cases).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate case ids in the dataset: {', '.join(duplicates)}")
    return cases
