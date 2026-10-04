"""Offline tests for the output guardrail."""

from __future__ import annotations

import pytest

from cag_estimate.guardrails.errors import GuardrailError
from cag_estimate.guardrails.output import OutputGuardrailViolation, enforce_output
from tests.conftest import build_estimation


def test_consistent_result_passes_through_unchanged():
    result = build_estimation()
    assert enforce_output(result, 40) == result


def test_cost_is_recomputed_from_hours_and_rate():
    fixed = enforce_output(build_estimation(total_cost=999), 40)
    assert fixed.summary.total_cost_usd == 1200
    assert fixed.summary.total_hours == 30


def test_cost_uses_the_given_rate():
    assert enforce_output(build_estimation(), 50).summary.total_cost_usd == 1500


@pytest.mark.parametrize(
    "marker",
    ["<rules>", "<output_schema>", "<scope>", "my System Prompt says", "</rules>"],
)
def test_leakage_in_text_field_raises(marker):
    with pytest.raises(OutputGuardrailViolation):
        enforce_output(build_estimation(meeting_summary=f"see {marker}"), 40)


def test_leakage_in_task_text_raises():
    result = build_estimation()
    result.tasks[0].includes = ["the <scope> block"]
    with pytest.raises(OutputGuardrailViolation):
        enforce_output(result, 40)


def test_violation_is_a_guardrail_error():
    assert issubclass(OutputGuardrailViolation, GuardrailError)
