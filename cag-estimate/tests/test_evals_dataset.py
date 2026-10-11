"""Offline tests for the golden dataset: structure, coverage and honesty of the expectations."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cag_estimate.evals.dataset import GoldenCase, load_dataset
from cag_estimate.guardrails.input import InputGuardrailViolation, check_input
from cag_estimate.schemas.estimation import (
    MAX_PROJECT_HOURS,
    MAX_TASKS,
    MIN_PROJECT_HOURS,
    MIN_TASKS,
)
from cag_estimate.schemas.session import ProjectMetadata
from cag_estimate.services.tier_resolver import Tier, resolve_tier

TURN = "We need a small bookstore web shop with a catalog and checkout."


def _case(**overrides: Any) -> dict[str, Any]:
    return {"id": "case-one", "description": "d", "turns": [TURN], **overrides}


@pytest.fixture(scope="module")
def cases() -> list[GoldenCase]:
    return load_dataset()


def test_the_packaged_dataset_has_at_least_twelve_cases(cases: list[GoldenCase]) -> None:
    assert len(cases) >= 12


def test_ids_are_unique(cases: list[GoldenCase]) -> None:
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))


def test_every_category_is_covered(cases: list[GoldenCase]) -> None:
    assert any(len(c.turns) == 1 and c.expect_guardrail_reason is None for c in cases)
    assert any(c.expected_resolved_tier for c in cases)
    assert any(c.expect_guardrail_reason for c in cases)
    assert any(c.check_memory and len(c.turns) >= 3 for c in cases)


def test_ranges_are_sane(cases: list[GoldenCase]) -> None:
    for case in cases:
        if case.expected_task_count_range:
            low, high = case.expected_task_count_range
            assert MIN_TASKS <= low <= high <= MAX_TASKS, case.id
        if case.expected_hours_range:
            low, high = case.expected_hours_range
            assert MIN_PROJECT_HOURS <= low <= high <= MAX_PROJECT_HOURS, case.id


def test_memory_cases_have_several_turns(cases: list[GoldenCase]) -> None:
    for case in cases:
        assert not case.check_memory or len(case.turns) >= 2, case.id


def test_expected_terms_can_be_derived_from_the_case_text(cases: list[GoldenCase]) -> None:
    """A model repeats the user's words: an expected term the text never uses is a dataset bug.

    A live run once failed `leave` on a transcript that only said "time off".
    """
    for case in cases:
        text = " ".join(case.turns).lower()
        for term in case.expected_in_text:
            assert term.lower() in text, f"{case.id}: '{term}' never appears in the turns"


def test_guardrail_cases_really_trigger_the_declared_reason(cases: list[GoldenCase]) -> None:
    guarded = [case for case in cases if case.expect_guardrail_reason]
    assert {case.expect_guardrail_reason for case in guarded} >= {"prompt_injection", "pii"}
    for case in guarded:
        with pytest.raises(InputGuardrailViolation) as raised:
            check_input(case.turns[-1])
        assert raised.value.reason == case.expect_guardrail_reason, case.id


def test_other_cases_do_not_trip_the_input_guardrails(cases: list[GoldenCase]) -> None:
    for case in cases:
        if case.expect_guardrail_reason is None:
            for turn in case.turns:
                check_input(turn)


def test_tier_cases_really_resolve_to_their_expected_tier(cases: list[GoldenCase]) -> None:
    tiered = [case for case in cases if case.expected_resolved_tier]
    assert {case.expected_resolved_tier for case in tiered} >= {"executive", "pm", "developer"}
    for case in tiered:
        override = None if case.tier == "auto" else Tier(case.tier)
        tier, _rule = resolve_tier("\n".join(case.turns), ProjectMetadata(), override)
        assert tier.value == case.expected_resolved_tier, case.id


def test_at_least_three_tier_cases_rely_on_the_text_alone(cases: list[GoldenCase]) -> None:
    inferred = [c for c in cases if c.expected_resolved_tier and c.tier == "auto"]
    assert {c.expected_resolved_tier for c in inferred} == {"executive", "pm", "developer"}


def test_load_dataset_reads_a_custom_file(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([_case()]), encoding="utf-8")
    assert [case.id for case in load_dataset(path)] == ["case-one"]


def test_a_duplicate_id_is_rejected_with_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([_case(), _case()]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate.*case-one"):
        load_dataset(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "ab"},
        {"id": "x" * 65},
        {"turns": []},
        {"turns": ["too short"]},
        {"turns": ["x" * 80_001]},
        {"hourly_rate": 29},
        {"hourly_rate": 151},
        {"tier": "wizard"},
        {"expected_hours_range": [100, 50]},
        {"expected_hours_range": [1, 50]},
        {"expected_hours_range": [100, 501]},
        {"expected_task_count_range": [10, 5]},
        {"expected_task_count_range": [2, 10]},
        {"expected_task_count_range": [5, 21]},
        {"expect_guardrail_reason": "toxicity"},
        {"expected_resolved_tier": "intern"},
    ],
    ids=lambda overrides: next(iter(overrides)) + "=" + str(next(iter(overrides.values())))[:12],
)
def test_invalid_cases_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GoldenCase.model_validate(_case(**overrides))


def test_defaults() -> None:
    case = GoldenCase.model_validate(_case())
    assert case.hourly_rate == 40 and case.tier == "auto" and case.check_memory is False
    assert case.expected_in_text == [] and case.expected_technologies_any_of == []
