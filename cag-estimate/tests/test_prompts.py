"""Tests for the versioned Jinja2 estimation prompt templates."""

import pytest
from jinja2 import UndefinedError

from cag_estimate.context.examples import ESTIMATION_EXAMPLES
from cag_estimate.prompts import loader, render_estimation_prompt, render_system_prompt
from cag_estimate.schemas.estimation import EstimationRequest
from cag_estimate.schemas.session import ProjectMetadata

DESCRIPTION = "We need a booking app with payments and an admin panel."


def make_request(hourly_rate: int = 55) -> EstimationRequest:
    return EstimationRequest(transcription=DESCRIPTION, hourly_rate=hourly_rate)


def test_returns_system_and_user_prompts() -> None:
    system, user = render_estimation_prompt(make_request())
    assert "<role>" in system
    assert DESCRIPTION in user


def test_user_prompt_wraps_description_and_renders_rate() -> None:
    _, user = render_estimation_prompt(make_request(hourly_rate=77))
    assert f"<project_description>\n{DESCRIPTION}\n</project_description>" in user
    assert "$77/hour" in user


def test_user_prompt_keeps_current_instructions() -> None:
    _, user = render_estimation_prompt(make_request())
    assert "testing, documentation, and deployment" in user
    assert "assumptions" in user
    assert "Return only valid JSON" in user


def test_json_output_format_block() -> None:
    system, user = render_estimation_prompt(make_request())
    assert "<output_schema>" in system
    assert '"estimated_hours"' in system
    assert "**Project:**" not in system
    assert "Return only valid JSON" in user


def test_markdown_output_format_block() -> None:
    system, user = render_estimation_prompt(make_request(), output_format="markdown")
    assert "**Project:**" in system
    assert '"estimated_hours": number' not in system
    assert "Return only valid JSON" not in user
    assert "Markdown content described in the system prompt" in user


def test_system_prompt_states_validator_rules() -> None:
    system, _ = render_estimation_prompt(make_request())
    assert "sum of every task's estimated_hours" in system
    assert "between 3 and 20 tasks" in system
    assert "between 30 and 150" in system
    assert "Simple: 4-24" in system


def test_examples_included_from_data_and_workout_excluded() -> None:
    system, _ = render_estimation_prompt(make_request())
    for example in ESTIMATION_EXAMPLES:
        assert example["project_name"] in system
    assert "hyrox_20250914_001" not in system


def test_unknown_version_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="v99"):
        render_estimation_prompt(make_request(), version="v99")


def test_unknown_output_format_raises() -> None:
    with pytest.raises(ValueError, match="output_format"):
        render_estimation_prompt(make_request(), output_format="xml")


def test_strict_undefined_raises_on_missing_variable() -> None:
    template = loader._env.from_string("{{ missing_variable }}")
    with pytest.raises(UndefinedError):
        template.render()


# ------------------------------------------------------------ project metadata


def make_metadata() -> ProjectMetadata:
    return ProjectMetadata(
        project_name="Booking Hub",
        assumed_team_size=3,
        mentioned_technologies=["FastAPI", "PostgreSQL"],
        agreed_scope="Booking flow and admin panel.",
    )


def test_metadata_block_rendered_with_known_facts() -> None:
    system, _ = render_estimation_prompt(make_request(), project_metadata=make_metadata())
    assert "<project_metadata>" in system
    assert "Booking Hub" in system
    assert "3" in system.split("<project_metadata>")[1].split("</project_metadata>")[0]
    assert "FastAPI, PostgreSQL" in system
    assert "Booking flow and admin panel." in system
    assert "consistent" in system


def test_metadata_block_absent_without_or_with_empty_metadata() -> None:
    baseline, _ = render_estimation_prompt(make_request())
    for metadata in (None, ProjectMetadata()):
        system, _ = render_estimation_prompt(make_request(), project_metadata=metadata)
        assert "<project_metadata>" not in system
        assert system == baseline


def test_metadata_block_skips_unknown_fields() -> None:
    system = render_system_prompt(ProjectMetadata(project_name="Solo"))
    assert "Solo" in system
    assert "Team size" not in system
    assert "Technologies" not in system


def test_render_system_prompt_needs_no_request() -> None:
    system = render_system_prompt(make_metadata())
    assert "<role>" in system
    assert "Booking Hub" in system


def test_render_system_prompt_matches_estimation_system_part() -> None:
    expected, _ = render_estimation_prompt(make_request(), project_metadata=make_metadata())
    assert render_system_prompt(make_metadata()) == expected


def test_render_system_prompt_validates_inputs() -> None:
    with pytest.raises(ValueError, match="v99"):
        render_system_prompt(ProjectMetadata(), version="v99")
    with pytest.raises(ValueError, match="output_format"):
        render_system_prompt(ProjectMetadata(), output_format="xml")
