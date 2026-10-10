"""Tests for the versioned Jinja2 estimation prompt templates."""

import hashlib

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


V1_PROMPT_SHA256 = {
    "json": (
        "1f454f4f3a217a185bc09f88e6241f7092a0797bf96e675d71ea64e130179ac6",
        "ad731a719b61a82ded4506411b1df01ad8633db4f386667f307c211d3464e84d",
    ),
    "markdown": (
        "85b7847716f018503d5b596cd351f65d65eea60c6c13bfec0ec1a086fdc95e62",
        "3d9eddcf46ae402b6a9784e21c2574e4e0d3e5b4b1496cfe6221995827bf3de5",
    ),
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("output_format", ["json", "markdown"])
def test_v1_prompts_stay_byte_identical(output_format: str) -> None:
    system, user = render_estimation_prompt(make_request(), output_format=output_format)
    assert (_sha256(system), _sha256(user)) == V1_PROMPT_SHA256[output_format]


def test_v1_ignores_the_tier() -> None:
    assert render_estimation_prompt(make_request(), tier="executive") == render_estimation_prompt(
        make_request()
    )


def test_v2_system_keeps_the_v1_schema_rules_and_metadata() -> None:
    v1, _ = render_estimation_prompt(make_request(), project_metadata=make_metadata())
    v2, _ = render_estimation_prompt(
        make_request(), version="v2", project_metadata=make_metadata()
    )
    assert v2.startswith(v1.rstrip("\n"))
    assert "<output_schema>" in v2
    assert "<project_metadata>" in v2


def test_v2_conversation_rules_demand_the_updated_full_estimate() -> None:
    system, _ = render_estimation_prompt(make_request(), version="v2")
    assert "<conversation_rules>" in system
    assert "updated FULL estimate" in system
    assert "source of truth" in system
    assert "project_name" in system


@pytest.mark.parametrize(
    ("tier", "marker"),
    [
        ("executive", "business outcomes"),
        ("pm", "dependencies"),
        ("developer", "integration points"),
        ("default", "balanced"),
        (None, "balanced"),
    ],
)
def test_v2_audience_block_follows_the_tier(tier: str | None, marker: str) -> None:
    system, _ = render_estimation_prompt(make_request(), version="v2", tier=tier)
    audience = system.split("<audience>")[1].split("</audience>")[0]
    assert marker in audience
    others = {"business outcomes", "dependencies", "integration points", "balanced"} - {marker}
    assert not any(other in audience for other in others)


def test_v2_user_prompt_adds_the_integration_closing_line() -> None:
    _, v1_user = render_estimation_prompt(make_request())
    _, v2_user = render_estimation_prompt(make_request(), version="v2")
    assert v2_user.startswith(v1_user.rstrip("\n"))
    assert "full updated estimate" in v2_user


def test_render_system_prompt_supports_v2() -> None:
    system = render_system_prompt(make_metadata(), version="v2")
    assert "<conversation_rules>" in system
    assert "Booking Hub" in system
