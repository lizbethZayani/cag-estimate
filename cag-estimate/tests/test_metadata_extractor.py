"""Offline tests for the heuristic project-metadata extractor."""

from cag_estimate.schemas.estimation import ProjectEstimation
from cag_estimate.services.metadata_extractor import extract_metadata


def make_result(
    *,
    name: str = "Booking Hub",
    summary: str = "Booking flow and admin panel.",
    team: str = "3 developers",
    task_text: str = "Build the API",
) -> ProjectEstimation:
    task = {
        "task_id": 1,
        "name": task_text,
        "description": task_text,
        "estimated_hours": 10,
        "estimated_cost_usd": 400,
        "complexity": "Medium",
        "includes": [task_text],
    }
    return ProjectEstimation.model_validate(
        {
            "project_name": name,
            "meeting_summary": summary,
            "tasks": [{**task, "task_id": i} for i in (1, 2, 3)],
            "summary": {
                "total_hours": 30,
                "total_cost_usd": 1200,
                "team_size": team,
                "estimated_duration_weeks": 4,
                "hourly_rate": 40,
            },
        }
    )


def test_name_team_size_and_scope_come_from_result() -> None:
    metadata = extract_metadata(make_result(), "transcript")
    assert metadata.project_name == "Booking Hub"
    assert metadata.assumed_team_size == 3
    assert metadata.agreed_scope == "Booking flow and admin panel."


def test_team_size_none_when_no_number() -> None:
    assert extract_metadata(make_result(team="a small team"), "").assumed_team_size is None


def test_scope_collapses_whitespace_and_truncates_at_word_boundary() -> None:
    long_summary = "word  \n" * 200
    scope = extract_metadata(make_result(summary=long_summary), "").agreed_scope
    assert scope is not None
    assert len(scope) <= 300
    assert "  " not in scope and "\n" not in scope
    assert scope.endswith("word")


def test_blank_scope_becomes_none() -> None:
    assert extract_metadata(make_result(summary="   "), "").agreed_scope is None


def test_technologies_found_with_canonical_casing_and_special_names() -> None:
    transcript = "we use fastapi, postgresql and c++ plus c# with .net and NODE.JS"
    techs = extract_metadata(make_result(), transcript).mentioned_technologies
    assert {"FastAPI", "PostgreSQL", "C++", "C#", ".NET", "Node.js"} <= set(techs)


def test_technologies_scan_task_fields() -> None:
    result = make_result(task_text="Deploy with Docker and Kubernetes")
    techs = extract_metadata(result, "").mentioned_technologies
    assert "Docker" in techs and "Kubernetes" in techs


def test_no_false_positives_for_ambiguous_short_tokens() -> None:
    transcript = "Let's go and review the road map, rather than rest. R is not here."
    assert extract_metadata(make_result(), transcript).mentioned_technologies == []


def test_no_match_inside_larger_words() -> None:
    assert extract_metadata(make_result(), "Reactive Javascripty").mentioned_technologies == []


def test_technologies_are_deduplicated() -> None:
    transcript = "Docker docker DOCKER"
    techs = extract_metadata(make_result(task_text="Docker"), transcript).mentioned_technologies
    assert techs.count("Docker") == 1
