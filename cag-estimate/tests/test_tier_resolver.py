"""Offline tests for the audience tier resolver (pure heuristics)."""

import pytest

from cag_estimate.schemas.session import ProjectMetadata
from cag_estimate.services.tier_resolver import Tier, resolve_tier

EMPTY = ProjectMetadata()


@pytest.mark.parametrize(
    ("text", "tier", "rule"),
    [
        ("The CEO needs the numbers for the board.", Tier.EXECUTIVE, "executive_cues"),
        ("We need budget approval and a clear ROI.", Tier.EXECUTIVE, "executive_cues"),
        ("Please send an executive summary.", Tier.EXECUTIVE, "executive_cues"),
        ("Our investors want a quick view.", Tier.EXECUTIVE, "executive_cues"),
        ("Plan the next sprint and the backlog.", Tier.PM, "pm_cues"),
        ("Stakeholders want a roadmap with milestones.", Tier.PM, "pm_cues"),
        ("What is the timeline for each deliverable?", Tier.PM, "pm_cues"),
        ("Expose a REST API endpoint for orders.", Tier.DEVELOPER, "developer_cues"),
        ("We will refactor the schema and split a microservice.", Tier.DEVELOPER, "developer_cues"),
        ("The stack is React with FastAPI.", Tier.DEVELOPER, "developer_cues"),
        ("We want a simple booking website.", Tier.DEFAULT, "default"),
        ("", Tier.DEFAULT, "default"),
    ],
)
def test_each_rule_decides_its_tier_and_names_itself(text: str, tier: Tier, rule: str) -> None:
    assert resolve_tier(text, EMPTY, None) == (tier, rule)


def test_matching_is_case_insensitive_and_word_bounded() -> None:
    assert resolve_tier("the ceo is here", EMPTY, None)[0] is Tier.EXECUTIVE
    # "roi" inside another word and "api" inside "rapid" must not match.
    assert resolve_tier("a heroic and rapid launch", EMPTY, None) == (Tier.DEFAULT, "default")


def test_executive_cues_beat_pm_and_developer_cues() -> None:
    text = "CEO wants the sprint plan for the API."
    assert resolve_tier(text, EMPTY, None) == (Tier.EXECUTIVE, "executive_cues")


def test_pm_cues_beat_developer_cues() -> None:
    text = "Milestones for the API endpoint work."
    assert resolve_tier(text, EMPTY, None) == (Tier.PM, "pm_cues")


def test_three_technologies_in_metadata_select_developer() -> None:
    metadata = ProjectMetadata(mentioned_technologies=["Alpha", "Beta", "Gamma"])
    assert resolve_tier("A plain request.", metadata, None) == (Tier.DEVELOPER, "technology_count")


def test_two_technologies_are_not_enough() -> None:
    metadata = ProjectMetadata(mentioned_technologies=["Alpha", "Beta"])
    assert resolve_tier("A plain request.", metadata, None) == (Tier.DEFAULT, "default")


def test_text_cues_beat_the_technology_count() -> None:
    metadata = ProjectMetadata(mentioned_technologies=["Alpha", "Beta", "Gamma"])
    assert resolve_tier("The CFO asks.", metadata, None) == (Tier.EXECUTIVE, "executive_cues")


@pytest.mark.parametrize("override", [Tier.EXECUTIVE, Tier.PM, Tier.DEVELOPER, Tier.DEFAULT])
def test_override_always_wins(override: Tier) -> None:
    text = "CEO sprint API"
    assert resolve_tier(text, EMPTY, override) == (override, "override")


def test_tier_values_are_the_wire_names() -> None:
    assert [tier.value for tier in Tier] == ["executive", "pm", "developer", "default"]
