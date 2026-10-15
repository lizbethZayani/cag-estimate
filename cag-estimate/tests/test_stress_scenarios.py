"""Offline tests for the stress-test scenarios."""

import pytest

from cag_estimate.evals.stress.scenarios import (
    SCENARIO_NAMES,
    TURN_COUNTS,
    Scenario,
    get_scenario,
    slice_scenario,
)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_every_profile_has_twenty_numbered_turns_with_real_text(name: str) -> None:
    scenario = get_scenario(name)

    assert scenario.name == name
    assert [turn.turn_index for turn in scenario.turns] == list(range(1, 21))
    assert all(len(turn.transcript) >= 20 for turn in scenario.turns)


def test_profiles_are_growing_pivot_and_contradiction() -> None:
    assert set(SCENARIO_NAMES) == {"growing", "pivot", "contradiction"}


def test_turn_counts_are_the_documented_sweep() -> None:
    assert TURN_COUNTS == (1, 3, 6, 10, 20)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_turn_one_introduces_the_tracked_project_name(name: str) -> None:
    scenario = get_scenario(name)
    first = scenario.turns[0]

    assert first.fact_introduced == "Nimbus"
    assert first.fact_field == "project_name"
    assert "Nimbus" in first.transcript
    assert scenario.tracked_fact == "Nimbus"


def test_pivot_switches_from_react_to_flutter_at_turn_five() -> None:
    turns = get_scenario("pivot").turns

    assert "React" in " ".join(turn.transcript for turn in turns[:4])
    assert "Flutter" not in " ".join(turn.transcript for turn in turns[:4])
    assert "Flutter" in turns[4].transcript
    assert turns[4].fact_introduced == "Flutter"


def test_contradiction_changes_the_budget_at_turn_eight() -> None:
    turns = get_scenario("contradiction").turns

    assert "30k EUR" in turns[2].transcript
    assert turns[2].fact_introduced == "30k EUR"
    assert "80k EUR" in turns[7].transcript
    assert turns[7].fact_introduced == "80k EUR"
    assert turns[7].fact_field == turns[2].fact_field == "budget"


def test_slice_keeps_the_first_n_turns() -> None:
    sliced = slice_scenario(get_scenario("growing"), 6)

    assert isinstance(sliced, Scenario)
    assert [turn.turn_index for turn in sliced.turns] == [1, 2, 3, 4, 5, 6]
    assert sliced.name == "growing"


def test_slice_beyond_the_length_returns_every_turn() -> None:
    assert len(slice_scenario(get_scenario("pivot"), 50).turns) == 20


def test_slice_rejects_a_non_positive_count() -> None:
    with pytest.raises(ValueError):
        slice_scenario(get_scenario("pivot"), 0)


def test_get_scenario_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError):
        get_scenario("missing")
