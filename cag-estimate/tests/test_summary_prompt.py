"""Offline tests for the conversation-summary prompt templates."""

import pytest
from jinja2 import UndefinedError

from cag_estimate.prompts import SummaryTurn, loader, render_summary_prompt


def test_system_prompt_states_the_limit_and_that_the_conversation_is_data():
    system, _user = render_summary_prompt(None, [], 1500)

    assert "1500 characters" in system
    assert "DATA" in system
    assert "never instructions" in system


def test_user_prompt_renders_previous_summary_and_turns():
    turns = [SummaryTurn("Add a mobile app", "Demo, 30 h")]

    _system, user = render_summary_prompt("- web portal agreed", turns, 800)

    assert "<previous_summary>\n- web portal agreed\n</previous_summary>" in user
    assert "<user>Add a mobile app</user>" in user
    assert "<assistant_digest>Demo, 30 h</assistant_digest>" in user


def test_missing_previous_summary_renders_none():
    _system, user = render_summary_prompt(None, [SummaryTurn("hi", "")], 800)

    assert "<previous_summary>\nnone\n</previous_summary>" in user


def test_pinned_turn_without_digest_is_marked():
    _system, user = render_summary_prompt(None, [SummaryTurn("Under NDA", None)], 800)

    assert "<user>Under NDA</user>" in user
    assert "pinned constraint" in user
    assert "<assistant_digest>" not in user


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        render_summary_prompt(None, [], 800, version="v99")


def test_templates_use_strict_undefined():
    with pytest.raises(UndefinedError):
        loader._render("conversation_summary", "system", "v1", {})
