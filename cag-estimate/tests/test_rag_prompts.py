"""Tests for the versioned RAG estimation prompts."""

import pytest

from cag_estimate.prompts.loader import RAG_ESTIMATION_PROMPT_VERSION, render_rag_estimation_prompt

CONTEXT = '<sources>\n<source id="BUD-1::CART-001" document_id="BUD-1">Cart</source>\n</sources>'
QUERY = "A bookstore with a checkout"


def test_prompt_version_is_v1() -> None:
    assert RAG_ESTIMATION_PROMPT_VERSION == "v1"


@pytest.mark.parametrize(
    "rule",
    [
        "HOURS",
        "<sources>",
        "BUDGET::COMPONENT",
        "EXACTLY",
        "VERBATIM",
        "grounded=false",
        "sources=[]",
        "hours=null",
        "rationale",
        "assumptions",
        'confidence="insufficient"',
        "insufficient_context_explanation",
        "lines=[]",
        "outside knowledge",
    ],
)
def test_system_prompt_states_the_key_rules(rule: str) -> None:
    system, _ = render_rag_estimation_prompt(QUERY, CONTEXT)

    assert rule in system


def test_user_prompt_contains_the_query_and_the_context() -> None:
    _, user = render_rag_estimation_prompt(QUERY, CONTEXT)

    assert QUERY in user
    assert CONTEXT in user


def test_feedback_section_is_absent_without_feedback() -> None:
    _, user = render_rag_estimation_prompt(QUERY, CONTEXT)

    assert "rejected" not in user.lower()


def test_feedback_section_is_rendered_when_feedback_is_given() -> None:
    _, user = render_rag_estimation_prompt(QUERY, CONTEXT, feedback="invented id BUD-9::X-1")

    assert "rejected" in user.lower()
    assert "invented id BUD-9::X-1" in user


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown rag_estimation prompt version"):
        render_rag_estimation_prompt(QUERY, CONTEXT, version="v99")
