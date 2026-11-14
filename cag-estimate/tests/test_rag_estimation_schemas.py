"""Tests for the cited RAG estimate schemas and their integrity rule."""

from typing import Any

import pytest
from pydantic import ValidationError

from cag_estimate.schemas.rag_estimation import (
    EstimateLineItem,
    RagEstimate,
    SourceReference,
)


def make_source(**overrides: Any) -> SourceReference:
    data: dict[str, Any] = {
        "chunk_id": "BUD-2024-005::CART-002",
        "document_id": "BUD-2024-005",
        "evidence": "Shopping cart: 24 hours",
    }
    data.update(overrides)
    return SourceReference.model_validate(data)


def grounded_line(hours: float | None = 24.0, **overrides: Any) -> EstimateLineItem:
    data: dict[str, Any] = {
        "component": "Shopping cart",
        "hours": hours,
        "rationale": "Same scope as a past project",
        "grounded": True,
        "sources": [make_source()],
    }
    data.update(overrides)
    return EstimateLineItem.model_validate(data)


def ungrounded_line(**overrides: Any) -> EstimateLineItem:
    data: dict[str, Any] = {
        "component": "Loyalty program",
        "hours": None,
        "rationale": "No historical budget covers a loyalty program",
        "grounded": False,
        "sources": [],
    }
    data.update(overrides)
    return EstimateLineItem.model_validate(data)


def make_estimate(lines: list[EstimateLineItem], **overrides: Any) -> RagEstimate:
    data: dict[str, Any] = {
        "lines": lines,
        "assumptions": [],
        "confidence": "high",
        "insufficient_context_explanation": None,
    }
    data.update(overrides)
    return RagEstimate.model_validate(data)


# --- SourceReference ---------------------------------------------------------


def test_source_reference_accepts_valid_ids():
    source = make_source()

    assert source.chunk_id == "BUD-2024-005::CART-002"
    assert source.document_id == "BUD-2024-005"


@pytest.mark.parametrize("field", ["chunk_id", "document_id", "evidence"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_source_reference_rejects_blank_fields(field, blank):
    with pytest.raises(ValidationError, match=field):
        make_source(**{field: blank})


def test_source_reference_strips_evidence():
    assert make_source(evidence="  24 hours  ").evidence == "24 hours"


# --- EstimateLineItem integrity rule -----------------------------------------


def test_grounded_line_with_source_and_hours_is_valid():
    line = grounded_line(hours=0.0)

    assert line.grounded is True
    assert line.hours == 0.0


def test_grounded_line_without_sources_is_rejected():
    with pytest.raises(ValidationError, match="grounded line must cite at least one source"):
        grounded_line(sources=[])


def test_grounded_line_with_null_hours_is_rejected():
    with pytest.raises(ValidationError, match="grounded line must have hours"):
        grounded_line(hours=None)


def test_grounded_line_with_negative_hours_is_rejected():
    with pytest.raises(ValidationError, match="hours must be >= 0"):
        grounded_line(hours=-1.0)


def test_ungrounded_line_is_valid():
    line = ungrounded_line()

    assert line.hours is None
    assert line.sources == []


def test_ungrounded_line_with_sources_is_rejected():
    with pytest.raises(ValidationError, match="ungrounded line must not cite sources"):
        ungrounded_line(sources=[make_source()])


def test_ungrounded_line_with_hours_is_rejected():
    with pytest.raises(ValidationError, match="ungrounded line must not invent hours"):
        ungrounded_line(hours=8.0)


@pytest.mark.parametrize("blank", ["", "  "])
def test_ungrounded_line_with_blank_rationale_is_rejected(blank):
    with pytest.raises(ValidationError, match="ungrounded line must explain what data is missing"):
        ungrounded_line(rationale=blank)


# --- RagEstimate -------------------------------------------------------------


def test_total_hours_sums_grounded_lines_only():
    estimate = make_estimate(
        [grounded_line(24.0), grounded_line(10.5, component="Checkout"), ungrounded_line()]
    )

    assert estimate.total_hours == 34.5


def test_total_hours_is_none_when_no_line_is_grounded():
    estimate = make_estimate(
        [ungrounded_line()],
        confidence="insufficient",
        insufficient_context_explanation="The retrieved budgets do not cover this work",
    )

    assert estimate.total_hours is None


def test_total_hours_is_none_for_an_empty_estimate():
    estimate = make_estimate(
        [],
        confidence="insufficient",
        insufficient_context_explanation="Nothing relevant was retrieved",
    )

    assert estimate.total_hours is None


def test_total_hours_is_not_part_of_the_validation_schema():
    schema = RagEstimate.model_json_schema(mode="validation")

    assert "total_hours" not in schema["properties"]


def test_total_hours_is_serialized():
    estimate = make_estimate([grounded_line(24.0)])

    assert estimate.model_dump()["total_hours"] == 24.0


def test_no_grounded_line_requires_insufficient_confidence():
    with pytest.raises(ValidationError, match='confidence must be "insufficient"'):
        make_estimate(
            [ungrounded_line()],
            confidence="low",
            insufficient_context_explanation="Missing data",
        )


@pytest.mark.parametrize("explanation", [None, "", "   "])
def test_no_grounded_line_requires_an_explanation(explanation):
    with pytest.raises(ValidationError, match="insufficient_context_explanation"):
        make_estimate(
            [ungrounded_line()],
            confidence="insufficient",
            insufficient_context_explanation=explanation,
        )


def test_grounded_line_forbids_insufficient_confidence():
    with pytest.raises(ValidationError, match='confidence must not be "insufficient"'):
        make_estimate(
            [grounded_line()],
            confidence="insufficient",
            insufficient_context_explanation="Missing data",
        )


def test_partially_grounded_estimate_is_valid():
    estimate = make_estimate([grounded_line(), ungrounded_line()], confidence="medium")

    assert estimate.confidence == "medium"


def test_json_round_trip_ignores_the_computed_total():
    estimate = make_estimate([grounded_line(24.0), ungrounded_line()], confidence="medium")

    restored = RagEstimate.model_validate_json(estimate.model_dump_json())

    assert restored == estimate
    assert restored.total_hours == 24.0
