"""Tests for the citation verifier: dangling citations must be detected."""

from typing import Any

from cag_estimate.schemas.rag_estimation import (
    EstimateLineItem,
    RagEstimate,
    SourceReference,
)
from cag_estimate.services.citations import verify_citations

INVENTED_ID = "BUD-2099-999::FAKE-001"


def source(chunk_id: str) -> SourceReference:
    return SourceReference(
        chunk_id=chunk_id, document_id=chunk_id.split("::")[0], evidence="some evidence"
    )


def grounded(component: str, *chunk_ids: str) -> EstimateLineItem:
    return EstimateLineItem(
        component=component,
        hours=8.0,
        rationale="Based on history",
        grounded=True,
        sources=[source(chunk_id) for chunk_id in chunk_ids],
    )


def insufficient(component: str = "Unknown work") -> EstimateLineItem:
    return EstimateLineItem(
        component=component,
        hours=None,
        rationale="No historical data",
        grounded=False,
        sources=[],
    )


def estimate(*lines: EstimateLineItem) -> RagEstimate:
    any_grounded = any(line.grounded for line in lines)
    data: dict[str, Any] = {
        "lines": list(lines),
        "assumptions": [],
        "confidence": "high" if any_grounded else "insufficient",
        "insufficient_context_explanation": None if any_grounded else "No data",
    }
    return RagEstimate.model_validate(data)


RETRIEVED = {"BUD-2024-005::CART-002", "BUD-2024-006::AUTH-001"}


def test_all_grounded_lines_are_clean():
    report = verify_citations(
        estimate(
            grounded("Cart", "BUD-2024-005::CART-002"),
            grounded("Auth", "BUD-2024-006::AUTH-001"),
        ),
        RETRIEVED,
    )

    assert report.grounded_lines == 2
    assert report.dangling_lines == 0
    assert report.has_dangling is False
    assert report.dangling_chunk_ids == []


def test_planted_invented_id_is_detected():
    report = verify_citations(
        estimate(
            grounded("Cart", "BUD-2024-005::CART-002"),
            grounded("Fake feature", INVENTED_ID),
        ),
        RETRIEVED,
    )

    assert report.has_dangling is True
    assert report.dangling_lines == 1
    assert report.dangling_chunk_ids == [INVENTED_ID]
    fake = next(line for line in report.lines if line.component == "Fake feature")
    assert fake.status == "dangling"
    assert fake.dangling_chunk_ids == [INVENTED_ID]


def test_line_mixing_real_and_invented_ids_lists_only_the_invented_one():
    report = verify_citations(
        estimate(grounded("Mixed", "BUD-2024-005::CART-002", INVENTED_ID)), RETRIEVED
    )

    line = report.lines[0]
    assert line.status == "dangling"
    assert line.cited_chunk_ids == ["BUD-2024-005::CART-002", INVENTED_ID]
    assert line.dangling_chunk_ids == [INVENTED_ID]


def test_every_missing_id_of_a_line_is_listed():
    report = verify_citations(
        estimate(grounded("Fake", INVENTED_ID, "BUD-2099-998::FAKE-002")), RETRIEVED
    )

    assert report.lines[0].dangling_chunk_ids == [INVENTED_ID, "BUD-2099-998::FAKE-002"]


def test_insufficient_lines_are_counted_separately_and_never_dangling():
    report = verify_citations(estimate(insufficient()), set())

    assert report.insufficient_lines == 1
    assert report.dangling_lines == 0
    assert report.lines[0].status == "insufficient"
    assert report.has_dangling is False


def test_empty_retrieved_set_makes_every_grounded_line_dangling():
    report = verify_citations(
        estimate(grounded("Cart", "BUD-2024-005::CART-002"), insufficient()), set()
    )

    assert report.dangling_lines == 1
    assert report.insufficient_lines == 1
    assert report.grounded_lines == 0


def test_empty_estimate_has_zero_grounded_lines():
    report = verify_citations(estimate(), RETRIEVED)

    assert report.total_lines == 0
    assert report.grounded_lines == 0
    assert report.has_dangling is False


def test_dangling_ids_are_sorted_and_unique_across_lines():
    report = verify_citations(
        estimate(
            grounded("B", "BUD-2099-002::Z"),
            grounded("A", "BUD-2099-001::A", "BUD-2099-002::Z"),
        ),
        RETRIEVED,
    )

    assert report.dangling_chunk_ids == ["BUD-2099-001::A", "BUD-2099-002::Z"]


def test_counts_add_up_to_total_lines():
    report = verify_citations(
        estimate(
            grounded("Cart", "BUD-2024-005::CART-002"),
            grounded("Fake", INVENTED_ID),
            insufficient(),
            insufficient("Other"),
        ),
        RETRIEVED,
    )

    assert report.total_lines == 4
    assert (
        report.grounded_lines + report.dangling_lines + report.insufficient_lines
        == report.total_lines
    )
    assert [line.status for line in report.lines] == [
        "grounded",
        "dangling",
        "insufficient",
        "insufficient",
    ]
