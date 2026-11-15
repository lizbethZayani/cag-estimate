"""Shared offline builders for the RAG generation tests: estimates, search items and a retriever."""

from typing import Any

from cag_estimate.embedding_pipeline.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from cag_estimate.schemas.rag_estimation import EstimateLineItem, RagEstimate, SourceReference

QUERY = "An online store with a product catalog and a checkout"
EVIDENCE_TEXT = "Estimated hours: 140"


def make_item(
    budget_id: str | None = "BUD-1",
    component_id: str | None = "CART-001",
    content: str = "Shopping cart. Estimated hours: 140",
    chunk_id: int = 1,
) -> SearchResultItem:
    metadata: dict[str, Any] = {}
    if budget_id is not None:
        metadata["budget_id"] = budget_id
    if component_id is not None:
        metadata["component_id"] = component_id
    return SearchResultItem(
        chunk_id=chunk_id,
        document_id=1,
        chunk_type="component",
        content=content,
        distance=0.1,
        metadata=metadata,
    )


def make_response(items: list[SearchResultItem], query: str = QUERY) -> SearchResponse:
    return SearchResponse(
        query=query,
        k=10,
        search_time_ms=1,
        search_mode="hybrid",
        reranked=False,
        results=items,
    )


def grounded_line(chunk_id: str = "BUD-1::CART-001", hours: float = 140.0) -> EstimateLineItem:
    return EstimateLineItem(
        component="Shopping cart",
        hours=hours,
        rationale="Derived from the cited historical component.",
        grounded=True,
        sources=[
            SourceReference(
                chunk_id=chunk_id,
                document_id=chunk_id.split("::")[0],
                evidence=EVIDENCE_TEXT,
            )
        ],
    )


def make_estimate(*chunk_ids: str) -> RagEstimate:
    """A valid estimate with one grounded line per cited chunk id."""
    return RagEstimate(
        lines=[grounded_line(chunk_id) for chunk_id in chunk_ids],
        assumptions=["Same scope as the cited budgets."],
        confidence="medium",
        insufficient_context_explanation=None,
    )


class FakeRetriever:
    """Returns a fixed response and records every request."""

    def __init__(self, response: SearchResponse) -> None:
        self._response = response
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        return self._response
