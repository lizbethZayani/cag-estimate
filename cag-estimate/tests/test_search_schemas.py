"""Offline tests for the POST /search request and response schemas."""

import pytest
from pydantic import ValidationError

from cag_estimate.embedding_pipeline.schemas import SearchRequest, SearchResponse, SearchResultItem


def test_k_defaults_to_five():
    assert SearchRequest(query="payments api").k == 5


def test_query_is_stripped():
    assert SearchRequest(query="  payments api \n").query == "payments api"


@pytest.mark.parametrize("k", [1, 5, 50])
def test_k_inside_the_bounds_is_accepted(k):
    assert SearchRequest(query="x", k=k).k == k


@pytest.mark.parametrize("k", [0, -1, 51])
def test_k_outside_the_bounds_is_rejected(k):
    with pytest.raises(ValidationError):
        SearchRequest(query="x", k=k)


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(query):
    with pytest.raises(ValidationError):
        SearchRequest(query=query)


def test_response_serialises_the_documented_shape():
    item = SearchResultItem(
        chunk_id=17,
        document_id=3,
        chunk_type="budget_component",
        content="OAuth backend",
        distance=0.2314,
        metadata={"year": 2024},
    )

    response = SearchResponse(query="oauth", k=5, search_time_ms=123, results=[item])

    assert response.model_dump() == {
        "query": "oauth",
        "k": 5,
        "search_time_ms": 123,
        "results": [
            {
                "chunk_id": 17,
                "document_id": 3,
                "chunk_type": "budget_component",
                "content": "OAuth backend",
                "distance": 0.2314,
                "metadata": {"year": 2024},
            }
        ],
    }
