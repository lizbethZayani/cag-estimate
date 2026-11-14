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

    response = SearchResponse(
        query="oauth",
        k=5,
        search_time_ms=123,
        search_mode="vector",
        reranked=False,
        results=[item],
    )

    assert response.model_dump() == {
        "query": "oauth",
        "k": 5,
        "search_time_ms": 123,
        "search_mode": "vector",
        "reranked": False,
        "results": [
            {
                "chunk_id": 17,
                "document_id": 3,
                "chunk_type": "budget_component",
                "content": "OAuth backend",
                "distance": 0.2314,
                "score": None,
                "metadata": {"year": 2024},
            }
        ],
    }


def test_mode_and_rerank_default_to_none_meaning_the_configured_default():
    request = SearchRequest(query="x")

    assert request.search_mode is None
    assert request.rerank is None


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_known_search_modes_are_accepted(mode):
    request = SearchRequest(query="x", search_mode=mode, rerank=True)

    assert (request.search_mode, request.rerank) == (mode, True)


@pytest.mark.parametrize("mode", ["lexical", "", "HYBRID"])
def test_unknown_search_modes_are_rejected(mode):
    with pytest.raises(ValidationError):
        SearchRequest(query="x", search_mode=mode)


def test_a_lexical_only_item_has_no_distance_and_carries_a_score():
    item = SearchResultItem(
        chunk_id=1,
        document_id=1,
        chunk_type="c",
        content="x",
        distance=None,
        score=0.5,
        metadata={},
    )

    assert item.distance is None
    assert item.score == 0.5


def test_response_echoes_the_effective_mode_and_rerank_flag():
    response = SearchResponse(
        query="q", k=1, search_time_ms=1, search_mode="hybrid", reranked=True, results=[]
    )

    dumped = response.model_dump()
    assert (dumped["search_mode"], dumped["reranked"]) == ("hybrid", True)
