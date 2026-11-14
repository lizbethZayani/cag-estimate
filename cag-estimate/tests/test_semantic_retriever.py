"""Offline tests for the semantic retriever (fake embedder, session factory and store)."""

from collections.abc import Callable, Sequence
from typing import Any, Self

import pytest
from structlog.testing import capture_logs

from cag_estimate.embedding_pipeline.repository import ChunkHit
from cag_estimate.embedding_pipeline.retriever import SearchEmbeddingError, SemanticRetriever
from cag_estimate.embedding_pipeline.schemas import SearchMode, SearchRequest

QUERY = "oauth banking backend"
VECTOR = [0.1, 0.2, 0.3]


class FakeSession:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    async def __aenter__(self) -> Self:
        self._journal.append("open")
        return self

    async def __aexit__(self, *_: object) -> None:
        self._journal.append("close")


class FakeEmbedder:
    def __init__(self, journal: list[str], error: Exception | None = None) -> None:
        self._journal = journal
        self.error = error
        self.queries: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self._journal.append("embed")
        self.queries.append(text)
        if self.error:
            raise self.error
        return VECTOR


class FakeStore:
    def __init__(
        self, journal: list[str], hits: list[ChunkHit], lexical_hits: list[ChunkHit]
    ) -> None:
        self._journal = journal
        self.hits = hits
        self.lexical_hits = lexical_hits
        self.calls: list[tuple[Any, list[float], int]] = []
        self.lexical_calls: list[tuple[Any, str, int]] = []

    async def search(self, session: Any, query_vector: list[float], k: int) -> list[ChunkHit]:
        self._journal.append("search")
        self.calls.append((session, query_vector, k))
        return self.hits

    async def search_lexical(self, session: Any, query_text: str, k: int) -> list[ChunkHit]:
        self._journal.append("search_lexical")
        self.lexical_calls.append((session, query_text, k))
        return self.lexical_hits


class FakeReranker:
    """Scores a candidate by its text (``scores``), highest first; optionally raises."""

    def __init__(self, scores: dict[str, float], error: Exception | None = None) -> None:
        self._scores = scores
        self.error = error
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self, query: str, candidates: Sequence[Any], *, top_n: int, text_of: Callable[[Any], str]
    ) -> list[tuple[Any, float]]:
        self.calls.append((query, [text_of(c) for c in candidates], top_n))
        if self.error:
            raise self.error
        scored = [(c, self._scores[text_of(c)]) for c in candidates]
        return sorted(scored, key=lambda pair: -pair[1])[:top_n]


def make_lexical_hit(chunk_id: int, score: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        document_id=3,
        chunk_type="budget_component",
        content=f"content {chunk_id}",
        distance=None,
        metadata={"year": 2024},
        score=score,
    )


def make_hit(chunk_id: int, distance: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        document_id=3,
        chunk_type="budget_component",
        content=f"content {chunk_id}",
        distance=distance,
        metadata={"year": 2024},
    )


class Harness:
    def __init__(
        self,
        hits: list[ChunkHit] | None = None,
        embed_error: Exception | None = None,
        lexical_hits: list[ChunkHit] | None = None,
        reranker: FakeReranker | None = None,
        **settings: Any,
    ) -> None:
        self.journal: list[str] = []
        self.embedder = FakeEmbedder(self.journal, embed_error)
        self.store = FakeStore(
            self.journal,
            hits if hits is not None else [],
            lexical_hits if lexical_hits is not None else [],
        )
        self.reranker = reranker
        ticks = iter([10.0, 10.123])
        self.retriever = SemanticRetriever(
            lambda: FakeSession(self.journal),
            self.embedder,
            store=self.store,
            clock=lambda: next(ticks),
            reranker=reranker,
            **settings,
        )


def ids(response: Any) -> list[int]:
    return [item.chunk_id for item in response.results]


async def test_the_query_is_embedded_once_before_the_database_is_opened():
    harness = Harness()

    await harness.retriever.search(SearchRequest(query=QUERY))

    assert harness.journal == ["embed", "open", "search", "close"]
    assert harness.embedder.queries == [QUERY]


async def test_the_vector_and_k_are_passed_to_the_store():
    harness = Harness()

    await harness.retriever.search(SearchRequest(query=QUERY, k=7))

    _, vector, k = harness.store.calls[0]
    assert vector == VECTOR
    assert k == 7


async def test_response_keeps_the_store_order_and_reports_the_elapsed_time():
    harness = Harness(hits=[make_hit(5, 0.1), make_hit(2, 0.4)])

    response = await harness.retriever.search(SearchRequest(query=QUERY, k=2))

    assert response.query == QUERY
    assert response.k == 2
    assert response.search_time_ms == 123
    assert [(r.chunk_id, r.distance) for r in response.results] == [(5, 0.1), (2, 0.4)]
    assert response.results[0].model_dump() == {
        "chunk_id": 5,
        "document_id": 3,
        "chunk_type": "budget_component",
        "content": "content 5",
        "distance": 0.1,
        "score": None,
        "metadata": {"year": 2024},
    }


async def test_an_empty_corpus_gives_an_empty_result_list():
    response = await Harness(hits=[]).retriever.search(SearchRequest(query=QUERY))

    assert response.results == []


async def test_embedding_failure_raises_search_embedding_error_and_never_opens_the_database():
    harness = Harness(embed_error=ValueError("boom"))

    with pytest.raises(SearchEmbeddingError) as caught:
        await harness.retriever.search(SearchRequest(query=QUERY))

    assert isinstance(caught.value.__cause__, ValueError)
    assert harness.journal == ["embed"]


async def test_completed_log_has_counts_and_latency_but_not_the_query_text():
    harness = Harness(hits=[make_hit(1, 0.1)])

    with capture_logs() as logs:
        await harness.retriever.search(SearchRequest(query=QUERY, k=3))

    event = next(entry for entry in logs if entry["event"] == "rag_search_completed")
    assert event["k"] == 3
    assert event["results"] == 1
    assert event["latency_ms"] == 123
    assert event["query_length"] == len(QUERY)
    assert QUERY not in repr(logs)


# --- hybrid search and reranking ---

VECTOR_HITS = [make_hit(1, 0.1), make_hit(2, 0.2), make_hit(3, 0.3)]
RERANK_SCORES = {"content 1": 0.1, "content 2": 0.9, "content 3": 0.5, "content 9": 0.7}


async def test_the_default_configuration_is_the_plain_vector_search():
    harness = Harness(hits=VECTOR_HITS[:2], lexical_hits=[make_lexical_hit(9, 1.0)])

    response = await harness.retriever.search(SearchRequest(query=QUERY, k=2))

    assert harness.journal == ["embed", "open", "search", "close"]
    assert [call[2] for call in harness.store.calls] == [2]
    assert harness.store.lexical_calls == []
    assert (response.search_mode, response.reranked) == ("vector", False)
    assert [(r.chunk_id, r.distance, r.score) for r in response.results] == [
        (1, 0.1, None),
        (2, 0.2, None),
    ]


async def test_vector_mode_never_runs_the_lexical_branch():
    harness = Harness(hits=VECTOR_HITS, lexical_hits=[make_lexical_hit(9, 1.0)])

    await harness.retriever.search(SearchRequest(query=QUERY, search_mode="vector"))

    assert harness.store.lexical_calls == []


async def test_vector_with_rerank_recalls_recall_k_then_reranks_down_to_k():
    reranker = FakeReranker(RERANK_SCORES)
    harness = Harness(hits=VECTOR_HITS, reranker=reranker, recall_k=20)

    response = await harness.retriever.search(SearchRequest(query=QUERY, k=2, rerank=True))

    assert [call[2] for call in harness.store.calls] == [20]
    assert harness.store.lexical_calls == []
    assert reranker.calls == [(QUERY, ["content 1", "content 2", "content 3"], 2)]
    assert [(r.chunk_id, r.score) for r in response.results] == [(2, 0.9), (3, 0.5)]
    assert response.results[0].distance == 0.2
    assert (response.search_mode, response.reranked) == ("vector", True)


async def test_hybrid_fuses_both_branches_and_cuts_to_k_with_the_rrf_score():
    harness = Harness(
        hits=VECTOR_HITS,
        lexical_hits=[make_lexical_hit(3, 5.0)],
        recall_k=20,
        rrf_k=60,
    )

    response = await harness.retriever.search(SearchRequest(query=QUERY, k=2, search_mode="hybrid"))

    assert [call[2] for call in harness.store.calls] == [20]
    assert harness.store.lexical_calls == [(harness.store.lexical_calls[0][0], QUERY, 20)]
    assert ids(response) == [3, 1]
    assert response.results[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert response.results[1].score == pytest.approx(1 / 61)
    assert (response.search_mode, response.reranked) == ("hybrid", False)
    assert harness.journal == ["embed", "open", "search", "search_lexical", "close"]


async def test_hybrid_pool_is_never_smaller_than_k():
    harness = Harness(hits=VECTOR_HITS, recall_k=3)

    await harness.retriever.search(SearchRequest(query=QUERY, k=10, search_mode="hybrid"))

    assert [call[2] for call in harness.store.calls] == [10]
    assert [call[2] for call in harness.store.lexical_calls] == [10]


async def test_hybrid_with_rerank_reranks_the_fused_candidates_down_to_k():
    reranker = FakeReranker(RERANK_SCORES)
    harness = Harness(hits=VECTOR_HITS, lexical_hits=[make_lexical_hit(9, 5.0)], reranker=reranker)

    response = await harness.retriever.search(
        SearchRequest(query=QUERY, k=2, search_mode="hybrid", rerank=True)
    )

    assert reranker.calls[0][2] == 2
    assert sorted(reranker.calls[0][1]) == ["content 1", "content 2", "content 3", "content 9"]
    assert ids(response) == [2, 9]
    assert [r.score for r in response.results] == [0.9, 0.7]
    assert (response.search_mode, response.reranked) == ("hybrid", True)


async def test_a_chunk_found_by_both_branches_keeps_the_vector_distance():
    harness = Harness(hits=[make_hit(1, 0.1)], lexical_hits=[make_lexical_hit(1, 5.0)])

    response = await harness.retriever.search(SearchRequest(query=QUERY, search_mode="hybrid"))

    assert [(r.chunk_id, r.distance) for r in response.results] == [(1, 0.1)]


async def test_a_lexical_only_hit_has_no_distance_but_a_score():
    harness = Harness(hits=[make_hit(1, 0.1)], lexical_hits=[make_lexical_hit(9, 5.0)])

    response = await harness.retriever.search(SearchRequest(query=QUERY, search_mode="hybrid"))

    lexical_only = next(r for r in response.results if r.chunk_id == 9)
    assert lexical_only.distance is None
    assert lexical_only.score == pytest.approx(1 / 61)


async def test_empty_lexical_results_leave_the_vector_order():
    harness = Harness(hits=VECTOR_HITS, lexical_hits=[])

    response = await harness.retriever.search(SearchRequest(query=QUERY, search_mode="hybrid"))

    assert ids(response) == [1, 2, 3]


async def test_the_request_overrides_the_configured_defaults():
    reranker = FakeReranker(RERANK_SCORES)
    harness = Harness(
        hits=VECTOR_HITS, reranker=reranker, default_search_mode="hybrid", default_rerank=True
    )

    response = await harness.retriever.search(
        SearchRequest(query=QUERY, search_mode="vector", rerank=False)
    )

    assert (response.search_mode, response.reranked) == ("vector", False)
    assert harness.store.lexical_calls == []
    assert reranker.calls == []


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
async def test_missing_request_fields_fall_back_to_the_configured_defaults(mode: SearchMode):
    reranker = FakeReranker(RERANK_SCORES)
    harness = Harness(
        hits=VECTOR_HITS, reranker=reranker, default_search_mode=mode, default_rerank=True
    )

    response = await harness.retriever.search(SearchRequest(query=QUERY))

    assert (response.search_mode, response.reranked) == (mode, True)
    assert len(reranker.calls) == 1
    assert bool(harness.store.lexical_calls) == (mode == "hybrid")


async def test_a_failing_reranker_falls_back_to_the_fused_order_and_logs_a_warning():
    reranker = FakeReranker(RERANK_SCORES, error=RuntimeError(f"secret {QUERY} content 1"))
    harness = Harness(hits=VECTOR_HITS, reranker=reranker)

    with capture_logs() as logs:
        response = await harness.retriever.search(SearchRequest(query=QUERY, k=2, rerank=True))

    assert ids(response) == [1, 2]
    assert response.reranked is False
    assert [r.score for r in response.results] == [None, None]
    warning = next(entry for entry in logs if entry["event"] == "rerank_failed")
    assert warning["log_level"] == "warning"
    assert warning["error_type"] == "RuntimeError"
    assert QUERY not in repr(logs)
    assert "content 1" not in repr(logs)


async def test_a_failing_reranker_keeps_the_rrf_scores_of_the_hybrid_fallback():
    reranker = FakeReranker(RERANK_SCORES, error=ValueError("boom"))
    harness = Harness(hits=VECTOR_HITS, lexical_hits=[make_lexical_hit(3, 5.0)], reranker=reranker)

    response = await harness.retriever.search(
        SearchRequest(query=QUERY, k=2, search_mode="hybrid", rerank=True)
    )

    assert ids(response) == [3, 1]
    assert response.reranked is False
    assert response.results[0].score == pytest.approx(1 / 63 + 1 / 61)


async def test_rerank_requested_without_a_reranker_fails_soft():
    harness = Harness(hits=VECTOR_HITS, reranker=None)

    with capture_logs() as logs:
        response = await harness.retriever.search(SearchRequest(query=QUERY, k=2, rerank=True))

    assert ids(response) == [1, 2]
    assert response.reranked is False
    assert any(entry["event"] == "rerank_failed" for entry in logs)
    assert QUERY not in repr(logs)


async def test_completed_log_reports_mode_rerank_and_candidates():
    reranker = FakeReranker(RERANK_SCORES)
    harness = Harness(hits=VECTOR_HITS, lexical_hits=[make_lexical_hit(9, 5.0)], reranker=reranker)

    with capture_logs() as logs:
        await harness.retriever.search(
            SearchRequest(query=QUERY, k=2, search_mode="hybrid", rerank=True)
        )

    event = next(entry for entry in logs if entry["event"] == "rag_search_completed")
    assert event["search_mode"] == "hybrid"
    assert event["reranked"] is True
    assert event["candidates"] == 4
    assert event["results"] == 2
    assert QUERY not in repr(logs)
