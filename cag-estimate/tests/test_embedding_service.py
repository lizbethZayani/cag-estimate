"""Offline tests for the ingest orchestration (real chunker/embedder, fake OpenAI client)."""

from types import SimpleNamespace
from typing import Any

import pytest

from cag_estimate.embedding_pipeline.chunker import JSONStructuralChunker
from cag_estimate.embedding_pipeline.embedder import OpenAIEmbedder
from cag_estimate.embedding_pipeline.schemas import Budget
from cag_estimate.embedding_pipeline.service import EmbeddingIngestService


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class FakeEmbeddings:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        data = [
            SimpleNamespace(index=i, embedding=[float(i), 1.0])
            for i, _ in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=data, usage=None)


def make_budget(budget_id: str, component_ids: list[str]) -> Budget:
    return Budget.model_validate(
        {
            "budget_id": budget_id,
            "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
            "project_summary": "Mobile banking API",
            "main_technology": "ruby_on_rails",
            "year": 2024,
            "total_estimated_hours": 100,
            "components": [
                {
                    "component_id": component_id,
                    "name": "Auth backend",
                    "description": "OAuth flows.",
                    "estimated_hours": 10,
                    "complexity": "high",
                }
                for component_id in component_ids
            ],
        }
    )


def make_service(client: Any) -> EmbeddingIngestService:
    chunker = JSONStructuralChunker(token_counter=WordCounter())
    return EmbeddingIngestService(chunker, OpenAIEmbedder(client, sleep=lambda _s: None))


def test_ingest_returns_embedded_chunks_in_order():
    client = SimpleNamespace(embeddings=FakeEmbeddings())
    budgets = [make_budget("B1", ["A", "B"]), make_budget("B2", ["C"])]

    response = make_service(client).ingest(budgets)

    assert [chunk.chunk_id for chunk in response.chunks] == ["B1::A", "B1::B", "B2::C"]
    assert [chunk.embedding for chunk in response.chunks] == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    assert len(client.embeddings.calls) == 1


def test_ingest_stats_have_correct_totals_and_cost():
    client = SimpleNamespace(embeddings=FakeEmbeddings())
    budgets = [make_budget("B1", ["A", "B"]), make_budget("B2", ["C"])]

    response = make_service(client).ingest(budgets)

    tokens = sum(chunk.token_count for chunk in response.chunks)
    assert response.stats.total_budgets == 2
    assert response.stats.total_chunks == 3
    assert response.stats.total_tokens == tokens
    assert response.stats.estimated_cost_usd == pytest.approx(tokens / 1_000_000 * 0.02)


def test_embedder_error_propagates():
    client = SimpleNamespace(embeddings=FakeEmbeddings(error=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        make_service(client).ingest([make_budget("B1", ["A"])])
