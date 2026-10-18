"""Offline tests for JSONStructuralChunker (fake token counter, no tiktoken)."""

import sys
from typing import Any

import pytest
from structlog.testing import capture_logs

from cag_estimate.embedding_pipeline.chunker import JSONStructuralChunker
from cag_estimate.embedding_pipeline.schemas import Budget


class FakeCounter:
    """Counts whitespace-separated words."""

    def count(self, text: str) -> int:
        return len(text.split())


def make_component(component_id: str = "AUTH-001", **overrides: Any) -> dict[str, Any]:
    component: dict[str, Any] = {
        "component_id": component_id,
        "name": "OAuth 2.0 authentication backend",
        "description": "OAuth 2.0 flows with JWT sessions.",
        "tech_stack": ["ruby_on_rails", "postgresql", "redis"],
        "estimated_hours": 120,
        "complexity": "high",
        "dependencies": [],
    }
    component.update(overrides)
    return component


def make_budget(budget_id: str = "BUD-1", components: list[dict[str, Any]] | None = None) -> Budget:
    return Budget.model_validate(
        {
            "budget_id": budget_id,
            "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
            "project_summary": "Mobile banking API",
            "main_technology": "ruby_on_rails",
            "year": 2024,
            "total_estimated_hours": 480,
            "components": components if components is not None else [make_component()],
        }
    )


def make_chunker(max_chunk_tokens: int = 8191) -> JSONStructuralChunker:
    return JSONStructuralChunker(token_counter=FakeCounter(), max_chunk_tokens=max_chunk_tokens)


def test_one_chunk_per_component_with_composite_ids():
    budget = make_budget(components=[make_component(f"C-{i}") for i in range(3)])

    chunks = make_chunker().chunk([budget])

    assert [chunk.chunk_id for chunk in chunks] == ["BUD-1::C-0", "BUD-1::C-1", "BUD-1::C-2"]


def test_text_has_the_contextual_header_and_component_fields():
    chunks = make_chunker().chunk([make_budget()])

    assert chunks[0].text == (
        "[Project: Mobile banking API]\n"
        "[Client sector: finance | Year: 2024 | Main tech: ruby_on_rails]\n"
        "\n"
        "Component: OAuth 2.0 authentication backend\n"
        "Description: OAuth 2.0 flows with JWT sessions.\n"
        "Tech stack: ruby_on_rails, postgresql, redis\n"
        "Complexity: high\n"
        "Estimated hours: 120"
    )


def test_metadata_is_filterable_and_not_in_the_text():
    chunk = make_chunker().chunk([make_budget()])[0]

    assert chunk.metadata == {
        "budget_id": "BUD-1",
        "component_id": "AUTH-001",
        "client_sector": "finance",
        "main_technology": "ruby_on_rails",
        "year": 2024,
        "complexity": "high",
        "estimated_hours": 120,
    }


def test_token_count_comes_from_the_injected_counter():
    chunk = make_chunker().chunk([make_budget()])[0]

    assert chunk.token_count == len(chunk.text.split())


def test_long_description_is_neither_split_nor_overlapped():
    long_description = "word " * 5000
    budget = make_budget(components=[make_component(description=long_description)])

    chunks = make_chunker().chunk([budget])

    assert len(chunks) == 1
    assert long_description in chunks[0].text


def test_oversized_chunk_logs_a_warning_and_is_still_returned():
    budget = make_budget(components=[make_component(description="word " * 50)])

    with capture_logs() as logs:
        chunks = make_chunker(max_chunk_tokens=20).chunk([budget])

    assert len(chunks) == 1
    warnings = [entry for entry in logs if entry["event"] == "chunk_exceeds_token_limit"]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert warnings[0]["chunk_id"] == "BUD-1::AUTH-001"
    assert warnings[0]["token_count"] == chunks[0].token_count
    assert all("text" not in entry and "word" not in str(entry.values()) for entry in warnings)


def test_chunk_within_the_limit_logs_nothing():
    with capture_logs() as logs:
        make_chunker().chunk([make_budget()])

    assert logs == []


def test_duplicate_chunk_ids_raise_a_clear_error():
    budget = make_budget(components=[make_component("DUP"), make_component("DUP")])

    with pytest.raises(ValueError, match=r"Duplicate chunk_id 'BUD-1::DUP'"):
        make_chunker().chunk([budget])


def test_duplicate_budget_ids_across_the_input_raise():
    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        make_chunker().chunk([make_budget("BUD-1"), make_budget("BUD-1")])


def test_two_budgets_keep_input_order():
    budgets = [
        make_budget("B-2", [make_component("X"), make_component("Y")]),
        make_budget("B-1", [make_component("X")]),
    ]

    chunks = make_chunker().chunk(budgets)

    assert [chunk.chunk_id for chunk in chunks] == ["B-2::X", "B-2::Y", "B-1::X"]


def test_empty_input_yields_no_chunks():
    assert make_chunker().chunk([]) == []


def test_default_counter_is_built_lazily(monkeypatch):
    import tiktoken

    calls: list[str] = []
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: calls.append(model))
    sys.modules.pop("cag_estimate.embedding_pipeline.chunker", None)

    from cag_estimate.embedding_pipeline.chunker import JSONStructuralChunker as Fresh

    chunker = Fresh()
    assert calls == []

    class FakeEncoding:
        def encode(self, text: str) -> list[int]:
            return [0] * len(text.split())

    monkeypatch.setattr(
        tiktoken, "encoding_for_model", lambda model: calls.append(model) or FakeEncoding()
    )
    chunker.chunk([make_budget()])
    assert calls == ["text-embedding-3-small"]
