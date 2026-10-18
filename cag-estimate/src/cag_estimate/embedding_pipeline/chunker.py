"""Structural chunker: one chunk per budget component, no overlap, no splitting.

Every chunk repeats the parent budget's project, sector, year and technology as a
contextual header so a component never loses its context once embedded alone.
"""

from typing import Any, Protocol

import structlog
import tiktoken

from cag_estimate.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_EMBEDDING_INPUT_TOKENS = 8191

log = structlog.get_logger()


class TokenCounter(Protocol):
    """Counts the tokens of a text."""

    def count(self, text: str) -> int: ...


class TiktokenCounter:
    """Counts with the embedding model's tokenizer, built on first use.

    tiktoken downloads its BPE file on first use, so nothing is built at import.
    """

    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self._model = model
        self._encoding: tiktoken.Encoding | None = None

    def count(self, text: str) -> int:
        if self._encoding is None:
            self._encoding = tiktoken.encoding_for_model(self._model)
        return len(self._encoding.encode(text))


def _render_text(budget: Budget, component: BudgetComponent) -> str:
    return "\n".join(
        [
            f"[Project: {budget.project_summary}]",
            (
                f"[Client sector: {budget.client_metadata.sector} | Year: {budget.year}"
                f" | Main tech: {budget.main_technology}]"
            ),
            "",
            f"Component: {component.name}",
            f"Description: {component.description}",
            f"Tech stack: {', '.join(component.tech_stack)}",
            f"Complexity: {component.complexity}",
            f"Estimated hours: {component.estimated_hours}",
        ]
    )


def _metadata(budget: Budget, component: BudgetComponent) -> dict[str, Any]:
    return {
        "budget_id": budget.budget_id,
        "component_id": component.component_id,
        "client_sector": budget.client_metadata.sector,
        "main_technology": budget.main_technology,
        "year": budget.year,
        "complexity": component.complexity,
        "estimated_hours": component.estimated_hours,
    }


class JSONStructuralChunker:
    """Turns budgets into one chunk per component, in input order."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        max_chunk_tokens: int = MAX_EMBEDDING_INPUT_TOKENS,
    ) -> None:
        self._counter = token_counter or TiktokenCounter()
        self._max_chunk_tokens = max_chunk_tokens

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        """One chunk per component; raises ``ValueError`` on a duplicate chunk id."""
        chunks: list[Chunk] = []
        seen: set[str] = set()
        for budget in budgets:
            for component in budget.components:
                chunk = self._build_chunk(budget, component)
                if chunk.chunk_id in seen:
                    raise ValueError(
                        f"Duplicate chunk_id '{chunk.chunk_id}': budget_id and component_id "
                        "pairs must be unique across the input"
                    )
                seen.add(chunk.chunk_id)
                chunks.append(chunk)
        return chunks

    def _build_chunk(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = _render_text(budget, component)
        chunk = Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata=_metadata(budget, component),
            token_count=self._counter.count(text),
        )
        if chunk.token_count > self._max_chunk_tokens:
            log.warning(
                "chunk_exceeds_token_limit",
                chunk_id=chunk.chunk_id,
                token_count=chunk.token_count,
                max_chunk_tokens=self._max_chunk_tokens,
            )
        return chunk
