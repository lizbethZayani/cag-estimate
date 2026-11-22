"""Turns search results into id-tagged chunks and the ``<sources>`` block of the prompt.

A chunk id is ``"<budget_id>::<component_id>"``: stable, human readable and checkable
against the ids that were actually retrieved. Chunk contents are never logged.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from cag_estimate.embedding_pipeline.schemas import SearchResultItem

log = structlog.get_logger()

# A closing tag inside the content could end its source early or inject sources.
_CLOSING_TAG = re.compile(r"</(sources?)\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved component, addressable by the id the LLM must cite."""

    chunk_id: str
    document_id: str
    content: str


def to_retrieved_chunks(items: Sequence[SearchResultItem]) -> list[RetrievedChunk]:
    """Map search results to chunks, skipping items without ids and repeated ids."""
    chunks: dict[str, RetrievedChunk] = {}
    for item in items:
        chunk = _to_chunk(item)
        if chunk is not None:
            chunks.setdefault(chunk.chunk_id, chunk)
    return list(chunks.values())


def _to_chunk(item: SearchResultItem) -> RetrievedChunk | None:
    budget_id = item.metadata.get("budget_id")
    component_id = item.metadata.get("component_id")
    if not budget_id or not component_id:
        log.warning("rag_chunk_skipped", chunk_id=item.chunk_id, reason="missing_id_metadata")
        return None
    return RetrievedChunk(
        chunk_id=f"{budget_id}::{component_id}", document_id=str(budget_id), content=item.content
    )


def build_context_block(chunks: Sequence[RetrievedChunk]) -> str:
    """Render the chunks as ``<sources>`` with one ``<source id=... document_id=...>`` each."""
    sources = [_render_source(chunk) for chunk in chunks]
    return "\n".join(["<sources>", *sources, "</sources>"])


def _render_source(chunk: RetrievedChunk) -> str:
    content = _CLOSING_TAG.sub(r"<\\/\1>", chunk.content)
    return f'<source id="{chunk.chunk_id}" document_id="{chunk.document_id}">{content}</source>'


def retrieved_chunk_ids(chunks: Sequence[RetrievedChunk]) -> set[str]:
    """The ids the LLM was allowed to cite."""
    return {chunk.chunk_id for chunk in chunks}
