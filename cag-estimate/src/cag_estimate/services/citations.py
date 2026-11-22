"""
Citation verification for cited RAG estimates (pure: no I/O, no logging).

A dangling citation points to a source that was never given to the LLM, i.e. a
hallucination dressed up as rigor. ``verify_citations`` checks every grounded
line against the chunk ids that were actually retrieved.
"""

from typing import Literal

from pydantic import BaseModel

from cag_estimate.schemas.rag_estimation import EstimateLineItem, RagEstimate

LineStatus = Literal["grounded", "dangling", "insufficient"]


class LineCitation(BaseModel):
    """Verification outcome of one estimate line."""

    component: str
    status: LineStatus
    cited_chunk_ids: list[str]
    dangling_chunk_ids: list[str]


class CitationReport(BaseModel):
    """Aggregated verification outcome of a whole estimate."""

    total_lines: int
    grounded_lines: int
    dangling_lines: int
    insufficient_lines: int
    dangling_chunk_ids: list[str]  # sorted and unique across all lines
    lines: list[LineCitation]

    @property
    def has_dangling(self) -> bool:
        return self.dangling_lines > 0


def verify_citations(estimate: RagEstimate, retrieved_chunk_ids: set[str]) -> CitationReport:
    """Classify every line as grounded, dangling (cites unretrieved ids) or insufficient."""
    lines = [_verify_line(line, retrieved_chunk_ids) for line in estimate.lines]
    dangling_ids = sorted({cid for line in lines for cid in line.dangling_chunk_ids})
    return CitationReport(
        total_lines=len(lines),
        grounded_lines=_count(lines, "grounded"),
        dangling_lines=_count(lines, "dangling"),
        insufficient_lines=_count(lines, "insufficient"),
        dangling_chunk_ids=dangling_ids,
        lines=lines,
    )


def _verify_line(line: EstimateLineItem, retrieved_chunk_ids: set[str]) -> LineCitation:
    cited = [source.chunk_id for source in line.sources]
    if not line.grounded:
        return LineCitation(
            component=line.component, status="insufficient", cited_chunk_ids=cited,
            dangling_chunk_ids=[],
        )
    missing = [chunk_id for chunk_id in cited if chunk_id not in retrieved_chunk_ids]
    return LineCitation(
        component=line.component,
        status="dangling" if missing else "grounded",
        cited_chunk_ids=cited,
        dangling_chunk_ids=missing,
    )


def _count(lines: list[LineCitation], status: LineStatus) -> int:
    return sum(1 for line in lines if line.status == status)
