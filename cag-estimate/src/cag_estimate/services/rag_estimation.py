"""Cited RAG estimation: retrieve, generate, verify citations, one corrective retry, reject.

Logs carry counts, ids and the request id only: never the query, chunk contents,
evidence or rationale text (consistent with the retriever).
"""

from dataclasses import dataclass
from typing import Protocol

import structlog

from cag_estimate.embedding_pipeline.schemas import SearchRequest, SearchResponse
from cag_estimate.prompts.loader import RAG_ESTIMATION_PROMPT_VERSION
from cag_estimate.request_context import current_request_id
from cag_estimate.schemas.rag_estimation import RagEstimate
from cag_estimate.services.citations import CitationReport, verify_citations
from cag_estimate.services.rag_context import (
    RetrievedChunk,
    build_context_block,
    retrieved_chunk_ids,
    to_retrieved_chunks,
)
from cag_estimate.services.rag_errors import DanglingCitationError, RagOutputInvalidError

log = structlog.get_logger()

NO_BUDGET_MATCHED = "No historical budget matched the project description."


class Retriever(Protocol):
    async def search(self, request: SearchRequest) -> SearchResponse: ...


class Generator(Protocol):
    async def generate(
        self, query: str, context_block: str, feedback: str | None = None
    ) -> RagEstimate: ...


@dataclass(frozen=True)
class RagEstimationResult:
    """An accepted estimate with its citation report and the chunks it was grounded on."""

    estimate: RagEstimate
    citation_report: CitationReport
    chunks: list[RetrievedChunk]
    prompt_version: str


@dataclass(frozen=True)
class _Verified:
    estimate: RagEstimate
    report: CitationReport
    attempts: int


class RagEstimationService:
    """Produces a cited estimate; an estimate with dangling citations is never returned."""

    def __init__(self, retriever: Retriever, generator: Generator, *, top_k: int) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    async def estimate(self, query: str, request_id: str | None = None) -> RagEstimationResult:
        """Estimate ``query`` from the retrieved history.

        Raises:
            DanglingCitationError: Both attempts cited sources that were not retrieved.
            RagOutputInvalidError: Both attempts broke the estimate integrity rules.
            RagGenerationError: The generator failed.
        """
        request_id = request_id or current_request_id()
        chunks = await self._retrieve(query)
        if not chunks:
            return self._accept(_insufficient(), chunks, request_id, attempts=0)
        try:
            verified = await self._generate_verified(query, build_context_block(chunks), chunks)
        except DanglingCitationError as exc:
            log.warning(
                "citations_rejected",
                request_id=request_id,
                dangling_chunk_ids=exc.report.dangling_chunk_ids,
                attempts=2,
            )
            raise
        return self._accept(verified.estimate, chunks, request_id, verified.attempts, verified.report)

    async def _retrieve(self, query: str) -> list[RetrievedChunk]:
        request = SearchRequest(query=query, k=self._top_k, search_mode="hybrid", rerank=False)
        response = await self._retriever.search(request)
        return to_retrieved_chunks(response.results)

    async def _generate_verified(
        self, query: str, context_block: str, chunks: list[RetrievedChunk]
    ) -> _Verified:
        ids = retrieved_chunk_ids(chunks)
        try:
            estimate = await self._generator.generate(query, context_block)
        except RagOutputInvalidError as exc:
            return await self._retry(query, context_block, ids, _invalid_feedback(exc))
        report = verify_citations(estimate, ids)
        if not report.has_dangling:
            return _Verified(estimate, report, attempts=1)
        return await self._retry(query, context_block, ids, _dangling_feedback(report))

    async def _retry(
        self, query: str, context_block: str, ids: set[str], feedback: str
    ) -> _Verified:
        estimate = await self._generator.generate(query, context_block, feedback)
        report = verify_citations(estimate, ids)
        if report.has_dangling:
            raise DanglingCitationError(report)
        return _Verified(estimate, report, attempts=2)

    def _accept(
        self,
        estimate: RagEstimate,
        chunks: list[RetrievedChunk],
        request_id: str | None,
        attempts: int,
        report: CitationReport | None = None,
    ) -> RagEstimationResult:
        report = report or verify_citations(estimate, set())
        log.info(
            "citation_report",
            request_id=request_id,
            total_lines=report.total_lines,
            grounded_lines=report.grounded_lines,
            dangling_lines=report.dangling_lines,
            insufficient_lines=report.insufficient_lines,
            dangling_chunk_ids=report.dangling_chunk_ids,
            attempts=attempts,
            prompt_version=RAG_ESTIMATION_PROMPT_VERSION,
        )
        return RagEstimationResult(estimate, report, chunks, RAG_ESTIMATION_PROMPT_VERSION)


def _insufficient() -> RagEstimate:
    return RagEstimate(
        lines=[],
        assumptions=[],
        confidence="insufficient",
        insufficient_context_explanation=NO_BUDGET_MATCHED,
    )


def _invalid_feedback(error: RagOutputInvalidError) -> str:
    return f"it broke a validation rule: {error}"


def _dangling_feedback(report: CitationReport) -> str:
    ids = ", ".join(report.dangling_chunk_ids)
    return (
        f"it cited source ids that are not in the context: {ids}. "
        "Cite only ids present in the context, copied exactly, "
        "or mark the line as not grounded."
    )
