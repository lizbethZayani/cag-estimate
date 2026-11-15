#!/usr/bin/env python3
"""Score cited RAG estimates with RAGAS (faithfulness, relevancy, precision, recall).

Two phases, so scoring can be repeated without paying for generation again:

1. Generation: every golden query runs through the real ``RagEstimationService`` (hybrid
   retrieval, cited generation, citation verification) in this process. The answers, their
   retrieved contexts, the ground truths and the citation reports are saved to ``--samples``.
2. Scoring: RAGAS judges every sample with ``--judge-model`` and ``--embedding-model`` and the
   raw scores are saved to ``--out``. ``--score-only`` skips phase 1 and reloads ``--samples``.

It needs the pgvector container that already holds the ingested budgets (no docker command is
needed when it is running) and these variables, for example in ``.env``:

    DATABASE_URL=postgresql+asyncpg://estimator:estimator@localhost:5433/estimator
    OPENAI_API_KEY=...

    PYTHONPATH=src uv run --group evals python scripts/eval_ragas.py

A query the service rejects (dangling citations, invalid output, generation failure) is recorded
with its error and scored ``n/a``; the run continues.

Exit codes: 0 success, 1 at least one query was rejected, 2 bad usage, input or configuration.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, Protocol, TextIO

from pydantic import BaseModel, ValidationError
from structlog.contextvars import bind_contextvars, unbind_contextvars

from cag_estimate.config import get_settings
from cag_estimate.evals.ragas_eval import (
    METRICS,
    QueryScores,
    RagasSample,
    SampleScorer,
    average_scores,
    build_ragas_scorer,
    render_answer_text,
    render_ragas_table,
)
from cag_estimate.evals.retrieval_eval import (
    GOLDEN_SET_PATH,
    GoldenQuery,
    load_golden_set,
    require_ground_truth,
)
from cag_estimate.services.citations import CitationReport
from cag_estimate.services.rag_errors import (
    DanglingCitationError,
    RagGenerationError,
    RagOutputInvalidError,
)
from cag_estimate.services.rag_estimation import RagEstimationResult

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_USAGE = 2

DEFAULT_SAMPLES = "session11/ragas_samples.json"
DEFAULT_OUT = "session11/ragas_raw.json"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

NOT_CONFIGURED = """the RAG estimation service is not configured.
Set these variables (for example in .env) and keep the pgvector container running:
  DATABASE_URL=postgresql+asyncpg://estimator:estimator@localhost:5433/estimator
  OPENAI_API_KEY=<your key>
Then run:
  PYTHONPATH=src uv run --group evals python scripts/eval_ragas.py"""


class InputError(Exception):
    """An option, the golden set, the samples file or the configuration is unusable."""


class RagEstimator(Protocol):
    async def estimate(self, query: str) -> RagEstimationResult: ...


class QueryRecord(BaseModel):
    """What generation produced for one query; the unit saved in the samples file."""

    query_id: str
    status: Literal["ok", "rejected"]
    error: str | None = None
    dangling_chunk_ids: list[str] = []
    citation_report: CitationReport | None = None
    sample: RagasSample | None = None


class SamplesFile(BaseModel):
    records: list[QueryRecord]


ServiceFactory = Callable[[], RagEstimator | None]
ScorerFactory = Callable[[str, str], SampleScorer]


def default_service_factory() -> RagEstimator | None:
    from cag_estimate.dependencies import get_rag_estimation_service

    return get_rag_estimation_service()


def default_scorer_factory(judge_model: str, embedding_model: str) -> SampleScorer:
    api_key = get_settings().openai_api_key
    if not api_key:
        raise InputError("OPENAI_API_KEY is not set; RAGAS needs it for the judge model")
    return build_ragas_scorer(api_key, judge_model, embedding_model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score cited RAG estimates with RAGAS.")
    parser.add_argument("--golden", default=str(GOLDEN_SET_PATH), help="golden set JSON")
    parser.add_argument("--limit", type=int, default=None, help="only the first N queries")
    parser.add_argument("--samples", default=DEFAULT_SAMPLES, help="samples JSON (read or written)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="raw scores JSON")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="RAGAS judge model")
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="RAGAS embeddings"
    )
    parser.add_argument(
        "--score-only", action="store_true", help="score the saved --samples without generating"
    )
    return parser


def read_queries(path: str, limit: int | None) -> list[GoldenQuery]:
    try:
        queries = require_ground_truth(load_golden_set(Path(path)))
    except (OSError, ValueError, ValidationError) as exc:
        raise InputError(f"cannot use the golden set {path}: {exc}") from exc
    return queries[:limit]


def read_records(path: str, limit: int | None) -> list[QueryRecord]:
    try:
        samples = SamplesFile.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read the samples file {path}: {type(exc).__name__}") from exc
    return samples.records[:limit]


def _rejected(query: GoldenQuery, exc: Exception) -> QueryRecord:
    report = exc.report if isinstance(exc, DanglingCitationError) else None
    return QueryRecord(
        query_id=query.id,
        status="rejected",
        error=type(exc).__name__,
        dangling_chunk_ids=report.dangling_chunk_ids if report else [],
        citation_report=report,
    )


def _accepted(query: GoldenQuery, result: RagEstimationResult) -> QueryRecord:
    assert query.ground_truth is not None  # guaranteed by require_ground_truth
    sample = RagasSample(
        query_id=query.id,
        question=query.query,
        answer=render_answer_text(result.estimate),
        contexts=[chunk.content for chunk in result.chunks],
        ground_truth=query.ground_truth,
    )
    return QueryRecord(
        query_id=query.id, status="ok", citation_report=result.citation_report, sample=sample
    )


async def generate_record(service: RagEstimator, query: GoldenQuery) -> QueryRecord:
    bind_contextvars(request_id=f"eval-{query.id}")
    try:
        return _accepted(query, await service.estimate(query.query))
    except (DanglingCitationError, RagOutputInvalidError, RagGenerationError) as exc:
        return _rejected(query, exc)
    finally:
        unbind_contextvars("request_id")


async def generate_records(
    service: RagEstimator, queries: Sequence[GoldenQuery]
) -> list[QueryRecord]:
    return [await generate_record(service, query) for query in queries]


async def score_records(scorer: SampleScorer, records: Sequence[QueryRecord]) -> list[QueryScores]:
    rows = []
    for record in records:
        scores = await scorer.score(record.sample) if record.sample else {}
        rows.append(
            QueryScores(query_id=record.query_id, scores={m: scores.get(m) for m in METRICS})
        )
    return rows


def render_citation_table(records: Sequence[QueryRecord]) -> str:
    header = "| Query | Status | Lines | Grounded | Dangling | Insufficient |"
    rows = [
        _citation_row(record.query_id, record.status, record.citation_report) for record in records
    ]
    reports = [r.citation_report for r in records if r.citation_report is not None]
    total = CitationReport(
        total_lines=sum(r.total_lines for r in reports),
        grounded_lines=sum(r.grounded_lines for r in reports),
        dangling_lines=sum(r.dangling_lines for r in reports),
        insufficient_lines=sum(r.insufficient_lines for r in reports),
        dangling_chunk_ids=[],
        lines=[],
    )
    return "\n".join(
        [header, "|---|---|---|---|---|---|", *rows, _citation_row("**Total**", "", total)]
    )


def _citation_row(label: str, status: str, report: CitationReport | None) -> str:
    counts = (
        ["-"] * 4
        if report is None
        else [
            str(report.total_lines),
            str(report.grounded_lines),
            str(report.dangling_lines),
            str(report.insufficient_lines),
        ]
    )
    return f"| {label} | {status} | " + " | ".join(counts) + " |"


def _write_json(path: str, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_samples(path: str, records: Sequence[QueryRecord]) -> None:
    _write_json(path, SamplesFile(records=list(records)).model_dump(mode="json"))


def write_raw(
    path: str,
    records: Sequence[QueryRecord],
    rows: Sequence[QueryScores],
    judge_model: str,
    embedding_model: str,
) -> None:
    queries = [
        {
            "query_id": record.query_id,
            "status": record.status,
            "error": record.error,
            "scores": row.scores,
            "citation_report": record.citation_report.model_dump(mode="json")
            if record.citation_report
            else None,
        }
        for record, row in zip(records, rows, strict=True)
    ]
    _write_json(
        path,
        {
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "queries": queries,
            "averages": average_scores(rows),
        },
    )


def require_service(service_factory: ServiceFactory) -> RagEstimator:
    service = service_factory()
    if service is None:
        raise InputError(NOT_CONFIGURED)
    return service


def generate_phase(
    args: argparse.Namespace, service_factory: ServiceFactory
) -> Callable[[], list[QueryRecord]]:
    """Validate everything generation needs up front; return the paid step to run later."""
    queries = read_queries(args.golden, args.limit)
    service = require_service(service_factory)

    def generate() -> list[QueryRecord]:
        records = asyncio.run(generate_records(service, queries))
        write_samples(args.samples, records)
        return records

    return generate


def run(
    args: argparse.Namespace,
    service_factory: ServiceFactory,
    scorer_factory: ScorerFactory,
    out: TextIO,
) -> int:
    if args.limit is not None and args.limit < 1:
        raise InputError("--limit must be at least 1")
    obtain = (
        partial(read_records, args.samples, args.limit)
        if args.score_only
        else generate_phase(args, service_factory)
    )
    scorer = scorer_factory(args.judge_model, args.embedding_model)
    records = obtain()
    rows = asyncio.run(score_records(scorer, records))
    write_raw(args.out, records, rows, args.judge_model, args.embedding_model)
    out.write(render_ragas_table(rows) + "\n\n")
    out.write(render_citation_table(records) + "\n\n")
    out.write(f"samples: {args.samples}\nraw scores: {args.out}\n")
    return EXIT_REJECTED if any(r.status == "rejected" for r in records) else EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    service_factory: ServiceFactory = default_service_factory,
    scorer_factory: ScorerFactory = default_scorer_factory,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        return run(args, service_factory, scorer_factory, out)
    except InputError as exc:
        err.write(f"error: {exc}\n")
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
