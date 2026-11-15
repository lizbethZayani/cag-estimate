"""RAGAS evaluation of cited estimates: sample rendering, scoring port and markdown table.

``ragas`` is an optional dependency (the ``evals`` group) and is imported lazily inside
``build_ragas_scorer`` so importing this module never needs it. Scoring uses the modern
``ragas.metrics.collections`` interface. Logs carry the metric name and the exception class
only, never the sample text.
"""

from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any, Protocol

import structlog
from pydantic import BaseModel

from cag_estimate.schemas.rag_estimation import EstimateLineItem, RagEstimate

log = structlog.get_logger()

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_TITLES = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}
# Sample field behind each ragas keyword argument.
_ARGUMENT_FIELDS = {
    "user_input": "question",
    "response": "answer",
    "retrieved_contexts": "contexts",
    "reference": "ground_truth",
}
_METRIC_ARGUMENTS = {
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "context_precision": ("user_input", "reference", "retrieved_contexts"),
    "context_recall": ("user_input", "retrieved_contexts", "reference"),
}


class RagasSample(BaseModel):
    """One question with the answer to judge, the contexts it was built on and the reference."""

    query_id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


class QueryScores(BaseModel):
    """The metric scores of one query; ``None`` when a metric could not be computed."""

    query_id: str
    scores: dict[str, float | None]


class SampleScorer(Protocol):
    async def score(self, sample: RagasSample) -> dict[str, float | None]: ...


def _hours(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


def _render_line(line: EstimateLineItem) -> str:
    if not line.grounded or line.hours is None:
        return f"- {line.component}: no data. {line.rationale}"
    evidence = " ".join(source.evidence for source in line.sources)
    return f"- {line.component}: {_hours(line.hours)} hours. {line.rationale} Evidence: {evidence}"


def render_answer_text(estimate: RagEstimate) -> str:
    """Plain text of an estimate for the judge: hours only, nothing truncated."""
    if estimate.confidence == "insufficient":
        return f"Insufficient context: {estimate.insufficient_context_explanation}"
    parts = [_render_line(line) for line in estimate.lines]
    if estimate.total_hours is not None:
        parts.append(f"Total: {_hours(estimate.total_hours)} hours.")
    if estimate.assumptions:
        parts.append("Assumptions:")
        parts.extend(f"- {assumption}" for assumption in estimate.assumptions)
    parts.append(f"Confidence: {estimate.confidence}")
    return "\n".join(parts)


def average_scores(rows: Sequence[QueryScores]) -> dict[str, float | None]:
    """Mean of every metric over the queries that have a value for it."""
    averages: dict[str, float | None] = {}
    for metric in METRICS:
        values = [row.scores[metric] for row in rows if row.scores.get(metric) is not None]
        averages[metric] = fmean(values) if values else None
    return averages


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _row(label: str, scores: Mapping[str, float | None]) -> str:
    return f"| {label} | " + " | ".join(_cell(scores.get(metric)) for metric in METRICS) + " |"


def render_ragas_table(rows: Sequence[QueryScores]) -> str:
    """Markdown table: one row per query and a final average row."""
    header = "| Query | " + " | ".join(_TITLES[metric] for metric in METRICS) + " |"
    separator = "|---" * (len(METRICS) + 1) + "|"
    body = [_row(row.query_id, row.scores) for row in rows]
    return "\n".join([header, separator, *body, _row("**Average**", average_scores(rows))])


class RagasScorer:
    """Scores a sample with the four ragas metrics; one failing metric does not lose the rest."""

    def __init__(self, metrics: Mapping[str, Any]) -> None:
        self._metrics = metrics

    async def score(self, sample: RagasSample) -> dict[str, float | None]:
        return {name: await self._score_metric(name, sample) for name in METRICS}

    async def _score_metric(self, name: str, sample: RagasSample) -> float | None:
        arguments = {
            argument: getattr(sample, _ARGUMENT_FIELDS[argument])
            for argument in _METRIC_ARGUMENTS[name]
        }
        try:
            result = await self._metrics[name].ascore(**arguments)
            return float(result.value)
        except Exception as exc:  # noqa: BLE001 - a metric failure must not lose the others
            log.warning("ragas_metric_failed", metric=name, error_type=type(exc).__name__)
            return None


def build_ragas_scorer(api_key: str, judge_model: str, embedding_model: str) -> RagasScorer:
    """Real scorer backed by OpenAI clients (constructing them does not call the network)."""
    from openai import AsyncOpenAI, OpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    llm = llm_factory(judge_model, client=AsyncOpenAI(api_key=api_key))
    embeddings = OpenAIEmbeddings(client=OpenAI(api_key=api_key), model=embedding_model)
    return RagasScorer(
        {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }
    )
