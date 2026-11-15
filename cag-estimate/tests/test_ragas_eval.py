"""Tests for the RAGAS evaluation helpers: answer rendering, tables and the scoring port."""

import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import pytest
from structlog.testing import capture_logs

from cag_estimate.evals.ragas_eval import (
    METRICS,
    QueryScores,
    RagasSample,
    RagasScorer,
    average_scores,
    build_ragas_scorer,
    render_answer_text,
    render_ragas_table,
)
from cag_estimate.schemas.rag_estimation import EstimateLineItem, RagEstimate, SourceReference


def _source(chunk_id: str, evidence: str) -> SourceReference:
    return SourceReference(
        chunk_id=chunk_id, document_id=chunk_id.split("::")[0], evidence=evidence
    )


def _grounded(component: str, hours: float, rationale: str, *sources: SourceReference):
    return EstimateLineItem(
        component=component, hours=hours, rationale=rationale, grounded=True, sources=list(sources)
    )


def _ungrounded(component: str, rationale: str) -> EstimateLineItem:
    return EstimateLineItem(
        component=component, hours=None, rationale=rationale, grounded=False, sources=[]
    )


def _estimate(*lines: EstimateLineItem, assumptions: list[str] | None = None) -> RagEstimate:
    return RagEstimate(
        lines=list(lines),
        assumptions=assumptions or [],
        confidence="medium",
        insufficient_context_explanation=None,
    )


def test_render_grounded_line_uses_hours_as_the_only_unit_and_quotes_every_evidence() -> None:
    estimate = _estimate(
        _grounded(
            "Cart",
            120.0,
            "Similar to a past cart.",
            _source("BUD-1::CART", "Cart took 100 hours"),
            _source("BUD-2::CART", "Cart took 140 hours"),
        ),
        assumptions=["Stripe is used"],
    )

    text = render_answer_text(estimate)

    assert (
        "- Cart: 120 hours. Similar to a past cart. "
        "Evidence: Cart took 100 hours Cart took 140 hours" in text
    )
    assert "Total: 120 hours." in text
    assert "Assumptions:\n- Stripe is used" in text
    assert text.rstrip().endswith("Confidence: medium")
    assert "days" not in text.lower() and "weeks" not in text.lower()


def test_render_ungrounded_line_says_no_data() -> None:
    estimate = _estimate(
        _grounded("Cart", 10.0, "r", _source("BUD-1::CART", "ev")),
        _ungrounded("Chat", "No chat in the history."),
    )

    assert "- Chat: no data. No chat in the history." in render_answer_text(estimate)


def test_render_insufficient_estimate_is_one_line_with_the_explanation() -> None:
    estimate = RagEstimate(
        lines=[],
        assumptions=[],
        confidence="insufficient",
        insufficient_context_explanation="No budget matched.",
    )

    text = render_answer_text(estimate)

    assert "\n" not in text.strip()
    assert "No budget matched." in text
    assert "Total:" not in text


def test_render_never_truncates_long_text() -> None:
    long_evidence = "e" * 5000
    long_rationale = "r" * 5000
    estimate = _estimate(
        _grounded("Cart", 1.5, long_rationale, _source("BUD-1::CART", long_evidence))
    )

    text = render_answer_text(estimate)

    assert long_evidence in text and long_rationale in text
    assert "- Cart: 1.5 hours." in text


def test_render_omits_total_when_no_line_is_grounded() -> None:
    estimate = RagEstimate(
        lines=[_ungrounded("Chat", "missing")],
        assumptions=[],
        confidence="insufficient",
        insufficient_context_explanation="nothing relevant",
    )

    assert "Total:" not in render_answer_text(estimate)


def test_average_ignores_none_and_is_none_without_values() -> None:
    rows = [
        QueryScores(query_id="q1", scores={"faithfulness": 1.0, "answer_relevancy": None}),
        QueryScores(query_id="q2", scores={"faithfulness": 0.5, "answer_relevancy": None}),
    ]

    averages = average_scores(rows)

    assert averages["faithfulness"] == pytest.approx(0.75)
    assert averages["answer_relevancy"] is None
    assert set(averages) == set(METRICS)


def test_table_has_header_one_row_per_query_and_an_average_row() -> None:
    rows = [QueryScores(query_id=f"q{i}", scores={m: 0.5 for m in METRICS}) for i in range(1, 6)]
    rows[1] = QueryScores(query_id="q2", scores={m: None for m in METRICS})

    lines = render_ragas_table(rows).splitlines()

    assert lines[0] == (
        "| Query | Faithfulness | Answer relevancy | Context precision | Context recall |"
    )
    body = lines[2:]
    assert len(body) == 6
    assert body[0] == "| q1 | 0.50 | 0.50 | 0.50 | 0.50 |"
    assert body[1] == "| q2 | n/a | n/a | n/a | n/a |"
    assert body[-1] == "| **Average** | 0.50 | 0.50 | 0.50 | 0.50 |"


def test_ragas_module_does_not_import_ragas() -> None:
    code = "import sys, cag_estimate.evals.ragas_eval; sys.exit(int('ragas' in sys.modules))"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)

    assert result.returncode == 0, result.stderr.decode()


@dataclass
class _Result:
    value: float


class FakeMetric:
    def __init__(self, value: float = 0.8, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def ascore(self, **kwargs: Any) -> _Result:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _Result(self.value)


SAMPLE = RagasSample(
    query_id="q1",
    question="Build a shop",
    answer="- Cart: 10 hours.",
    contexts=["ctx one", "ctx two"],
    ground_truth="reference text",
)


def _metrics(**overrides: FakeMetric) -> dict[str, FakeMetric]:
    metrics = {name: FakeMetric() for name in METRICS}
    metrics.update(overrides)
    return metrics


async def test_scorer_calls_each_metric_with_its_own_keyword_arguments() -> None:
    metrics = _metrics()

    scores = await RagasScorer(metrics).score(SAMPLE)

    assert scores == {name: 0.8 for name in METRICS}
    assert metrics["faithfulness"].calls == [
        {
            "user_input": "Build a shop",
            "response": "- Cart: 10 hours.",
            "retrieved_contexts": ["ctx one", "ctx two"],
        }
    ]
    assert metrics["answer_relevancy"].calls == [
        {"user_input": "Build a shop", "response": "- Cart: 10 hours."}
    ]
    assert metrics["context_precision"].calls == [
        {
            "user_input": "Build a shop",
            "reference": "reference text",
            "retrieved_contexts": ["ctx one", "ctx two"],
        }
    ]
    assert metrics["context_recall"].calls == [
        {
            "user_input": "Build a shop",
            "retrieved_contexts": ["ctx one", "ctx two"],
            "reference": "reference text",
        }
    ]


async def test_one_failing_metric_yields_none_only_for_that_metric() -> None:
    failing = FakeMetric(error=ValueError("secret sample text: Build a shop"))
    metrics = _metrics(context_recall=failing)

    with capture_logs() as logs:
        scores = await RagasScorer(metrics).score(SAMPLE)

    assert scores["context_recall"] is None
    assert [scores[m] for m in METRICS if m != "context_recall"] == [0.8, 0.8, 0.8]
    warnings = [e for e in logs if e["log_level"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["metric"] == "context_recall"
    assert warnings[0]["error_type"] == "ValueError"
    assert "Build a shop" not in repr(warnings)


def test_build_ragas_scorer_constructs_without_network() -> None:
    pytest.importorskip("ragas")

    scorer = build_ragas_scorer("sk-test", "gpt-4o-mini", "text-embedding-3-small")

    assert isinstance(scorer, RagasScorer)
