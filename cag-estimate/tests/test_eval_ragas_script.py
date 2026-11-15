"""Tests for scripts/eval_ragas.py, loaded as a module (scripts/ is not a package)."""

import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import structlog

from cag_estimate.evals.ragas_eval import METRICS, RagasSample
from cag_estimate.evals.retrieval_eval import GOLDEN_SET_PATH, load_golden_set
from cag_estimate.schemas.rag_estimation import EstimateLineItem, RagEstimate, SourceReference
from cag_estimate.services.citations import verify_citations
from cag_estimate.services.rag_context import RetrievedChunk
from cag_estimate.services.rag_errors import (
    DanglingCitationError,
    RagGenerationError,
    RagOutputInvalidError,
)
from cag_estimate.services.rag_estimation import RagEstimationResult

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ragas.py"
QUERIES = load_golden_set(GOLDEN_SET_PATH).queries
CHUNK = RetrievedChunk(chunk_id="BUD-1::CART", document_id="BUD-1", content="Cart took 10 hours")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_ragas_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


def _estimate(chunk_id: str = "BUD-1::CART") -> RagEstimate:
    line = EstimateLineItem(
        component="Cart",
        hours=10.0,
        rationale="Seen before.",
        grounded=True,
        sources=[
            SourceReference(chunk_id=chunk_id, document_id="BUD-1", evidence="Cart took 10 hours")
        ],
    )
    return RagEstimate(
        lines=[line], assumptions=[], confidence="high", insufficient_context_explanation=None
    )


def _result() -> RagEstimationResult:
    estimate = _estimate()
    return RagEstimationResult(
        estimate=estimate,
        citation_report=verify_citations(estimate, {CHUNK.chunk_id}),
        chunks=[CHUNK],
        prompt_version="v1",
    )


class FakeService:
    def __init__(self, failures: dict[str, Exception] | None = None) -> None:
        self.failures = failures or {}
        self.queries: list[str] = []
        self.bound_ids: list[Any] = []

    async def estimate(self, query: str, request_id: str | None = None) -> RagEstimationResult:
        self.queries.append(query)
        self.bound_ids.append(structlog.contextvars.get_contextvars().get("request_id"))
        if query in self.failures:
            raise self.failures[query]
        return _result()


class FakeScorer:
    def __init__(self) -> None:
        self.samples: list[RagasSample] = []

    async def score(self, sample: RagasSample) -> dict[str, float | None]:
        self.samples.append(sample)
        return {name: 0.5 for name in METRICS}


class Harness:
    def __init__(self, tmp_path: Path, service: Any = None) -> None:
        self.service = service if service is not None else FakeService()
        self.scorer = FakeScorer()
        self.factory_calls: list[tuple[str, str]] = []
        self.samples_path = tmp_path / "samples.json"
        self.out_path = tmp_path / "raw.json"
        self.out = io.StringIO()
        self.err = io.StringIO()

    def scorer_factory(self, judge_model: str, embedding_model: str) -> FakeScorer:
        self.factory_calls.append((judge_model, embedding_model))
        return self.scorer

    def run(self, script: ModuleType, *argv: str, service: Any = "default") -> int:
        chosen = self.service if service == "default" else service
        return script.main(
            [*argv, "--samples", str(self.samples_path), "--out", str(self.out_path)],
            service_factory=lambda: chosen,
            scorer_factory=self.scorer_factory,
            stdout=self.out,
            stderr=self.err,
        )


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def test_happy_path_prints_both_tables_and_writes_both_files(
    script: ModuleType, harness: Harness
) -> None:
    code = harness.run(script)

    out = harness.out.getvalue()
    assert code == 0
    assert "| Query | Faithfulness | Answer relevancy | Context precision | Context recall |" in out
    assert "| **Average** | 0.50 | 0.50 | 0.50 | 0.50 |" in out
    assert "| Query | Status | Lines | Grounded | Dangling | Insufficient |" in out
    assert str(harness.samples_path) in out and str(harness.out_path) in out
    assert len(harness.service.queries) == len(QUERIES) == 5
    samples = json.loads(harness.samples_path.read_text())
    assert len(samples["records"]) == 5
    first = samples["records"][0]
    assert first["status"] == "ok"
    assert first["citation_report"]["grounded_lines"] == 1
    assert first["sample"]["contexts"] == [CHUNK.content]
    assert first["sample"]["ground_truth"] == QUERIES[0].ground_truth
    assert "- Cart: 10 hours." in first["sample"]["answer"]
    raw = json.loads(harness.out_path.read_text())
    assert raw["judge_model"] == "gpt-4o-mini"
    assert raw["embedding_model"] == "text-embedding-3-small"
    assert raw["averages"]["faithfulness"] == pytest.approx(0.5)
    assert raw["queries"][0]["scores"]["context_recall"] == pytest.approx(0.5)
    assert raw["queries"][0]["citation_report"]["total_lines"] == 1


def test_limit_restricts_the_queries(script: ModuleType, harness: Harness) -> None:
    code = harness.run(script, "--limit", "2")

    assert code == 0
    assert len(harness.service.queries) == 2
    assert len(json.loads(harness.samples_path.read_text())["records"]) == 2


def test_models_are_passed_to_the_scorer_factory(script: ModuleType, harness: Harness) -> None:
    harness.run(script, "--judge-model", "judge-x", "--embedding-model", "emb-y", "--limit", "1")

    assert harness.factory_calls == [("judge-x", "emb-y")]
    assert json.loads(harness.out_path.read_text())["judge_model"] == "judge-x"


def test_request_id_is_bound_while_the_service_runs_and_cleared_after(
    script: ModuleType, harness: Harness
) -> None:
    structlog.contextvars.clear_contextvars()

    harness.run(script, "--limit", "2")

    assert harness.service.bound_ids == [f"eval-{q.id}" for q in QUERIES[:2]]
    assert "request_id" not in structlog.contextvars.get_contextvars()


def test_score_only_reuses_the_samples_and_never_calls_the_service(
    script: ModuleType, harness: Harness
) -> None:
    harness.run(script)
    first_calls = len(harness.service.queries)
    harness.scorer.samples.clear()
    harness.out_path.unlink()

    code = harness.run(script, "--score-only", service=None)

    assert code == 0
    assert len(harness.service.queries) == first_calls
    assert len(harness.scorer.samples) == 5
    assert json.loads(harness.out_path.read_text())["averages"]["faithfulness"] == 0.5


def test_score_only_with_a_missing_samples_file_exits_2(
    script: ModuleType, harness: Harness
) -> None:
    code = harness.run(script, "--score-only")

    assert code == 2
    assert "error" in harness.err.getvalue()


def test_a_rejected_query_gets_na_scores_and_exit_code_1(
    script: ModuleType, harness: Harness
) -> None:
    dangling_estimate = _estimate("BUD-9::FAKE")
    report = verify_citations(dangling_estimate, {CHUNK.chunk_id})
    harness.service = FakeService({QUERIES[1].query: DanglingCitationError(report)})

    code = harness.run(script)

    out = harness.out.getvalue()
    assert code == 1
    assert f"| {QUERIES[1].id} | n/a | n/a | n/a | n/a |" in out
    assert len(harness.scorer.samples) == 4
    records = json.loads(harness.samples_path.read_text())["records"]
    rejected = records[1]
    assert rejected["status"] == "rejected"
    assert rejected["error"] == "DanglingCitationError"
    assert rejected["dangling_chunk_ids"] == ["BUD-9::FAKE"]
    assert rejected["sample"] is None
    raw = json.loads(harness.out_path.read_text())
    assert raw["queries"][1]["status"] == "rejected"
    assert raw["queries"][1]["scores"] == {name: None for name in METRICS}


@pytest.mark.parametrize("error", [RagOutputInvalidError("rule"), RagGenerationError("boom")])
def test_other_service_errors_are_also_rejections(
    script: ModuleType, harness: Harness, error: Exception
) -> None:
    harness.service = FakeService({q.query: error for q in QUERIES})

    code = harness.run(script)

    records = json.loads(harness.samples_path.read_text())["records"]
    assert code == 1
    assert {r["error"] for r in records} == {type(error).__name__}
    assert all(r["citation_report"] is None for r in records)


def test_missing_ground_truth_exits_2(script: ModuleType, harness: Harness, tmp_path: Path) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "version": 1,
                "queries": [
                    {
                        "id": "q1",
                        "query": "x",
                        "relevant_budget_ids": ["BUD-2024-001"],
                        "rationale": "r",
                    }
                ],
            }
        )
    )

    code = harness.run(script, "--golden", str(golden))

    assert code == 2
    assert "ground_truth" in harness.err.getvalue()
    assert harness.service.queries == []


def test_unreadable_golden_set_exits_2(
    script: ModuleType, harness: Harness, tmp_path: Path
) -> None:
    code = harness.run(script, "--golden", str(tmp_path / "nope.json"))

    assert code == 2
    assert "error" in harness.err.getvalue()


@pytest.mark.parametrize("flag_value", ["0", "-1"])
def test_non_positive_limit_exits_2(script: ModuleType, harness: Harness, flag_value: str) -> None:
    assert harness.run(script, "--limit", flag_value) == 2


def test_unconfigured_service_exits_2_with_the_guidance(
    script: ModuleType, harness: Harness
) -> None:
    code = harness.run(script, service=None)

    err = harness.err.getvalue()
    assert code == 2
    assert "DATABASE_URL=postgresql+asyncpg://estimator:estimator@localhost:5433/estimator" in err
    assert "OPENAI_API_KEY" in err
    assert "PYTHONPATH=src uv run --group evals python scripts/eval_ragas.py" in err
    assert harness.scorer.samples == []


def test_help_works_without_a_server(script: ModuleType) -> None:
    with pytest.raises(SystemExit) as exit_info:
        script.main(["--help"])

    assert exit_info.value.code == 0
