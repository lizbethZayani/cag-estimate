"""Offline tests for the stress-test metrics."""

import pytest

from cag_estimate.evals.metrics import MetricResult
from cag_estimate.evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)
from cag_estimate.schemas.session import TurnObservation


def _observation(latency_ms: float = 1000.0, cost_usd: float = 0.01) -> TurnObservation:
    return TurnObservation(
        turn_index=1,
        session_id="s1",
        enriched_transcript_chars=10,
        attachments_total_chars=0,
        messages_in_window=2,
        anchors_count=0,
        summary_chars=0,
        tokens_in=1,
        tokens_out=1,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cache_hit_kind="none",
        last_resolved_tier="default",
    )


def _snapshot(summary: str = "", anchors: list[str] | None = None, **metadata: object) -> dict:
    return {"summary": summary, "anchors": anchors or [], "project_metadata": metadata}


def test_latency_within_budget_passes() -> None:
    result = LatencyBudgetMetric(8000).evaluate(_observation(latency_ms=7999))

    assert isinstance(result, MetricResult)
    assert result.name == "latency_budget"
    assert result.passed and result.score == 1.0


def test_latency_over_budget_fails_with_a_proportional_score() -> None:
    result = LatencyBudgetMetric(1000).evaluate(_observation(latency_ms=4000))

    assert not result.passed
    assert result.score == pytest.approx(0.25)
    assert "4000" in result.details


def test_latency_exactly_at_the_budget_passes() -> None:
    assert LatencyBudgetMetric(1000).evaluate(_observation(latency_ms=1000)).passed


def test_cost_within_budget_passes_and_over_budget_fails() -> None:
    metric = CostBudgetMetric(0.02)

    assert metric.evaluate(_observation(cost_usd=0.02)).passed
    failed = metric.evaluate(_observation(cost_usd=0.04))
    assert not failed.passed
    assert failed.score == pytest.approx(0.5)
    assert failed.name == "cost_budget"


def test_zero_cost_passes() -> None:
    assert CostBudgetMetric(0.02).evaluate(_observation(cost_usd=0.0)).score == 1.0


def test_budgets_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LatencyBudgetMetric(0)
    with pytest.raises(ValueError):
        CostBudgetMetric(-1)


def test_fact_in_the_summary_passes() -> None:
    result = MemoryDriftMetric("Nimbus").evaluate(_snapshot(summary="- the Nimbus portal"))

    assert result.passed
    assert result.name == "memory_drift"
    assert "summary" in result.details


def test_fact_in_an_anchor_passes_case_insensitively() -> None:
    assert MemoryDriftMetric("nimbus").evaluate(_snapshot(anchors=["NIMBUS is under NDA"])).passed


def test_fact_in_metadata_passes() -> None:
    result = MemoryDriftMetric("Nimbus").evaluate(_snapshot(project_name="Nimbus"))

    assert result.passed


def test_missing_fact_fails_with_zero_score() -> None:
    result = MemoryDriftMetric("Nimbus").evaluate(_snapshot(summary="- other", anchors=["x"]))

    assert not result.passed
    assert result.score == 0.0


def test_empty_summary_and_anchors_fail_without_error() -> None:
    assert not MemoryDriftMetric("Nimbus").evaluate(_snapshot()).passed


def test_where_subset_ignores_the_other_locations() -> None:
    snapshot = _snapshot(summary="Nimbus is here", anchors=[])

    assert MemoryDriftMetric("Nimbus", where=["summary"]).evaluate(snapshot).passed
    assert not MemoryDriftMetric("Nimbus", where=["anchors", "metadata"]).evaluate(snapshot).passed


def test_score_is_the_fraction_of_locations_holding_the_fact() -> None:
    snapshot = _snapshot(summary="Nimbus", anchors=["Nimbus"], project_name="other")

    assert MemoryDriftMetric("Nimbus").evaluate(snapshot).score == pytest.approx(2 / 3)


def test_non_ascii_facts_match_without_escaping() -> None:
    snapshot = _snapshot(agreed_scope="Presupuesto máximo: 30 000 € — aplicación móvil")

    assert MemoryDriftMetric("aplicación móvil", where=["metadata"]).evaluate(snapshot).passed
    assert MemoryDriftMetric("€", where=["metadata"]).evaluate(snapshot).passed


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        MemoryDriftMetric("")
    with pytest.raises(ValueError):
        MemoryDriftMetric("Nimbus", where=["nowhere"])
    with pytest.raises(ValueError):
        MemoryDriftMetric("Nimbus", where=[])
