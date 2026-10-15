"""Metrics of the memory stress test.

These live here and not in ``evals/metrics.py`` because they consume a different
shape: the per-turn ``TurnObservation`` (latency, cost) and the session snapshot
(summary, anchors, metadata) of a multi-turn run, while the golden-set metrics
judge a final estimate from a ``CaseOutcome``. They still return the shared
``MetricResult`` so reports can treat both families alike.

All checks are deterministic: numeric comparisons and case-insensitive substring
matching, no LLM and no randomness.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from cag_estimate.evals.metrics import MetricResult
from cag_estimate.schemas.session import TurnObservation

MEMORY_LOCATIONS = ("summary", "anchors", "metadata")


def _budget_result(name: str, unit: str, value: float, budget: float) -> MetricResult:
    passed = value <= budget
    score = 1.0 if passed else budget / value
    details = f"{value:g} {unit} {'<=' if passed else '>'} budget {budget:g} {unit}"
    return MetricResult(name, passed, score, details)


class LatencyBudgetMetric:
    """Server-side LLM latency of one turn must stay within ``budget_ms``."""

    def __init__(self, budget_ms: float) -> None:
        if budget_ms <= 0:
            raise ValueError("budget_ms must be positive")
        self.budget_ms = budget_ms

    def evaluate(self, observation: TurnObservation) -> MetricResult:
        return _budget_result("latency_budget", "ms", observation.latency_ms, self.budget_ms)


class CostBudgetMetric:
    """LLM cost of one turn must stay within ``budget_usd``."""

    def __init__(self, budget_usd: float) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        self.budget_usd = budget_usd

    def evaluate(self, observation: TurnObservation) -> MetricResult:
        return _budget_result("cost_budget", "USD", observation.cost_usd, self.budget_usd)


class MemoryDriftMetric:
    """A fact told early must still appear somewhere in the session memory.

    ``where`` selects the locations searched (summary, anchors, metadata). Passes
    when at least one holds the fact; the score is the fraction that do.
    """

    def __init__(self, fact: str, where: Sequence[str] = MEMORY_LOCATIONS) -> None:
        if not fact.strip():
            raise ValueError("fact must not be empty")
        locations = tuple(where)
        if not locations or not set(locations) <= set(MEMORY_LOCATIONS):
            raise ValueError(f"where must be a non-empty subset of {MEMORY_LOCATIONS}")
        self.fact = fact
        self.where = locations

    def evaluate(self, snapshot: Mapping[str, Any]) -> MetricResult:
        needle = self.fact.casefold()
        found = [loc for loc in self.where if needle in self._text(snapshot, loc).casefold()]
        score = len(found) / len(self.where)
        where = ", ".join(found) if found else "nowhere"
        return MetricResult("memory_drift", bool(found), score, f"{self.fact!r} found in {where}")

    @staticmethod
    def _text(snapshot: Mapping[str, Any], location: str) -> str:
        if location == "summary":
            return str(snapshot.get("summary") or "")
        if location == "anchors":
            return "\n".join(str(anchor) for anchor in snapshot.get("anchors") or [])
        return json.dumps(snapshot.get("project_metadata") or {}, ensure_ascii=False)
