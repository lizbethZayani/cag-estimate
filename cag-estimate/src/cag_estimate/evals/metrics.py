"""Deterministic, offline metrics over recorded eval outcomes.

Every metric is a pure function of a ``CaseOutcome``: no network, no LLM, no
randomness. They check STRUCTURE (totals add up, bounds hold, guardrails fire,
memory is consistent) and REGRESSIONS (loose ranges, expected vocabulary); they
do not judge whether an estimate is good.

A metric that does not apply to a case returns ``passed=True, score=1.0,
details="n/a"``. An outcome with a transport ``error`` fails every metric.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from cag_estimate.evals.dataset import GoldenCase
from cag_estimate.schemas.estimation import COMPLEXITY_HOURS, MAX_TASKS, MIN_TASKS

NOT_APPLICABLE = "n/a"
COST_TOLERANCE_USD = 1
MIN_ECHO_FRAGMENT_CHARS = 15
_SENTENCE_BREAKS = re.compile(r"[.!?\n]+")


@dataclass(frozen=True)
class TurnResult:
    """One HTTP exchange: status, decoded JSON body (``{}`` when not JSON) and latency."""

    status_code: int
    body: dict[str, Any]
    latency_ms: float


@dataclass(frozen=True)
class CaseOutcome:
    """What running a case produced; ``error`` is set when transport failed."""

    case: GoldenCase
    turns: list[TurnResult] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class MetricResult:
    name: str
    passed: bool
    score: float
    details: str


class Check(NamedTuple):
    """One assertion of a metric: a human label and whether it held."""

    label: str
    ok: bool


class Metric(ABC):
    """Template: subclasses declare when they apply and which checks they run."""

    name: str

    def evaluate(self, outcome: CaseOutcome) -> MetricResult:
        if outcome.error is not None:
            return MetricResult(self.name, False, 0.0, outcome.error)
        if not self.applies(outcome.case):
            return MetricResult(self.name, True, 1.0, NOT_APPLICABLE)
        incomplete = _incomplete_run(outcome)
        if incomplete:
            return MetricResult(self.name, False, 0.0, incomplete)
        try:
            checks = self.checks(outcome)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            detail = f"malformed response ({type(exc).__name__}: {exc})"
            return MetricResult(self.name, False, 0.0, detail)
        return _summarize(self.name, checks)

    @abstractmethod
    def applies(self, case: GoldenCase) -> bool: ...

    @abstractmethod
    def checks(self, outcome: CaseOutcome) -> list[Check]: ...


def _incomplete_run(outcome: CaseOutcome) -> str | None:
    expected, ran = len(outcome.case.turns), len(outcome.turns)
    if ran == expected:
        return None
    return f"run stopped after {ran} of {expected} turns"


def _summarize(name: str, checks: list[Check]) -> MetricResult:
    failed = [check.label for check in checks if not check.ok]
    score = (len(checks) - len(failed)) / len(checks) if checks else 1.0
    return MetricResult(name, not failed, score, "; ".join(failed) or "ok")


def _final(outcome: CaseOutcome) -> TurnResult:
    return outcome.turns[-1]


def _haystack(body: dict[str, Any]) -> str:
    """Project name, summary, task names and descriptions, lowercased."""
    result = body["result"]
    parts = [result["project_name"], result["meeting_summary"]]
    for task in result["tasks"]:
        parts.extend((task["name"], task["description"]))
    return " ".join(parts).casefold()


class FinalEstimateMetric(Metric):
    """A metric about the estimate of the final turn, which must be HTTP 200."""

    def applies(self, case: GoldenCase) -> bool:
        return case.expect_guardrail_reason is None and self.relevant(case)

    @abstractmethod
    def relevant(self, case: GoldenCase) -> bool: ...

    def checks(self, outcome: CaseOutcome) -> list[Check]:
        final = _final(outcome)
        if final.status_code != 200:
            return [Check(f"final turn returned HTTP {final.status_code}", False)]
        return self.body_checks(outcome.case, final.body)

    @abstractmethod
    def body_checks(self, case: GoldenCase, body: dict[str, Any]) -> list[Check]: ...


class SchemaConsistencyMetric(FinalEstimateMetric):
    """Totals add up, cost matches the rate, counts and complexity are valid."""

    name = "schema_consistency"

    def relevant(self, case: GoldenCase) -> bool:
        return True

    def body_checks(self, case: GoldenCase, body: dict[str, Any]) -> list[Check]:
        result = body["result"]
        tasks, summary = result["tasks"], result["summary"]
        hours = sum(task["estimated_hours"] for task in tasks)
        expected_cost = summary["total_hours"] * case.hourly_rate
        return [
            Check(
                f"summary.total_hours {summary['total_hours']} != sum of task hours {hours}",
                summary["total_hours"] == hours,
            ),
            Check(
                f"summary.total_cost_usd {summary['total_cost_usd']} != "
                f"total_hours x rate {expected_cost}",
                abs(summary["total_cost_usd"] - expected_cost) <= COST_TOLERANCE_USD,
            ),
            Check(
                f"{len(tasks)} tasks outside {MIN_TASKS}..{MAX_TASKS}",
                MIN_TASKS <= len(tasks) <= MAX_TASKS,
            ),
            Check(
                "a task has an unknown complexity",
                all(task["complexity"] in COMPLEXITY_HOURS for task in tasks),
            ),
            Check("prompt_version is missing", bool(body.get("prompt_version"))),
        ]


class RangeBoundsMetric(FinalEstimateMetric):
    """Task count and total hours stay inside the case's loose ranges."""

    name = "range_bounds"

    def relevant(self, case: GoldenCase) -> bool:
        return bool(case.expected_task_count_range or case.expected_hours_range)

    def body_checks(self, case: GoldenCase, body: dict[str, Any]) -> list[Check]:
        result = body["result"]
        checks: list[Check] = []
        if case.expected_task_count_range:
            low, high = case.expected_task_count_range
            count = len(result["tasks"])
            checks.append(Check(f"{count} tasks outside {low}..{high}", low <= count <= high))
        if case.expected_hours_range:
            low, high = case.expected_hours_range
            hours = result["summary"]["total_hours"]
            checks.append(Check(f"{hours} hours outside {low}..{high}", low <= hours <= high))
        return checks


class ContentRecallMetric(FinalEstimateMetric):
    """Expected terms appear in the estimate text; at least one expected technology does."""

    name = "content_recall"

    def relevant(self, case: GoldenCase) -> bool:
        return bool(case.expected_in_text or case.expected_technologies_any_of)

    def body_checks(self, case: GoldenCase, body: dict[str, Any]) -> list[Check]:
        haystack = _haystack(body)
        checks = [
            Check(f"missing term '{term}'", term.casefold() in haystack)
            for term in case.expected_in_text
        ]
        if case.expected_technologies_any_of:
            known = f"{haystack} {_mentioned_technologies(body)}"
            wanted = case.expected_technologies_any_of
            found = any(tech.casefold() in known for tech in wanted)
            checks.append(Check(f"none of the technologies found: {', '.join(wanted)}", found))
        return checks


def _mentioned_technologies(body: dict[str, Any]) -> str:
    return " ".join(body.get("project_metadata", {}).get("mentioned_technologies", [])).casefold()


class TierExpectationMetric(FinalEstimateMetric):
    """The final response reports the audience tier the case expects."""

    name = "tier_expectation"

    def relevant(self, case: GoldenCase) -> bool:
        return case.expected_resolved_tier is not None

    def body_checks(self, case: GoldenCase, body: dict[str, Any]) -> list[Check]:
        actual = body["resolved_tier"]
        return [
            Check(
                f"resolved_tier '{actual}' != '{case.expected_resolved_tier}'",
                actual == case.expected_resolved_tier,
            )
        ]


class GuardrailBehaviorMetric(Metric):
    """A guardrail case is rejected with its reason and no echo; others are never rejected."""

    name = "guardrail_behavior"

    def applies(self, case: GoldenCase) -> bool:
        return True

    def checks(self, outcome: CaseOutcome) -> list[Check]:
        expected = outcome.case.expect_guardrail_reason
        final = _final(outcome)
        if expected is None:
            return [
                Check(
                    f"unexpected HTTP 400 (reason={final.body.get('reason')})",
                    final.status_code != 400,
                )
            ]
        return [
            Check(
                f"final turn returned HTTP {final.status_code}, expected 400",
                final.status_code == 400,
            ),
            Check(
                f"reason '{final.body.get('reason')}' != '{expected}'",
                final.body.get("reason") == expected,
            ),
            Check(
                "rejection body echoes the input text", not _echoes_input(final.body, outcome.case)
            ),
        ]


def _echo_fragments(text: str) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_BREAKS.split(text)]
    long_sentences = [part for part in sentences if len(part) >= MIN_ECHO_FRAGMENT_CHARS]
    return [text.strip(), *long_sentences]


def _echoes_input(body: dict[str, Any], case: GoldenCase) -> bool:
    dumped = json.dumps(body, ensure_ascii=False).casefold()
    return any(fragment.casefold() in dumped for fragment in _echo_fragments(case.turns[-1]))


class MemoryConsistencyMetric(Metric):
    """Across turns of one session: all 200, counter +1, same project name, hours not shrinking.

    The name rule is deliberately strict: ``project_name`` must be the same on every
    turn, compared case-insensitively. The v2 prompt asks the model to keep it stable,
    and a renamed project is exactly the regression this metric exists to catch. The
    hours rule assumes ADDITIVE follow-ups ("Add X"), which never reduce scope.
    """

    name = "memory_consistency"

    def applies(self, case: GoldenCase) -> bool:
        return case.check_memory

    def checks(self, outcome: CaseOutcome) -> list[Check]:
        turns = outcome.turns
        statuses = [turn.status_code for turn in turns]
        if any(status != 200 for status in statuses):
            return [Check(f"turn statuses {statuses}, expected all 200", False)]
        counters = [turn.body["turns"] for turn in turns]
        names = {turn.body["result"]["project_name"].casefold() for turn in turns}
        hours = [turn.body["result"]["summary"]["total_hours"] for turn in turns]
        return [
            Check(
                f"turns counter {counters} != {_expected_counters(turns)}",
                counters == _expected_counters(turns),
            ),
            Check(f"project_name changed across turns: {sorted(names)}", len(names) == 1),
            Check(f"total hours shrank across turns: {hours}", hours == sorted(hours)),
        ]


def _expected_counters(turns: list[TurnResult]) -> list[int]:
    return list(range(1, len(turns) + 1))


METRICS: tuple[Metric, ...] = (
    SchemaConsistencyMetric(),
    RangeBoundsMetric(),
    ContentRecallMetric(),
    GuardrailBehaviorMetric(),
    TierExpectationMetric(),
    MemoryConsistencyMetric(),
)


def run_all_metrics(outcome: CaseOutcome) -> list[MetricResult]:
    """Evaluate every metric, always in the same order."""
    return [metric.evaluate(outcome) for metric in METRICS]
