"""Run golden cases against a RUNNING API over HTTP and aggregate the metric results.

Nothing is faked: every turn is a real request. The runner needs a client with
``post(url, data=None, files=None, timeout=...)``, ``get(url, timeout=...)`` and
``delete(url, timeout=...)`` whose responses expose ``.status_code`` and
``.json()``. ``requests.Session`` and Starlette's ``TestClient`` both fit.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

import structlog

from cag_estimate.evals.dataset import GoldenCase
from cag_estimate.evals.metrics import (
    METRICS,
    CaseOutcome,
    MetricResult,
    TurnResult,
    run_all_metrics,
)

log = structlog.get_logger()

Mode = Literal["standard", "acb"]
ESTIMATE_PATHS: dict[str, str] = {"standard": "estimate", "acb": "estimate-acb"}
DEFAULT_TIMEOUT_SECONDS = 300.0
SESSIONS_PATH = "/api/v1/sessions"
CREATED = 201


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def post(
        self, url: str, data: Any = None, files: Any = None, timeout: Any = None
    ) -> HttpResponse: ...

    def get(self, url: str, timeout: Any = None) -> HttpResponse: ...

    def delete(self, url: str, timeout: Any = None) -> HttpResponse: ...


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _json_body(response: HttpResponse) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _create_session(client: HttpClient, base: str, timeout: float) -> str:
    response = client.post(f"{base}{SESSIONS_PATH}", timeout=timeout)
    if response.status_code != CREATED:
        raise RuntimeError(f"session creation returned HTTP {response.status_code}")
    return _json_body(response)["session_id"]


def _delete_session(client: HttpClient, base: str, session_id: str, timeout: float) -> None:
    """Best effort: a leftover session is only memory in the API, never a failed eval."""
    try:
        client.delete(f"{base}{SESSIONS_PATH}/{session_id}", timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - client-agnostic transport boundary
        log.warning("eval_session_delete_failed", error=_describe(exc))


def _form(case: GoldenCase, text: str) -> dict[str, str]:
    form = {"transcript": text, "hourly_rate": str(case.hourly_rate)}
    if case.tier != "auto":
        form["tier"] = case.tier
    return form


def _send_turn(client: HttpClient, url: str, form: dict[str, str], timeout: float) -> TurnResult:
    started = time.perf_counter()
    response = client.post(url, data=form, timeout=timeout)
    latency_ms = (time.perf_counter() - started) * 1000
    return TurnResult(response.status_code, _json_body(response), latency_ms)


def _send_turns(
    case: GoldenCase, client: HttpClient, url: str, timeout: float, done: list[TurnResult]
) -> None:
    """Send every turn in order into ``done``; stop at the first non-200 answer."""
    for text in case.turns:
        done.append(_send_turn(client, url, _form(case, text), timeout))
        if done[-1].status_code != 200:
            return


def run_case(
    case: GoldenCase,
    client: HttpClient,
    base_url: str,
    mode: Mode = "standard",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> CaseOutcome:
    """Run one case in a fresh session, always deleting it. Transport errors never pass."""
    base = base_url.rstrip("/")
    try:
        session_id = _create_session(client, base, timeout)
    except Exception as exc:  # noqa: BLE001 - any failure here is an errored case
        return CaseOutcome(case, error=_describe(exc))
    url = f"{base}{SESSIONS_PATH}/{session_id}/{ESTIMATE_PATHS[mode]}"
    done: list[TurnResult] = []
    try:
        _send_turns(case, client, url, timeout, done)
        return CaseOutcome(case, done)
    except Exception as exc:  # noqa: BLE001 - any failure here is an errored case
        return CaseOutcome(case, done, error=_describe(exc))
    finally:
        _delete_session(client, base, session_id, timeout)


@dataclass(frozen=True)
class CaseReport:
    """One row of the report: the metric results of a case and how long it took."""

    case_id: str
    results: list[MetricResult]
    turn_latencies_ms: list[float]
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return sum(self.turn_latencies_ms)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "passed": self.passed,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metrics": [asdict(result) for result in self.results],
        }


@dataclass(frozen=True)
class EvalReport:
    mode: str
    base_url: str
    cases: list[CaseReport] = field(default_factory=list)

    @property
    def passed_cases(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def metric_pass_counts(self) -> dict[str, int]:
        return {
            metric.name: sum(
                result.passed
                for case in self.cases
                for result in case.results
                if result.name == metric.name
            )
            for metric in METRICS
        }

    @property
    def average_turn_latency_ms(self) -> float:
        latencies = [latency for case in self.cases for latency in case.turn_latencies_ms]
        return sum(latencies) / len(latencies) if latencies else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "base_url": self.base_url,
            "total_cases": len(self.cases),
            "passed_cases": self.passed_cases,
            "failed_cases": len(self.cases) - self.passed_cases,
            "metric_pass_counts": self.metric_pass_counts,
            "average_turn_latency_ms": self.average_turn_latency_ms,
            "cases": [case.to_dict() for case in self.cases],
        }


def build_case_report(outcome: CaseOutcome) -> CaseReport:
    return CaseReport(
        case_id=outcome.case.id,
        results=run_all_metrics(outcome),
        turn_latencies_ms=[turn.latency_ms for turn in outcome.turns],
        error=outcome.error,
    )


def build_report(outcomes: list[CaseOutcome], mode: str, base_url: str) -> EvalReport:
    return EvalReport(mode, base_url, [build_case_report(outcome) for outcome in outcomes])


def evaluate_cases(
    cases: list[GoldenCase],
    client: HttpClient,
    base_url: str,
    mode: Mode,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    on_case: Callable[[CaseReport], None] | None = None,
) -> EvalReport:
    """Run the cases one after another, reporting each row as soon as it is ready."""
    reports: list[CaseReport] = []
    for case in cases:
        report = build_case_report(run_case(case, client, base_url, mode, timeout))
        reports.append(report)
        if on_case is not None:
            on_case(report)
    return EvalReport(mode, base_url, reports)
