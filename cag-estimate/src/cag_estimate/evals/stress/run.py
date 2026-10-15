"""CLI: ``python -m cag_estimate.evals.stress.run``.

Plays the synthetic scenarios against a RUNNING API (``/api/v1/sessions``) and writes
one CSV row per turn: the server-side ``observation`` plus run columns (wall clock,
HTTP status, budget and memory verdicts). Nothing is faked and every turn is a real
LLM call, so run it deliberately; the tests use ``httpx.MockTransport``.

Attachments (``--attachment-sizes``, KB, 0 = none) are sent on EVERY turn. A turn that
fails (413, 502, transport error) writes an error row and the run continues. The CSV is
flushed per row so an interrupted run keeps what it measured.

Exit codes: 0 finished; 2 the service is not healthy (nothing was estimated).
"""

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import httpx

from cag_estimate.evals.stress.metrics import (
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)
from cag_estimate.evals.stress.scenarios import (
    SCENARIO_NAMES,
    TOTAL_TURNS,
    Scenario,
    get_scenario,
    slice_scenario,
)
from cag_estimate.schemas.session import TurnObservation

EXIT_OK, EXIT_UNHEALTHY = 0, 2
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_OUTPUT = Path("session6/stress/results.csv")
DEFAULT_FIXTURES_DIR = Path("session6/stress/fixtures")
HOURLY_RATE = 40
TIMEOUT_SECONDS = 300.0
HEALTH_TIMEOUT_SECONDS = 10.0

OBSERVATION_FIELDS = tuple(TurnObservation.model_fields)
RUN_FIELDS = (
    "scenario",
    "attachment_size_kb",
    "repeat",
    "wall_clock_ms",
    "status_code",
    "latency_budget_passed",
    "cost_budget_passed",
    "memory_drift_passed",
    "tracked_fact",
    "error",
)
CSV_FIELDS = (*OBSERVATION_FIELDS, *RUN_FIELDS)


def write(text: str = "") -> None:
    """The single output channel of the CLI."""
    sys.stdout.write(f"{text}\n")


def percentile(values: Sequence[float], rank: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(rank / 100 * len(ordered)) - 1, 0)
    return ordered[index]


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cag_estimate.evals.stress.run")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--scenarios", type=_csv_list, default=list(SCENARIO_NAMES))
    parser.add_argument(
        "--attachment-sizes",
        type=lambda v: [int(item) for item in _csv_list(v)],
        default=[0, 5, 20, 50, 100],
        help="comma separated KB sizes sent on every turn (0 = no attachment)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--turns", type=int, default=TOTAL_TURNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--latency-budget-ms", type=float, default=8000.0)
    parser.add_argument("--cost-budget-usd", type=float, default=0.02)
    return parser


class StressRunner:
    """Runs scenarios through one HTTP client and appends rows to a CSV."""

    def __init__(
        self,
        client: httpx.Client,
        sink: csv.DictWriter,
        stream: TextIO,
        latency: LatencyBudgetMetric,
        cost: CostBudgetMetric,
    ) -> None:
        self._client = client
        self._sink = sink
        self._stream = stream
        self._latency = latency
        self._cost = cost
        self.rows: list[dict[str, Any]] = []

    def run_session(
        self, scenario: Scenario, size_kb: int, repeat: int, attachment: Path | None
    ) -> None:
        session_id = self._client.post("/sessions").json()["session_id"]
        memory = MemoryDriftMetric(scenario.tracked_fact)
        file_bytes = attachment.read_bytes() if attachment else None
        try:
            for turn in scenario.turns:
                row: dict[str, Any] = {
                    "turn_index": turn.turn_index,
                    "session_id": session_id,
                    "scenario": scenario.name,
                    "attachment_size_kb": size_kb,
                    "repeat": repeat,
                    "tracked_fact": scenario.tracked_fact,
                }
                self._play_turn(row, session_id, turn.transcript, attachment, file_bytes, memory)
                self._emit(row)
        finally:
            self._client.delete(f"/sessions/{session_id}")

    def _play_turn(
        self,
        row: dict[str, Any],
        session_id: str,
        transcript: str,
        attachment: Path | None,
        file_bytes: bytes | None,
        memory: MemoryDriftMetric,
    ) -> None:
        files = (
            [("attachments", (attachment.name, file_bytes, "application/pdf"))]
            if attachment and file_bytes is not None
            else None
        )
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"/sessions/{session_id}/estimate",
                data={"transcript": transcript, "hourly_rate": str(HOURLY_RATE)},
                files=files,
            )
        except httpx.HTTPError as exc:
            row["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 1)
            row["error"] = f"{type(exc).__name__}: {exc}"
            return
        row["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 1)
        row["status_code"] = response.status_code
        if response.status_code != httpx.codes.OK:
            row["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
            return
        observation_data = response.json().get("observation")
        if not observation_data:
            row["error"] = "response carried no observation"
            return
        observation = TurnObservation.model_validate(observation_data)
        row.update(observation.model_dump())
        row["latency_budget_passed"] = self._latency.evaluate(observation).passed
        row["cost_budget_passed"] = self._cost.evaluate(observation).passed
        snapshot = self._client.get(f"/sessions/{session_id}")
        if snapshot.status_code == httpx.codes.OK:
            row["memory_drift_passed"] = memory.evaluate(snapshot.json()).passed

    def _emit(self, row: dict[str, Any]) -> None:
        self._sink.writerow(row)
        self._stream.flush()
        self.rows.append(row)


def _values(group: list[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in group if r.get(key) not in ("", None)]


def _fmt(values: list[float], rank: int) -> str:
    value = percentile(values, rank)
    return "n/a" if value is None else f"{value:.4g}"


def _summarize(rows: list[dict[str, Any]]) -> None:
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_size[row["attachment_size_kb"]].append(row)
    write(f"{len(rows)} turns, {sum(1 for r in rows if r.get('error'))} errors")
    for size in sorted(by_size):
        group = by_size[size]
        columns = [
            f"{label} P50 {_fmt(vals, 50)} P95 {_fmt(vals, 95)}"
            for label, vals in (
                ("latency_ms", _values(group, "latency_ms")),
                ("wall_ms", _values(group, "wall_clock_ms")),
                ("cost_usd", _values(group, "cost_usd")),
            )
        ]
        write(f"attachment {size:>3} KB | " + " | ".join(columns))


def _healthy(client: httpx.Client, base_url: str) -> bool:
    try:
        return client.get(f"{base_url}/health", timeout=HEALTH_TIMEOUT_SECONDS).is_success
    except httpx.HTTPError:
        return False


def main(argv: Sequence[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_url = args.base_url.rstrip("/")
    with httpx.Client(
        base_url=f"{base_url}/api/v1", timeout=TIMEOUT_SECONDS, transport=transport
    ) as client:
        if not _healthy(client, base_url):
            write(f"Service at {base_url} is not healthy; nothing was run.")
            return EXIT_UNHEALTHY
        # Imported lazily: fpdf2 is a dev dependency, only needed to build missing PDFs.
        from cag_estimate.evals.stress.fixtures.build_pdfs import ensure_fixtures

        sizes = [size for size in args.attachment_sizes if size > 0]
        attachments = ensure_fixtures(args.fixtures_dir, tuple(sizes)) if sizes else {}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            sink = csv.DictWriter(stream, fieldnames=CSV_FIELDS, restval="")
            sink.writeheader()
            runner = StressRunner(
                client,
                sink,
                stream,
                LatencyBudgetMetric(args.latency_budget_ms),
                CostBudgetMetric(args.cost_budget_usd),
            )
            for name in args.scenarios:
                scenario = slice_scenario(get_scenario(name), args.turns)
                for size in args.attachment_sizes:
                    for repeat in range(1, args.repeats + 1):
                        runner.run_session(scenario, size, repeat, attachments.get(size))
        _summarize(runner.rows)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
