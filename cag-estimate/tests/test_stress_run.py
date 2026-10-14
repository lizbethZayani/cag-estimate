"""Offline tests for the stress runner (httpx.MockTransport, no network, no LLM)."""

import csv
from pathlib import Path

import httpx
import pytest

from cag_estimate.evals.stress.run import CSV_FIELDS, OBSERVATION_FIELDS, main, percentile

OVERSIZED_BYTES = 90_000


class FakeApi:
    """Minimal fake of the session API: counts turns, remembers the first transcript."""

    def __init__(self, healthy: bool = True, fail_turn: int | None = None) -> None:
        self.healthy = healthy
        self.fail_turn = fail_turn
        self.turns: dict[str, int] = {}
        self.deleted: list[str] = []
        self.requests: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))
        if path == "/health":
            return httpx.Response(200 if self.healthy else 503, json={"status": "ok"})
        if request.method == "POST" and path == "/api/v1/sessions":
            session_id = f"s{len(self.turns) + 1}"
            self.turns[session_id] = 0
            return httpx.Response(201, json={"session_id": session_id})
        session_id = path.split("/")[4]
        if request.method == "DELETE":
            self.deleted.append(session_id)
            return httpx.Response(204)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"summary": "- the Nimbus portal", "anchors": [], "project_metadata": {}},
            )
        return self._estimate(request, session_id)

    def _estimate(self, request: httpx.Request, session_id: str) -> httpx.Response:
        if len(request.content) > OVERSIZED_BYTES:
            return httpx.Response(413, json={"detail": "too large"})
        self.turns[session_id] += 1
        turn = self.turns[session_id]
        if turn == self.fail_turn:
            return httpx.Response(502, json={"detail": "upstream"})
        observation = {
            "turn_index": turn,
            "session_id": session_id,
            "enriched_transcript_chars": 100 * turn,
            "attachments_total_chars": 0,
            "messages_in_window": 2 * turn,
            "anchors_count": 0,
            "summary_chars": 0,
            "tokens_in": 10 * turn,
            "tokens_out": 5,
            "cost_usd": 0.001 * turn,
            "latency_ms": 100.0 * turn,
            "cache_hit_kind": "none",
            "last_resolved_tier": "default",
        }
        return httpx.Response(200, json={"observation": observation})


def _run(tmp_path: Path, api: FakeApi, *extra: str) -> tuple[int, list[dict[str, str]]]:
    output = tmp_path / "out" / "results.csv"
    argv = [
        "--base-url", "http://test",
        "--scenarios", "growing",
        "--attachment-sizes", "0",
        "--repeats", "1",
        "--turns", "3",
        "--output", str(output),
        "--fixtures-dir", str(tmp_path / "fixtures"),
        *extra,
    ]  # fmt: skip
    code = main(argv, transport=httpx.MockTransport(api))
    with output.open(newline="", encoding="utf-8") as handle:
        return code, list(csv.DictReader(handle))


def test_csv_header_has_the_observation_and_run_columns() -> None:
    assert len(OBSERVATION_FIELDS) == 13
    for column in (*OBSERVATION_FIELDS, "scenario", "attachment_size_kb", "repeat", "wall_clock_ms",
                   "status_code", "latency_budget_passed", "cost_budget_passed",
                   "memory_drift_passed", "tracked_fact", "error"):  # fmt: skip
        assert column in CSV_FIELDS


def test_one_row_per_turn_with_observation_and_metric_columns(tmp_path: Path) -> None:
    api = FakeApi()

    code, rows = _run(tmp_path, api, "--latency-budget-ms", "250", "--cost-budget-usd", "0.0025")

    assert code == 0
    assert [row["turn_index"] for row in rows] == ["1", "2", "3"]
    assert {row["scenario"] for row in rows} == {"growing"}
    assert rows[0]["tokens_in"] == "10"
    assert rows[0]["status_code"] == "200"
    assert float(rows[0]["wall_clock_ms"]) >= 0
    assert rows[0]["tracked_fact"] == "Nimbus"
    assert rows[0]["memory_drift_passed"] == "True"
    assert [row["latency_budget_passed"] for row in rows] == ["True", "True", "False"]
    assert [row["cost_budget_passed"] for row in rows] == ["True", "True", "False"]
    assert rows[0]["error"] == ""


def test_the_session_is_deleted_at_the_end_of_each_run(tmp_path: Path) -> None:
    api = FakeApi()

    _run(tmp_path, api, "--repeats", "2")

    assert api.deleted == ["s1", "s2"]


def test_a_failed_turn_writes_an_error_row_and_the_run_continues(tmp_path: Path) -> None:
    code, rows = _run(tmp_path, FakeApi(fail_turn=2))

    assert code == 0
    assert [row["status_code"] for row in rows] == ["200", "502", "200"]
    assert rows[1]["error"] != ""
    assert rows[1]["tokens_in"] == ""
    assert rows[1]["memory_drift_passed"] == ""


def test_an_oversized_attachment_is_recorded_as_413_for_every_turn(tmp_path: Path) -> None:
    code, rows = _run(tmp_path, FakeApi(), "--attachment-sizes", "100")

    assert code == 0
    assert [row["status_code"] for row in rows] == ["413", "413", "413"]
    assert {row["attachment_size_kb"] for row in rows} == {"100"}
    assert all("413" in row["error"] for row in rows)
    assert (tmp_path / "fixtures" / "attach_100kb.pdf").exists()


def test_an_unhealthy_service_aborts_before_creating_sessions(tmp_path: Path) -> None:
    api = FakeApi(healthy=False)
    output = tmp_path / "results.csv"

    code = main(
        ["--output", str(output), "--fixtures-dir", str(tmp_path)],
        transport=httpx.MockTransport(api),
    )

    assert code == 2
    assert api.requests == [("GET", "/health")]
    assert not output.exists()


def test_the_summary_prints_p50_and_p95(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(tmp_path, FakeApi())

    out = capsys.readouterr().out
    assert "P50" in out and "P95" in out


def test_percentile_uses_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 50) == 20.0
    assert percentile(values, 95) == 40.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 50) is None
