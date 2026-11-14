"""Tests for scripts/eval_retrieval.py, loaded as a module (scripts/ is not a package)."""

import importlib.util
import io
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_retrieval.py"
GOLDEN = {
    "version": 1,
    "queries": [
        {
            "id": "q1",
            "query": "first query",
            "relevant_budget_ids": ["BUD-2024-001"],
            "rationale": "r",
        },
        {
            "id": "q2",
            "query": "second query",
            "relevant_budget_ids": ["BUD-2024-002"],
            "rationale": "r",
        },
    ],
}

Handler = Callable[[httpx.Request], httpx.Response]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_retrieval_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


@pytest.fixture
def golden_file(tmp_path: Path) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(GOLDEN))
    return path


def _hit(budget_id: str) -> dict[str, object]:
    return {
        "chunk_id": 1,
        "document_id": 1,
        "chunk_type": "component",
        "content": "x",
        "distance": 0.1,
        "metadata": {"budget_id": budget_id},
    }


class Api:
    """Fake API: echoes the requested strategy unless told otherwise, records requests."""

    def __init__(self, reranked: bool | None = None, status: int = 200) -> None:
        self.forced_reranked = reranked
        self.status = status
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "secret-body"})
        body = json.loads(request.content)
        reranked = body["rerank"] if self.forced_reranked is None else self.forced_reranked
        return httpx.Response(
            200,
            json={
                "query": body["query"],
                "k": body["k"],
                "search_time_ms": 5,
                "search_mode": body["search_mode"],
                "reranked": reranked,
                "results": [_hit("BUD-2024-001")],
            },
        )

    def bodies(self) -> list[dict[str, object]]:
        return [json.loads(r.content) for r in self.requests]


def _run(script: ModuleType, argv: list[str], handler: Handler) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t") as client:
        code = script.main(argv, client=client, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_each_config_sends_its_payload_with_both_fields_explicit(
    script: ModuleType, golden_file: Path
) -> None:
    api = Api()

    code, _, _ = _run(script, ["--golden", str(golden_file), "--runs", "1"], api.handle)

    assert code == 0
    sent = {(b["search_mode"], b["rerank"]) for b in api.bodies()}
    assert sent == {("vector", False), ("hybrid", False), ("vector", True), ("hybrid", True)}
    assert all(set(b) == {"query", "k", "search_mode", "rerank"} for b in api.bodies())
    assert {b["k"] for b in api.bodies()} == {5}


def test_warm_up_plus_runs_requests_per_query_and_config(
    script: ModuleType, golden_file: Path
) -> None:
    api = Api()

    _run(script, ["--golden", str(golden_file), "--runs", "3"], api.handle)

    # 2 queries x 4 configs x (1 warm-up + 3 timed)
    assert len(api.requests) == 2 * 4 * 4


def test_warm_up_is_excluded_from_the_timings(script: ModuleType, golden_file: Path) -> None:
    api = Api()
    output = golden_file.parent / "raw.json"

    _run(
        script,
        ["--golden", str(golden_file), "--runs", "2", "--output", str(output)],
        api.handle,
    )

    raw = json.loads(output.read_text())
    assert all(len(m["latencies_ms"]) == 2 for m in raw["measurements"])
    assert all(len(m["server_times_ms"]) == 2 for m in raw["measurements"])


def test_prints_both_tables_and_the_parameters(script: ModuleType, golden_file: Path) -> None:
    api = Api()

    _, out, _ = _run(script, ["--golden", str(golden_file), "--runs", "1", "--k", "3"], api.handle)

    assert "| Config | Search | Reranking |" in out
    assert "| Query | A | B | C | D |" in out
    assert "| q1 | 1.00 |" not in out  # k=3 gives 1/3, not 1.00
    assert "queries=2" in out and "runs=1" in out and "k=3" in out


def test_a_reranked_mismatch_aborts_with_exit_1(script: ModuleType, golden_file: Path) -> None:
    api = Api(reranked=False)

    code, out, err = _run(script, ["--golden", str(golden_file), "--runs", "1"], api.handle)

    assert code == 1
    assert "reranked" in err
    assert "| Config |" not in out


def test_an_http_error_aborts_with_exit_1_without_leaking_the_body(
    script: ModuleType, golden_file: Path
) -> None:
    api = Api(status=500)

    code, out, err = _run(script, ["--golden", str(golden_file)], api.handle)

    assert code == 1
    assert "500" in err
    assert "secret-body" not in err + out


def test_a_transport_error_aborts_with_exit_1(script: ModuleType, golden_file: Path) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    code, _, err = _run(script, ["--golden", str(golden_file)], boom)

    assert code == 1
    assert "error" in err


def test_missing_golden_file_exits_2(script: ModuleType, tmp_path: Path) -> None:
    api = Api()

    code, _, err = _run(script, ["--golden", str(tmp_path / "nope.json")], api.handle)

    assert code == 2
    assert "error" in err
    assert api.requests == []


def test_invalid_golden_file_exits_2(script: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"version": 1, "queries": []}')
    api = Api()

    code, _, _ = _run(script, ["--golden", str(path)], api.handle)

    assert code == 2
    assert api.requests == []


@pytest.mark.parametrize("flag", ["--runs", "--k"])
def test_non_positive_runs_or_k_exit_2(script: ModuleType, golden_file: Path, flag: str) -> None:
    api = Api()

    code, _, err = _run(script, ["--golden", str(golden_file), flag, "0"], api.handle)

    assert code == 2
    assert "error" in err
    assert api.requests == []


def test_output_writes_every_raw_measurement(script: ModuleType, golden_file: Path) -> None:
    api = Api()
    output = golden_file.parent / "raw.json"

    code, _, _ = _run(
        script,
        ["--golden", str(golden_file), "--runs", "1", "--output", str(output)],
        api.handle,
    )

    raw = json.loads(output.read_text())
    assert code == 0
    assert len(raw["measurements"]) == 2 * 4
    assert {m["config"] for m in raw["measurements"]} == {"A", "B", "C", "D"}
    assert raw["measurements"][0]["precision"] == pytest.approx(0.2)
    assert raw["k"] == 5 and raw["runs"] == 1
