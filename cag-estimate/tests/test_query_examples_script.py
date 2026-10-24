"""Tests for scripts/query_examples.py, loaded as a module (scripts/ is not a package)."""

import importlib.util
import io
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "query_examples.py"
QUERIES = [
    "OAuth 2.0 authentication with JWT tokens",
    "payment processing with Stripe",
    "restaurant reservations",
    "high-complexity backend components",
    "real-time dashboard",
]
BUDGETS = [{"budget_id": "BUD-1", "year": 2024}, {"budget_id": "BUD-2", "year": 2023}]

Handler = Callable[[httpx.Request], httpx.Response]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("query_examples_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps(BUDGETS))
    return path


def _hit(rank: int, content: str = "[Project: X]\nbody") -> dict[str, object]:
    return {
        "chunk_id": rank,
        "document_id": 1,
        "chunk_type": "component",
        "content": content,
        "distance": 0.123456 * rank,
        "metadata": {},
    }


class Api:
    """Fake API: records requests and answers ingest/search through MockTransport."""

    def __init__(
        self,
        ingest_status: int = 200,
        results: list[dict[str, object]] | None = None,
    ) -> None:
        self.ingest_status = ingest_status
        self.results = [_hit(1)] if results is None else results
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/embeddings/ingest":
            return httpx.Response(self.ingest_status, json={"detail": "secret-body"})
        return httpx.Response(200, json={"search_time_ms": 7, "results": self.results})

    def bodies(self, path: str) -> list[dict[str, object]]:
        return [json.loads(r.content) for r in self.requests if r.url.path == path]


def _run(script: ModuleType, argv: list[str], handler: Handler) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://t") as client:
        code = script.main(argv, client=client, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_ingests_each_budget_with_source_path_and_type(script: ModuleType, data_file: Path) -> None:
    api = Api()

    code, out, _ = _run(script, ["--data", str(data_file)], api.handle)

    assert code == 0
    bodies = api.bodies("/embeddings/ingest")
    assert [b["source_path"] for b in bodies] == [f"{data_file}::BUD-1", f"{data_file}::BUD-2"]
    assert all(b["document_type"] == "historical_budget" for b in bodies)
    assert [b["content"] for b in bodies] == BUDGETS
    assert "Ingested 2, already present 0" in out


def test_409_counts_as_already_present(script: ModuleType, data_file: Path) -> None:
    api = Api(ingest_status=409)

    code, out, _ = _run(script, ["--data", str(data_file)], api.handle)

    assert code == 0
    assert "Ingested 0, already present 2" in out


def test_other_ingest_status_exits_1_without_leaking_the_body(
    script: ModuleType, data_file: Path
) -> None:
    api = Api(ingest_status=500)

    code, out, err = _run(script, ["--data", str(data_file)], api.handle)

    assert code == 1
    assert "500" in err
    assert "secret-body" not in err + out
    assert api.bodies("/search") == []


def test_transport_error_exits_1(script: ModuleType, data_file: Path) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    code, _, err = _run(script, ["--data", str(data_file)], boom)

    assert code == 1
    assert "error" in err


def test_search_failure_exits_1(script: ModuleType, data_file: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        status = 200 if request.url.path == "/embeddings/ingest" else 502
        return httpx.Response(status, json={})

    code, _, err = _run(script, ["--data", str(data_file)], handler)

    assert code == 1
    assert "502" in err


def test_sends_the_five_queries_in_order_with_k(script: ModuleType, data_file: Path) -> None:
    api = Api()

    _run(script, ["--data", str(data_file), "--k", "3"], api.handle)

    assert api.bodies("/search") == [{"query": q, "k": 3} for q in QUERIES]


def test_default_k_is_5(script: ModuleType, data_file: Path) -> None:
    api = Api()

    _run(script, ["--data", str(data_file)], api.handle)

    assert {b["k"] for b in api.bodies("/search")} == {5}


def test_prints_headers_ranked_lines_and_search_time(script: ModuleType, data_file: Path) -> None:
    api = Api(results=[_hit(1), _hit(2, "[Project: X]\n\n  second\tline")])

    _, out, _ = _run(script, ["--data", str(data_file)], api.handle)

    assert "Query 1: OAuth 2.0 authentication with JWT tokens" in out
    assert "Query 5: real-time dashboard" in out
    assert "  1. chunk_id=1 distance=0.1235 type=component | [Project: X] body" in out
    assert "  2. chunk_id=2 distance=0.2469 type=component | [Project: X] second line" in out
    assert "search_time_ms=7" in out


def test_long_content_is_truncated_to_120_chars(script: ModuleType, data_file: Path) -> None:
    api = Api(results=[_hit(1, "x" * 500)])

    _, out, _ = _run(script, ["--data", str(data_file)], api.handle)

    line = next(ln for ln in out.splitlines() if ln.startswith("  1."))
    assert line.endswith("| " + "x" * 120)


def test_empty_results_print_no_results(script: ModuleType, data_file: Path) -> None:
    api = Api(results=[])

    _, out, _ = _run(script, ["--data", str(data_file)], api.handle)

    assert out.count("  (no results)") == 5


def test_missing_data_file_exits_2(script: ModuleType, tmp_path: Path) -> None:
    api = Api()

    code, _, err = _run(script, ["--data", str(tmp_path / "nope.json")], api.handle)

    assert code == 2
    assert "error" in err
    assert api.requests == []


@pytest.mark.parametrize("content", ["not json", '{"budget_id": "x"}', '[{"year": 1}]', "[1]"])
def test_invalid_data_file_exits_2(script: ModuleType, tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content)
    api = Api()

    code, _, err = _run(script, ["--data", str(path)], api.handle)

    assert code == 2
    assert "error" in err
    assert api.requests == []
