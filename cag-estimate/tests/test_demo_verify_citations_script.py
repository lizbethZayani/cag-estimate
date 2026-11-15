"""Tests for scripts/demo_verify_citations.py, loaded as a module (scripts/ is not a package)."""

import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pytest

from cag_estimate.services.citations import CitationReport

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "demo_verify_citations.py"
PLANTED_ID = "BUD-2099-999::FAKE-001"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_verify_citations_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


def _run(script: ModuleType, **kwargs: object) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = script.main([], stdout=out, stderr=err, **kwargs)
    return code, out.getvalue(), err.getvalue()


def test_demo_exits_zero_when_the_planted_id_is_detected(script: ModuleType) -> None:
    code, _, err = _run(script)

    assert code == 0
    assert err == ""


def test_output_shows_the_three_statuses_and_the_invented_id(script: ModuleType) -> None:
    _, out, _ = _run(script)

    for status in ("grounded", "dangling", "insufficient"):
        assert status in out
    assert PLANTED_ID in out
    assert "DANGLING CITATION DETECTED" in out


def test_output_cites_a_real_chunk_id_of_the_sample_budget(script: ModuleType) -> None:
    _, out, _ = _run(script)

    assert "BUD-2024-005::CATALOG-001" in out


def test_output_has_a_table_header_and_a_summary_line(script: ModuleType) -> None:
    _, out, _ = _run(script)

    assert "| Line | Status | Cited ids | Dangling ids |" in out
    assert "lines=3 grounded=1 dangling=1 insufficient=1" in out


def test_exit_one_when_the_verifier_misses_the_planted_id(script: ModuleType) -> None:
    def broken_verify(estimate: object, retrieved_chunk_ids: set[str]) -> CitationReport:
        return CitationReport(
            total_lines=3,
            grounded_lines=2,
            dangling_lines=0,
            insufficient_lines=1,
            dangling_chunk_ids=[],
            lines=[],
        )

    code, out, err = _run(script, verify=broken_verify)

    assert code == 1
    assert "DANGLING CITATION DETECTED" not in out
    assert "verifier" in err


def test_the_script_needs_no_environment_variables(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)

    code, _, _ = _run(script)

    assert code == 0
