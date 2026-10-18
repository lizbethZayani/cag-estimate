"""Tests for scripts/compare.py, loaded as a module (scripts/ is not a package)."""

import importlib.util
import io
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare.py"
SECRET = "sk-test-secret-key"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_one(self, text: str) -> list[float]:
        return self._vectors[text]


class FailingEmbedder:
    def embed_one(self, text: str) -> list[float]:
        raise RuntimeError("boom: service unavailable")


@pytest.fixture
def script() -> ModuleType:
    return _load_script()


def _run(script: ModuleType, argv: list[str], embedder: object | None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = script.main(argv, embedder=embedder, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_prints_the_three_line_format_for_identical_vectors(script: ModuleType) -> None:
    embedder = FakeEmbedder({"a": [1.0, 2.0], "b": [1.0, 2.0]})

    code, out, err = _run(script, ["--text-a", "a", "--text-b", "b"], embedder)

    assert code == 0
    assert out == "Text A: a\nText B: b\nCosine similarity: 1.0000\n"
    assert err == ""


def test_orthogonal_vectors_print_zero(script: ModuleType) -> None:
    embedder = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})

    code, out, _ = _run(script, ["--text-a", "a", "--text-b", "b"], embedder)

    assert code == 0
    assert out.splitlines()[2] == "Cosine similarity: 0.0000"


def test_similarity_is_rounded_to_four_decimals(script: ModuleType) -> None:
    embedder = FakeEmbedder({"a": [1.0, 0.0], "b": [3.0, 4.0]})

    _, out, _ = _run(script, ["--text-a", "a", "--text-b", "b"], embedder)

    assert out.splitlines()[2] == "Cosine similarity: 0.6000"


def test_missing_openai_key_exits_2_without_building_a_client(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script, "get_settings", lambda: SimpleNamespace(openai_api_key=None))
    monkeypatch.setattr(
        script, "OpenAI", lambda **_: pytest.fail("no client may be built without a key")
    )

    code, out, err = _run(script, ["--text-a", "a", "--text-b", "b"], None)

    assert code == 2
    assert out == ""
    assert "OPENAI_API_KEY" in err
    assert "Traceback" not in err


def test_missing_text_b_is_a_usage_error(script: ModuleType) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run(script, ["--text-a", "a"], FakeEmbedder({}))

    assert exc_info.value.code == 2


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_text_exits_2(script: ModuleType, blank: str) -> None:
    code, out, err = _run(script, ["--text-a", blank, "--text-b", "b"], FakeEmbedder({}))

    assert code == 2
    assert out == ""
    assert "empty" in err


def test_embedding_failure_exits_1_with_the_error_on_stderr(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)

    code, out, err = _run(script, ["--text-a", "a", "--text-b", "b"], FailingEmbedder())

    assert code == 1
    assert out == ""
    assert "RuntimeError" in err
    assert "boom: service unavailable" in err
    assert SECRET not in out + err


def test_main_builds_the_real_embedder_lazily_from_settings(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> str:
        built.update(kwargs)
        return "client"

    class RecordingEmbedder(FakeEmbedder):
        def __init__(self, client: object, model: str) -> None:
            super().__init__({"a": [1.0], "b": [1.0]})
            built["embedder"] = (client, model)

    settings = SimpleNamespace(openai_api_key=SECRET, embedding_model="custom-model")
    monkeypatch.setattr(script, "get_settings", lambda: settings)
    monkeypatch.setattr(script, "OpenAI", fake_openai)
    monkeypatch.setattr(script, "OpenAIEmbedder", RecordingEmbedder)

    code, _, _ = _run(script, ["--text-a", "a", "--text-b", "b"], None)

    assert code == 0
    assert built["api_key"] == SECRET
    assert built["max_retries"] == 0
    assert built["embedder"] == ("client", "custom-model")


def test_script_does_not_use_print() -> None:
    assert "print(" not in SCRIPT.read_text(encoding="utf-8")


def test_script_is_executable_with_a_shebang() -> None:
    assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3\n")
