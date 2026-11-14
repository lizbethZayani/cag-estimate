"""Offline tests for the lazy cross-encoder reranker (the real model is never loaded)."""

import subprocess
import sys
import threading
import time
import types
from collections.abc import Sequence

import pytest

from cag_estimate.embedding_pipeline.reranker import CrossEncoderReranker


class FakeScorer:
    """Scores a pair by the length of the document; records every call."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        self.calls.append(list(pairs))
        return [self.scores.get(doc, float(len(doc))) for _, doc in pairs]


def test_score_builds_query_document_pairs_in_input_order() -> None:
    scorer = FakeScorer()
    reranker = CrossEncoderReranker("m", model=scorer)

    scores = reranker.score("q", ["aaa", "b"])

    assert scorer.calls == [[("q", "aaa"), ("q", "b")]]
    assert scores == [3.0, 1.0]


def test_rerank_orders_by_score_descending_and_keeps_top_n() -> None:
    scorer = FakeScorer({"low": 0.1, "high": 0.9, "mid": 0.5})
    reranker = CrossEncoderReranker("m", model=scorer)

    ranked = reranker.rerank("q", ["low", "high", "mid"], top_n=2, text_of=str)

    assert ranked == [("high", 0.9), ("mid", 0.5)]


def test_rerank_ties_keep_input_order() -> None:
    scorer = FakeScorer({"a": 1.0, "b": 1.0, "c": 1.0})
    reranker = CrossEncoderReranker("m", model=scorer)

    ranked = reranker.rerank("q", ["b", "a", "c"], top_n=3, text_of=str)

    assert [item for item, _ in ranked] == ["b", "a", "c"]


def test_rerank_uses_text_of_to_read_the_document() -> None:
    scorer = FakeScorer({"x": 2.0, "y": 1.0})
    reranker = CrossEncoderReranker("m", model=scorer)

    ranked = reranker.rerank("q", [{"t": "y"}, {"t": "x"}], top_n=5, text_of=lambda c: c["t"])

    assert ranked == [({"t": "x"}, 2.0), ({"t": "y"}, 1.0)]


def test_rerank_rejects_top_n_below_one() -> None:
    reranker = CrossEncoderReranker("m", model=FakeScorer())

    with pytest.raises(ValueError, match="top_n"):
        reranker.rerank("q", ["a"], top_n=0, text_of=str)


def test_empty_inputs_do_not_load_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = CrossEncoderReranker("m")
    monkeypatch.setattr(reranker, "_load", lambda: pytest.fail("model loaded"))

    assert reranker.score("q", []) == []
    assert reranker.rerank("q", [], top_n=3, text_of=str) == []


def test_construction_does_not_load_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    CrossEncoderReranker("m")

    assert "sentence_transformers" not in sys.modules


def install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch, created: list[str], delay: float = 0.0
) -> None:
    class FakeCrossEncoder:
        def __init__(self, name: str) -> None:
            time.sleep(delay)
            created.append(name)

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [1.0 for _ in pairs]

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_model_loads_lazily_once_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []
    install_fake_sentence_transformers(monkeypatch, created)
    reranker = CrossEncoderReranker("my/model")
    assert created == []

    reranker.score("q", ["a"])
    reranker.score("q", ["b"])
    reranker.rerank("q", ["c"], top_n=1, text_of=str)

    assert created == ["my/model"]


def test_model_loads_once_across_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []
    install_fake_sentence_transformers(monkeypatch, created, delay=0.05)
    reranker = CrossEncoderReranker("m")
    threads = [threading.Thread(target=reranker.score, args=("q", ["a"])) for _ in range(8)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert created == ["m"]


def test_importing_the_module_does_not_import_sentence_transformers() -> None:
    code = (
        "import sys; import cag_estimate.embedding_pipeline.reranker; "
        "sys.exit(int('sentence_transformers' in sys.modules or 'torch' in sys.modules))"
    )

    result = subprocess.run([sys.executable, "-c", code], check=False)

    assert result.returncode == 0
