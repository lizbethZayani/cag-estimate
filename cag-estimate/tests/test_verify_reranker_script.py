"""Tests for scripts/verify_reranker.py, loaded as a module (scripts/ is not a package)."""

import importlib.util
import io
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from cag_estimate.embedding_pipeline.reranker import CrossEncoderReranker

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_reranker.py"


@pytest.fixture
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_reranker_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeScorer:
    """Gives the first pair's document ``first`` and the second ``second``."""

    def __init__(self, first: float, second: float) -> None:
        self.values = [first, second]

    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        return self.values[: len(pairs)]


class BrokenScorer:
    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]:
        raise RuntimeError("secret-detail")


def run(script: ModuleType, scorer: object) -> tuple[int, str, str]:
    reranker = CrossEncoderReranker("test/model", model=scorer)  # type: ignore[arg-type]
    out, err = io.StringIO(), io.StringIO()
    code = script.main([], reranker=reranker, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_check_ranking_ok_when_relevant_scores_higher(script: ModuleType) -> None:
    assert script.check_ranking(2.0, 1.0) == 0


@pytest.mark.parametrize(("relevant", "irrelevant"), [(1.0, 2.0), (1.0, 1.0)])
def test_check_ranking_wrong_when_relevant_is_not_higher(
    script: ModuleType, relevant: float, irrelevant: float
) -> None:
    assert script.check_ranking(relevant, irrelevant) == 2


def test_exit_0_and_prints_both_scores_and_the_model(script: ModuleType) -> None:
    code, out, _ = run(script, FakeScorer(3.5, -1.25))

    assert code == 0
    assert "test/model" in out
    assert "3.5000" in out
    assert "-1.2500" in out


def test_exit_2_when_the_ranking_is_wrong(script: ModuleType) -> None:
    code, _, err = run(script, FakeScorer(-1.0, 4.0))

    assert code == 2
    assert "ranking" in err


def test_exit_1_when_the_model_fails_without_leaking_the_detail(script: ModuleType) -> None:
    code, out, err = run(script, BrokenScorer())

    assert code == 1
    assert "error" in err
    assert "secret-detail" not in out + err
