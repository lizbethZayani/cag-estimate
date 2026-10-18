"""Offline tests for the OpenAI embedder, cost helpers and cosine similarity."""

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from structlog.testing import capture_logs

from cag_estimate.embedding_pipeline.chunker import EMBEDDING_MODEL
from cag_estimate.embedding_pipeline.embedder import (
    OpenAIEmbedder,
    build_ingest_stats,
    cosine_similarity,
    estimate_cost_usd,
)
from cag_estimate.embedding_pipeline.schemas import Chunk, EmbeddedChunk

SECRET_TEXT = "super secret chunk text"


def _response(status: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return httpx.Response(status, request=request)


def rate_limit_error(code: str | None = "rate_limit_exceeded") -> openai.RateLimitError:
    body = {"message": "slow down", "type": "error", "code": code}
    return openai.RateLimitError("slow down", response=_response(429), body=body)


def nested_quota_error() -> openai.RateLimitError:
    body = {"error": {"message": "no credit", "code": "insufficient_quota"}}
    return openai.RateLimitError("no credit", response=_response(429), body=body)


def auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError("bad key", response=_response(401), body=None)


def make_chunks(count: int) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i}", text=f"{SECRET_TEXT} {i}", metadata={"i": i}, token_count=10)
        for i in range(count)
    ]


class FakeEmbeddings:
    def __init__(
        self,
        failures: list[Exception] | None = None,
        reverse: bool = False,
        drop_last: bool = False,
        usage: bool = True,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._failures = list(failures or [])
        self._reverse = reverse
        self._drop_last = drop_last
        self._usage = usage

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._failures:
            raise self._failures.pop(0)
        inputs = kwargs["input"]
        items = [
            SimpleNamespace(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(inputs)
        ]
        if self._drop_last:
            items = items[:-1]
        if self._reverse:
            items = list(reversed(items))
        usage = SimpleNamespace(total_tokens=7 * len(inputs)) if self._usage else None
        return SimpleNamespace(data=items, usage=usage)


class FakeClient:
    def __init__(self, embeddings: FakeEmbeddings) -> None:
        self.embeddings = embeddings


class Harness:
    def __init__(self, **fake_kwargs: Any) -> None:
        self.embeddings = FakeEmbeddings(**fake_kwargs)
        self.sleeps: list[float] = []
        ticks = iter(float(n) * 0.0123 for n in range(1000))
        self.embedder = OpenAIEmbedder(
            FakeClient(self.embeddings),
            sleep=self.sleeps.append,
            clock=lambda: next(ticks),
        )


def test_empty_input_makes_no_call() -> None:
    h = Harness()
    assert h.embedder.embed_many([]) == []
    assert h.embeddings.calls == []


def test_250_chunks_go_in_three_batches_keeping_order() -> None:
    h = Harness(reverse=True)
    chunks = make_chunks(250)

    result = h.embedder.embed_many(chunks)

    assert [len(c["input"]) for c in h.embeddings.calls] == [100, 100, 50]
    assert [r.chunk_id for r in result] == [c.chunk_id for c in chunks]
    assert all(isinstance(r, EmbeddedChunk) for r in result)
    assert result[0].embedding == [float(len(chunks[0].text)), 1.0]
    assert result[0].metadata == {"i": 0}
    assert result[249].embedding == [float(len(chunks[249].text)), 1.0]


def test_sends_model_and_never_dimensions() -> None:
    h = Harness()
    h.embedder.embed_many(make_chunks(2))
    call = h.embeddings.calls[0]
    assert call["model"] == EMBEDDING_MODEL
    assert "dimensions" not in call
    assert call["input"] == [c.text for c in make_chunks(2)]


def test_embed_one_returns_the_vector() -> None:
    h = Harness()
    assert h.embedder.embed_one("abcd") == [4.0, 1.0]
    assert h.embeddings.calls[0]["input"] == ["abcd"]


def test_wrong_vector_count_raises() -> None:
    h = Harness(drop_last=True)
    with pytest.raises(ValueError, match="expected 3 embeddings"):
        h.embedder.embed_many(make_chunks(3))


def test_rate_limit_twice_then_success_waits_1_then_2() -> None:
    h = Harness(failures=[rate_limit_error(), rate_limit_error()])
    result = h.embedder.embed_many(make_chunks(1))
    assert h.sleeps == [1, 2]
    assert len(h.embeddings.calls) == 3
    assert len(result) == 1


def test_rate_limit_exhausts_retries_and_raises() -> None:
    h = Harness(failures=[rate_limit_error() for _ in range(4)])
    with pytest.raises(openai.RateLimitError):
        h.embedder.embed_many(make_chunks(1))
    assert h.sleeps == [1, 2, 4]
    assert len(h.embeddings.calls) == 4


@pytest.mark.parametrize("error", [rate_limit_error("insufficient_quota"), nested_quota_error()])
def test_insufficient_quota_propagates_immediately(error: openai.RateLimitError) -> None:
    h = Harness(failures=[error])
    with pytest.raises(openai.RateLimitError):
        h.embedder.embed_many(make_chunks(1))
    assert h.sleeps == []
    assert len(h.embeddings.calls) == 1


def _error_with(body: dict[str, Any]) -> openai.RateLimitError:
    return openai.RateLimitError("no credit", response=_response(429), body=body)


NO_CREDIT_INNER = {
    "message": "You have no credits remaining.",
    "type": "insufficient_quota",
    "param": None,
    "code": "credit_balance_exhausted",
}


@pytest.mark.parametrize(
    "body",
    [
        NO_CREDIT_INNER,
        {"error": NO_CREDIT_INNER},
        {"message": "x", "type": "insufficient_quota", "code": None},
        {"message": "x", "type": "requests", "code": "credit_balance_exhausted"},
        {"error": {"message": "x", "type": "insufficient_quota"}},
        {"error": {"message": "x", "code": "credit_balance_exhausted"}},
    ],
)
def test_credit_errors_are_not_retried(body: dict[str, Any]) -> None:
    h = Harness(failures=[_error_with(body)])
    with pytest.raises(openai.RateLimitError):
        h.embedder.embed_many(make_chunks(1))
    assert h.sleeps == []
    assert len(h.embeddings.calls) == 1


def test_ordinary_429_with_requests_type_still_retries() -> None:
    body = {"message": "slow down", "type": "requests", "code": "rate_limit_exceeded"}
    h = Harness(failures=[_error_with(body) for _ in range(4)])
    with pytest.raises(openai.RateLimitError):
        h.embedder.embed_many(make_chunks(1))
    assert h.sleeps == [1, 2, 4]


@pytest.mark.parametrize("error", [auth_error(), RuntimeError("boom")])
def test_other_errors_propagate_without_retry(error: Exception) -> None:
    h = Harness(failures=[error])
    with pytest.raises(type(error)):
        h.embedder.embed_many(make_chunks(1))
    assert h.sleeps == []
    assert len(h.embeddings.calls) == 1


def test_logs_batch_with_usage_tokens_and_latency_but_no_text() -> None:
    h = Harness()
    with capture_logs() as logs:
        h.embedder.embed_many(make_chunks(3))
    sent = [e for e in logs if e["event"] == "embedding_batch_sent"]
    assert len(sent) == 1
    assert sent[0]["chunks"] == 3
    assert sent[0]["tokens"] == 21
    assert sent[0]["latency_ms"] == 12.3
    assert SECRET_TEXT not in repr(logs)


def test_tokens_fall_back_to_chunk_token_counts_without_usage() -> None:
    h = Harness(usage=False)
    with capture_logs() as logs:
        h.embedder.embed_many(make_chunks(3))
    sent = next(e for e in logs if e["event"] == "embedding_batch_sent")
    assert sent["tokens"] == 30


def test_logs_retry_and_failure_without_text() -> None:
    h = Harness(failures=[rate_limit_error() for _ in range(4)])
    with capture_logs() as logs, pytest.raises(openai.RateLimitError):
        h.embedder.embed_many(make_chunks(1))
    retries = [e for e in logs if e["event"] == "embedding_retry"]
    assert [(e["attempt"], e["wait_seconds"], e["error_type"]) for e in retries] == [
        (1, 1, "RateLimitError"),
        (2, 2, "RateLimitError"),
        (3, 4, "RateLimitError"),
    ]
    failed = [e for e in logs if e["event"] == "embedding_failed"]
    assert len(failed) == 1
    assert failed[0]["error_type"] == "RateLimitError"
    assert failed[0]["log_level"] == "warning"
    assert SECRET_TEXT not in repr(logs)


def test_estimate_cost_usd() -> None:
    assert estimate_cost_usd(1_000_000) == 0.02
    assert estimate_cost_usd(0) == 0.0
    assert estimate_cost_usd(123) == round(123 / 1_000_000 * 0.02, 8)


def test_build_ingest_stats() -> None:
    chunks = [EmbeddedChunk(**c.model_dump(), embedding=[0.1]) for c in make_chunks(4)]
    stats = build_ingest_stats(2, chunks)
    assert stats.total_budgets == 2
    assert stats.total_chunks == 4
    assert stats.total_tokens == 40
    assert stats.estimated_cost_usd == estimate_cost_usd(40)


def test_cosine_identical_orthogonal_opposite() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_known_value() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(0.70710678)


@pytest.mark.parametrize(
    ("a", "b", "message"),
    [
        ([1.0], [1.0, 2.0], "same length"),
        ([], [], "empty"),
        ([0.0, 0.0], [1.0, 1.0], "zero"),
        ([1.0, 1.0], [0.0, 0.0], "zero"),
    ],
)
def test_cosine_invalid_inputs_raise(a: list[float], b: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cosine_similarity(a, b)
