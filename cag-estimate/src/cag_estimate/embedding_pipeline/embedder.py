"""OpenAI embedder (batched, with rate-limit retries), cost helpers and cosine similarity.

Chunk texts and API keys are never logged: only counts, tokens, latency and error types.
"""

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import openai
import structlog

from cag_estimate.embedding_pipeline.chunker import EMBEDDING_MODEL
from cag_estimate.embedding_pipeline.schemas import Chunk, EmbeddedChunk

# Current price of text-embedding-3-small in USD per 1M tokens; it changes over time.
EMBEDDING_PRICE_PER_M_TOKENS_USD = 0.02
COST_DECIMALS = 8
# Error types/codes of a 429 that means "no credit": waiting never fixes it.
OUT_OF_CREDIT_MARKERS = frozenset({"insufficient_quota", "credit_balance_exhausted"})

log = structlog.get_logger()


def _is_out_of_credit(exc: openai.APIError) -> bool:
    """True when the error ``type`` or ``code`` says the account has no credit left.

    Both fields are read from the SDK attributes and from the raw body, flat or
    nested under ``error``, because the payload shape varies.
    """
    body = exc.body if isinstance(exc.body, dict) else {}
    nested = body.get("error")
    sources = [body, nested if isinstance(nested, dict) else {}]
    values = {getattr(exc, "code", None), getattr(exc, "type", None)}
    values.update(source.get(field) for source in sources for field in ("type", "code"))
    return not values.isdisjoint(OUT_OF_CREDIT_MARKERS)


class OpenAIEmbedder:
    """Embeds chunks through an OpenAI client, ``batch_size`` chunks per request.

    A ``RateLimitError`` is retried up to ``max_retries`` times, waiting 1s, 2s, 4s...
    before giving up with the last error. The one exception is an out-of-credit error
    (type or code ``insufficient_quota`` / ``credit_balance_exhausted``): retrying
    never helps, so it propagates immediately without waiting. Every other exception propagates unchanged.
    """

    def __init__(
        self,
        client: Any,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 100,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text (used for queries, not for ingestion)."""
        return self._create([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed all chunks in order; an empty list makes no API call."""
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = self._embed_batch(batch)
            embedded.extend(
                EmbeddedChunk(**chunk.model_dump(), embedding=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            )
        return embedded

    def _embed_batch(self, batch: list[Chunk]) -> list[list[float]]:
        started = self._clock()
        response = self._request([chunk.text for chunk in batch])
        latency_ms = round((self._clock() - started) * 1000, 1)
        vectors = self._ordered_vectors(response, len(batch))
        log.info(
            "embedding_batch_sent",
            chunks=len(batch),
            tokens=self._tokens(response, batch),
            latency_ms=latency_ms,
        )
        return vectors

    def _create(self, texts: list[str]) -> list[list[float]]:
        return self._ordered_vectors(self._request(texts), len(texts))

    def _request(self, texts: list[str]) -> Any:
        attempt = 0
        while True:
            try:
                return self._client.embeddings.create(model=self._model, input=texts)
            except openai.RateLimitError as exc:
                if _is_out_of_credit(exc) or attempt >= self._max_retries:
                    log.warning("embedding_failed", error_type=type(exc).__name__)
                    raise
                wait_seconds = 2**attempt
                attempt += 1
                log.info(
                    "embedding_retry",
                    attempt=attempt,
                    wait_seconds=wait_seconds,
                    error_type=type(exc).__name__,
                )
                self._sleep(wait_seconds)
            except Exception as exc:
                log.warning("embedding_failed", error_type=type(exc).__name__)
                raise

    @staticmethod
    def _ordered_vectors(response: Any, expected: int) -> list[list[float]]:
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != expected:
            raise ValueError(f"expected {expected} embeddings but the API returned {len(items)}")
        return [list(item.embedding) for item in items]

    @staticmethod
    def _tokens(response: Any, batch: list[Chunk]) -> int:
        usage = getattr(response, "usage", None)
        if usage is not None and usage.total_tokens is not None:
            return usage.total_tokens
        return sum(chunk.token_count for chunk in batch)


def estimate_cost_usd(total_tokens: int) -> float:
    """Estimated embedding cost in USD for ``total_tokens`` tokens."""
    return round(total_tokens / 1_000_000 * EMBEDDING_PRICE_PER_M_TOKENS_USD, COST_DECIMALS)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product over the product of the norms, by hand (no numpy)."""
    if not a or not b:
        raise ValueError("cannot compare empty vectors")
    if len(a) != len(b):
        raise ValueError(f"vectors must have the same length, got {len(a)} and {len(b)}")
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cannot compare a zero-norm vector")
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)
