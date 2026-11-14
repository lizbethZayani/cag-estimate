"""Cross-encoder reranking of recalled candidates; the model loads lazily on first use."""

import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

import structlog

log = structlog.get_logger()

T = TypeVar("T")


class PairScorer(Protocol):
    """Anything that scores ``(query, document)`` pairs (a sentence-transformers CrossEncoder)."""

    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[float]: ...


class CrossEncoderReranker:
    """Rescore candidates against the query with a cross-encoder (higher is better).

    Construction is free: the model is loaded (once, thread-safe) by the first call that
    has something to score, so importing or building this class never pulls in torch.
    """

    def __init__(self, model_name: str, *, model: PairScorer | None = None) -> None:
        self._model_name = model_name
        self._model = model
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """One score per document, in input order; no model load for an empty input."""
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        return [float(value) for value in self._load().predict(pairs)]

    def rerank(
        self,
        query: str,
        candidates: Sequence[T],
        *,
        top_n: int,
        text_of: Callable[[T], str],
    ) -> list[tuple[T, float]]:
        """The ``top_n`` candidates by score, best first; ties keep their input order."""
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        if not candidates:
            return []
        scores = self.score(query, [text_of(candidate) for candidate in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])
        return ranked[:top_n]

    def _load(self) -> PairScorer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    def _load_model(self) -> PairScorer:
        # The only import of sentence_transformers (and so torch) in the code base.
        from sentence_transformers import CrossEncoder

        started = time.perf_counter()
        model = CrossEncoder(self._model_name)
        log.info(
            "reranker_model_loaded",
            model=self._model_name,
            seconds=round(time.perf_counter() - started, 2),
        )
        return model
