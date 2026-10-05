"""Lazy OpenAI vectorizer: no network at construction, static dims."""

from unittest.mock import MagicMock, patch

import pytest

from cag_estimate import dependencies
from cag_estimate.cache.vectorizer import LazyOpenAIVectorizer


def _failing_client() -> MagicMock:
    client = MagicMock()
    client.embeddings.create.side_effect = RuntimeError("no credit")
    return client


def test_construction_makes_no_network_call() -> None:
    client = _failing_client()
    LazyOpenAIVectorizer(model="text-embedding-3-small", client=client)
    client.embeddings.create.assert_not_called()


def test_default_model_dims() -> None:
    vectorizer = LazyOpenAIVectorizer(model="text-embedding-3-small", client=_failing_client())
    assert vectorizer.dims == 1536


def test_explicit_dims_override_unknown_model() -> None:
    vectorizer = LazyOpenAIVectorizer(model="custom", dims=64, client=_failing_client())
    assert vectorizer.dims == 64


def test_unknown_model_without_dims_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="embedding_dims"):
        LazyOpenAIVectorizer(model="mystery-model", client=_failing_client())


def test_embed_calls_client_lazily() -> None:
    client = MagicMock()
    client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2])]
    vectorizer = LazyOpenAIVectorizer(model="text-embedding-3-small", client=client)
    assert vectorizer.embed("hello") == [0.1, 0.2]
    client.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input="hello")


def test_embed_failure_propagates_for_cache_to_soften() -> None:
    vectorizer = LazyOpenAIVectorizer(model="text-embedding-3-small", client=_failing_client())
    with pytest.raises(RuntimeError):
        vectorizer.embed("hello")


def test_build_semantic_cache_survives_failing_embeddings(monkeypatch) -> None:
    settings = MagicMock(
        semantic_cache_enabled=True,
        openai_api_key="sk-bad",
        embedding_model="text-embedding-3-small",
        embedding_dims=None,
        redis_url="redis://localhost:1/0",
        semantic_cache_threshold=0.9,
        semantic_cache_ttl=60,
        semantic_cache_log_only=False,
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    with (
        patch.object(dependencies, "OpenAI", return_value=_failing_client()),
        patch.object(dependencies.redis, "from_url", return_value=MagicMock()),
        patch("cag_estimate.cache.semantic.SearchIndex") as index_cls,
    ):
        cache = dependencies.build_semantic_cache()
    assert cache is not None
    assert index_cls.from_dict.called


def test_lookup_returns_none_when_embedding_fails() -> None:
    from cag_estimate.cache.semantic import EstimationSemanticCache
    from cag_estimate.schemas.estimation import EstimationRequest

    vectorizer = LazyOpenAIVectorizer(model="text-embedding-3-small", client=_failing_client())
    with patch("cag_estimate.cache.semantic.SearchIndex"):
        cache = EstimationSemanticCache(
            redis_client=MagicMock(), vectorizer=vectorizer, threshold=0.9, ttl=60
        )
    request = EstimationRequest(transcription="A small blog with comments and login.")
    assert cache.lookup(request, "v1") is None
