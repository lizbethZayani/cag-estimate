"""Offline tests for the semantic retriever dependency builder."""

import sys

import pytest

from cag_estimate import dependencies
from cag_estimate.config import Settings
from cag_estimate.embedding_pipeline.reranker import CrossEncoderReranker
from cag_estimate.embedding_pipeline.retriever import SemanticRetriever

DATABASE_URL = "postgresql+asyncpg://u:p@localhost:5433/db"


def test_build_semantic_retriever_needs_a_database_url(monkeypatch):
    settings = Settings(anthropic_api_key="test", openai_api_key="sk-test", database_url=None)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "get_embedder", lambda: object())

    assert dependencies.build_semantic_retriever() is None


def test_build_semantic_retriever_needs_an_embedder(monkeypatch):
    settings = Settings(anthropic_api_key="test", openai_api_key=None, database_url=DATABASE_URL)
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "get_embedder", lambda: None)

    assert dependencies.build_semantic_retriever() is None


def test_build_semantic_retriever_is_built_when_configured(monkeypatch):
    settings = Settings(
        anthropic_api_key="test", openai_api_key="sk-test", database_url=DATABASE_URL
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.setattr(dependencies, "get_embedder", lambda: object())
    monkeypatch.setattr(dependencies, "get_async_session_factory", lambda: object())

    assert isinstance(dependencies.build_semantic_retriever(), SemanticRetriever)


@pytest.mark.parametrize("rerank_enabled", [False, True])
def test_build_reranker_ignores_the_enabled_flag_and_does_not_load(monkeypatch, rerank_enabled):
    settings = Settings(
        anthropic_api_key="test", rerank_enabled=rerank_enabled, rerank_model="org/some-model"
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    reranker = dependencies.build_reranker()

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "org/some-model"
    assert "sentence_transformers" not in sys.modules


def test_get_reranker_is_cached(monkeypatch):
    dependencies.get_reranker.cache_clear()
    settings = Settings(anthropic_api_key="test")
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)

    assert dependencies.get_reranker() is dependencies.get_reranker()
    dependencies.get_reranker.cache_clear()
