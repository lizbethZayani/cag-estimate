"""Offline tests for the semantic retriever dependency builder."""

from cag_estimate import dependencies
from cag_estimate.config import Settings
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
